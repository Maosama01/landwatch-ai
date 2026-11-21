# LandWatch AI - train one change-detection model on an NVIDIA GPU under Windows 10/11.
#
# Run it from the project folder after scripts\setup-windows.ps1 has succeeded once:
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train-gpu-windows.ps1 -Job check
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train-gpu-windows.ps1 -Job transformer
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\train-gpu-windows.ps1 -Job no-augmentation
#
# Jobs:
#   check            ~30 training steps on 240 patches, to confirm the GPU works and see the speed.
#                    The first run downloads the CUDA 12.8 build of PyTorch (about 2-3 GB) into
#                    %USERPROFILE%\.djl.ai, so it can take a while.
#   transformer      FC-Siam-diff + Transformer, 13 epochs  -> models\siamese_unet_levir_transformer
#   no-augmentation  ablation: FC-Siam-diff without augmentation -> models\siamese_unet_levir_no_aug
#   no-dice          ablation: FC-Siam-diff with weighted BCE only -> models\siamese_unet_levir_no_dice
#
# The training data are the LEVIR-CD 256x256 tiles (folders train, val and metadata of processed_256).
# The script looks for them in -Data, then datasets\LEVIR-CD\processed_256, then C:\landwatch-data\processed_256.
# Keeping them outside OneDrive (C:\landwatch-data) stops OneDrive from uploading 1.9 GB of images.
#
# Training uses the TRAIN and VAL splits only; the TEST split is never read here. At the end the script
# writes send_back_<job>.zip (model weights + logs, a few MB) to the project folder.
# It never deletes anything except its own check output under target\gpu-check.

param(
    [ValidateSet('check', 'transformer', 'no-augmentation', 'no-dice')]
    [string]$Job = 'check',
    [string]$Data = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Say($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   !!  $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "   XX  $text" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1. NVIDIA GPU
Say 'NVIDIA GPU'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    Fail 'nvidia-smi was not found, so no NVIDIA driver is installed. Install the latest driver from nvidia.com (or the NVIDIA App) and run this again.'
}
$gpu = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Select-Object -First 1)
Ok "GPU: $gpu"
$cudaLine = (& nvidia-smi | Select-String -Pattern 'CUDA Version:\s*([0-9.]+)' | Select-Object -First 1)
if ($cudaLine) {
    $cuda = [version]$cudaLine.Matches[0].Groups[1].Value
    if ($cuda -lt [version]'12.8') {
        Warn "The driver supports CUDA $cuda. PyTorch's CUDA 12.8 build may need a newer driver;"
        Warn 'if the check fails with a CUDA error, update the NVIDIA driver and try again.'
    } else {
        Ok "Driver supports CUDA $cuda"
    }
}

# ---------------------------------------------------------------- 2. Java 17
Say 'Java 17'
if (-not $env:JAVA_HOME -or -not (Test-Path -LiteralPath (Join-Path $env:JAVA_HOME 'bin\java.exe'))) {
    Fail 'JAVA_HOME is not set. Run scripts\setup-windows.ps1 first, then open a new terminal.'
}
$java = Join-Path $env:JAVA_HOME 'bin\java.exe'
$release = Join-Path $env:JAVA_HOME 'release'
$version = (Get-Content -LiteralPath $release | Where-Object { $_ -like 'JAVA_VERSION=*' }) -replace 'JAVA_VERSION=', ''
if (-not $version.Trim('"').StartsWith('17')) { Fail "JAVA_HOME points to Java $version; Java 17 is required." }
Ok "Java $($version.Trim('"')) at JAVA_HOME"

# ---------------------------------------------------------------- 3. Training tiles
Say 'LEVIR-CD training tiles'
$candidates = @()
if ($Data) { $candidates += $Data }
$candidates += 'datasets\LEVIR-CD\processed_256', 'C:\landwatch-data\processed_256'
$dataDir = $null
foreach ($c in $candidates) {
    if ((Test-Path -LiteralPath (Join-Path $c 'metadata\train_patches.csv')) -and (Test-Path -LiteralPath (Join-Path $c 'train\A'))) {
        $dataDir = $c; break
    }
}
if (-not $dataDir) {
    Fail ('Tiles not found. Unzip LEVIR-CD_train_val_tiles.zip so that C:\landwatch-data\processed_256\train\A exists, ' +
          'or pass -Data <folder that contains train, val and metadata>.')
}
foreach ($sub in 'train\A', 'train\B', 'train\label', 'val\A', 'val\B', 'val\label') {
    if (-not (Test-Path -LiteralPath (Join-Path $dataDir $sub))) { Fail "Missing folder $sub in $dataDir" }
}
$nTrain = (Get-ChildItem -LiteralPath (Join-Path $dataDir 'train\A') -File).Count
$nVal = (Get-ChildItem -LiteralPath (Join-Path $dataDir 'val\A') -File).Count
Ok "Tiles in $dataDir : train $nTrain, val $nVal (expected 7120 and 1024)"
if ($nTrain -ne 7120 -or $nVal -ne 1024) { Warn 'Unexpected tile counts: was the zip fully extracted?' }

# ---------------------------------------------------------------- 4. Build and classpath
Say 'Building (Maven wrapper)'
& .\mvnw.cmd -B -q -DskipTests compile
if ($LASTEXITCODE -ne 0) { Fail 'Build failed.' }
& .\mvnw.cmd -B -q dependency:build-classpath '-Dmdep.outputFile=target\classpath.txt' '-Dmdep.includeScope=runtime'
if ($LASTEXITCODE -ne 0) { Fail 'Could not compute the classpath.' }
$cp = 'target\classes;' + (Get-Content -LiteralPath 'target\classpath.txt' -Raw).Trim()
Ok 'Built'

# ---------------------------------------------------------------- 5. Train
$env:PYTORCH_FLAVOR = 'cu128'   # DJL downloads the CUDA 12.8 build of PyTorch 2.7.1 instead of the CPU build
$common = @('--data', $dataDir, '--batch', '8', '--lr', '1e-3', '--device', 'gpu', '--workers', '4', '--seed', '42', '--pos-weight', 'auto')
switch ($Job) {
    'check' {
        $name = 'gpu-check'
        $out = 'target\gpu-check\model'; $res = 'target\gpu-check\results'
        if (Test-Path -LiteralPath 'target\gpu-check') { Remove-Item -LiteralPath 'target\gpu-check' -Recurse -Force }
        $jobArgs = @('--epochs', '1', '--train-limit', '240', '--val-limit', '32', '--max-minutes', '60')
    }
    'transformer' {
        $name = 'siamese_unet_levir_transformer'; $out = "models\$name"; $res = 'results\deeplearning_transformer'
        $jobArgs = @('--epochs', '13', '--max-minutes', '900', '--arch', 'fc-siam-diff-transformer')
    }
    'no-augmentation' {
        $name = 'siamese_unet_levir_no_aug'; $out = "models\$name"; $res = 'results\deeplearning_no_aug'
        $jobArgs = @('--epochs', '13', '--max-minutes', '900', '--augment', 'false')
    }
    'no-dice' {
        $name = 'siamese_unet_levir_no_dice'; $out = "models\$name"; $res = 'results\deeplearning_no_dice'
        $jobArgs = @('--epochs', '13', '--max-minutes', '900', '--dice-weight', '0')
    }
}
if ($Job -ne 'check' -and (Test-Path -LiteralPath (Join-Path $out 'model.properties'))) {
    Fail "$out already holds a trained model. Move it away first if you really want to train it again."
}
New-Item -ItemType Directory -Force -Path $res | Out-Null
$log = Join-Path $res 'training_console.log'
Say "Training job '$Job' on the GPU (log: $log)"
Write-Host '   Progress is printed every 100 steps and after every epoch. Keep the laptop plugged in and awake.'

$ErrorActionPreference = 'Continue'   # Java writes log lines to stderr; they are not errors
& $java '-Xmx3g' '-Dfile.encoding=UTF-8' -cp $cp com.landwatch.deeplearning.TrainChangeDetector `
    --out $out --results $res @common @jobArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
$exit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($exit -ne 0) { Fail "Training stopped with exit code $exit. Send the log $log." }

if ($Job -eq 'check') {
    $line = Select-String -LiteralPath $log -Pattern '(\d+) s train' | Select-Object -Last 1
    Ok 'The GPU works.'
    if ($line) {
        $sec = [double]$line.Matches[0].Groups[1].Value
        Write-Host ("   30 steps took {0:N0} s ({1:N2} s per step, first steps include start-up)." -f $sec, ($sec / 30))
        Write-Host ("   A full 13-epoch run is about 11,570 steps: roughly {0:N0} minutes at this speed." -f ($sec / 30 * 11570 / 60))
    }
    Write-Host '   Next: run this script with -Job transformer'
    exit 0
}

Say 'Packing the results to send back'
$zip = "send_back_$Job.zip"
if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
Compress-Archive -Path $out, $res -DestinationPath $zip
Ok "Wrote $zip - send this file back. It holds the weights and logs only (no images)."
