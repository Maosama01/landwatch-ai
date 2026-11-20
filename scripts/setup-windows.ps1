# LandWatch AI - one-time setup and check on Windows 10/11.
#
# Run it from the project folder (VS Code: Terminal > Run Task > "LandWatch: Windows setup"), or:
#     powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup-windows.ps1
#
# What it does, in order:
#   1. finds a JDK 17, or installs Eclipse Temurin 17 (winget, else a zip into %USERPROFILE%\.jdks);
#   2. sets JAVA_HOME for your user account to that JDK (asks first if JAVA_HOME points elsewhere);
#   3. checks for the Microsoft Visual C++ runtime that PyTorch needs (offers to install it);
#   4. installs the VS Code "Extension Pack for Java" if the "code" command is available;
#   5. builds the project and runs the test suite with the Maven wrapper (downloads Maven,
#      the dependencies and the PyTorch native library on first use; takes several minutes).
# It never deletes anything and does not touch the project's data or results.
# -Unattended answers "no" to every question (used by the automated Windows check on GitHub).

param([switch]$Unattended)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Say($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   !!  $text" -ForegroundColor Yellow }
function Ask($question) {
    if ($Unattended) { Write-Host "   $question -> no (unattended)"; return 'n' }
    return (Read-Host $question)
}

function Get-JdkVersion($dir) {
    $release = Join-Path $dir 'release'
    if (-not (Test-Path -LiteralPath $release)) { return $null }
    $line = Get-Content -LiteralPath $release | Where-Object { $_ -like 'JAVA_VERSION=*' } | Select-Object -First 1
    if ($line) { return ($line -replace 'JAVA_VERSION=', '').Trim('"') }
    return $null
}

function Find-Jdk17 {
    $candidates = @()
    if ($env:JAVA_HOME) { $candidates += $env:JAVA_HOME }
    $patterns = @(
        'C:\Program Files\Eclipse Adoptium\jdk-17*',
        'C:\Program Files\Java\jdk-17*',
        'C:\Program Files\Microsoft\jdk-17*',
        'C:\Program Files\Zulu\zulu-17*',
        (Join-Path $env:USERPROFILE '.jdks\jdk-17*')
    )
    foreach ($p in $patterns) {
        $candidates += Get-ChildItem -Path $p -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    }
    foreach ($c in $candidates) {
        $v = Get-JdkVersion $c
        if ($v -and $v.StartsWith('17') -and (Test-Path -LiteralPath (Join-Path $c 'bin\java.exe'))) { return $c }
    }
    return $null
}

Write-Host "LandWatch AI setup in: $root"
if ($root -match 'OneDrive') {
    Warn 'The project is inside OneDrive. It works, but OneDrive will also sync the build folder (target).'
    Warn 'If anything behaves oddly, copy the folder to e.g. C:\dev\landwatch-ai and open that copy instead.'
}

# ---------------------------------------------------------------- 1. JDK 17
Say 'Java 17 (the project and its tests require Java 17)'
$jdk = Find-Jdk17
if (-not $jdk) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Host '   Installing Eclipse Temurin 17 with winget (Windows may ask for permission)...'
        $ErrorActionPreference = 'Continue'
        & winget install --id EclipseAdoptium.Temurin.17.JDK -e --silent --accept-package-agreements --accept-source-agreements
        $ErrorActionPreference = 'Stop'
        $jdk = Find-Jdk17
    }
    if (-not $jdk) {
        Write-Host '   Downloading Eclipse Temurin 17 (zip) into your user folder...'
        $dest = Join-Path $env:USERPROFILE '.jdks'
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        $zip = Join-Path $env:TEMP 'temurin17.zip'
        $url = 'https://api.adoptium.net/v3/binary/latest/17/ga/windows/x64/jdk/hotspot/normal/eclipse?project=jdk'
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
        Remove-Item -LiteralPath $zip
        $jdk = Find-Jdk17
    }
}
if (-not $jdk) { throw 'Could not find or install a JDK 17. Install "Eclipse Temurin 17 (JDK)" from https://adoptium.net and run this script again.' }
Ok "JDK 17 found: $jdk ($(Get-JdkVersion $jdk))"

$userJavaHome = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'User')
$machineJavaHome = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'Machine')
$current = if ($userJavaHome) { $userJavaHome } else { $machineJavaHome }
if ($current -ne $jdk) {
    $answer = if ($Unattended) { 'n' } else { 'y' }
    if ($current) {
        Warn "JAVA_HOME currently points to: $current"
        $answer = Ask '   Point JAVA_HOME (for your user account) to the JDK 17 above? [y/N]'
    }
    if ($answer -match '^[yY]') {
        [Environment]::SetEnvironmentVariable('JAVA_HOME', $jdk, 'User')
        Ok 'JAVA_HOME set for your user account. Restart VS Code once so it picks this up.'
    } else {
        Warn 'JAVA_HOME left unchanged; this script still uses JDK 17 for its own build below.'
    }
}
$env:JAVA_HOME = $jdk
$env:Path = (Join-Path $jdk 'bin') + ';' + $env:Path

# ---------------------------------------------------------------- 2. Visual C++ runtime for PyTorch
Say 'Microsoft Visual C++ runtime (needed by the PyTorch library behind the deep model)'
if (Test-Path (Join-Path $env:WINDIR 'System32\vcruntime140_1.dll')) {
    Ok 'present'
} else {
    Warn 'Not found. Without it the deep model cannot load (the classical engine still works).'
    $answer = Ask '   Install "Microsoft Visual C++ Redistributable 2015-2022 (x64)" now with winget? [y/N]'
    if ($answer -match '^[yY]' -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        $ErrorActionPreference = 'Continue'
        & winget install --id Microsoft.VCRedist.2015+.x64 -e --silent --accept-package-agreements --accept-source-agreements
        $ErrorActionPreference = 'Stop'
    } else {
        Warn 'Install it later from https://aka.ms/vs/17/release/vc_redist.x64.exe'
    }
}

# ---------------------------------------------------------------- 3. VS Code Java extensions
Say 'VS Code Extension Pack for Java'
if ($Unattended) {
    Write-Host '   skipped (unattended)'
} elseif (Get-Command code -ErrorAction SilentlyContinue) {
    $ErrorActionPreference = 'Continue'
    & code --install-extension vscjava.vscode-java-pack
    $ErrorActionPreference = 'Stop'
    Ok 'installed or already present'
} else {
    Warn 'The "code" command is not on the PATH. In VS Code, open Extensions and install "Extension Pack for Java".'
}

# ---------------------------------------------------------------- 4. Build and test
Say 'Build and test (first run downloads Maven, the libraries and PyTorch: several minutes)'
$ErrorActionPreference = 'Continue'
& (Join-Path $root 'mvnw.cmd') -B test
$code = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($code -ne 0) {
    Warn "The build or a test failed (exit code $code). Scroll up for the first error and send it over."
    exit $code
}
Ok 'build and tests passed (tests that need the LEVIR-CD images are skipped when the images are absent)'

Say 'Done. To start the dashboard:'
Write-Host '   VS Code: Run and Debug (Ctrl+Shift+D) > "LandWatch AI dashboard" > Start (F5)'
Write-Host '   or:      Terminal > Run Task > "LandWatch: run dashboard"'
Write-Host '   or:      .\mvnw.cmd javafx:run'
Write-Host '   In the app, click "Demo pair" and then "Analyze changes".'
