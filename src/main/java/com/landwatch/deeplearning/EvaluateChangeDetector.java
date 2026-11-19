package com.landwatch.deeplearning;

import com.landwatch.evaluation.BinaryChangeMetrics;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Evaluates a trained model on one LEVIR-CD split through the public {@link DeepChangeDetector} API
 * (exactly what the dashboard uses), and writes {@code results/deeplearning/<split>_metrics.txt}.
 *
 * <p>Usage: {@code EvaluateChangeDetector <val|test|train> [--model models/siamese_unet_levir]
 * [--data <processed_256>] [--results results/deeplearning] [--device auto] [--limit 0]
 * [--cpu-latency-patches 50] [--force false]}</p>
 *
 * <p>Pixel metrics are accumulated with {@link BinaryChangeMetrics#accumulate} over every pixel of
 * every patch, using the decision threshold stored with the model (tuned on VAL during training).
 * The test split is meant to be evaluated once, with the final model only: if
 * {@code test_metrics.txt} already exists the program refuses to overwrite it unless
 * {@code --force true} is given.</p>
 */
public final class EvaluateChangeDetector {

    private static final int WARMUP_CALLS = 5;

    private EvaluateChangeDetector() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args[0].startsWith("--")) {
            System.err.println("Usage: EvaluateChangeDetector <val|test|train> [--model dir] [--data dir] [--results dir]"
                    + " [--device auto|cpu|mps] [--limit N] [--cpu-latency-patches N] [--force true]");
            System.exit(2);
        }
        String split = args[0];
        Map<String, String> opt = TrainChangeDetector.parseArgs(Arrays.copyOfRange(args, 1, args.length));
        Path modelDir = Path.of(opt.getOrDefault("model", "models/siamese_unet_levir"));
        Path data = opt.containsKey("data") ? Path.of(opt.get("data")) : LevirCdPatch.locateDatasetRoot();
        if (data == null) {
            throw new IllegalArgumentException("Dataset not found; pass --data <.../LEVIR-CD/processed_256>");
        }
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/deeplearning"));
        int limit = Integer.parseInt(opt.getOrDefault("limit", "0"));
        int cpuLatencyPatches = Integer.parseInt(opt.getOrDefault("cpu-latency-patches", "50"));
        boolean force = Boolean.parseBoolean(opt.getOrDefault("force", "false"));
        Path report = resultsDir.resolve(split + "_metrics.txt");
        if ("test".equals(split) && Files.exists(report) && !force) {
            System.err.println("Refusing to re-evaluate TEST: " + report + " already exists (use --force true deliberately).");
            System.exit(3);
        }

        List<LevirCdPatch> patches = LevirCdPatch.readSplit(data, split);
        if (limit > 0 && limit < patches.size()) {
            patches = patches.subList(0, limit);
        }

        BinaryChangeMetrics metrics = new BinaryChangeMetrics();
        BinaryChangeMetrics metricsAt05 = new BinaryChangeMetrics();
        long[] latencyNs = new long[patches.size()];
        String deviceName;
        float threshold;
        String description;
        long start = System.nanoTime();
        try (DeepChangeDetector detector = DeepChangeDetector.load(modelDir, opt.getOrDefault("device", "auto"))) {
            threshold = detector.getThreshold();
            deviceName = detector.getDevice().toString();
            System.out.println("Evaluating " + split + " (" + patches.size() + " patches) on " + deviceName
                    + " with threshold " + threshold);
            for (int i = 0; i < patches.size(); i++) {
                LevirCdPatch p = patches.get(i);
                BufferedImage a = ImageTensors.read(p.imageA());
                BufferedImage b = ImageTensors.read(p.imageB());
                BufferedImage label = ImageTensors.read(p.label());
                long t0 = System.nanoTime();
                float[][] prob = detector.predictProbabilities(a, b);
                boolean[][] mask = DeepChangeDetector.applyThreshold(prob, threshold); // == predictMask(a, b)
                latencyNs[i] = System.nanoTime() - t0;
                boolean[][] truth = ImageTensors.toBooleanMask(ImageTensors.toBinaryMask(label),
                        label.getWidth(), label.getHeight());
                metrics.accumulate(mask, truth);
                metricsAt05.accumulate(DeepChangeDetector.applyThreshold(prob, 0.5f), truth);
                if ((i + 1) % 256 == 0) {
                    System.out.printf(Locale.ROOT, "  %d/%d  running F1 %.4f%n", i + 1, patches.size(), metrics.f1());
                }
            }
            description = detector.describe();
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        String latency = latencySummary(latencyNs, deviceName);

        String cpuLatency = "";
        if (cpuLatencyPatches > 0 && !"cpu()".equals(deviceName)) {
            cpuLatency = measureCpuLatency(modelDir, patches, cpuLatencyPatches);
        }

        StringBuilder sb = new StringBuilder();
        sb.append("LandWatch AI - FC-Siam-diff change detection, evaluation on LEVIR-CD ").append(split.toUpperCase(Locale.ROOT))
                .append(" split\n");
        sb.append("Generated: ").append(LocalDateTime.now()).append('\n');
        sb.append("Model: ").append(modelDir.toAbsolutePath()).append('\n');
        sb.append("Patches: ").append(patches.size()).append(" of 256x256 (").append(data.resolve("metadata/" + split + "_patches.csv")).append(")\n");
        sb.append("Decision threshold: ").append(threshold).append(" (tuned on VAL during training)\n");
        sb.append("Inference device: ").append(deviceName).append(", ").append(DjlRuntime.describe()).append('\n');
        sb.append("Metrics: global pixel confusion matrix over all patches (com.landwatch.evaluation.BinaryChangeMetrics)\n\n");
        sb.append(metrics.toReport());
        sb.append(String.format(Locale.ROOT, "%nFor reference only, at the untuned threshold 0.50: precision %.4f, recall %.4f, F1 %.4f, IoU %.4f%n",
                metricsAt05.precision(), metricsAt05.recall(), metricsAt05.f1(), metricsAt05.iou()));
        sb.append('\n').append(latency);
        sb.append(cpuLatency);
        sb.append(String.format(Locale.ROOT, "Total evaluation wall time (incl. PNG decoding): %.1f s%n", seconds));
        Files.createDirectories(resultsDir);
        Files.writeString(report, sb.toString(), StandardCharsets.UTF_8);
        System.out.println(sb);
        System.out.println("Wrote " + report.toAbsolutePath());

        if (limit <= 0) {
            ModelConfig config = ModelConfig.load(modelDir);
            TrainChangeDetector.putMetrics(config, "metrics." + split, metrics);
            config.set("metrics." + split + ".patches", patches.size());
            config.set("metrics." + split + ".source", "results/deeplearning/" + split + "_metrics.txt");
            config.save(modelDir);
        }
        System.out.println();
        System.out.println(description);
    }

    private static String measureCpuLatency(Path modelDir, List<LevirCdPatch> patches, int n) throws IOException {
        int count = Math.min(n, patches.size());
        long[] ns = new long[count];
        try (DeepChangeDetector cpu = DeepChangeDetector.load(modelDir, "cpu")) {
            for (int i = 0; i < count; i++) {
                BufferedImage a = ImageTensors.read(patches.get(i).imageA());
                BufferedImage b = ImageTensors.read(patches.get(i).imageB());
                long t0 = System.nanoTime();
                cpu.predictMask(a, b);
                ns[i] = System.nanoTime() - t0;
            }
        }
        return latencySummary(ns, "cpu() [separate run on the first " + count + " patches, for machines without MPS]");
    }

    private static String latencySummary(long[] ns, String device) {
        int skip = Math.min(WARMUP_CALLS, Math.max(0, ns.length - 1));
        long[] timed = Arrays.copyOfRange(ns, skip, ns.length);
        Arrays.sort(timed);
        double mean = Arrays.stream(timed).average().orElse(0) / 1e6;
        double median = timed.length == 0 ? 0 : timed[timed.length / 2] / 1e6;
        double p95 = timed.length == 0 ? 0 : timed[Math.min(timed.length - 1, (int) Math.ceil(timed.length * 0.95) - 1)] / 1e6;
        return String.format(Locale.ROOT,
                "Inference latency per 256x256 patch on %s (predictProbabilities + threshold, batch of 1, "
                        + "excluding %d warm-up calls, n=%d): mean %.1f ms, median %.1f ms, p95 %.1f ms%n",
                device, skip, timed.length, mean, median, p95);
    }
}
