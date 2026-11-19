package com.landwatch.evaluation;

import com.landwatch.geospatial.BaselineChangeDetector;
import com.landwatch.geospatial.ChangeDetectionResult;

import java.io.IOException;
import java.io.PrintStream;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;

/**
 * Benchmarks the classical {@link BaselineChangeDetector} on a LEVIR-CD split.
 *
 * <p>Every full-resolution A/B pair of the split is run through
 * {@link BaselineChangeDetector#analyze(java.awt.image.BufferedImage, java.awt.image.BufferedImage)},
 * the overload with the detector's default parameters. Predictions are compared with the
 * binarised ground truth (label &gt; 0) in one global {@link BinaryChangeMetrics} confusion
 * matrix. Per-image metrics and latency are recorded as well.</p>
 *
 * <p>Usage (after {@code mvn -B compile}):</p>
 * <pre>
 * java -cp target/classes com.landwatch.evaluation.BaselineBenchmark [split] [datasetDir] [outputDir]
 * </pre>
 * Defaults: split {@code test}, dataset {@code datasets/LEVIR-CD} and output
 * {@code results/benchmarks}, both relative to the working directory.
 *
 * <p>Outputs: {@code baseline_<split>_metrics.txt} and {@code baseline_<split>_per_image.csv}.</p>
 */
public final class BaselineBenchmark {

    public static final String DEFAULT_SPLIT = "test";
    public static final String DEFAULT_DATASET_DIR = "datasets/LEVIR-CD";
    public static final String DEFAULT_OUTPUT_DIR = "results/benchmarks";

    static final String USAGE = "Usage: java -cp target/classes " + BaselineBenchmark.class.getName()
            + " [split=" + DEFAULT_SPLIT + "] [datasetDir=" + DEFAULT_DATASET_DIR + "] [outputDir=" + DEFAULT_OUTPUT_DIR + "]";

    static final String CSV_HEADER = "image,width,height,true_change_px,pred_change_px,tp,fp,fn,tn,"
            + "precision,recall,f1,iou,otsu_threshold,analyze_ms,detector_reported_ms,load_ms";

    private static final Pattern SPLIT_NAME = Pattern.compile("[A-Za-z0-9_-]+");

    private BaselineBenchmark() {
    }

    /** Measurements for one image pair. */
    public record ImageResult(String name, int width, int height, BinaryChangeMetrics metrics,
                              int otsuThreshold, double analyzeMs, long detectorReportedMs, double loadMs) {
        public long trueChangePixels() {
            return metrics.getTruePositives() + metrics.getFalseNegatives();
        }

        public long predictedChangePixels() {
            return metrics.getTruePositives() + metrics.getFalsePositives();
        }
    }

    /** Mean, median, minimum and maximum of a set of measurements. */
    public record Stats(double mean, double median, double min, double max) {
        public static Stats of(double[] values) {
            if (values.length == 0) {
                return new Stats(0, 0, 0, 0);
            }
            double[] sorted = values.clone();
            Arrays.sort(sorted);
            int n = sorted.length;
            double median = (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
            return new Stats(Arrays.stream(sorted).average().orElse(0), median, sorted[0], sorted[n - 1]);
        }

        String format() {
            return String.format(Locale.ROOT, "mean %.1f | median %.1f | min %.1f | max %.1f", mean, median, min, max);
        }
    }

    /** The outcome of one benchmark run. */
    public record Result(String split, Path datasetDir, List<ImageResult> images, BinaryChangeMetrics global,
                         ZonedDateTime startedAt, double totalSeconds,
                         double loadAverageAtStart, double loadAverageAtEnd) {

        public Stats analyzeLatency() {
            return Stats.of(images.stream().mapToDouble(ImageResult::analyzeMs).toArray());
        }

        public Stats detectorReportedLatency() {
            return Stats.of(images.stream().mapToDouble(ImageResult::detectorReportedMs).toArray());
        }

        public Stats loadLatency() {
            return Stats.of(images.stream().mapToDouble(ImageResult::loadMs).toArray());
        }
    }

    public static void main(String[] args) {
        int code = runCli(args, System.out, System.err);
        if (code != 0) {
            System.exit(code);
        }
    }

    /** CLI entry point without {@code System.exit}; returns the process exit code. */
    static int runCli(String[] args, PrintStream out, PrintStream err) {
        if (args.length > 0 && ("-h".equals(args[0]) || "--help".equals(args[0]))) {
            out.println(USAGE);
            return 0;
        }
        if (args.length > 3) {
            err.println("Too many arguments." + System.lineSeparator() + USAGE);
            return 2;
        }
        String split = args.length > 0 ? args[0] : DEFAULT_SPLIT;
        Path datasetDir = Paths.get(args.length > 1 ? args[1] : DEFAULT_DATASET_DIR).toAbsolutePath().normalize();
        Path outputDir = Paths.get(args.length > 2 ? args[2] : DEFAULT_OUTPUT_DIR).toAbsolutePath().normalize();
        try {
            Result result = evaluate(split, datasetDir, out);
            List<Path> written = writeOutputs(result, outputDir);
            out.println();
            out.print(result.global().toReport());
            out.println("Analyze latency (ms): " + result.analyzeLatency().format());
            for (Path p : written) {
                out.println("Wrote " + p);
            }
            return 0;
        } catch (IOException | RuntimeException e) {
            err.println("ERROR: " + e.getMessage());
            return 1;
        }
    }

    /**
     * Runs the default-parameter baseline on every triplet of {@code split}.
     *
     * @param progress where to print one progress line per image (may be null)
     */
    public static Result evaluate(String split, Path datasetDir, PrintStream progress) throws IOException {
        if (split == null || !SPLIT_NAME.matcher(split).matches()) {
            throw new IllegalArgumentException("Invalid split name '" + split + "' (letters, digits, '_' and '-' only)");
        }
        LevirCdPairReader reader = new LevirCdPairReader(datasetDir);
        List<LevirCdPairReader.Triplet> triplets = reader.listTriplets(split);
        if (triplets.isEmpty()) {
            throw new IOException("No image triplets found in split '" + split + "' under " + datasetDir);
        }

        ZonedDateTime startedAt = ZonedDateTime.now();
        double loadAverageAtStart = systemLoadAverage();
        long runStart = System.nanoTime();
        BinaryChangeMetrics global = new BinaryChangeMetrics();
        List<ImageResult> images = new ArrayList<>(triplets.size());

        for (int i = 0; i < triplets.size(); i++) {
            LevirCdPairReader.Triplet triplet = triplets.get(i);

            long loadStart = System.nanoTime();
            LevirCdPairReader.Sample sample = reader.load(triplet);
            long analyzeStart = System.nanoTime();
            ChangeDetectionResult detection = BaselineChangeDetector.analyze(sample.pre(), sample.post());
            long analyzeEnd = System.nanoTime();

            BinaryChangeMetrics perImage = new BinaryChangeMetrics();
            perImage.accumulate(detection.getBinaryMask(), sample.changeMask());
            global.merge(perImage);

            long predicted = perImage.getTruePositives() + perImage.getFalsePositives();
            if (predicted != detection.getFilteredChangedPixels()) {
                throw new IllegalStateException("Internal inconsistency for " + triplet.name() + ": mask has "
                        + predicted + " change pixels but detector reported " + detection.getFilteredChangedPixels());
            }

            ImageResult row = new ImageResult(triplet.name(), sample.width(), sample.height(), perImage,
                    detection.getOtsuThreshold(), (analyzeEnd - analyzeStart) / 1e6,
                    detection.getExecutionTimeMs(), (analyzeStart - loadStart) / 1e6);
            images.add(row);

            if (progress != null) {
                progress.printf(Locale.ROOT, "[%d/%d] %s  F1=%.4f  IoU=%.4f  otsu=%d  analyze=%.1f ms%n",
                        i + 1, triplets.size(), triplet.name(), perImage.f1(), perImage.iou(),
                        row.otsuThreshold(), row.analyzeMs());
            }
        }

        double totalSeconds = (System.nanoTime() - runStart) / 1e9;
        return new Result(split, datasetDir, Collections.unmodifiableList(images), global, startedAt, totalSeconds,
                loadAverageAtStart, systemLoadAverage());
    }

    public static String metricsFileName(String split) {
        return "baseline_" + split + "_metrics.txt";
    }

    public static String perImageFileName(String split) {
        return "baseline_" + split + "_per_image.csv";
    }

    /** Writes the metrics report and the per-image CSV; returns the two paths. */
    public static List<Path> writeOutputs(Result result, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);
        Path csv = outputDir.resolve(perImageFileName(result.split()));
        Path txt = outputDir.resolve(metricsFileName(result.split()));
        Files.writeString(csv, formatCsv(result), StandardCharsets.UTF_8);
        Files.writeString(txt, formatReport(result, csv.getFileName().toString()), StandardCharsets.UTF_8);
        return List.of(txt, csv);
    }

    static String formatCsv(Result result) {
        StringBuilder sb = new StringBuilder();
        sb.append(CSV_HEADER).append('\n');
        for (ImageResult r : result.images()) {
            BinaryChangeMetrics m = r.metrics();
            sb.append(String.format(Locale.ROOT,
                    "%s,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%.3f,%d,%.3f\n",
                    r.name(), r.width(), r.height(), r.trueChangePixels(), r.predictedChangePixels(),
                    m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives(), m.getTrueNegatives(),
                    m.precision(), m.recall(), m.f1(), m.iou(), r.otsuThreshold(),
                    r.analyzeMs(), r.detectorReportedMs(), r.loadMs()));
        }
        return sb.toString();
    }

    static String formatReport(Result result, String csvFileName) {
        BinaryChangeMetrics g = result.global();
        List<ImageResult> images = result.images();
        long total = g.getTotal();
        long trueChange = g.getTruePositives() + g.getFalseNegatives();
        long predictedChange = g.getTruePositives() + g.getFalsePositives();
        long noTruthImages = images.stream().filter(r -> r.trueChangePixels() == 0).count();
        long emptyPredictionImages = images.stream().filter(r -> r.predictedChangePixels() == 0).count();
        Set<String> sizes = new TreeSet<>();
        images.forEach(r -> sizes.add(r.width() + "x" + r.height()));

        String nl = System.lineSeparator();
        String rule = "--------------------------------------------------------------------------------";
        StringBuilder sb = new StringBuilder();
        sb.append("================================================================================").append(nl);
        sb.append(" LANDWATCH AI - Classical baseline change detection on LEVIR-CD").append(nl);
        sb.append("================================================================================").append(nl);
        sb.append("Dataset        : LEVIR-CD (").append(result.datasetDir()).append(')').append(nl);
        sb.append("Split          : ").append(result.split()).append(nl);
        sb.append("Images         : ").append(images.size()).append(" image pairs, full resolution (")
                .append(String.join(", ", sizes)).append("), no tiling, no resizing").append(nl);
        sb.append("Method         : com.landwatch.geospatial.BaselineChangeDetector.analyze(pre, post)").append(nl);
        sb.append("                 classical pixel differencing, no learning, no training data used").append(nl);
        sb.append("Pipeline       : BT.601 luminance -> |post - pre| -> per-image Otsu threshold").append(nl);
        sb.append("                 -> 3x3 morphological opening -> 8-connected components,").append(nl);
        sb.append("                 components smaller than minRegionSize pixels removed").append(nl);
        sb.append("Parameters     : minRegionSize = ").append(BaselineChangeDetector.DEFAULT_MIN_REGION_SIZE)
                .append(" (DEFAULT_MIN_REGION_SIZE, untouched default; not tuned)").append(nl);
        sb.append("                 Otsu threshold chosen automatically for each image").append(nl);
        sb.append("Ground truth   : label pixel value > 0 = change (also catches stray 156/254 values)").append(nl);
        sb.append("Protocol       : one global pixel confusion matrix over all images,").append(nl);
        sb.append("                 positive class = change (BinaryChangeMetrics)").append(nl);
        sb.append("Per-image data : ").append(csvFileName).append(nl);
        sb.append(nl);

        sb.append("Global metrics (").append(result.split()).append(" split)").append(nl).append(rule).append(nl);
        sb.append(g.toReport());
        sb.append(nl);
        sb.append(String.format(Locale.ROOT, "True change pixels      : %d (%.4f%% of all pixels)%n",
                trueChange, pct(trueChange, total)));
        sb.append(String.format(Locale.ROOT, "Predicted change pixels : %d (%.4f%% of all pixels)%n",
                predictedChange, pct(predictedChange, total)));
        sb.append(String.format(Locale.ROOT, "Images with no true change pixels      : %d of %d%n",
                noTruthImages, images.size()));
        sb.append(String.format(Locale.ROOT, "Images with no predicted change pixels : %d of %d%n",
                emptyPredictionImages, images.size()));
        sb.append(nl);

        sb.append("Latency per image pair, milliseconds (").append(images.size()).append(" images)").append(nl)
                .append(rule).append(nl);
        sb.append("analyze() wall clock      : ").append(result.analyzeLatency().format()).append(nl);
        sb.append("  System.nanoTime around BaselineChangeDetector.analyze(pre, post); includes the").append(nl);
        sb.append("  binary and overlay visualisation images that analyze() always builds; excludes").append(nl);
        sb.append("  PNG decoding and metric computation.").append(nl);
        sb.append("detector-reported time    : ").append(result.detectorReportedLatency().format()).append(nl);
        sb.append("  ChangeDetectionResult.getExecutionTimeMs() (System.currentTimeMillis, 1 ms").append(nl);
        sb.append("  resolution); covers the detection steps, not the visualisation images.").append(nl);
        sb.append("PNG loading (A + B + label): ").append(result.loadLatency().format()).append(nl);
        sb.append(String.format(Locale.ROOT, "Total benchmark wall time : %.1f s%n", result.totalSeconds()));
        sb.append("  Single thread, images processed sequentially. No warm-up runs were excluded, so the").append(nl);
        sb.append("  first images include JVM/JIT warm-up; the median is the most robust figure.").append(nl);
        sb.append(String.format(Locale.ROOT, "System load average (1 min) at start / end: %s / %s on %d cores;%n",
                formatLoad(result.loadAverageAtStart()), formatLoad(result.loadAverageAtEnd()),
                Runtime.getRuntime().availableProcessors()));
        sb.append("  a load well above 1 means other processes shared the CPU, so latency is indicative only.").append(nl);
        sb.append(nl);

        Runtime rt = Runtime.getRuntime();
        sb.append("Environment").append(nl).append(rule).append(nl);
        sb.append("Timestamp (run start) : ")
                .append(result.startedAt().truncatedTo(ChronoUnit.SECONDS).format(DateTimeFormatter.ISO_OFFSET_DATE_TIME))
                .append(nl);
        sb.append("Java                  : ").append(System.getProperty("java.version")).append(" (")
                .append(System.getProperty("java.vendor")).append(", ").append(System.getProperty("java.vm.name"))
                .append(')').append(nl);
        sb.append("OS                    : ").append(System.getProperty("os.name")).append(' ')
                .append(System.getProperty("os.version")).append(", os.arch=").append(System.getProperty("os.arch"))
                .append(nl);
        sb.append("CPU cores visible     : ").append(rt.availableProcessors()).append(" (benchmark uses 1)").append(nl);
        sb.append("Max JVM heap          : ").append(rt.maxMemory() / (1024 * 1024)).append(" MB").append(nl);
        sb.append("Reproduce             : java -cp target/classes ").append(BaselineBenchmark.class.getName())
                .append(' ').append(result.split()).append(' ').append(result.datasetDir()).append(" <outputDir>").append(nl);
        sb.append(nl);

        sb.append("Limitations (read before comparing with a learned model)").append(nl).append(rule).append(nl);
        sb.append("* Pixel differencing reacts to ANY brightness difference, not only to real change.").append(nl);
        sb.append("  Illumination and sun-angle differences, shadows, seasonal vegetation, haze and").append(nl);
        sb.append("  small registration offsets between A and B all produce false positives.").append(nl);
        sb.append("* LEVIR-CD labels only building changes (construction and demolition). Genuine").append(nl);
        sb.append("  non-building changes (roads, bare soil, vegetation, water, vehicles) are scored").append(nl);
        sb.append("  as false positives under this protocol.").append(nl);
        sb.append("* Only luminance is compared: a new structure whose brightness is close to what was").append(nl);
        sb.append("  there before is missed (false negatives), whatever its colour or texture.").append(nl);
        sb.append("* Otsu always splits the difference histogram into two classes, so even an image").append(nl);
        sb.append("  with little or no real change gets its most-different pixels flagged.").append(nl);
        sb.append("* Metrics come from one global confusion matrix, so images with many pixels of").append(nl);
        sb.append("  change weigh more. In the per-image CSV, precision/recall/F1/IoU are 0.0 when").append(nl);
        sb.append("  their denominator is 0 (BinaryChangeMetrics convention), e.g. F1 = 0.0 for an").append(nl);
        sb.append("  image with no true and no predicted change.").append(nl);
        sb.append("* Latency depends on the machine and JVM; treat it as indicative only.").append(nl);
        return sb.toString();
    }

    /** 1-minute system load average, or a negative value when the platform does not report it. */
    private static double systemLoadAverage() {
        return ManagementFactory.getOperatingSystemMXBean().getSystemLoadAverage();
    }

    private static String formatLoad(double load) {
        return load < 0 ? "n/a" : String.format(Locale.ROOT, "%.2f", load);
    }

    private static double pct(long part, long total) {
        return total == 0 ? 0.0 : 100.0 * part / total;
    }
}
