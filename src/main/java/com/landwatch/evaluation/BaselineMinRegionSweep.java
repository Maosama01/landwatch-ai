package com.landwatch.evaluation;

import com.landwatch.geospatial.BaselineChangeDetector;
import com.landwatch.geospatial.ChangeDetectionResult;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Exploratory sweep of the baseline's {@code minRegionSize} parameter.
 *
 * <p><b>Validation data only.</b> The sweep refuses to run on the {@code test} split, so the
 * test-split benchmark ({@link BaselineBenchmark}) always reflects the untouched default
 * parameters. Each image is loaded once and analysed with every value in the sweep.</p>
 *
 * <pre>
 * java -cp target/classes com.landwatch.evaluation.BaselineMinRegionSweep [split=val] [datasetDir] [outputDir] [sizes=1,12,25,...]
 * </pre>
 *
 * <p>Outputs: {@code baseline_<split>_minregion_sweep.txt} and {@code .csv}.</p>
 */
public final class BaselineMinRegionSweep {

    public static final String DEFAULT_SPLIT = "val";
    public static final int[] DEFAULT_SIZES = {1, 12, 25, 50, 100, 200, 400, 800};
    static final String FORBIDDEN_SPLIT = "test";

    static final String USAGE = "Usage: java -cp target/classes " + BaselineMinRegionSweep.class.getName()
            + " [split=" + DEFAULT_SPLIT + "] [datasetDir=" + BaselineBenchmark.DEFAULT_DATASET_DIR
            + "] [outputDir=" + BaselineBenchmark.DEFAULT_OUTPUT_DIR + "] [sizes=1,12,25,50,100,200,400,800]";

    private static final Pattern SPLIT_NAME = Pattern.compile("[A-Za-z0-9_-]+");

    private BaselineMinRegionSweep() {
    }

    /** Global metrics for one minRegionSize value. */
    public record Variant(int minRegionSize, BinaryChangeMetrics metrics) {
        public boolean isDefault() {
            return minRegionSize == BaselineChangeDetector.DEFAULT_MIN_REGION_SIZE;
        }
    }

    /** The outcome of a sweep. */
    public record Result(String split, Path datasetDir, int imageCount, List<Variant> variants,
                         ZonedDateTime startedAt, double totalSeconds) {
    }

    public static void main(String[] args) {
        int code = runCli(args, System.out, System.err);
        if (code != 0) {
            System.exit(code);
        }
    }

    static int runCli(String[] args, PrintStream out, PrintStream err) {
        if (args.length > 0 && ("-h".equals(args[0]) || "--help".equals(args[0]))) {
            out.println(USAGE);
            return 0;
        }
        if (args.length > 4) {
            err.println("Too many arguments." + System.lineSeparator() + USAGE);
            return 2;
        }
        try {
            String split = args.length > 0 ? args[0] : DEFAULT_SPLIT;
            Path datasetDir = Paths.get(args.length > 1 ? args[1] : BaselineBenchmark.DEFAULT_DATASET_DIR)
                    .toAbsolutePath().normalize();
            Path outputDir = Paths.get(args.length > 2 ? args[2] : BaselineBenchmark.DEFAULT_OUTPUT_DIR)
                    .toAbsolutePath().normalize();
            int[] sizes = args.length > 3 ? parseSizes(args[3]) : DEFAULT_SIZES;

            Result result = evaluate(split, datasetDir, sizes, out);
            List<Path> written = writeOutputs(result, outputDir);
            out.println();
            out.print(formatTable(result));
            for (Path p : written) {
                out.println("Wrote " + p);
            }
            return 0;
        } catch (IOException | RuntimeException e) {
            err.println("ERROR: " + e.getMessage());
            return 1;
        }
    }

    static int[] parseSizes(String csv) {
        int[] sizes = Arrays.stream(csv.split(",")).map(String::trim).filter(s -> !s.isEmpty())
                .mapToInt(Integer::parseInt).distinct().sorted().toArray();
        if (sizes.length == 0 || sizes[0] < 1) {
            throw new IllegalArgumentException("minRegionSize values must be positive integers: " + csv);
        }
        return sizes;
    }

    /**
     * Runs the baseline with each minRegionSize on every triplet of a non-test split.
     *
     * @throws IllegalArgumentException if {@code split} is the test split
     */
    public static Result evaluate(String split, Path datasetDir, int[] sizes, PrintStream progress) throws IOException {
        if (split == null || !SPLIT_NAME.matcher(split).matches()) {
            throw new IllegalArgumentException("Invalid split name '" + split + "'");
        }
        if (FORBIDDEN_SPLIT.equalsIgnoreCase(split)) {
            throw new IllegalArgumentException("Refusing to explore parameters on the test split. "
                    + "Parameter exploration is for val (or train) only; the test result must use the defaults.");
        }
        if (sizes == null || sizes.length == 0) {
            throw new IllegalArgumentException("At least one minRegionSize value is required");
        }
        LevirCdPairReader reader = new LevirCdPairReader(datasetDir);
        List<LevirCdPairReader.Triplet> triplets = reader.listTriplets(split);
        if (triplets.isEmpty()) {
            throw new IOException("No image triplets found in split '" + split + "' under " + datasetDir);
        }

        ZonedDateTime startedAt = ZonedDateTime.now();
        long start = System.nanoTime();
        BinaryChangeMetrics[] metrics = new BinaryChangeMetrics[sizes.length];
        for (int k = 0; k < sizes.length; k++) {
            metrics[k] = new BinaryChangeMetrics();
        }
        for (int i = 0; i < triplets.size(); i++) {
            LevirCdPairReader.Sample sample = reader.load(triplets.get(i));
            for (int k = 0; k < sizes.length; k++) {
                ChangeDetectionResult detection = BaselineChangeDetector.analyze(sample.pre(), sample.post(), sizes[k]);
                metrics[k].accumulate(detection.getBinaryMask(), sample.changeMask());
            }
            if (progress != null) {
                progress.printf(Locale.ROOT, "[%d/%d] %s%n", i + 1, triplets.size(), sample.name());
            }
        }
        List<Variant> variants = new ArrayList<>();
        for (int k = 0; k < sizes.length; k++) {
            variants.add(new Variant(sizes[k], metrics[k]));
        }
        return new Result(split, datasetDir, triplets.size(), List.copyOf(variants), startedAt,
                (System.nanoTime() - start) / 1e9);
    }

    public static List<Path> writeOutputs(Result result, Path outputDir) throws IOException {
        Files.createDirectories(outputDir);
        String base = "baseline_" + result.split() + "_minregion_sweep";
        Path csv = outputDir.resolve(base + ".csv");
        Path txt = outputDir.resolve(base + ".txt");
        Files.writeString(csv, formatCsv(result), StandardCharsets.UTF_8);
        Files.writeString(txt, formatReport(result), StandardCharsets.UTF_8);
        return List.of(txt, csv);
    }

    static String formatCsv(Result result) {
        StringBuilder sb = new StringBuilder("min_region_size,is_default,tp,fp,fn,tn,precision,recall,f1,iou,kappa\n");
        for (Variant v : result.variants()) {
            BinaryChangeMetrics m = v.metrics();
            sb.append(String.format(Locale.ROOT, "%d,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    v.minRegionSize(), v.isDefault(), m.getTruePositives(), m.getFalsePositives(),
                    m.getFalseNegatives(), m.getTrueNegatives(), m.precision(), m.recall(), m.f1(), m.iou(), m.kappa()));
        }
        return sb.toString();
    }

    static String formatTable(Result result) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT, "%-14s %9s %9s %9s %9s %9s %12s %12s %12s%n",
                "minRegionSize", "precision", "recall", "F1", "IoU", "kappa", "TP", "FP", "FN"));
        for (Variant v : result.variants()) {
            BinaryChangeMetrics m = v.metrics();
            sb.append(String.format(Locale.ROOT, "%-14s %9.4f %9.4f %9.4f %9.4f %9.4f %12d %12d %12d%n",
                    v.minRegionSize() + (v.isDefault() ? " (default)" : ""), m.precision(), m.recall(),
                    m.f1(), m.iou(), m.kappa(), m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives()));
        }
        return sb.toString();
    }

    static String formatReport(Result result) {
        String nl = System.lineSeparator();
        String rule = "--------------------------------------------------------------------------------";
        Variant best = result.variants().get(0);
        for (Variant v : result.variants()) {
            if (v.metrics().f1() > best.metrics().f1()) {
                best = v;
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("================================================================================").append(nl);
        sb.append(" LANDWATCH AI - Baseline minRegionSize exploration (").append(result.split().toUpperCase(Locale.ROOT))
                .append(" SPLIT ONLY)").append(nl);
        sb.append("================================================================================").append(nl);
        sb.append("This is an exploration on the ").append(result.split()).append(" split only. It was NOT used to").append(nl);
        sb.append("choose the parameters of the reported test result: baseline_test_metrics.txt uses").append(nl);
        sb.append("the untouched default minRegionSize = ").append(BaselineChangeDetector.DEFAULT_MIN_REGION_SIZE)
                .append(". No value from this sweep has been run on test.").append(nl);
        sb.append(nl);
        sb.append("Dataset        : LEVIR-CD (").append(result.datasetDir()).append(')').append(nl);
        sb.append("Split          : ").append(result.split()).append(" (").append(result.imageCount())
                .append(" full-resolution image pairs)").append(nl);
        sb.append("Method         : BaselineChangeDetector.analyze(pre, post, minRegionSize); everything").append(nl);
        sb.append("                 else unchanged (luminance difference, per-image Otsu, 3x3 opening)").append(nl);
        sb.append("Ground truth   : label pixel value > 0 = change").append(nl);
        sb.append("Protocol       : one global pixel confusion matrix per minRegionSize value").append(nl);
        sb.append("Timestamp      : ").append(result.startedAt().truncatedTo(ChronoUnit.SECONDS)
                .format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)).append(nl);
        sb.append("Java / os.arch : ").append(System.getProperty("java.version")).append(" / ")
                .append(System.getProperty("os.arch")).append(nl);
        sb.append(String.format(Locale.ROOT, "Wall time      : %.1f s%n", result.totalSeconds()));
        sb.append(nl);
        sb.append("Global metrics per minRegionSize").append(nl).append(rule).append(nl);
        sb.append(formatTable(result));
        sb.append(nl);
        sb.append(String.format(Locale.ROOT, "Highest %s change-F1 in this sweep: minRegionSize = %d (F1 = %.4f).%n",
                result.split(), best.minRegionSize(), best.metrics().f1()));
        sb.append("Any value chosen from this table would have to be evaluated once on test, as a").append(nl);
        sb.append("separate and clearly labelled run. That has not been done.").append(nl);
        return sb.toString();
    }
}
