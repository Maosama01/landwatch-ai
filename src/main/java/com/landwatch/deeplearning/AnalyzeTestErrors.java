package com.landwatch.deeplearning;

import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.evaluation.LevirCdPairReader;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Post-hoc error analysis of a trained change detector on the full-resolution LEVIR-CD images of
 * one split: where the model is wrong, not how good it is.
 *
 * <p>For every image pair it predicts the change mask through {@link DeepChangeDetector} (the same
 * 256 x 256 tiling as training and evaluation) and records:</p>
 * <ul>
 *   <li>the per-image confusion matrix, precision, recall, F1 and IoU;</li>
 *   <li>how many wrong pixels lie in a thin band around the true building outlines
 *       ({@value #BAND} px on either side), i.e. boundary errors rather than whole-object errors;</li>
 *   <li>object-level recall: every 8-connected true change region counts as found when at least
 *       half of its pixels are predicted as change, reported by region size;</li>
 *   <li>false-alarm objects: predicted regions that do not touch any true change at all.</li>
 * </ul>
 *
 * <p>Nothing is tuned here: the model, its threshold and every setting are used as shipped, and the
 * summed confusion matrix is compared with the split's recorded {@code <split>_metrics.txt}.</p>
 *
 * <p>Usage: {@code AnalyzeTestErrors [split=test] [--model models/siamese_unet_levir]
 * [--data datasets/LEVIR-CD] [--device auto] [--results results/deeplearning] [--force false]}</p>
 */
public final class AnalyzeTestErrors {

    /** Half-width in pixels of the band around true outlines counted as "near the outline". */
    static final int BAND = 2;
    /** Upper bounds (exclusive) of the object-size bins in pixels; the last bin is open-ended. */
    static final int[] SIZE_BINS = {100, 400, 1600};

    private AnalyzeTestErrors() {
    }

    /** Result for one image. */
    record ImageResult(String name, BinaryChangeMetrics metrics, int trueObjects, int foundObjects,
                       int falseAlarmObjects, long falseAlarmObjectPixels, long fpNearOutline, long fnNearOutline,
                       long[] objectsPerBin, long[] foundPerBin) {
        double changeShare() {
            long truth = metrics.getTruePositives() + metrics.getFalseNegatives();
            return truth / (double) metrics.getTotal();
        }
    }

    public static void main(String[] args) throws Exception {
        String split = args.length > 0 && !args[0].startsWith("--") ? args[0] : "test";
        String[] rest = args.length > 0 && !args[0].startsWith("--") ? Arrays.copyOfRange(args, 1, args.length) : args;
        Map<String, String> opt = TrainChangeDetector.parseArgs(rest);
        Path modelDir = Path.of(opt.getOrDefault("model", "models/siamese_unet_levir"));
        Path data = Path.of(opt.getOrDefault("data", "datasets/LEVIR-CD"));
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/deeplearning"));
        String device = opt.getOrDefault("device", "auto");
        boolean force = Boolean.parseBoolean(opt.getOrDefault("force", "false"));

        Path csvOut = resultsDir.resolve(split + "_per_image.csv");
        Path txtOut = resultsDir.resolve(split + "_error_analysis.txt");
        if (!force && (Files.exists(csvOut) || Files.exists(txtOut))) {
            System.err.println("Refusing to overwrite " + TrainingSupport.displayPath(txtOut)
                    + " (pass --force true to recompute it deliberately).");
            System.exit(2);
        }

        LevirCdPairReader reader = new LevirCdPairReader(data);
        List<LevirCdPairReader.Triplet> triplets = reader.listTriplets(split);
        List<ImageResult> results = new ArrayList<>();
        BinaryChangeMetrics total = new BinaryChangeMetrics();
        String deviceName;
        float threshold;
        long start = System.nanoTime();
        try (DeepChangeDetector detector = DeepChangeDetector.load(modelDir, device)) {
            deviceName = detector.getDevice().toString();
            threshold = detector.getThreshold();
            System.out.println(detector.describe());
            System.out.printf(Locale.ROOT, "Analysing %d %s pairs from %s%n", triplets.size(), split,
                    TrainingSupport.displayPath(data));
            int i = 0;
            for (LevirCdPairReader.Triplet t : triplets) {
                LevirCdPairReader.Sample s = reader.load(t);
                boolean[][] predicted = detector.predictMask(s.pre(), s.post());
                ImageResult r = analyse(t.name(), predicted, s.changeMask());
                results.add(r);
                total.merge(r.metrics());
                if (++i % 16 == 0) {
                    System.out.printf(Locale.ROOT, "  %d / %d%n", i, triplets.size());
                }
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;

        Files.createDirectories(resultsDir);
        writeCsv(csvOut, results);
        String report = report(split, modelDir, data, deviceName, threshold, seconds, results, total,
                resultsDir.resolve(split + "_metrics.txt"));
        Files.writeString(txtOut, report, StandardCharsets.UTF_8);
        System.out.print(report);
        System.out.println("Wrote " + TrainingSupport.displayPath(csvOut) + " and " + TrainingSupport.displayPath(txtOut));
    }

    /** Analyses one prediction against its ground truth (both indexed {@code [y][x]}). */
    static ImageResult analyse(String name, boolean[][] predicted, boolean[][] truth) {
        int h = truth.length;
        int w = truth[0].length;
        BinaryChangeMetrics m = new BinaryChangeMetrics();
        m.accumulate(predicted, truth);

        boolean[][] band = outlineBand(truth, BAND);
        long fpNear = 0;
        long fnNear = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (band[y][x] && predicted[y][x] != truth[y][x]) {
                    if (predicted[y][x]) {
                        fpNear++;
                    } else {
                        fnNear++;
                    }
                }
            }
        }

        long[] perBin = new long[SIZE_BINS.length + 1];
        long[] foundBin = new long[SIZE_BINS.length + 1];
        int trueObjects = 0;
        int found = 0;
        for (List<int[]> region : components(truth)) {
            trueObjects++;
            long hit = region.stream().filter(p -> predicted[p[1]][p[0]]).count();
            int bin = sizeBin(region.size());
            perBin[bin]++;
            if (hit * 2 >= region.size()) {
                found++;
                foundBin[bin]++;
            }
        }

        int falseAlarms = 0;
        long falseAlarmPixels = 0;
        for (List<int[]> region : components(predicted)) {
            boolean touchesTruth = region.stream().anyMatch(p -> truth[p[1]][p[0]]);
            if (!touchesTruth) {
                falseAlarms++;
                falseAlarmPixels += region.size();
            }
        }
        return new ImageResult(name, m, trueObjects, found, falseAlarms, falseAlarmPixels, fpNear, fnNear,
                perBin, foundBin);
    }

    static int sizeBin(int size) {
        for (int i = 0; i < SIZE_BINS.length; i++) {
            if (size < SIZE_BINS[i]) {
                return i;
            }
        }
        return SIZE_BINS.length;
    }

    /**
     * Pixels whose (2r+1) x (2r+1) neighbourhood contains both change and no-change pixels of the
     * ground truth: a band of about r pixels on each side of every true outline.
     */
    static boolean[][] outlineBand(boolean[][] mask, int r) {
        int h = mask.length;
        int w = mask[0].length;
        long[][] integral = new long[h + 1][w + 1];
        for (int y = 0; y < h; y++) {
            long row = 0;
            for (int x = 0; x < w; x++) {
                row += mask[y][x] ? 1 : 0;
                integral[y + 1][x + 1] = integral[y][x + 1] + row;
            }
        }
        boolean[][] band = new boolean[h][w];
        for (int y = 0; y < h; y++) {
            int y0 = Math.max(0, y - r);
            int y1 = Math.min(h, y + r + 1);
            for (int x = 0; x < w; x++) {
                int x0 = Math.max(0, x - r);
                int x1 = Math.min(w, x + r + 1);
                long sum = integral[y1][x1] - integral[y0][x1] - integral[y1][x0] + integral[y0][x0];
                long area = (long) (y1 - y0) * (x1 - x0);
                band[y][x] = sum > 0 && sum < area;
            }
        }
        return band;
    }

    /** 8-connected regions of a mask, each as a list of {x, y} points. */
    static List<List<int[]>> components(boolean[][] mask) {
        int h = mask.length;
        int w = mask[0].length;
        boolean[][] seen = new boolean[h][w];
        List<List<int[]>> regions = new ArrayList<>();
        ArrayDeque<int[]> queue = new ArrayDeque<>();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                if (!mask[y][x] || seen[y][x]) {
                    continue;
                }
                List<int[]> region = new ArrayList<>();
                seen[y][x] = true;
                queue.add(new int[] {x, y});
                while (!queue.isEmpty()) {
                    int[] p = queue.poll();
                    region.add(p);
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = p[0] + dx;
                            int ny = p[1] + dy;
                            if (nx >= 0 && ny >= 0 && nx < w && ny < h && mask[ny][nx] && !seen[ny][nx]) {
                                seen[ny][nx] = true;
                                queue.add(new int[] {nx, ny});
                            }
                        }
                    }
                }
                regions.add(region);
            }
        }
        return regions;
    }

    private static void writeCsv(Path out, List<ImageResult> results) throws IOException {
        try (PrintWriter pw = new PrintWriter(Files.newBufferedWriter(out, StandardCharsets.UTF_8))) {
            pw.println("image,true_change_px,pred_change_px,tp,fp,fn,tn,precision,recall,f1,iou,"
                    + "true_objects,found_objects,false_alarm_objects,fp_near_outline,fn_near_outline");
            for (ImageResult r : results) {
                BinaryChangeMetrics m = r.metrics();
                pw.printf(Locale.ROOT, "%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%d,%d,%d,%d%n", r.name(),
                        m.getTruePositives() + m.getFalseNegatives(), m.getTruePositives() + m.getFalsePositives(),
                        m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives(), m.getTrueNegatives(),
                        m.precision(), m.recall(), m.f1(), m.iou(), r.trueObjects(), r.foundObjects(),
                        r.falseAlarmObjects(), r.fpNearOutline(), r.fnNearOutline());
            }
        }
    }

    private static String report(String split, Path modelDir, Path data, String device, float threshold, double seconds,
                                 List<ImageResult> results, BinaryChangeMetrics total, Path recordedMetrics) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("=== LandWatch AI - post-hoc error analysis of the deep model on the ").append(split)
                .append(" split ===\n");
        sb.append(String.format(Locale.ROOT, "Date: %s | Java %s | device %s%n", LocalDateTime.now().withNano(0),
                System.getProperty("java.version"), device));
        sb.append(String.format(Locale.ROOT, "Model: %s (decision threshold %.3f, used as shipped) | data: %s/%s%n",
                TrainingSupport.displayPath(modelDir), threshold, TrainingSupport.displayPath(data), split));
        sb.append("Purpose: describe where the shipped model is wrong. Nothing was tuned or changed afterwards.\n");
        sb.append(String.format(Locale.ROOT, "Reproduce: java -cp \"$CP\" com.landwatch.deeplearning.AnalyzeTestErrors %s%n%n", split));

        // 1. consistency with the recorded evaluation
        sb.append("1. Consistency with the recorded evaluation\n");
        sb.append(String.format(Locale.ROOT, "   Summed over %d full-resolution images: TP=%d  FP=%d  FN=%d  TN=%d  (F1 %.4f, IoU %.4f)%n",
                results.size(), total.getTruePositives(), total.getFalsePositives(), total.getFalseNegatives(),
                total.getTrueNegatives(), total.f1(), total.iou()));
        if (Files.exists(recordedMetrics)) {
            Matcher mt = Pattern.compile("TP=(\\d+)\\s+FP=(\\d+)\\s+FN=(\\d+)\\s+TN=(\\d+)")
                    .matcher(Files.readString(recordedMetrics, StandardCharsets.UTF_8));
            if (mt.find()) {
                long[] rec = {Long.parseLong(mt.group(1)), Long.parseLong(mt.group(2)), Long.parseLong(mt.group(3)),
                        Long.parseLong(mt.group(4))};
                long[] now = {total.getTruePositives(), total.getFalsePositives(), total.getFalseNegatives(),
                        total.getTrueNegatives()};
                boolean same = Arrays.equals(rec, now);
                sb.append(String.format(Locale.ROOT, "   Recorded in %s:          TP=%d  FP=%d  FN=%d  TN=%d  -> %s%n",
                        TrainingSupport.displayPath(recordedMetrics), rec[0], rec[1], rec[2], rec[3],
                        same ? "identical" : String.format(Locale.ROOT, "differs by %d / %d / %d / %d pixels",
                                now[0] - rec[0], now[1] - rec[1], now[2] - rec[2], now[3] - rec[3])));
            }
        }
        sb.append('\n');

        // 2. per-image spread
        List<ImageResult> withChange = results.stream().filter(r -> r.changeShare() > 0).toList();
        List<ImageResult> noChange = results.stream().filter(r -> r.changeShare() == 0).toList();
        double[] f1s = withChange.stream().mapToDouble(r -> r.metrics().f1()).sorted().toArray();
        sb.append(String.format(Locale.ROOT, "2. Per-image F1 (%d images with true change; F1 = 0 when nothing is found)%n", f1s.length));
        if (f1s.length > 0) {
            sb.append(String.format(Locale.ROOT, "   min %.4f | 25 %% %.4f | median %.4f | 75 %% %.4f | max %.4f%n",
                    f1s[0], quantile(f1s, 0.25), quantile(f1s, 0.5), quantile(f1s, 0.75), f1s[f1s.length - 1]));
            sb.append(String.format(Locale.ROOT, "   images below F1 0.80: %d | below 0.70: %d | below 0.50: %d%n",
                    count(f1s, 0.80), count(f1s, 0.70), count(f1s, 0.50)));
        }
        sb.append("   Ten lowest-F1 images:\n");
        sb.append("     image          change %  pred %    F1      P       R     true obj  found  false-alarm obj\n");
        withChange.stream().sorted(Comparator.comparingDouble(r -> r.metrics().f1())).limit(10).forEach(r -> {
            BinaryChangeMetrics m = r.metrics();
            sb.append(String.format(Locale.ROOT, "     %-13s %7.2f  %7.2f  %6.4f  %6.4f  %6.4f  %8d  %5d  %8d%n",
                    r.name(), 100 * r.changeShare(),
                    100.0 * (m.getTruePositives() + m.getFalsePositives()) / m.getTotal(),
                    m.f1(), m.precision(), m.recall(), r.trueObjects(), r.foundObjects(), r.falseAlarmObjects()));
        });
        long noChangeFlagged = noChange.stream().mapToLong(r -> r.metrics().getFalsePositives()).sum();
        long noChangeClean = noChange.stream().filter(r -> r.metrics().getFalsePositives() == 0).count();
        sb.append(String.format(Locale.ROOT, "   %d images contain no true change: %d of them get an empty mask; together %d pixels are flagged (%.3f %% of their pixels)%n%n",
                noChange.size(), noChangeClean, noChangeFlagged,
                noChange.isEmpty() ? 0.0 : 100.0 * noChangeFlagged / noChange.stream().mapToLong(r -> r.metrics().getTotal()).sum()));

        // 3. boundary errors
        long fp = total.getFalsePositives();
        long fn = total.getFalseNegatives();
        long fpNear = results.stream().mapToLong(ImageResult::fpNearOutline).sum();
        long fnNear = results.stream().mapToLong(ImageResult::fnNearOutline).sum();
        sb.append(String.format(Locale.ROOT, "3. Errors near true building outlines (band of %d px on each side)%n", BAND));
        sb.append(String.format(Locale.ROOT, "   false positives near an outline: %d of %d (%.1f %%)%n", fpNear, fp, pct(fpNear, fp)));
        sb.append(String.format(Locale.ROOT, "   false negatives near an outline: %d of %d (%.1f %%)%n", fnNear, fn, pct(fnNear, fn)));
        sb.append(String.format(Locale.ROOT, "   all wrong pixels near an outline: %d of %d (%.1f %%)%n%n", fpNear + fnNear, fp + fn,
                pct(fpNear + fnNear, fp + fn)));

        // 4. object-level recall by size
        long[] perBin = new long[SIZE_BINS.length + 1];
        long[] foundBin = new long[SIZE_BINS.length + 1];
        for (ImageResult r : results) {
            for (int i = 0; i < perBin.length; i++) {
                perBin[i] += r.objectsPerBin()[i];
                foundBin[i] += r.foundPerBin()[i];
            }
        }
        long objects = Arrays.stream(perBin).sum();
        long foundObjects = Arrays.stream(foundBin).sum();
        sb.append("4. True change regions (8-connected) found, by size (found = at least half its pixels predicted)\n");
        sb.append(String.format(Locale.ROOT, "   all regions: %d of %d found (%.1f %%)%n", foundObjects, objects, pct(foundObjects, objects)));
        for (int i = 0; i < perBin.length; i++) {
            String label = i == 0 ? String.format(Locale.ROOT, "< %d px", SIZE_BINS[0])
                    : i == SIZE_BINS.length ? String.format(Locale.ROOT, ">= %d px", SIZE_BINS[i - 1])
                    : String.format(Locale.ROOT, "%d-%d px", SIZE_BINS[i - 1], SIZE_BINS[i] - 1);
            double m2lo = (i == 0 ? 0 : SIZE_BINS[i - 1]) * 0.25;
            String area = i == SIZE_BINS.length ? String.format(Locale.ROOT, ">= %.0f m2", m2lo)
                    : String.format(Locale.ROOT, "%.0f-%.0f m2", m2lo, SIZE_BINS[i] * 0.25);
            sb.append(String.format(Locale.ROOT, "   %-12s (%-12s): %5d of %5d found (%.1f %%)%n", label, area, foundBin[i], perBin[i],
                    pct(foundBin[i], perBin[i])));
        }
        sb.append('\n');

        // 5. false-alarm objects
        long faObjects = results.stream().mapToLong(ImageResult::falseAlarmObjects).sum();
        long faPixels = results.stream().mapToLong(ImageResult::falseAlarmObjectPixels).sum();
        sb.append("5. Predicted regions that touch no true change at all (whole false alarms)\n");
        sb.append(String.format(Locale.ROOT, "   %d regions, %d pixels = %.1f %% of all false-positive pixels; the rest are overshoot attached to true changes%n%n",
                faObjects, faPixels, pct(faPixels, fp)));
        sb.append(String.format(Locale.ROOT, "Time: %.1f s for %d image pairs%n", seconds, results.size()));
        return sb.toString();
    }

    private static double quantile(double[] sorted, double q) {
        double pos = q * (sorted.length - 1);
        int lo = (int) Math.floor(pos);
        int hi = Math.min(sorted.length - 1, lo + 1);
        return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
    }

    private static long count(double[] values, double below) {
        return Arrays.stream(values).filter(v -> v < below).count();
    }

    private static double pct(long part, long whole) {
        return whole == 0 ? 0.0 : 100.0 * part / whole;
    }
}
