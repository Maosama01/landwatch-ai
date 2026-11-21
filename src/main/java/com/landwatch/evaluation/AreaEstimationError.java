package com.landwatch.evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Area estimation error: how well a method estimates HOW MUCH changed, in square metres.
 *
 * <p>A planning office often wants "how many square metres of new buildings?" more than an exact
 * outline. This tool converts changed-pixel counts to areas with the ground sample distance (GSD;
 * LEVIR-CD is 0.5 m per pixel, so one pixel is 0.25 m^2) and compares the predicted with the true
 * changed area, image by image. It reads per-image CSV files that the evaluations already wrote
 * (columns {@code image, true_change_px, pred_change_px}), so it does not run any model and does not
 * look at the images again.</p>
 *
 * <p>Usage: {@code AreaEstimationError [--gsd 0.5] [--out results/deeplearning/test_area_error.txt]
 * "Label one=path/one.csv" "Label two=path/two.csv" ...}</p>
 *
 * <p>Caveat printed with the results: an area can be right while the location is wrong (false alarms
 * and misses cancel), so area error complements IoU and F1; it does not replace them.</p>
 */
public final class AreaEstimationError {

    /** Images with at least this many true change pixels (250 m^2 at 0.5 m) get a relative error. */
    public static final long MIN_PIXELS_FOR_RELATIVE = 1000;

    private AreaEstimationError() {
    }

    /** Area statistics of one method over a set of images. */
    public record Summary(int images, double gsd, double trueArea, double predictedArea, double meanAbsoluteError,
                          double medianAbsoluteError, int relativeImages, double medianRelative, double q1Relative,
                          double q3Relative, double within10, double within25, int noChangeImages,
                          double meanFalseAreaOnNoChange, double correlation) {

        /** Signed relative error of the total area: (predicted - true) / true. */
        public double totalRelativeError() {
            return (predictedArea - trueArea) / trueArea;
        }
    }

    /**
     * @param truePx changed pixels in the ground truth, per image
     * @param predPx changed pixels in the prediction, per image
     * @param gsd    ground sample distance in metres per pixel
     */
    public static Summary summarize(long[] truePx, long[] predPx, double gsd) {
        if (truePx.length != predPx.length || truePx.length == 0) {
            throw new IllegalArgumentException("Need the same, non-zero number of true and predicted counts");
        }
        if (!(gsd > 0)) {
            throw new IllegalArgumentException("gsd must be positive");
        }
        double pixelArea = gsd * gsd;
        int n = truePx.length;
        double trueSum = 0;
        double predSum = 0;
        double[] absErr = new double[n];
        List<Double> rel = new ArrayList<>();
        int noChange = 0;
        double falseArea = 0;
        for (int i = 0; i < n; i++) {
            trueSum += truePx[i] * pixelArea;
            predSum += predPx[i] * pixelArea;
            absErr[i] = Math.abs(predPx[i] - truePx[i]) * pixelArea;
            if (truePx[i] >= MIN_PIXELS_FOR_RELATIVE) {
                rel.add(Math.abs(predPx[i] - truePx[i]) / (double) truePx[i]);
            }
            if (truePx[i] == 0) {
                noChange++;
                falseArea += predPx[i] * pixelArea;
            }
        }
        double[] r = rel.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        double within10 = r.length == 0 ? Double.NaN : Arrays.stream(r).filter(v -> v <= 0.10).count() / (double) r.length;
        double within25 = r.length == 0 ? Double.NaN : Arrays.stream(r).filter(v -> v <= 0.25).count() / (double) r.length;
        return new Summary(n, gsd, trueSum, predSum, Arrays.stream(absErr).average().orElse(0), quantile(absErr, 0.5),
                r.length, quantile(r, 0.5), quantile(r, 0.25), quantile(r, 0.75), within10, within25,
                noChange, noChange == 0 ? Double.NaN : falseArea / noChange, correlation(truePx, predPx));
    }

    /** Linear-interpolated quantile of an array (sorted copy), NaN for an empty array. */
    static double quantile(double[] values, double q) {
        if (values.length == 0) {
            return Double.NaN;
        }
        double[] s = values.clone();
        Arrays.sort(s);
        double pos = q * (s.length - 1);
        int lo = (int) Math.floor(pos);
        int hi = (int) Math.ceil(pos);
        return s[lo] + (s[hi] - s[lo]) * (pos - lo);
    }

    /** Pearson correlation between true and predicted changed areas across images. */
    static double correlation(long[] a, long[] b) {
        int n = a.length;
        double ma = Arrays.stream(a).average().orElse(0);
        double mb = Arrays.stream(b).average().orElse(0);
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < n; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return saa == 0 || sbb == 0 ? Double.NaN : sab / Math.sqrt(saa * sbb);
    }

    /** Reads {@code true_change_px} and {@code pred_change_px} from a per-image CSV (header names located by name). */
    static long[][] readCounts(Path csv) throws IOException {
        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        List<String> header = Arrays.asList(lines.get(0).split(","));
        int t = header.indexOf("true_change_px");
        int p = header.indexOf("pred_change_px");
        if (t < 0 || p < 0) {
            throw new IOException(csv + " has no true_change_px / pred_change_px columns");
        }
        List<long[]> rows = new ArrayList<>();
        for (String line : lines.subList(1, lines.size())) {
            if (line.isBlank()) {
                continue;
            }
            String[] f = line.split(",");
            rows.add(new long[] {Long.parseLong(f[t].trim()), Long.parseLong(f[p].trim())});
        }
        long[][] out = new long[2][rows.size()];
        for (int i = 0; i < rows.size(); i++) {
            out[0][i] = rows.get(i)[0];
            out[1][i] = rows.get(i)[1];
        }
        return out;
    }

    static String report(List<String> labels, List<Path> files, List<Summary> summaries) {
        StringBuilder sb = new StringBuilder();
        Summary first = summaries.get(0);
        sb.append("LandWatch AI - area estimation error on LEVIR-CD\n");
        sb.append("Generated: ").append(LocalDate.now()).append(" by com.landwatch.evaluation.AreaEstimationError\n");
        sb.append(String.format(Locale.ROOT, "Ground sample distance %.2f m/pixel, so 1 pixel = %.4f m^2. Computed from per-image counts "
                + "already recorded; no model was run again.%n", first.gsd(), first.gsd() * first.gsd()));
        sb.append("Relative errors use images with at least ").append(MIN_PIXELS_FOR_RELATIVE)
                .append(" true change pixels (250 m^2); below that a few pixels give huge percentages.\n\n");
        for (int k = 0; k < labels.size(); k++) {
            Summary s = summaries.get(k);
            sb.append("== ").append(labels.get(k)).append("  (").append(files.get(k).toString().replace('\\', '/')).append(")\n");
            sb.append(String.format(Locale.ROOT, "Images: %d%n", s.images()));
            sb.append(String.format(Locale.ROOT, "Total changed area: true %,.0f m^2 (%.2f ha), predicted %,.0f m^2 (%.2f ha): %+.2f %%%n",
                    s.trueArea(), s.trueArea() / 1e4, s.predictedArea(), s.predictedArea() / 1e4, 100 * s.totalRelativeError()));
            sb.append(String.format(Locale.ROOT, "Per-image absolute area error: mean %,.0f m^2, median %,.0f m^2%n",
                    s.meanAbsoluteError(), s.medianAbsoluteError()));
            sb.append(String.format(Locale.ROOT, "Per-image relative area error (%d images): median %.1f %% (quartiles %.1f-%.1f %%); "
                            + "within 10 %%: %.0f %% of images, within 25 %%: %.0f %%%n",
                    s.relativeImages(), 100 * s.medianRelative(), 100 * s.q1Relative(), 100 * s.q3Relative(),
                    100 * s.within10(), 100 * s.within25()));
            sb.append(String.format(Locale.ROOT, "Images without change: %d, mean predicted (false) area on them %,.0f m^2%n",
                    s.noChangeImages(), s.meanFalseAreaOnNoChange()));
            sb.append(String.format(Locale.ROOT, "Correlation of true and predicted area across images: r = %.4f%n%n", s.correlation()));
        }
        sb.append("Caveat: equal areas do not mean the same places. False alarms and misses can cancel in the total,\n");
        sb.append("so read this together with IoU and F1, which measure where the change is.\n");
        return sb.toString();
    }

    public static void main(String[] args) throws IOException {
        double gsd = 0.5;
        Path out = Path.of("results/deeplearning/test_area_error.txt");
        List<String> labels = new ArrayList<>();
        List<Path> files = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            if ("--gsd".equals(args[i]) && i + 1 < args.length) {
                gsd = Double.parseDouble(args[++i]);
            } else if ("--out".equals(args[i]) && i + 1 < args.length) {
                out = Path.of(args[++i]);
            } else {
                int eq = args[i].lastIndexOf('=');
                if (eq <= 0) {
                    throw new IllegalArgumentException("Expected \"Label=path.csv\", got " + args[i]);
                }
                labels.add(args[i].substring(0, eq));
                files.add(Path.of(args[i].substring(eq + 1)));
            }
        }
        if (labels.isEmpty()) {
            labels.add("Classical baseline");
            files.add(Path.of("results/benchmarks/baseline_test_per_image.csv"));
            labels.add("FC-Siam-diff");
            files.add(Path.of("results/deeplearning/test_per_image.csv"));
        }
        List<Summary> summaries = new ArrayList<>();
        for (Path f : files) {
            long[][] counts = readCounts(f);
            summaries.add(summarize(counts[0], counts[1], gsd));
        }
        String text = report(labels, files, summaries);
        Files.createDirectories(out.toAbsolutePath().getParent());
        Files.writeString(out, text, StandardCharsets.UTF_8);
        System.out.print(text);
        System.out.println("Wrote " + out.toString().replace('\\', '/'));
    }
}
