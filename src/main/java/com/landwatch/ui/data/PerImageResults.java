package com.landwatch.ui.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Per-image rows of a benchmark CSV such as {@code results/benchmarks/baseline_test_per_image.csv}
 * (header {@code image,width,height,true_change_px,pred_change_px,tp,fp,fn,tn,precision,recall,f1,iou,...}).
 * Columns are found by header name. Rows that cannot be parsed are skipped and counted.
 */
public final class PerImageResults {

    /** One image of the benchmark. Metric values are those written by the benchmark, not recomputed. */
    public record Row(String image, long trueChangePx, long predChangePx,
                      double precision, double recall, double f1, double iou) {
    }

    private static final String[] REQUIRED = {"image", "true_change_px", "pred_change_px",
            "precision", "recall", "f1", "iou"};

    private final Path file;
    private final MetricsReport.Status status;
    private final String problem;
    private final List<Row> rows;
    private final int skippedRows;

    private PerImageResults(Path file, MetricsReport.Status status, String problem, List<Row> rows, int skippedRows) {
        this.file = file;
        this.status = status;
        this.problem = problem;
        this.rows = rows;
        this.skippedRows = skippedRows;
    }

    public static PerImageResults read(Path file) {
        if (file == null || !Files.isRegularFile(file)) {
            return new PerImageResults(file, MetricsReport.Status.MISSING, "file not found: " + file, List.of(), 0);
        }
        try {
            return parse(file, Files.readAllLines(file, StandardCharsets.UTF_8));
        } catch (IOException | RuntimeException e) {
            return new PerImageResults(file, MetricsReport.Status.MALFORMED,
                    "cannot read " + file + ": " + e.getMessage(), List.of(), 0);
        }
    }

    public static PerImageResults parse(Path file, List<String> lines) {
        if (lines.isEmpty()) {
            return new PerImageResults(file, MetricsReport.Status.MALFORMED, "malformed " + file + ": empty file",
                    List.of(), 0);
        }
        String[] header = lines.get(0).split(",", -1);
        Map<String, Integer> index = new HashMap<>();
        for (int i = 0; i < header.length; i++) {
            index.put(header[i].trim().toLowerCase(), i);
        }
        for (String r : REQUIRED) {
            if (!index.containsKey(r)) {
                return new PerImageResults(file, MetricsReport.Status.MALFORMED,
                        "malformed " + file + ": no '" + r + "' column", List.of(), 0);
            }
        }
        List<Row> rows = new ArrayList<>();
        int skipped = 0;
        for (int n = 1; n < lines.size(); n++) {
            String line = lines.get(n);
            if (line.isBlank()) {
                continue;
            }
            String[] c = line.split(",", -1);
            try {
                rows.add(new Row(
                        c[index.get("image")].trim(),
                        Long.parseLong(c[index.get("true_change_px")].trim()),
                        Long.parseLong(c[index.get("pred_change_px")].trim()),
                        Double.parseDouble(c[index.get("precision")].trim()),
                        Double.parseDouble(c[index.get("recall")].trim()),
                        Double.parseDouble(c[index.get("f1")].trim()),
                        Double.parseDouble(c[index.get("iou")].trim())));
            } catch (RuntimeException e) {
                skipped++;
            }
        }
        if (rows.isEmpty()) {
            return new PerImageResults(file, MetricsReport.Status.MALFORMED,
                    "malformed " + file + ": no parsable rows", List.of(), skipped);
        }
        return new PerImageResults(file, MetricsReport.Status.OK, null, List.copyOf(rows), skipped);
    }

    public Path getFile() { return file; }
    public MetricsReport.Status getStatus() { return status; }
    public boolean isOk() { return status == MetricsReport.Status.OK; }
    public String getProblem() { return problem; }
    public List<Row> getRows() { return rows; }
    public int getSkippedRows() { return skippedRows; }

    /** Images that contain labelled change, sorted by F1 ascending (worst first), at most {@code limit}. */
    public List<Row> worstByF1(int limit) {
        return rows.stream()
                .filter(r -> r.trueChangePx() > 0)
                .sorted(Comparator.comparingDouble(Row::f1).thenComparing(Row::image))
                .limit(limit)
                .toList();
    }

    /** Number of images whose label contains no change at all. */
    public long imagesWithoutTrueChange() {
        return rows.stream().filter(r -> r.trueChangePx() == 0).count();
    }

    /** Median of the per-image F1 over the images that contain labelled change, or NaN if none. */
    public double medianF1WithChange() {
        double[] f1 = rows.stream().filter(r -> r.trueChangePx() > 0).mapToDouble(Row::f1).sorted().toArray();
        if (f1.length == 0) {
            return Double.NaN;
        }
        int mid = f1.length / 2;
        return f1.length % 2 == 1 ? f1[mid] : (f1[mid - 1] + f1[mid]) / 2.0;
    }
}
