package com.landwatch.ui.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A metrics file read at runtime, e.g. {@code results/benchmarks/baseline_test_metrics.txt}.
 *
 * <p>The files use the {@code key : value} lines of {@code BinaryChangeMetrics.toReport()}
 * ({@code Change F1        : 0.1008}); the EuroSAT report uses {@code Overall accuracy : ...} and
 * {@code Macro F1         : ...}. The first line for each key wins, so later reference lines (such as
 * "For reference only, at the untuned threshold 0.50: ... F1 ...") are never mistaken for the main
 * result. Nothing is ever filled in: a missing file is {@link Status#MISSING}, an unusable one is
 * {@link Status#MALFORMED} with the reason.</p>
 */
public final class MetricsReport {

    /** Whether the file could be used. */
    public enum Status { OK, MISSING, MALFORMED }

    /** Metrics the dashboard reads, with the exact key used in the files. */
    public enum Metric {
        F1("Change F1"),
        IOU("Change IoU"),
        PRECISION("Change precision"),
        RECALL("Change recall"),
        KAPPA("Cohen's kappa"),
        PIXEL_ACCURACY("Pixel accuracy"),
        MACRO_F1("Macro F1"),
        MEAN_IOU("Mean IoU"),
        OVERALL_ACCURACY("Overall accuracy");

        private final String key;

        Metric(String key) {
            this.key = key;
        }

        public String key() {
            return key;
        }

        /** Kappa can be negative; all other metrics are ratios in [0, 1]. */
        boolean inRange(double v) {
            return this == KAPPA ? v >= -1.0 && v <= 1.0 : v >= 0.0 && v <= 1.0;
        }
    }

    /** Required keys for a change-detection report (LEVIR-CD, baseline or deep model). */
    public static final Set<Metric> CHANGE_DETECTION_REQUIRED = Set.of(Metric.F1);
    /** Required keys for the EuroSAT classification report. */
    public static final Set<Metric> EUROSAT_REQUIRED = Set.of(Metric.OVERALL_ACCURACY, Metric.MACRO_F1);

    /** {@code key : value}, key starting at the beginning of the line (leading spaces allowed). */
    private static final Pattern LINE = Pattern.compile("^\\s*([A-Za-z][A-Za-z0-9' ]*?)\\s*:\\s*(\\S+)");
    private static final Pattern IMAGES = Pattern.compile("^\\s*Images\\s*:\\s*(\\d+)\\s+image pairs");
    private static final Pattern PATCHES = Pattern.compile("^\\s*Patches\\s*:\\s*(\\d+)\\s+of\\s+(\\d+x\\d+)");
    private static final Pattern TRUE_CHANGE = Pattern.compile(
            "^\\s*True change pixels\\s*:\\s*\\d+\\s*\\(([0-9.]+)%");

    private final Path file;
    private final Status status;
    private final String problem;
    private final Map<Metric, Double> values;
    private final String scope;
    private final Double trueChangePercent;

    private MetricsReport(Path file, Status status, String problem, Map<Metric, Double> values,
                          String scope, Double trueChangePercent) {
        this.file = file;
        this.status = status;
        this.problem = problem;
        this.values = values;
        this.scope = scope;
        this.trueChangePercent = trueChangePercent;
    }

    /** Reads a change-detection metrics file (requires a {@code Change F1} line). */
    public static MetricsReport readChangeDetection(Path file) {
        return read(file, CHANGE_DETECTION_REQUIRED);
    }

    /** Reads the EuroSAT metrics file (requires {@code Overall accuracy} and {@code Macro F1}). */
    public static MetricsReport readEuroSat(Path file) {
        return read(file, EUROSAT_REQUIRED);
    }

    /** Reads {@code file}; the report is {@link Status#MALFORMED} if any of {@code required} is absent. */
    public static MetricsReport read(Path file, Set<Metric> required) {
        if (file == null || !Files.isRegularFile(file)) {
            return new MetricsReport(file, Status.MISSING, "file not found: " + file, Map.of(), null, null);
        }
        List<String> lines;
        try {
            lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        } catch (IOException | RuntimeException e) {
            return new MetricsReport(file, Status.MALFORMED, "cannot read " + file + ": " + e.getMessage(),
                    Map.of(), null, null);
        }
        return parse(file, lines, required);
    }

    /** Parses already-read lines (visible for tests). */
    public static MetricsReport parse(Path file, List<String> lines, Set<Metric> required) {
        Map<Metric, Double> values = new EnumMap<>(Metric.class);
        String scope = null;
        Double trueChange = null;
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            if (scope == null) {
                Matcher im = IMAGES.matcher(line);
                Matcher pm = PATCHES.matcher(line);
                if (im.find()) {
                    scope = im.group(1) + " image pairs";
                } else if (pm.find()) {
                    scope = pm.group(1) + " patches of " + pm.group(2);
                }
            }
            if (trueChange == null) {
                Matcher tm = TRUE_CHANGE.matcher(line);
                if (tm.find()) {
                    trueChange = parseNumber(tm.group(1));
                }
            }
            Matcher m = LINE.matcher(line);
            if (!m.find()) {
                continue;
            }
            String key = m.group(1).trim();
            for (Metric metric : Metric.values()) {
                if (!metric.key.equalsIgnoreCase(key) || values.containsKey(metric)) {
                    continue;
                }
                Double v = parseValue(m.group(2), line.substring(m.end(2)));
                if (v == null) {
                    return malformed(file, String.format(Locale.ROOT, "'%s' is not a number (line %d)",
                            metric.key + " : " + m.group(2), i + 1));
                }
                if (!metric.inRange(v)) {
                    return malformed(file, String.format(Locale.ROOT, "%s = %s is out of range (line %d)",
                            metric.key, m.group(2), i + 1));
                }
                values.put(metric, v);
            }
        }
        for (Metric r : required) {
            if (!values.containsKey(r)) {
                return malformed(file, "no '" + r.key + " : <number>' line");
            }
        }
        return new MetricsReport(file, Status.OK, null, Collections.unmodifiableMap(values), scope, trueChange);
    }

    private static MetricsReport malformed(Path file, String reason) {
        return new MetricsReport(file, Status.MALFORMED, "malformed " + file + ": " + reason, Map.of(), null, null);
    }

    /**
     * A value token such as {@code 0.1008}; a percentage ({@code 97.25%} or {@code 97.25 %}) is
     * converted to a fraction so every metric is on the same 0..1 scale.
     */
    private static Double parseValue(String token, String rest) {
        boolean percent = token.endsWith("%") || rest.stripLeading().startsWith("%");
        String number = token.endsWith("%") ? token.substring(0, token.length() - 1) : token;
        Double v = parseNumber(number);
        if (v == null) {
            return null;
        }
        return percent ? Double.valueOf(v / 100.0) : v;
    }

    private static Double parseNumber(String s) {
        try {
            double v = Double.parseDouble(s.trim());
            return Double.isFinite(v) ? v : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }

    public Path getFile() { return file; }
    public Status getStatus() { return status; }
    public boolean isOk() { return status == Status.OK; }

    /** Why the report is not usable, or {@code null} when it is. */
    public String getProblem() { return problem; }

    /** The value of {@code metric}, empty if the file is not OK or has no such line. */
    public OptionalDouble get(Metric metric) {
        Double v = values.get(metric);
        return v == null ? OptionalDouble.empty() : OptionalDouble.of(v);
    }

    /** E.g. {@code 128 image pairs} or {@code 2048 patches of 256x256}, if the file states it. */
    public Optional<String> getScope() { return Optional.ofNullable(scope); }

    /** Share of true change pixels in % (baseline report line "True change pixels"), if present. */
    public OptionalDouble getTrueChangePercent() {
        return trueChangePercent == null ? OptionalDouble.empty() : OptionalDouble.of(trueChangePercent);
    }

    /** Table text: the value with 4 decimals, "not available" (missing file), "malformed file" or "n/a". */
    public String format(Metric metric) {
        switch (status) {
            case MISSING: return "not available";
            case MALFORMED: return "malformed file";
            default:
                OptionalDouble v = get(metric);
                return v.isPresent() ? String.format(Locale.ROOT, "%.4f", v.getAsDouble()) : "n/a";
        }
    }
}
