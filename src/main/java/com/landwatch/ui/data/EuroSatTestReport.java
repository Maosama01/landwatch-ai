package com.landwatch.ui.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The recorded EuroSAT TEST evaluation, {@code results/eurosat/test_metrics.txt} (written by
 * {@code EvaluateEuroSatClassifier}), read at runtime for the "Land use (EuroSAT)" page.
 *
 * <p>The headline values (overall accuracy, macro F1) come from {@link MetricsReport#readEuroSat}, the
 * same reader the Benchmarks page uses. This class adds what that page does not show: the image count
 * line, the per-class precision / recall / F1 table, the most frequent confusions and the protocol line.
 * Nothing is filled in: a part that is absent or unusable is simply empty, and the page says so.</p>
 */
public final class EuroSatTestReport {

    /** One row of the per-class table. */
    public record ClassRow(int index, String className, double precision, double recall, double f1, int support) {
    }

    /** One "actual -> predicted: count" line. */
    public record Confusion(String actual, String predicted, int count) {
    }

    private static final Pattern IMAGES = Pattern.compile(
            "^\\s*Images\\s*:\\s*(\\d+)\\s*\\(\\s*(\\d+)\\s+correct\\s*,\\s*(\\d+)\\s+wrong\\s*\\)");
    private static final Pattern CLASS_ROW = Pattern.compile(
            "^\\s*(\\d+)\\s+(\\S+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+([0-9.]+)\\s+(\\d+)\\s*$");
    private static final Pattern CONFUSION = Pattern.compile("^\\s*(\\S+)\\s*->\\s*(\\S+)\\s*:\\s*(\\d+)\\s*$");
    private static final Pattern PROTOCOL = Pattern.compile("^\\s*Protocol\\s*:\\s*(.+?)\\s*$");
    private static final Pattern GENERATED = Pattern.compile("^\\s*Generated\\s*:\\s*(\\S+)");

    private final MetricsReport headline;
    private final Integer images;
    private final Integer correct;
    private final Integer wrong;
    private final List<ClassRow> perClass;
    private final String perClassProblem;
    private final List<Confusion> confusions;
    private final String protocol;
    private final String generated;

    private EuroSatTestReport(MetricsReport headline, Integer images, Integer correct, Integer wrong,
                              List<ClassRow> perClass, String perClassProblem, List<Confusion> confusions,
                              String protocol, String generated) {
        this.headline = headline;
        this.images = images;
        this.correct = correct;
        this.wrong = wrong;
        this.perClass = perClass;
        this.perClassProblem = perClassProblem;
        this.confusions = confusions;
        this.protocol = protocol;
        this.generated = generated;
    }

    /** Reads {@code file}; never throws (a missing file gives a report whose headline is MISSING). */
    public static EuroSatTestReport read(Path file) {
        MetricsReport headline = MetricsReport.readEuroSat(file);
        if (!headline.isOk()) {
            return new EuroSatTestReport(headline, null, null, null, List.of(), null, List.of(), null, null);
        }
        try {
            return parse(headline, Files.readAllLines(file, StandardCharsets.UTF_8));
        } catch (IOException | RuntimeException e) {
            return new EuroSatTestReport(headline, null, null, null, List.of(), "cannot read the file again: "
                    + e.getMessage(), List.of(), null, null);
        }
    }

    /** Parses already-read lines (visible for tests); {@code headline} must come from the same file. */
    static EuroSatTestReport parse(MetricsReport headline, List<String> lines) {
        Integer images = null;
        Integer correct = null;
        Integer wrong = null;
        String protocol = null;
        String generated = null;
        List<ClassRow> rows = new ArrayList<>();
        String rowProblem = null;
        List<Confusion> confusions = new ArrayList<>();
        String section = "";
        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.startsWith("Per-class metrics")) {
                section = "per-class";
                continue;
            }
            if (trimmed.startsWith("Most frequent confusions")) {
                section = "confusions";
                continue;
            }
            if (trimmed.startsWith("Confusion matrix") || trimmed.startsWith("Inference")) {
                section = "";
            }
            if (images == null) {
                Matcher m = IMAGES.matcher(line);
                if (m.find()) {
                    images = Integer.parseInt(m.group(1));
                    correct = Integer.parseInt(m.group(2));
                    wrong = Integer.parseInt(m.group(3));
                }
            }
            if (protocol == null) {
                Matcher m = PROTOCOL.matcher(line);
                if (m.find()) {
                    protocol = m.group(1);
                }
            }
            if (generated == null) {
                Matcher m = GENERATED.matcher(line);
                if (m.find()) {
                    generated = m.group(1);
                }
            }
            if (section.equals("per-class")) {
                Matcher m = CLASS_ROW.matcher(line);
                if (m.matches()) {
                    double p = Double.parseDouble(m.group(3));
                    double r = Double.parseDouble(m.group(4));
                    double f = Double.parseDouble(m.group(5));
                    if (!inUnitRange(p) || !inUnitRange(r) || !inUnitRange(f)) {
                        rowProblem = "per-class value out of range: " + trimmed;
                    }
                    rows.add(new ClassRow(Integer.parseInt(m.group(1)), m.group(2), p, r, f,
                            Integer.parseInt(m.group(6))));
                }
            } else if (section.equals("confusions")) {
                Matcher m = CONFUSION.matcher(line);
                if (m.matches()) {
                    confusions.add(new Confusion(m.group(1), m.group(2), Integer.parseInt(m.group(3))));
                }
            }
        }
        if (rowProblem == null && rows.isEmpty()) {
            rowProblem = "no per-class table in the file";
        }
        List<ClassRow> perClass = rowProblem == null ? List.copyOf(rows) : List.of();
        if (images != null && correct + wrong != images) {
            // An inconsistent count line is not shown rather than "corrected".
            images = null;
            correct = null;
            wrong = null;
        }
        return new EuroSatTestReport(headline, images, correct, wrong, perClass, rowProblem,
                List.copyOf(confusions), protocol, generated);
    }

    private static boolean inUnitRange(double v) {
        return v >= 0.0 && v <= 1.0;
    }

    /** Overall accuracy and macro F1 (status MISSING / MALFORMED when the file is unusable). */
    public MetricsReport getHeadline() {
        return headline;
    }

    public boolean isOk() {
        return headline.isOk();
    }

    public Path getFile() {
        return headline.getFile();
    }

    public OptionalInt getImages() {
        return images == null ? OptionalInt.empty() : OptionalInt.of(images);
    }

    public OptionalInt getCorrect() {
        return correct == null ? OptionalInt.empty() : OptionalInt.of(correct);
    }

    public OptionalInt getWrong() {
        return wrong == null ? OptionalInt.empty() : OptionalInt.of(wrong);
    }

    /** The per-class table, empty if absent or unusable ({@link #getPerClassProblem()} says why). */
    public List<ClassRow> getPerClass() {
        return perClass;
    }

    public String getPerClassProblem() {
        return perClassProblem;
    }

    /** Most frequent confusions in file order (largest first, as the evaluator writes them). */
    public List<Confusion> getConfusions() {
        return confusions;
    }

    /** The "Protocol:" sentence, e.g. that TEST was evaluated once. */
    public Optional<String> getProtocol() {
        return Optional.ofNullable(protocol);
    }

    /** The "Generated:" timestamp as written. */
    public Optional<String> getGenerated() {
        return Optional.ofNullable(generated);
    }
}
