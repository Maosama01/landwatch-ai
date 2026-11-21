package com.landwatch.ui;

import com.landwatch.ui.data.EuroSatTestSet.DatasetMatch;
import com.landwatch.ui.data.LevirPair;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/**
 * Text and ordering for the "Land use (EuroSAT)" page (no JavaFX types, so it is unit-testable): the
 * probability list sorted from most to least likely, percentages that never round a non-zero value to
 * 0 or a value below 1 to 100, and the sentences that say where an image came from.
 */
public final class EuroSatResultFormat {

    private EuroSatResultFormat() {
    }

    /**
     * One line of the probability list.
     *
     * @param predicted true for the arg-max class (the prediction)
     * @param trueClass true for the image's known true class (never true when the class is unknown)
     */
    public record ProbabilityRow(int classIndex, String className, double probability, boolean predicted,
                                 boolean trueClass) {
    }

    /**
     * All classes sorted by decreasing probability (ties by class index, so the order is stable).
     *
     * @param trueIndex the true class index, or -1 when it is not known
     */
    public static List<ProbabilityRow> sortedRows(float[] probabilities, List<String> classNames, int trueIndex) {
        if (probabilities.length != classNames.size()) {
            throw new IllegalArgumentException(probabilities.length + " probabilities for " + classNames.size()
                    + " classes");
        }
        int best = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        List<ProbabilityRow> rows = new ArrayList<>(probabilities.length);
        for (int i = 0; i < probabilities.length; i++) {
            rows.add(new ProbabilityRow(i, classNames.get(i), probabilities[i], i == best, i == trueIndex));
        }
        rows.sort(Comparator.comparingDouble(ProbabilityRow::probability).reversed()
                .thenComparingInt(ProbabilityRow::classIndex));
        return List.copyOf(rows);
    }

    /**
     * A probability as a percentage with two decimals. Values that would round to 0.00 % but are not 0
     * are shown as "&lt;0.01 %", values that would round to 100.00 % but are below 1 as "&gt;99.99 %".
     */
    public static String percent(double p) {
        if (Double.isNaN(p)) {
            return "n/a";
        }
        double pct = p * 100.0;
        String text = String.format(Locale.ROOT, "%.2f", pct);
        if (p > 0 && text.equals("0.00")) {
            return "<0.01 %";
        }
        if (p < 1 && text.equals("100.00")) {
            return ">99.99 %";
        }
        return text + " %";
    }

    /** Width of a probability bar as a fraction of the track, clamped to 0..1. */
    public static double barFraction(double p) {
        if (Double.isNaN(p) || p <= 0) {
            return 0;
        }
        return Math.min(1.0, p);
    }

    /** "Correct" / "Wrong: true class is X" for a known true class; empty when the true class is unknown. */
    public static String verdict(String predicted, String trueClass) {
        if (trueClass == null) {
            return "";
        }
        return predicted.equals(trueClass) ? "Correct: the true class is " + trueClass
                : "Wrong: the true class is " + trueClass;
    }

    /**
     * Where the image came from, in one sentence: a EuroSAT test image (true label from the split), a
     * train or val image (with the warning that the model was trained or selected on it), a file in a
     * EuroSAT class folder that the split does not list, or the user's own file.
     */
    public static String sourceSentence(DatasetMatch match) {
        if (match == null) {
            return "Your own image: no true label is known, so this prediction cannot be checked here.";
        }
        if (match.isTestImage()) {
            return "EuroSAT test split · true label " + match.trueClass()
                    + " (from the split CSV). The model never saw this image during training or model selection.";
        }
        if ("train".equals(match.split())) {
            return "EuroSAT train split · true label " + match.trueClass()
                    + ". The model was trained on this image, so a correct answer here says little about unseen images.";
        }
        if ("val".equals(match.split())) {
            return "EuroSAT validation split · true label " + match.trueClass()
                    + ". This split chose the saved checkpoint, so it is not an unseen test image.";
        }
        return "EuroSAT class folder " + match.trueClass() + " · true label " + match.trueClass()
                + " (folder name); the split CSVs do not list this file.";
    }

    /**
     * The reason a file is NOT classified, if it is a LEVIR-CD pair image (a file in a split's A/, B/ or
     * label/ folder with its A and B counterparts, as {@link LevirPair#locate} recognises it). The
     * land-use classifier is never applied to LEVIR-CD imagery.
     */
    public static Optional<String> levirCdRefusal(Path file) {
        try {
            LevirPair pair = LevirPair.locate(file);
            return Optional.of("Not classified: this is a LEVIR-CD image (" + pair.shortName() + "), 0.5 m per pixel "
                    + "aerial imagery from the change-detection pairs. The land-use classifier was trained on 64×64 "
                    + "Sentinel-2 tiles at 10 m per pixel and is not applied to LEVIR-CD; analyse this pair on the "
                    + "Change detection page.");
        } catch (IOException | RuntimeException notLevir) {
            return Optional.empty();
        }
    }

    /** "64×64 px" style size text. */
    public static String sizeText(int width, int height) {
        return width + "×" + height + " px";
    }

    /** Gallery summary for {@code correct} of {@code total} live predictions in the shown sample. */
    public static String sampleSummary(int correct, int total, int skipped, String sampleDescription) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT, "%d of %d correct in this sample (classified live just now). ",
                correct, total));
        sb.append(sampleDescription);
        if (skipped > 0) {
            sb.append(String.format(Locale.ROOT, " %d listed image(s) could not be read and were left out.", skipped));
        }
        return sb.toString();
    }

    /** "12 ms on CPU" style latency text; the device is the DJL name, e.g. {@code cpu()} or {@code mps(0)}. */
    public static String latencyText(long millis, String djlDevice) {
        return millis + " ms on " + deviceText(djlDevice);
    }

    /** "mps(0)" → "Apple GPU (MPS)", "cpu()" → "CPU", anything else unchanged. */
    public static String deviceText(String djlDevice) {
        if (djlDevice == null) {
            return "?";
        }
        if (djlDevice.startsWith("mps")) {
            return "Apple GPU (MPS)";
        }
        if (djlDevice.startsWith("cpu")) {
            return "CPU";
        }
        if (djlDevice.startsWith("gpu")) {
            return "GPU";
        }
        return djlDevice;
    }
}
