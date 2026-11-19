package com.landwatch.ui;

import com.landwatch.geospatial.ChangeVisualizer;
import com.landwatch.ui.engine.EngineResult;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Saves one analysis run to disk: the black-and-white mask, the red overlay, the thematic change map
 * and a plain-text summary, all sharing the prefix {@code landwatch_<engine-id>_<yyyyMMdd_HHmmss>}.
 */
public final class ResultExporter {

    public static final DateTimeFormatter FILE_TIMESTAMP = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
    public static final String SUFFIX_SUMMARY = "_summary.txt";

    private static final DateTimeFormatter HUMAN_TIMESTAMP = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private ResultExporter() {
    }

    /** File-name prefix shared by every file of one run, e.g. {@code landwatch_baseline-otsu_20261119_143000}. */
    public static String baseName(EngineResult result, LocalDateTime when) {
        return "landwatch_" + result.getEngineId() + "_" + when.format(FILE_TIMESTAMP);
    }

    /**
     * Like {@link #baseName} but appends {@code -2}, {@code -3}, ... if files with that prefix already exist
     * in {@code outputDirectory}, so two runs within the same second never overwrite each other.
     */
    public static String uniqueBaseName(File outputDirectory, EngineResult result, LocalDateTime when) {
        String base = baseName(result, when);
        String candidate = base;
        for (int i = 2; prefixInUse(outputDirectory, candidate); i++) {
            candidate = base + "-" + i;
        }
        return candidate;
    }

    private static boolean prefixInUse(File dir, String prefix) {
        String[] suffixes = {ChangeVisualizer.SUFFIX_BINARY_MASK_BW, ChangeVisualizer.SUFFIX_RED_OVERLAY,
                ChangeVisualizer.SUFFIX_THEMATIC_MAP, SUFFIX_SUMMARY};
        for (String s : suffixes) {
            if (new File(dir, prefix + s).exists()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Writes all files of one run into {@code outputDirectory} under a collision-free prefix
     * (see {@link #uniqueBaseName}).
     *
     * @param inputDescription where the images came from (stated honestly, e.g. "synthetic demo pair")
     * @return the files written: binary mask, red overlay, thematic map, summary
     */
    public static List<File> export(File outputDirectory, EngineResult result, ChangeVisualizationSet visuals,
                                    String inputDescription, LocalDateTime when) throws IOException {
        return export(outputDirectory, uniqueBaseName(outputDirectory, result, when), result, visuals,
                inputDescription, when);
    }

    /** Writes all files of one run into {@code outputDirectory} using the given file-name prefix. */
    public static List<File> export(File outputDirectory, String base, EngineResult result,
                                    ChangeVisualizationSet visuals, String inputDescription,
                                    LocalDateTime when) throws IOException {
        return export(outputDirectory, base, result, visuals, inputDescription, when, null);
    }

    /**
     * Like {@link #export(File, String, EngineResult, ChangeVisualizationSet, String, LocalDateTime)} but
     * appends {@code extraSection} (for example the per-image comparison with a ground-truth label) to the
     * summary file. {@code null} or blank adds nothing.
     */
    public static List<File> export(File outputDirectory, String base, EngineResult result,
                                    ChangeVisualizationSet visuals, String inputDescription,
                                    LocalDateTime when, String extraSection) throws IOException {
        List<File> files = new ArrayList<>(ChangeVisualizer.saveVisualizationSet(outputDirectory, base,
                visuals.getBinaryMask(), visuals.getRedOverlay(), visuals.getThematicMap()));

        File summary = new File(outputDirectory, base + SUFFIX_SUMMARY);
        String text = buildSummary(result, inputDescription, when);
        if (extraSection != null && !extraSection.isBlank()) {
            text = text + '\n' + extraSection + (extraSection.endsWith("\n") ? "" : "\n");
        }
        Files.writeString(summary.toPath(), text, StandardCharsets.UTF_8);
        files.add(summary);
        return files;
    }

    /** Plain-text summary of one run. Areas are in pixels only (no ground-sample-distance calibration). */
    public static String buildSummary(EngineResult r, String inputDescription, LocalDateTime when) {
        StringBuilder sb = new StringBuilder();
        sb.append("LANDWATCH AI - change detection run\n");
        sb.append("Generated:        ").append(when.format(HUMAN_TIMESTAMP)).append('\n');
        sb.append("Engine:           ").append(r.getEngineName()).append('\n');
        sb.append("Input:            ").append(inputDescription == null ? "unspecified" : inputDescription).append('\n');
        sb.append(String.format(Locale.US, "Image size:       %d x %d px (%,d px)%n",
                r.getWidth(), r.getHeight(), r.getTotalPixels()));
        sb.append(String.format(Locale.US, "Changed pixels:   %,d px (%.2f %% of image)%n",
                r.getChangedPixels(), r.getChangePercentage()));
        sb.append(String.format(Locale.US, "Change regions:   %d (8-connected)%n", r.getRegionCount()));
        sb.append("Otsu threshold:   ")
                .append(r.getOtsuThreshold() != null ? r.getOtsuThreshold() + " / 255" : "n/a (not used by this engine)")
                .append('\n');
        if (r.getRawChangedPixels() != null) {
            sb.append(String.format(Locale.US, "Raw changed px:   %,d (before 3x3 opening and region filter)%n",
                    r.getRawChangedPixels()));
        }
        sb.append(String.format(Locale.US, "Engine latency:   %d ms%n", r.getLatencyMs()));
        sb.append("Change level:     ").append(r.getChangeLevel().getLabel())
                .append("  (heuristic from % of changed pixels only)\n");
        sb.append('\n');
        sb.append("Note: areas are reported in pixels. No ground-sample-distance calibration is applied,\n");
        sb.append("so no m2 / hectare figures are given.\n");
        sb.append('\n');
        sb.append("Files of this run:\n");
        sb.append("  *").append(ChangeVisualizer.SUFFIX_BINARY_MASK_BW)
                .append("       black = unchanged, white = changed\n");
        sb.append("  *").append(ChangeVisualizer.SUFFIX_RED_OVERLAY)
                .append("          post-event image, red tint = changed, amber boxes = change regions\n");
        sb.append("  *").append(ChangeVisualizer.SUFFIX_THEMATIC_MAP)
                .append("  dimmed grayscale post-event image, cyan = changed, yellow = region outline\n");
        return sb.toString();
    }
}
