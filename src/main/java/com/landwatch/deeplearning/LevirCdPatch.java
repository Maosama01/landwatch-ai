package com.landwatch.deeplearning;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * One 256x256 LEVIR-CD patch triplet (before image A, after image B, change label), as listed in
 * {@code processed_256/metadata/<split>_patches.csv}. Paths are absolute.
 *
 * @param name              patch file name, e.g. {@code train_1_patch_00.png}
 * @param imageA            earlier RGB image
 * @param imageB            later RGB image
 * @param label             8-bit grayscale mask; any value &gt; 0 means "change"
 * @param split             train, val or test
 * @param changePixelCount  number of change pixels according to the metadata
 * @param pixelCount        patch width x height
 */
public record LevirCdPatch(String name, Path imageA, Path imageB, Path label, String split,
                           long changePixelCount, long pixelCount) {

    /** The three LEVIR-CD splits used by this project. */
    public static final List<String> SPLITS = List.of("train", "val", "test");

    /**
     * Reads the metadata CSV of one split.
     *
     * @param datasetRoot the {@code processed_256} directory
     * @param split       train, val or test
     * @return the patches in file order
     */
    public static List<LevirCdPatch> readSplit(Path datasetRoot, String split) {
        if (!SPLITS.contains(split)) {
            throw new IllegalArgumentException("Unknown split '" + split + "', expected one of " + SPLITS);
        }
        Path csv = datasetRoot.resolve("metadata").resolve(split + "_patches.csv");
        List<String> lines;
        try {
            lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Cannot read " + csv, e);
        }
        if (lines.isEmpty()) {
            throw new IllegalStateException("Empty metadata file " + csv);
        }
        List<String> header = Arrays.asList(lines.get(0).trim().split(","));
        int iName = column(header, "patch_filename", csv);
        int iA = column(header, "relative_path_A", csv);
        int iB = column(header, "relative_path_B", csv);
        int iLabel = column(header, "relative_path_label", csv);
        int iSplit = column(header, "split", csv);
        int iCount = column(header, "change_pixel_count", csv);
        int iW = column(header, "patch_width", csv);
        int iH = column(header, "patch_height", csv);

        List<LevirCdPatch> patches = new ArrayList<>(lines.size() - 1);
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] f = line.split(",", -1);
            if (!split.equals(f[iSplit])) {
                throw new IllegalStateException("Row " + i + " of " + csv + " belongs to split " + f[iSplit]);
            }
            patches.add(new LevirCdPatch(f[iName],
                    datasetRoot.resolve(f[iA]), datasetRoot.resolve(f[iB]), datasetRoot.resolve(f[iLabel]),
                    f[iSplit], Long.parseLong(f[iCount]),
                    Long.parseLong(f[iW]) * Long.parseLong(f[iH])));
        }
        return Collections.unmodifiableList(patches);
    }

    private static int column(List<String> header, String name, Path csv) {
        int index = header.indexOf(name);
        if (index < 0) {
            throw new IllegalStateException("Column '" + name + "' missing in " + csv);
        }
        return index;
    }

    /**
     * Finds the LEVIR-CD {@code processed_256} directory. Checked in order: system property
     * {@code landwatch.levir.root}, environment variable {@code LEVIR_CD_ROOT}, then
     * {@code datasets/LEVIR-CD/processed_256} relative to the working directory.
     *
     * @return the directory, or {@code null} if none of the candidates contains the metadata
     */
    public static Path locateDatasetRoot() {
        List<String> candidates = new ArrayList<>();
        String prop = System.getProperty("landwatch.levir.root");
        if (prop != null && !prop.isBlank()) {
            candidates.add(prop);
        }
        String env = System.getenv("LEVIR_CD_ROOT");
        if (env != null && !env.isBlank()) {
            candidates.add(env);
        }
        candidates.add("datasets/LEVIR-CD/processed_256");
        for (String c : candidates) {
            Path p = Path.of(c).toAbsolutePath().normalize();
            if (Files.isRegularFile(p.resolve("metadata").resolve("train_patches.csv"))) {
                return p;
            }
        }
        return null;
    }
}
