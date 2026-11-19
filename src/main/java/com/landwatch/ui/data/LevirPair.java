package com.landwatch.ui.data;

import com.landwatch.evaluation.LevirCdPairReader;
import com.landwatch.geospatial.SatelliteImageLoader;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;

/**
 * A LEVIR-CD image pair located from ONE file the user picked inside a split folder.
 *
 * <p>LEVIR-CD layout: {@code <split>/A/<name>} (earlier image), {@code <split>/B/<name>} (later image)
 * and {@code <split>/label/<name>} (ground-truth change mask, any value &gt; 0 = change). Picking
 * {@code test/A/test_7.png} or {@code test/B/test_7.png} (or the label) finds the other files by name.</p>
 *
 * @param name  file name shared by A, B and label (e.g. {@code test_7.png})
 * @param split split folder name (e.g. {@code test})
 * @param pre   A image (before)
 * @param post  B image (after)
 * @param label ground-truth mask, or {@code null} if {@code label/<name>} does not exist
 */
public record LevirPair(String name, String split, Path pre, Path post, Path label) {

    /** LEVIR-CD ground sample distance: 0.5 m per pixel, so one pixel covers 0.25 m². */
    public static final double LEVIR_GSD_METRES = 0.5;

    private static final Set<String> PAIR_DIRS = Set.of(
            LevirCdPairReader.PRE_DIR, LevirCdPairReader.POST_DIR, LevirCdPairReader.LABEL_DIR);

    /** A loaded pair: both images as RGB and, if a label exists, the binarised mask {@code [y][x]}. */
    public record Loaded(LevirPair pair, BufferedImage pre, BufferedImage post, boolean[][] groundTruth) {
        public boolean hasGroundTruth() {
            return groundTruth != null;
        }
    }

    public boolean hasLabel() {
        return label != null;
    }

    /** E.g. {@code LEVIR-CD test/test_7.png}. */
    public String shortName() {
        return "LEVIR-CD " + split + "/" + name;
    }

    /**
     * Finds A, B and (optionally) label for a file inside {@code <split>/A}, {@code <split>/B} or
     * {@code <split>/label}.
     *
     * @throws IOException with a user-facing explanation if the file is not inside such a folder or A/B
     *                     counterpart is missing
     */
    public static LevirPair locate(Path picked) throws IOException {
        if (picked == null) {
            throw new IOException("No file selected.");
        }
        Path file = picked.toAbsolutePath().normalize();
        Path dir = file.getParent();
        Path splitDir = (dir != null) ? dir.getParent() : null;
        if (dir == null || splitDir == null || dir.getFileName() == null
                || !PAIR_DIRS.contains(dir.getFileName().toString())) {
            throw new IOException("Not a LEVIR-CD pair file: " + file + ". Pick an image inside a split's A/ or B/ "
                    + "folder, for example datasets/LEVIR-CD/test/A/test_7.png.");
        }
        String name = file.getFileName().toString();
        Path pre = splitDir.resolve(LevirCdPairReader.PRE_DIR).resolve(name);
        Path post = splitDir.resolve(LevirCdPairReader.POST_DIR).resolve(name);
        Path label = splitDir.resolve(LevirCdPairReader.LABEL_DIR).resolve(name);
        if (!Files.isRegularFile(pre)) {
            throw new IOException("Missing before image (A): " + pre);
        }
        if (!Files.isRegularFile(post)) {
            throw new IOException("Missing after image (B): " + post);
        }
        String split = splitDir.getFileName() != null ? splitDir.getFileName().toString() : "?";
        return new LevirPair(name, split, pre, post, Files.isRegularFile(label) ? label : null);
    }

    /**
     * Loads A and B (as RGB) and the label (value &gt; 0 = change, via
     * {@link LevirCdPairReader#toChangeMask}).
     *
     * @throws IOException if a file cannot be decoded or the sizes differ
     */
    public Loaded load() throws IOException {
        BufferedImage a = SatelliteImageLoader.load(pre.toFile());
        BufferedImage b = SatelliteImageLoader.load(post.toFile());
        if (a.getWidth() != b.getWidth() || a.getHeight() != b.getHeight()) {
            throw new IOException(String.format(Locale.ROOT, "Size mismatch for %s: A=%dx%d, B=%dx%d",
                    name, a.getWidth(), a.getHeight(), b.getWidth(), b.getHeight()));
        }
        boolean[][] gt = null;
        if (label != null) {
            BufferedImage labelImage = ImageIO.read(label.toFile());
            if (labelImage == null) {
                throw new IOException("Unsupported or corrupt label image: " + label);
            }
            if (labelImage.getWidth() != a.getWidth() || labelImage.getHeight() != a.getHeight()) {
                throw new IOException(String.format(Locale.ROOT, "Label size %dx%d does not match the images (%dx%d): %s",
                        labelImage.getWidth(), labelImage.getHeight(), a.getWidth(), a.getHeight(), label));
            }
            gt = LevirCdPairReader.toChangeMask(labelImage);
        }
        return new Loaded(this, a, b, gt);
    }

    /** Area in m² for a pixel count, ASSUMING the LEVIR-CD ground sample distance of 0.5 m/px. */
    public static double areaSquareMetres(long pixels) {
        return pixels * LEVIR_GSD_METRES * LEVIR_GSD_METRES;
    }
}
