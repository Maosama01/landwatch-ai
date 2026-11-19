package com.landwatch.evaluation;

import com.landwatch.geospatial.SatelliteImageLoader;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.IndexColorModel;
import java.awt.image.Raster;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.TreeSet;

/**
 * Reads image triplets from a LEVIR-CD style dataset.
 *
 * <p>Expected layout: {@code <root>/<split>/A/<name>} (earlier image),
 * {@code <root>/<split>/B/<name>} (later image) and {@code <root>/<split>/label/<name>}
 * (ground-truth change mask). Files are paired by identical file name.</p>
 *
 * <p>Hidden files (any name starting with '.', which includes macOS {@code ._*}
 * AppleDouble files and {@code .DS_Store}), directories and files without an image
 * extension are ignored. Triplets are returned in natural order, so {@code test_2.png}
 * comes before {@code test_10.png}.</p>
 *
 * <p>Ground truth is binarised as <b>value &gt; 0 = change</b>. LEVIR-CD masks use 255
 * for change, but some official masks contain a few stray 156 or 254 pixels, which also
 * count as change under this rule.</p>
 */
public final class LevirCdPairReader {

    /** Sub-folder holding the earlier (pre-change) image. */
    public static final String PRE_DIR = "A";
    /** Sub-folder holding the later (post-change) image. */
    public static final String POST_DIR = "B";
    /** Sub-folder holding the ground-truth change mask. */
    public static final String LABEL_DIR = "label";

    private static final List<String> SUB_DIRS = List.of(PRE_DIR, POST_DIR, LABEL_DIR);
    private static final Set<String> IMAGE_EXTENSIONS = Set.of("png", "jpg", "jpeg", "bmp", "tif", "tiff");
    private static final int MAX_LISTED_PROBLEMS = 20;

    /** Natural ("human") file-name order: digit runs compare by numeric value. */
    public static final Comparator<String> NATURAL_ORDER = LevirCdPairReader::compareNatural;

    private final Path datasetRoot;

    /** @param datasetRoot folder that contains the split folders (train, val, test) */
    public LevirCdPairReader(Path datasetRoot) {
        if (datasetRoot == null) {
            throw new IllegalArgumentException("Dataset root must not be null");
        }
        this.datasetRoot = datasetRoot;
    }

    public Path getDatasetRoot() {
        return datasetRoot;
    }

    /** One complete A / B / label file triplet, paired by file name. */
    public record Triplet(String name, Path pre, Path post, Path label) {
    }

    /** A loaded triplet: both images as TYPE_INT_RGB and the binarised mask indexed {@code [y][x]}. */
    public record Sample(String name, BufferedImage pre, BufferedImage post, boolean[][] changeMask) {
        public int width() {
            return pre.getWidth();
        }

        public int height() {
            return pre.getHeight();
        }
    }

    /** Thrown when a file in one of A, B or label has no counterpart with the same name in the others. */
    public static final class IncompleteSplitException extends IOException {
        private final List<String> problems;

        IncompleteSplitException(String message, List<String> problems) {
            super(message);
            this.problems = List.copyOf(problems);
        }

        /** One human-readable line per unmatched file name. */
        public List<String> getProblems() {
            return problems;
        }
    }

    /**
     * Lists every triplet of a split in natural order.
     *
     * @throws IncompleteSplitException if any file lacks a counterpart in one of the other folders
     *                                  (the message names each file and the folders it is missing from)
     * @throws IOException              if the split or one of its A / B / label folders does not exist
     */
    public List<Triplet> listTriplets(String split) throws IOException {
        if (split == null || split.isBlank()) {
            throw new IllegalArgumentException("Split name must not be empty");
        }
        Path splitDir = datasetRoot.resolve(split);
        if (!Files.isDirectory(splitDir)) {
            throw new IOException("Split folder not found: " + splitDir.toAbsolutePath()
                    + " (expected <dataset>/" + split + "/{A,B,label})");
        }

        List<Set<String>> namesPerDir = new ArrayList<>();
        for (String sub : SUB_DIRS) {
            Path dir = splitDir.resolve(sub);
            if (!Files.isDirectory(dir)) {
                throw new IOException("Missing folder '" + sub + "' in split '" + split + "': " + dir.toAbsolutePath());
            }
            namesPerDir.add(listImageNames(dir));
        }

        Set<String> allNames = new TreeSet<>(NATURAL_ORDER);
        namesPerDir.forEach(allNames::addAll);

        List<Triplet> triplets = new ArrayList<>();
        List<String> problems = new ArrayList<>();
        for (String name : allNames) {
            List<String> presentIn = new ArrayList<>();
            List<String> missingIn = new ArrayList<>();
            for (int i = 0; i < SUB_DIRS.size(); i++) {
                (namesPerDir.get(i).contains(name) ? presentIn : missingIn).add(SUB_DIRS.get(i));
            }
            if (missingIn.isEmpty()) {
                triplets.add(new Triplet(name,
                        splitDir.resolve(PRE_DIR).resolve(name),
                        splitDir.resolve(POST_DIR).resolve(name),
                        splitDir.resolve(LABEL_DIR).resolve(name)));
            } else {
                problems.add(name + ": present in " + String.join(", ", presentIn)
                        + "; missing in " + String.join(", ", missingIn));
            }
        }

        if (!problems.isEmpty()) {
            StringBuilder msg = new StringBuilder();
            msg.append("Split '").append(split).append("' at ").append(splitDir.toAbsolutePath())
                    .append(" is incomplete: ").append(problems.size())
                    .append(" file name(s) have no counterpart in A, B and label");
            for (int i = 0; i < Math.min(problems.size(), MAX_LISTED_PROBLEMS); i++) {
                msg.append(System.lineSeparator()).append("  ").append(problems.get(i));
            }
            if (problems.size() > MAX_LISTED_PROBLEMS) {
                msg.append(System.lineSeparator()).append("  ... and ")
                        .append(problems.size() - MAX_LISTED_PROBLEMS).append(" more");
            }
            throw new IncompleteSplitException(msg.toString(), problems);
        }
        return Collections.unmodifiableList(triplets);
    }

    /**
     * Loads a triplet: A and B through {@link SatelliteImageLoader#load(java.io.File)} and the label
     * binarised with {@link #toChangeMask(BufferedImage)}.
     *
     * @throws IOException if a file cannot be decoded or the three images differ in size
     */
    public Sample load(Triplet triplet) throws IOException {
        BufferedImage pre = SatelliteImageLoader.load(triplet.pre().toFile());
        BufferedImage post = SatelliteImageLoader.load(triplet.post().toFile());
        BufferedImage label = ImageIO.read(triplet.label().toFile());
        if (label == null) {
            throw new IOException("Unsupported or corrupt label image: " + triplet.label().toAbsolutePath());
        }
        if (pre.getWidth() != post.getWidth() || pre.getHeight() != post.getHeight()
                || pre.getWidth() != label.getWidth() || pre.getHeight() != label.getHeight()) {
            throw new IOException(String.format(Locale.ROOT,
                    "Size mismatch for %s: A=%dx%d, B=%dx%d, label=%dx%d", triplet.name(),
                    pre.getWidth(), pre.getHeight(), post.getWidth(), post.getHeight(),
                    label.getWidth(), label.getHeight()));
        }
        return new Sample(triplet.name(), pre, post, toChangeMask(label));
    }

    /**
     * Converts a ground-truth mask image to {@code boolean[y][x]} with <b>value &gt; 0 = change</b>.
     *
     * <ul>
     *   <li>Palette (indexed) masks, including 1-bit masks: a pixel is change when its palette
     *       colour is not black (any of R, G, B &gt; 0).</li>
     *   <li>Grayscale masks (8- or 16-bit, with or without alpha): the raw stored gray sample
     *       &gt; 0, read from the raster so no colour-space conversion is applied.</li>
     *   <li>Colour masks: any raw colour sample (R, G or B) &gt; 0. Alpha is ignored.</li>
     * </ul>
     */
    public static boolean[][] toChangeMask(BufferedImage label) {
        if (label == null) {
            throw new IllegalArgumentException("Label image must not be null");
        }
        int width = label.getWidth();
        int height = label.getHeight();
        boolean[][] mask = new boolean[height][width];
        Raster raster = label.getRaster();
        ColorModel cm = label.getColorModel();

        if (cm instanceof IndexColorModel icm) {
            boolean[] paletteIsChange = new boolean[icm.getMapSize()];
            for (int i = 0; i < paletteIsChange.length; i++) {
                paletteIsChange[i] = icm.getRed(i) > 0 || icm.getGreen(i) > 0 || icm.getBlue(i) > 0;
            }
            int[] row = new int[width];
            for (int y = 0; y < height; y++) {
                raster.getSamples(0, y, width, 1, 0, row);
                for (int x = 0; x < width; x++) {
                    int index = row[x];
                    mask[y][x] = index >= 0 && index < paletteIsChange.length && paletteIsChange[index];
                }
            }
            return mask;
        }

        int colourBands = Math.min(cm.getNumColorComponents(), raster.getNumBands());
        double[] row = new double[width];
        for (int band = 0; band < colourBands; band++) {
            for (int y = 0; y < height; y++) {
                raster.getSamples(0, y, width, 1, band, row);
                for (int x = 0; x < width; x++) {
                    if (row[x] > 0) {
                        mask[y][x] = true;
                    }
                }
            }
        }
        return mask;
    }

    /** Compares file names so that embedded numbers sort by value ("test_2" before "test_10"). */
    public static int compareNatural(String a, String b) {
        int i = 0;
        int j = 0;
        while (i < a.length() && j < b.length()) {
            char ca = a.charAt(i);
            char cb = b.charAt(j);
            if (isAsciiDigit(ca) && isAsciiDigit(cb)) {
                int startA = i;
                int startB = j;
                while (i < a.length() && isAsciiDigit(a.charAt(i))) {
                    i++;
                }
                while (j < b.length() && isAsciiDigit(b.charAt(j))) {
                    j++;
                }
                String numA = stripLeadingZeros(a.substring(startA, i));
                String numB = stripLeadingZeros(b.substring(startB, j));
                if (numA.length() != numB.length()) {
                    return Integer.compare(numA.length(), numB.length());
                }
                int cmp = numA.compareTo(numB);
                if (cmp != 0) {
                    return cmp;
                }
            } else {
                if (ca != cb) {
                    return Character.compare(ca, cb);
                }
                i++;
                j++;
            }
        }
        int cmp = Integer.compare(a.length() - i, b.length() - j);
        return cmp != 0 ? cmp : a.compareTo(b);
    }

    private static Set<String> listImageNames(Path dir) throws IOException {
        Set<String> names = new TreeSet<>(NATURAL_ORDER);
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path p : stream) {
                String name = p.getFileName().toString();
                if (name.startsWith(".") || !Files.isRegularFile(p) || !hasImageExtension(name)) {
                    continue;
                }
                names.add(name);
            }
        }
        return names;
    }

    private static boolean hasImageExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 && IMAGE_EXTENSIONS.contains(name.substring(dot + 1).toLowerCase(Locale.ROOT));
    }

    private static boolean isAsciiDigit(char c) {
        return c >= '0' && c <= '9';
    }

    private static String stripLeadingZeros(String digits) {
        int k = 0;
        while (k < digits.length() - 1 && digits.charAt(k) == '0') {
            k++;
        }
        return digits.substring(k);
    }
}
