package com.landwatch.deeplearning.eurosat;

import com.landwatch.deeplearning.TrainingSupport;

import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Stratified train / validation / test split of EuroSAT.
 *
 * <p>EuroSAT has no official split, so one is created here and committed to the repository
 * ({@code datasets/EuroSAT/splits/*.csv}) so that every run uses exactly the same images.</p>
 *
 * <h2>Procedure</h2>
 * <ol>
 *   <li>Classes are the sub-folders of the image folder ({@code EuroSAT_RGB}), sorted
 *       alphabetically; the position in that order is the class index.</li>
 *   <li>Within each class the {@code .jpg} files are put in a fixed order (by the number in
 *       {@code <Class>_<n>.jpg}, then by name) and shuffled with {@code java.util.Random(seed)}.
 *       One generator is used for all classes, visited in alphabetical order. Java specifies the
 *       algorithm of {@code Random} and {@code Collections.shuffle}, so the split is identical on
 *       every JVM.</li>
 *   <li>The first {@code round(0.70 n)} shuffled images of a class go to train, the next
 *       {@code round(0.15 n)} to val, and the rest to test. Doing this per class ("stratified")
 *       keeps the class proportions the same in all three splits.</li>
 * </ol>
 *
 * <p>Usage: {@code EuroSatSplit --root <EuroSAT folder with EuroSAT_RGB/> [--out <folder for
 * splits/ and dataset_summary.txt, default = root>] [--images EuroSAT_RGB] [--seed 42]}</p>
 */
public final class EuroSatSplit {

    /** Default image sub-folder of the EuroSAT root. */
    public static final String DEFAULT_IMAGE_DIR = "EuroSAT_RGB";
    public static final double TRAIN_FRACTION = 0.70;
    public static final double VAL_FRACTION = 0.15;
    public static final long DEFAULT_SEED = 42L;
    public static final String CSV_HEADER = "path,class_name,class_index";
    public static final List<String> SPLIT_NAMES = List.of("train", "val", "test");

    private static final Pattern NUMBERED = Pattern.compile("^(.*)_(\\d+)\\.[^.]+$");

    private EuroSatSplit() {
    }

    /** The three splits plus the class names in index order. */
    public record Split(List<String> classNames, List<EuroSatSample> train, List<EuroSatSample> val,
                        List<EuroSatSample> test) {

        public List<EuroSatSample> get(String name) {
            switch (name) {
                case "train":
                    return train;
                case "val":
                    return val;
                case "test":
                    return test;
                default:
                    throw new IllegalArgumentException("Unknown split " + name);
            }
        }

        /** {@code counts[splitIndex][classIndex]} for train, val, test. */
        public int[][] countsPerClass() {
            int[][] counts = new int[3][classNames.size()];
            for (int s = 0; s < 3; s++) {
                for (EuroSatSample x : get(SPLIT_NAMES.get(s))) {
                    counts[s][x.classIndex()]++;
                }
            }
            return counts;
        }
    }

    public static void main(String[] args) throws IOException {
        Map<String, String> opt = parseArgs(args);
        if (!opt.containsKey("root")) {
            System.err.println("Usage: EuroSatSplit --root <EuroSAT folder> [--out folder] [--images EuroSAT_RGB] [--seed 42]");
            System.exit(2);
        }
        Path root = Path.of(opt.get("root"));
        Path out = Path.of(opt.getOrDefault("out", opt.get("root")));
        String imageDir = opt.getOrDefault("images", DEFAULT_IMAGE_DIR);
        long seed = Long.parseLong(opt.getOrDefault("seed", String.valueOf(DEFAULT_SEED)));

        Split split = create(root, imageDir, seed);
        verifyDisjointAndComplete(split, listImages(root, imageDir));
        Path splitDir = out.resolve("splits");
        for (String name : SPLIT_NAMES) {
            writeCsv(splitDir.resolve(name + ".csv"), split.get(name));
        }
        String summary = summary(split, imageDir, seed);
        Files.writeString(out.resolve("dataset_summary.txt"), summary, StandardCharsets.UTF_8);
        System.out.print(summary);
        System.out.println("Wrote " + TrainingSupport.displayPath(splitDir) + "/{train,val,test}.csv and "
                + TrainingSupport.displayPath(out.resolve("dataset_summary.txt")));
    }

    /**
     * Finds the EuroSAT root folder (the one containing {@code EuroSAT_RGB/}): the system property
     * {@code landwatch.eurosat.root}, the environment variable {@code EUROSAT_ROOT}, or
     * {@code datasets/EuroSAT} relative to the working directory.
     *
     * @return the folder, or {@code null} if none of the candidates contains the images
     */
    public static Path locateDatasetRoot() {
        List<String> candidates = new ArrayList<>();
        String prop = System.getProperty("landwatch.eurosat.root");
        if (prop != null && !prop.isBlank()) {
            candidates.add(prop);
        }
        String env = System.getenv("EUROSAT_ROOT");
        if (env != null && !env.isBlank()) {
            candidates.add(env);
        }
        candidates.add("datasets/EuroSAT");
        for (String c : candidates) {
            Path p = Path.of(c).toAbsolutePath().normalize();
            if (Files.isDirectory(p.resolve(DEFAULT_IMAGE_DIR))) {
                return p;
            }
        }
        return null;
    }

    /** Class names: the sub-folders of {@code root/imageDir}, sorted alphabetically. */
    public static List<String> discoverClasses(Path root, String imageDir) throws IOException {
        Path dir = root.resolve(imageDir);
        if (!Files.isDirectory(dir)) {
            throw new IOException("Image folder not found: " + dir);
        }
        List<String> names = new ArrayList<>();
        try (Stream<Path> s = Files.list(dir)) {
            s.filter(Files::isDirectory)
                    .map(p -> p.getFileName().toString())
                    .filter(n -> !n.startsWith("."))
                    .forEach(names::add);
        }
        Collections.sort(names);
        if (names.size() < 2) {
            throw new IOException("Expected at least two class folders in " + dir + ", found " + names);
        }
        return List.copyOf(names);
    }

    /** All images, grouped by class (class-index order), each group in the fixed pre-shuffle order. */
    public static List<List<EuroSatSample>> listImages(Path root, String imageDir) throws IOException {
        List<String> classes = discoverClasses(root, imageDir);
        List<List<EuroSatSample>> perClass = new ArrayList<>();
        for (int c = 0; c < classes.size(); c++) {
            String name = classes.get(c);
            List<String> files = new ArrayList<>();
            try (Stream<Path> s = Files.list(root.resolve(imageDir).resolve(name))) {
                s.filter(Files::isRegularFile)
                        .map(p -> p.getFileName().toString())
                        .filter(n -> !n.startsWith(".") && n.toLowerCase(Locale.ROOT).endsWith(".jpg"))
                        .forEach(files::add);
            }
            if (files.isEmpty()) {
                throw new IOException("No .jpg images in class folder " + name);
            }
            files.sort(FILE_ORDER);
            List<EuroSatSample> samples = new ArrayList<>(files.size());
            for (String f : files) {
                samples.add(new EuroSatSample(imageDir + "/" + name + "/" + f, name, c));
            }
            perClass.add(samples);
        }
        return perClass;
    }

    /** Creates the stratified 70/15/15 split (see the class comment). */
    public static Split create(Path root, String imageDir, long seed) throws IOException {
        List<List<EuroSatSample>> perClass = listImages(root, imageDir);
        List<String> classes = new ArrayList<>();
        List<EuroSatSample> train = new ArrayList<>();
        List<EuroSatSample> val = new ArrayList<>();
        List<EuroSatSample> test = new ArrayList<>();
        Random rnd = new Random(seed);
        for (List<EuroSatSample> samples : perClass) {
            classes.add(samples.get(0).className());
            List<EuroSatSample> shuffled = new ArrayList<>(samples);
            Collections.shuffle(shuffled, rnd);
            int[] sizes = splitSizes(shuffled.size());
            train.addAll(shuffled.subList(0, sizes[0]));
            val.addAll(shuffled.subList(sizes[0], sizes[0] + sizes[1]));
            test.addAll(shuffled.subList(sizes[0] + sizes[1], shuffled.size()));
        }
        return new Split(List.copyOf(classes), sortedCopy(train), sortedCopy(val), sortedCopy(test));
    }

    /** {train, val, test} sizes for a class with {@code n} images: round(0.70n), round(0.15n), rest. */
    public static int[] splitSizes(int n) {
        int nTrain = (int) Math.round(n * TRAIN_FRACTION);
        int nVal = (int) Math.round(n * VAL_FRACTION);
        nTrain = Math.min(nTrain, n);
        nVal = Math.min(nVal, n - nTrain);
        return new int[] {nTrain, nVal, n - nTrain - nVal};
    }

    /**
     * Checks that no image appears in two splits (or twice in one) and, when {@code all} is given,
     * that the three splits together contain every image exactly once.
     *
     * @throws IllegalStateException if a check fails
     */
    public static void verifyDisjointAndComplete(Split split, List<List<EuroSatSample>> all) {
        Set<String> seen = new HashSet<>();
        for (String name : SPLIT_NAMES) {
            for (EuroSatSample s : split.get(name)) {
                if (!seen.add(s.relativePath())) {
                    throw new IllegalStateException("Image appears more than once across splits: " + s.relativePath());
                }
                if (!split.classNames().get(s.classIndex()).equals(s.className())) {
                    throw new IllegalStateException("Class index does not match class name: " + s);
                }
            }
        }
        if (all != null) {
            int total = 0;
            for (List<EuroSatSample> group : all) {
                for (EuroSatSample s : group) {
                    total++;
                    if (!seen.contains(s.relativePath())) {
                        throw new IllegalStateException("Image missing from all splits: " + s.relativePath());
                    }
                }
            }
            if (total != seen.size()) {
                throw new IllegalStateException("Splits contain " + seen.size() + " images, folder has " + total);
            }
        }
    }

    /** Writes a split CSV with header {@value #CSV_HEADER}. */
    public static void writeCsv(Path file, List<EuroSatSample> samples) throws IOException {
        if (file.getParent() != null) {
            Files.createDirectories(file.getParent());
        }
        try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            w.write(CSV_HEADER);
            w.write('\n');
            for (EuroSatSample s : samples) {
                w.write(s.relativePath() + "," + s.className() + "," + s.classIndex());
                w.write('\n');
            }
        }
    }

    /** Reads a split CSV written by {@link #writeCsv(Path, List)}. */
    public static List<EuroSatSample> readCsv(Path file) throws IOException {
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        if (lines.isEmpty() || !lines.get(0).trim().equals(CSV_HEADER)) {
            throw new IOException("Missing header '" + CSV_HEADER + "' in " + file);
        }
        List<EuroSatSample> out = new ArrayList<>(lines.size() - 1);
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }
            String[] parts = line.split(",");
            if (parts.length != 3) {
                throw new IOException("Bad line " + (i + 1) + " in " + file + ": " + line);
            }
            out.add(new EuroSatSample(parts[0], parts[1], Integer.parseInt(parts[2])));
        }
        return out;
    }

    /**
     * Reads {@code splitDir/{train,val,test}.csv} and checks that they are consistent: the same
     * class-index-to-name mapping everywhere, every index 0..K-1 used, names in alphabetical order.
     */
    public static Split read(Path splitDir) throws IOException {
        List<EuroSatSample> train = readCsv(splitDir.resolve("train.csv"));
        List<EuroSatSample> val = readCsv(splitDir.resolve("val.csv"));
        List<EuroSatSample> test = readCsv(splitDir.resolve("test.csv"));
        Map<Integer, String> names = new HashMap<>();
        for (List<EuroSatSample> list : List.of(train, val, test)) {
            for (EuroSatSample s : list) {
                String prev = names.putIfAbsent(s.classIndex(), s.className());
                if (prev != null && !prev.equals(s.className())) {
                    throw new IOException("Class index " + s.classIndex() + " is both " + prev + " and " + s.className());
                }
            }
        }
        List<String> classNames = new ArrayList<>();
        for (int c = 0; c < names.size(); c++) {
            String n = names.get(c);
            if (n == null) {
                throw new IOException("Class indices are not 0.." + (names.size() - 1) + " in " + splitDir);
            }
            classNames.add(n);
        }
        List<String> sorted = new ArrayList<>(classNames);
        Collections.sort(sorted);
        if (!sorted.equals(classNames)) {
            throw new IOException("Class indices are not in alphabetical order: " + classNames);
        }
        Split split = new Split(List.copyOf(classNames), train, val, test);
        verifyDisjointAndComplete(split, null);
        return split;
    }

    /** The text written to {@code dataset_summary.txt}. */
    public static String summary(Split split, String imageDir, long seed) {
        int[][] counts = split.countsPerClass();
        int k = split.classNames().size();
        StringBuilder sb = new StringBuilder();
        sb.append("EuroSAT (RGB) - stratified train / val / test split used by LandWatch AI\n");
        sb.append("=======================================================================\n");
        sb.append("Source     : EuroSAT, Helber et al., \"EuroSAT: A Novel Dataset and Deep Learning Benchmark for\n");
        sb.append("             Land Use and Land Cover Classification\", IEEE JSTARS 12(7), 2019.\n");
        sb.append("             Zenodo record 7711810 (EuroSAT_RGB.zip), MIT licence, MD5 verified on download.\n");
        sb.append("Imagery    : Sentinel-2 RGB bands, 64 x 64 px JPEG tiles at 10 m ground sample distance (640 m x 640 m).\n");
        sb.append("Images     : ").append(imageDir).append("/<Class>/<Class>_<n>.jpg (not redistributed in this repository).\n");
        sb.append("Created    : ").append(LocalDate.now()).append(" by com.landwatch.deeplearning.eurosat.EuroSatSplit\n");
        sb.append(String.format(Locale.ROOT,
                "Method     : per class, files sorted by <n>, shuffled with java.util.Random(%d) (one generator,%n"
                        + "             classes in alphabetical order); first round(%.2f n) -> train, next round(%.2f n) -> val,%n"
                        + "             rest -> test.%n", seed, TRAIN_FRACTION, VAL_FRACTION));
        sb.append("Class index: alphabetical order of the class folder names (0-based).\n");
        sb.append("Files      : splits/train.csv, splits/val.csv, splits/test.csv; columns ").append(CSV_HEADER)
                .append(";\n             paths are relative to datasets/EuroSAT/.\n\n");

        int nameWidth = "class".length();
        for (String n : split.classNames()) {
            nameWidth = Math.max(nameWidth, n.length());
        }
        String fmt = "%3s  %-" + nameWidth + "s  %6s  %6s  %6s  %6s%n";
        sb.append(String.format(Locale.ROOT, fmt, "idx", "class", "train", "val", "test", "total"));
        int[] totals = new int[3];
        for (int c = 0; c < k; c++) {
            int rowTotal = counts[0][c] + counts[1][c] + counts[2][c];
            sb.append(String.format(Locale.ROOT, "%3d  %-" + nameWidth + "s  %6d  %6d  %6d  %6d%n",
                    c, split.classNames().get(c), counts[0][c], counts[1][c], counts[2][c], rowTotal));
            for (int s = 0; s < 3; s++) {
                totals[s] += counts[s][c];
            }
        }
        int all = totals[0] + totals[1] + totals[2];
        sb.append(String.format(Locale.ROOT, "%3s  %-" + nameWidth + "s  %6d  %6d  %6d  %6d%n",
                "", "total", totals[0], totals[1], totals[2], all));
        sb.append(String.format(Locale.ROOT, "%3s  %-" + nameWidth + "s  %5.1f%%  %5.1f%%  %5.1f%%  %5.1f%%%n%n",
                "", "share", 100.0 * totals[0] / all, 100.0 * totals[1] / all, 100.0 * totals[2] / all, 100.0));
        sb.append("Checks     : no image appears in more than one split; the three splits together contain every\n");
        sb.append("             image of the folder exactly once; class indices match the class names.\n");
        return sb.toString();
    }

    /** Sorts by class index, then by file order. */
    private static List<EuroSatSample> sortedCopy(List<EuroSatSample> samples) {
        List<EuroSatSample> copy = new ArrayList<>(samples);
        copy.sort(Comparator.comparingInt(EuroSatSample::classIndex)
                .thenComparing(s -> s.relativePath().substring(s.relativePath().lastIndexOf('/') + 1), FILE_ORDER));
        return List.copyOf(copy);
    }

    /**
     * File order (a total order): names of the form {@code <prefix>_<n>.<ext>} first, by prefix and
     * then by the number n (so {@code Forest_2} comes before {@code Forest_10}); any other names after
     * them, alphabetically; remaining ties by full name.
     */
    static final Comparator<String> FILE_ORDER = Comparator
            .comparingInt((String n) -> NUMBERED.matcher(n).matches() ? 0 : 1)
            .thenComparing(n -> numberedPart(n, 1), Comparator.nullsFirst(Comparator.<String>naturalOrder()))
            .thenComparing(n -> numberedPart(n, 2) == null ? null : new BigInteger(numberedPart(n, 2)),
                    Comparator.nullsFirst(Comparator.<BigInteger>naturalOrder()))
            .thenComparing(Comparator.naturalOrder());

    private static String numberedPart(String name, int group) {
        Matcher m = NUMBERED.matcher(name);
        return m.matches() ? m.group(group) : null;
    }

    static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException("Expected --key value pairs, got: " + String.join(" ", args));
            }
            map.put(args[i].substring(2), args[++i]);
        }
        return map;
    }
}
