package com.landwatch.ui.data;

import com.landwatch.deeplearning.eurosat.EuroSatSample;
import com.landwatch.deeplearning.eurosat.EuroSatSplit;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;

/**
 * The EuroSAT TEST split as the "Land use (EuroSAT)" page uses it: the rows of {@code splits/test.csv}
 * (written by {@link EuroSatSplit}), whether the images themselves are installed, random picks for the
 * gallery, and the true label and split of any EuroSAT file the user opens.
 *
 * <p>No JavaFX and no model code, so everything here is unit-tested. The split CSVs are committed to
 * git; the images are not ({@code datasets/EuroSAT/EuroSAT_RGB/<ClassName>/*.jpg}, from
 * {@code EuroSAT_RGB.zip}, Zenodo record 7711810). Without the images the split still loads, but
 * {@link #imagesAvailable()} is false and nothing can be picked.</p>
 */
public final class EuroSatTestSet {

    /** Where a dataset image belongs: its true class (from the split CSV or its class folder) and split. */
    public record DatasetMatch(String trueClass, int trueIndex, String split) {

        /** True when the split CSVs list the image (otherwise only the folder name gave the class). */
        public boolean inKnownSplit() {
            return split != null;
        }

        public boolean isTestImage() {
            return "test".equals(split);
        }
    }

    private final Path datasetRoot;
    private final Path splitDir;
    private final List<EuroSatSample> test;
    private final List<String> classNames;
    /** Relative path ({@code EuroSAT_RGB/<Class>/<file>}) to split name, for train, val and test. */
    private final Map<String, String> splitByPath;
    private final String problem;

    private EuroSatTestSet(Path datasetRoot, Path splitDir, List<EuroSatSample> test, List<String> classNames,
                           Map<String, String> splitByPath, String problem) {
        this.datasetRoot = datasetRoot;
        this.splitDir = splitDir;
        this.test = test;
        this.classNames = classNames;
        this.splitByPath = splitByPath;
        this.problem = problem;
    }

    /** The split from the dashboard's configured locations ({@link AppPaths}). */
    public static EuroSatTestSet fromAppPaths() {
        return load(AppPaths.euroSatDatasetDir(), AppPaths.euroSatSplitDir());
    }

    /**
     * Reads {@code splitDir/test.csv} (required) and, for split lookups, {@code train.csv} and
     * {@code val.csv} when present. Never throws: a missing or unreadable test CSV gives an empty set
     * whose {@link #getProblem()} says why.
     */
    public static EuroSatTestSet load(Path datasetRoot, Path splitDir) {
        Path testCsv = splitDir.resolve("test.csv");
        if (!Files.isRegularFile(testCsv)) {
            return empty(datasetRoot, splitDir, "test split not found: " + AppPaths.shortForDisplay(testCsv));
        }
        List<EuroSatSample> test;
        try {
            test = EuroSatSplit.readCsv(testCsv);
        } catch (IOException | RuntimeException e) {
            return empty(datasetRoot, splitDir, "cannot read " + AppPaths.shortForDisplay(testCsv) + ": " + e.getMessage());
        }
        if (test.isEmpty()) {
            return empty(datasetRoot, splitDir, AppPaths.shortForDisplay(testCsv) + " has no rows");
        }
        Map<String, String> splitByPath = new HashMap<>();
        TreeMap<Integer, String> names = new TreeMap<>();
        for (EuroSatSample s : test) {
            splitByPath.put(s.relativePath(), "test");
            String prev = names.putIfAbsent(s.classIndex(), s.className());
            if (prev != null && !prev.equals(s.className())) {
                return empty(datasetRoot, splitDir, "class index " + s.classIndex() + " is both " + prev + " and "
                        + s.className() + " in " + AppPaths.shortForDisplay(testCsv));
            }
        }
        for (String other : List.of("train", "val")) {
            Path csv = splitDir.resolve(other + ".csv");
            if (!Files.isRegularFile(csv)) {
                continue;
            }
            try {
                for (EuroSatSample s : EuroSatSplit.readCsv(csv)) {
                    splitByPath.putIfAbsent(s.relativePath(), other);
                }
            } catch (IOException | RuntimeException ignored) {
                // Only used to say "this image is in the train split"; the test split itself is fine.
            }
        }
        List<String> classNames = new ArrayList<>();
        for (int c = 0; c <= names.lastKey(); c++) {
            String n = names.get(c);
            if (n == null) {
                return empty(datasetRoot, splitDir, "class indices in " + AppPaths.shortForDisplay(testCsv)
                        + " are not 0.." + names.lastKey());
            }
            classNames.add(n);
        }
        return new EuroSatTestSet(datasetRoot, splitDir, List.copyOf(test), List.copyOf(classNames),
                Collections.unmodifiableMap(splitByPath), null);
    }

    private static EuroSatTestSet empty(Path datasetRoot, Path splitDir, String problem) {
        return new EuroSatTestSet(datasetRoot, splitDir, List.of(), List.of(), Map.of(), problem);
    }

    /** True when the test CSV was read. */
    public boolean isOk() {
        return problem == null;
    }

    /** Why the split could not be read, or {@code null}. */
    public String getProblem() {
        return problem;
    }

    public Path getDatasetRoot() {
        return datasetRoot;
    }

    public Path getSplitDir() {
        return splitDir;
    }

    /** {@code <root>/EuroSAT_RGB}. */
    public Path getImagesDir() {
        return datasetRoot.resolve(AppPaths.EUROSAT_IMAGE_DIR);
    }

    /** True when the image folder exists (checked live, so unpacking the dataset while the app runs works). */
    public boolean imagesAvailable() {
        return Files.isDirectory(getImagesDir());
    }

    /** The test rows in file order. */
    public List<EuroSatSample> getSamples() {
        return test;
    }

    public int size() {
        return test.size();
    }

    /** Class names in class-index order, as the test CSV numbers them. */
    public List<String> getClassNames() {
        return classNames;
    }

    /** The image file of a row. */
    public Path resolve(EuroSatSample sample) {
        return sample.resolve(datasetRoot);
    }

    /** Short user-facing text saying where the images must go when {@link #imagesAvailable()} is false. */
    public String missingImagesMessage() {
        return "EuroSAT images not found at " + AppPaths.shortForDisplay(getImagesDir()) + ". The dataset is not "
                + "in git: unpack EuroSAT_RGB.zip (Zenodo record 7711810) so the images are at "
                + AppPaths.shortForDisplay(getImagesDir()) + "/<ClassName>/*.jpg, or start the app with -D"
                + AppPaths.EUROSAT_DIR_PROPERTY + "=<folder containing EuroSAT_RGB>. Browse and drag-and-drop "
                + "still classify any image with the model alone.";
    }

    // ------------------------------------------------------------------ picking

    /** One uniformly random test row. */
    public EuroSatSample pickOne(Random rnd) {
        if (test.isEmpty()) {
            throw new IllegalStateException("No test images: " + problem);
        }
        return test.get(rnd.nextInt(test.size()));
    }

    /** {@code n} distinct test rows, uniformly at random (all of them, shuffled, if {@code n >= size()}). */
    public List<EuroSatSample> pickRandom(int n, Random rnd) {
        return pickDistinct(test, n, rnd);
    }

    /** {@code perClass} random rows of every class, in class order. */
    public List<EuroSatSample> pickPerClass(int perClass, Random rnd) {
        List<List<EuroSatSample>> byClass = new ArrayList<>();
        for (int c = 0; c < classNames.size(); c++) {
            byClass.add(new ArrayList<>());
        }
        for (EuroSatSample s : test) {
            byClass.get(s.classIndex()).add(s);
        }
        List<EuroSatSample> out = new ArrayList<>();
        for (List<EuroSatSample> group : byClass) {
            out.addAll(pickDistinct(group, perClass, rnd));
        }
        return out;
    }

    /**
     * {@code n} random test rows among those whose relative path is in {@code relativePaths} (for example
     * the images the recorded evaluation got wrong). Paths that are not in the test split are ignored.
     */
    public List<EuroSatSample> pickAmong(Collection<String> relativePaths, int n, Random rnd) {
        Set<String> wanted = new HashSet<>(relativePaths);
        List<EuroSatSample> pool = new ArrayList<>();
        for (EuroSatSample s : test) {
            if (wanted.contains(s.relativePath())) {
                pool.add(s);
            }
        }
        return pickDistinct(pool, n, rnd);
    }

    /** Partial Fisher-Yates shuffle: the first {@code n} of a random permutation of {@code pool}. */
    static <T> List<T> pickDistinct(List<T> pool, int n, Random rnd) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be >= 0: " + n);
        }
        List<T> copy = new ArrayList<>(pool);
        int k = Math.min(n, copy.size());
        for (int i = 0; i < k; i++) {
            int j = i + rnd.nextInt(copy.size() - i);
            Collections.swap(copy, i, j);
        }
        return List.copyOf(copy.subList(0, k));
    }

    // ------------------------------------------------------------------ lookup

    /**
     * The true class of a EuroSAT image file, if it is one: a file {@code .../EuroSAT_RGB/<Class>/<name>}
     * whose class folder is one of the EuroSAT classes. The split is taken from the split CSVs (by the
     * relative path {@code EuroSAT_RGB/<Class>/<name>}, so a copy of the dataset in another folder is
     * recognised too); it is {@code null} when the CSVs do not list the file.
     */
    public Optional<DatasetMatch> lookup(Path file) {
        if (file == null || classNames.isEmpty()) {
            return Optional.empty();
        }
        Path abs = file.toAbsolutePath().normalize();
        Path classDir = abs.getParent();
        Path imagesDir = classDir == null ? null : classDir.getParent();
        if (classDir == null || imagesDir == null || imagesDir.getFileName() == null
                || !AppPaths.EUROSAT_IMAGE_DIR.equals(imagesDir.getFileName().toString())) {
            return Optional.empty();
        }
        String className = classDir.getFileName().toString();
        int index = classNames.indexOf(className);
        if (index < 0) {
            return Optional.empty();
        }
        String relative = AppPaths.EUROSAT_IMAGE_DIR + "/" + className + "/" + abs.getFileName();
        return Optional.of(new DatasetMatch(className, index, splitByPath.get(relative)));
    }

    // ------------------------------------------------------------------ recorded evaluation

    /**
     * Relative paths of the images that the recorded TEST evaluation classified wrongly: rows of
     * {@code results/eurosat/test_predictions.csv} (columns found by header name) whose {@code correct}
     * column is {@code no}, in file order.
     *
     * @throws IOException if the file cannot be read or lacks the {@code path} / {@code correct} columns
     */
    public static List<String> readRecordedErrors(Path predictionsCsv) throws IOException {
        List<String> lines = Files.readAllLines(predictionsCsv, StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("empty file " + AppPaths.shortForDisplay(predictionsCsv));
        }
        List<String> header = List.of(lines.get(0).trim().split(","));
        int pathCol = header.indexOf("path");
        int correctCol = header.indexOf("correct");
        if (pathCol < 0 || correctCol < 0) {
            throw new IOException("no 'path' and 'correct' columns in " + AppPaths.shortForDisplay(predictionsCsv));
        }
        Set<String> wrong = new LinkedHashSet<>();
        for (int i = 1; i < lines.size(); i++) {
            String[] parts = lines.get(i).trim().split(",");
            if (parts.length <= Math.max(pathCol, correctCol)) {
                continue;
            }
            if ("no".equals(parts[correctCol].trim().toLowerCase(Locale.ROOT))) {
                wrong.add(parts[pathCol].trim());
            }
        }
        return List.copyOf(wrong);
    }
}
