package com.landwatch.ui.data;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Locations the dashboard reads at runtime, all relative to the working directory (the project root
 * under {@code mvn javafx:run}) unless overridden by a system property.
 */
public final class AppPaths {

    /** System property overriding the LEVIR-CD dataset folder. */
    public static final String DATASET_DIR_PROPERTY = "landwatch.dataset.dir";
    public static final String DEFAULT_DATASET_DIR = "datasets/LEVIR-CD";

    /** System property overriding the results folder. */
    public static final String RESULTS_DIR_PROPERTY = "landwatch.results.dir";
    public static final String DEFAULT_RESULTS_DIR = "results";

    /**
     * System property overriding the EuroSAT folder (the one containing {@code EuroSAT_RGB/} and
     * {@code splits/}). Same name as the EuroSAT command-line tools use ({@code EuroSatSplit.locateDatasetRoot}).
     */
    public static final String EUROSAT_DIR_PROPERTY = "landwatch.eurosat.root";
    public static final String DEFAULT_EUROSAT_DIR = "datasets/EuroSAT";
    /** Image sub-folder of the EuroSAT folder: {@code EuroSAT_RGB/<ClassName>/*.jpg}. */
    public static final String EUROSAT_IMAGE_DIR = "EuroSAT_RGB";

    /** System property overriding the EuroSAT land-use classifier folder. */
    public static final String EUROSAT_MODEL_DIR_PROPERTY = "landwatch.eurosat.model.dir";
    public static final String DEFAULT_EUROSAT_MODEL_DIR = "models/eurosat_resnet";

    private AppPaths() {
    }

    /** LEVIR-CD root (contains train/, val/, test/). */
    public static Path levirDatasetDir() {
        return fromProperty(DATASET_DIR_PROPERTY, DEFAULT_DATASET_DIR);
    }

    /** Folder with the benchmark, deep-learning and EuroSAT results. */
    public static Path resultsDir() {
        return fromProperty(RESULTS_DIR_PROPERTY, DEFAULT_RESULTS_DIR);
    }

    /** {@code <dataset>/test/A} if it exists (initial folder of the "Open LEVIR-CD pair…" chooser), else null. */
    public static Path levirChooserStartDir() {
        Path testA = levirDatasetDir().resolve("test").resolve("A");
        return Files.isDirectory(testA) ? testA : null;
    }

    public static Path baselineMetrics(String split) {
        return resultsDir().resolve("benchmarks").resolve("baseline_" + split + "_metrics.txt");
    }

    public static Path baselinePerImage(String split) {
        return resultsDir().resolve("benchmarks").resolve("baseline_" + split + "_per_image.csv");
    }

    public static Path deepMetrics(String split) {
        return resultsDir().resolve("deeplearning").resolve(split + "_metrics.txt");
    }

    public static Path euroSatMetrics() {
        return resultsDir().resolve("eurosat").resolve("test_metrics.txt");
    }

    /** Per-image predictions recorded by the EuroSAT test evaluation ({@code EvaluateEuroSatClassifier}). */
    public static Path euroSatTestPredictions() {
        return resultsDir().resolve("eurosat").resolve("test_predictions.csv");
    }

    /** EuroSAT root folder (contains {@code EuroSAT_RGB/} when the images are installed, and {@code splits/}). */
    public static Path euroSatDatasetDir() {
        return fromProperty(EUROSAT_DIR_PROPERTY, DEFAULT_EUROSAT_DIR);
    }

    /** {@code <EuroSAT root>/EuroSAT_RGB}; the images are not in git, so it may be missing. */
    public static Path euroSatImagesDir() {
        return euroSatDatasetDir().resolve(EUROSAT_IMAGE_DIR);
    }

    /**
     * Folder with the split CSVs: {@code <EuroSAT root>/splits} if it has {@code test.csv}, otherwise the
     * copy committed with the project ({@code datasets/EuroSAT/splits}), so a root that only holds the
     * unpacked images still works.
     */
    public static Path euroSatSplitDir() {
        Path configured = euroSatDatasetDir().resolve("splits");
        return Files.isRegularFile(configured.resolve("test.csv"))
                ? configured : Path.of(DEFAULT_EUROSAT_DIR).resolve("splits");
    }

    /** Folder of the trained EuroSAT classifier ({@code model.properties} + {@code *.params}). */
    public static Path euroSatModelDir() {
        return fromProperty(EUROSAT_MODEL_DIR_PROPERTY, DEFAULT_EUROSAT_MODEL_DIR);
    }

    /**
     * Absolute, normalised form of {@code path} for display, with the user's home directory shown as
     * {@code ~} (as shells do). Keeps the account name off the screen in live demos and screenshots.
     */
    public static String forDisplay(Path path) {
        return abbreviateHome(path.toAbsolutePath().normalize(), System.getProperty("user.home"));
    }

    /**
     * Short form of {@code path} for display: relative to the working directory when it lies inside it
     * (e.g. {@code datasets/EuroSAT}), otherwise {@link #forDisplay} (home directory shown as {@code ~}).
     */
    public static String shortForDisplay(Path path) {
        return shortForDisplay(path, Path.of(""), System.getProperty("user.home"));
    }

    static String shortForDisplay(Path path, Path workingDir, String home) {
        Path abs = path.toAbsolutePath().normalize();
        Path cwd = workingDir.toAbsolutePath().normalize();
        if (abs.startsWith(cwd) && !abs.equals(cwd)) {
            return cwd.relativize(abs).toString().replace('\\', '/');
        }
        return abbreviateHome(abs, home);
    }

    static String abbreviateHome(Path absolute, String home) {
        if (home == null || home.isBlank()) {
            return absolute.toString();
        }
        Path homePath = Path.of(home).toAbsolutePath().normalize();
        if (absolute.equals(homePath)) {
            return "~";
        }
        if (absolute.startsWith(homePath)) {
            return "~/" + homePath.relativize(absolute).toString().replace('\\', '/');
        }
        return absolute.toString();
    }

    private static Path fromProperty(String property, String fallback) {
        String v = System.getProperty(property);
        return Path.of(v == null || v.isBlank() ? fallback : v.trim());
    }
}
