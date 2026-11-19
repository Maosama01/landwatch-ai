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

    /** {@code <dataset>/test/A} if it exists (initial folder of the "Load LEVIR-CD Pair" chooser), else null. */
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

    /**
     * Absolute, normalised form of {@code path} for display, with the user's home directory shown as
     * {@code ~} (as shells do). Keeps the account name off the screen in live demos and screenshots.
     */
    public static String forDisplay(Path path) {
        return abbreviateHome(path.toAbsolutePath().normalize(), System.getProperty("user.home"));
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
