package com.landwatch.ui.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AppPathsTest {

    private static final Path HOME = Path.of("").toAbsolutePath().getRoot().resolve("home-for-test").resolve("someone");

    @Test
    @DisplayName("Paths under the home directory are shown with ~ instead of the account name")
    void homeIsAbbreviated() {
        Path model = HOME.resolve("Projects").resolve("landwatch-AI").resolve("models").resolve("siamese_unet_levir");
        assertEquals("~/Projects/landwatch-AI/models/siamese_unet_levir", AppPaths.abbreviateHome(model, HOME.toString()));
        assertEquals("~", AppPaths.abbreviateHome(HOME, HOME.toString()));
    }

    @Test
    @DisplayName("Paths outside the home directory, or a sibling with the same prefix, are left unchanged")
    void otherPathsUnchanged() {
        Path sibling = HOME.getParent().resolve("someone-else").resolve("data");
        assertEquals(sibling.toString(), AppPaths.abbreviateHome(sibling, HOME.toString()));
        Path data = HOME.getRoot().resolve("data").resolve("LEVIR-CD");
        assertEquals(data.toString(), AppPaths.abbreviateHome(data, HOME.toString()));
        assertEquals(data.toString(), AppPaths.abbreviateHome(data, null));
    }

    @Test
    @DisplayName("shortForDisplay: relative inside the working directory, ~ under home, unchanged elsewhere")
    void shortForDisplay() {
        Path project = HOME.resolve("Projects").resolve("landwatch-AI");
        assertEquals("datasets/EuroSAT/EuroSAT_RGB", AppPaths.shortForDisplay(
                project.resolve("datasets").resolve("EuroSAT").resolve("EuroSAT_RGB"), project, HOME.toString()));
        assertEquals("~/Projects/landwatch-AI", AppPaths.shortForDisplay(project, project, HOME.toString()));
        assertEquals("~/Downloads/EuroSAT", AppPaths.shortForDisplay(HOME.resolve("Downloads").resolve("EuroSAT"),
                project, HOME.toString()));
        Path data = HOME.getRoot().resolve("data").resolve("EuroSAT");
        assertEquals(data.toString(), AppPaths.shortForDisplay(data, project, HOME.toString()));
    }

    @Test
    @DisplayName("EuroSAT locations: project-relative defaults, overridable by system properties")
    void euroSatPaths(@TempDir Path tmp) throws Exception {
        String previousRoot = System.getProperty(AppPaths.EUROSAT_DIR_PROPERTY);
        String previousModel = System.getProperty(AppPaths.EUROSAT_MODEL_DIR_PROPERTY);
        try {
            System.clearProperty(AppPaths.EUROSAT_DIR_PROPERTY);
            System.clearProperty(AppPaths.EUROSAT_MODEL_DIR_PROPERTY);
            assertEquals(Path.of("datasets/EuroSAT"), AppPaths.euroSatDatasetDir());
            assertEquals(Path.of("datasets/EuroSAT/EuroSAT_RGB"), AppPaths.euroSatImagesDir());
            assertEquals(Path.of("datasets/EuroSAT/splits"), AppPaths.euroSatSplitDir());
            assertEquals(Path.of("models/eurosat_resnet"), AppPaths.euroSatModelDir());
            assertEquals(Path.of("results/eurosat/test_predictions.csv"), AppPaths.euroSatTestPredictions());

            // a root holding only the unpacked images falls back to the committed split CSVs
            System.setProperty(AppPaths.EUROSAT_DIR_PROPERTY, tmp.toString());
            assertEquals(tmp.resolve("EuroSAT_RGB"), AppPaths.euroSatImagesDir());
            assertEquals(Path.of("datasets/EuroSAT/splits"), AppPaths.euroSatSplitDir());
            Files.createDirectories(tmp.resolve("splits"));
            Files.writeString(tmp.resolve("splits").resolve("test.csv"), "path,class_name,class_index\n");
            assertEquals(tmp.resolve("splits"), AppPaths.euroSatSplitDir());

            System.setProperty(AppPaths.EUROSAT_MODEL_DIR_PROPERTY, " other/model ");
            assertEquals(Path.of("other/model"), AppPaths.euroSatModelDir());
        } finally {
            restore(AppPaths.EUROSAT_DIR_PROPERTY, previousRoot);
            restore(AppPaths.EUROSAT_MODEL_DIR_PROPERTY, previousModel);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }

    @Test
    @DisplayName("forDisplay never shows the real home directory path")
    void forDisplayHidesRealHome() {
        String shown = AppPaths.forDisplay(Path.of("models"));
        assertTrue(shown.endsWith("models"));
        String home = System.getProperty("user.home");
        if (home != null && Path.of("").toAbsolutePath().startsWith(Path.of(home))) {
            assertFalse(shown.contains(home), shown);
            assertTrue(shown.startsWith("~"), shown);
        }
    }
}
