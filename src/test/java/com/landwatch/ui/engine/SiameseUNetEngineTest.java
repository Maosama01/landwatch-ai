package com.landwatch.ui.engine;

import com.landwatch.deeplearning.DeepChangeDetector;
import com.landwatch.deeplearning.UiTestModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The dashboard's deep engine: precise unavailability notes, model-dir resolution and a real load +
 * inference with a tiny model (CPU only, so the tests never touch the GPU).
 */
class SiameseUNetEngineTest {

    private static String previousDevice;

    @BeforeAll
    static void forceCpu() {
        previousDevice = System.getProperty(DeepChangeDetector.DEVICE_PROPERTY);
        System.setProperty(DeepChangeDetector.DEVICE_PROPERTY, "cpu");
    }

    @AfterAll
    static void restoreDevice() {
        if (previousDevice == null) {
            System.clearProperty(DeepChangeDetector.DEVICE_PROPERTY);
        } else {
            System.setProperty(DeepChangeDetector.DEVICE_PROPERTY, previousDevice);
        }
    }

    private static BufferedImage random(int w, int h, long seed) {
        Random rnd = new Random(seed);
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.setRGB(x, y, rnd.nextInt(0x1000000));
            }
        }
        return img;
    }

    @Test
    @DisplayName("Missing model directory: unavailable with the note 'model not found at <dir>', never any output")
    void missingModelDirIsUnavailableWithPreciseNote(@TempDir Path tmp) {
        Path dir = tmp.resolve("models").resolve("siamese_unet_levir");
        SiameseUNetEngine engine = new SiameseUNetEngine(dir);
        assertEquals(SiameseUNetEngine.State.NOT_LOADED, engine.getState());
        assertEquals(SiameseUNetEngine.UNAVAILABLE_NOTE, engine.getStatusNote());

        engine.markLoading();
        assertTrue(engine.isLoading());
        assertFalse(engine.isAvailable());
        assertEquals("loading…", engine.getStatusNote());

        assertFalse(engine.load());
        assertEquals(SiameseUNetEngine.State.FAILED, engine.getState());
        assertFalse(engine.isAvailable());
        assertFalse(engine.isLoading());
        assertEquals("model not found at " + dir, engine.getStatusNote());
        assertNull(engine.getDescription());

        BufferedImage img = random(8, 8, 1);
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> EngineRunner.run(engine, img, img));
        assertTrue(ex.getMessage().contains("model not found at " + dir), ex.getMessage());
        IllegalStateException direct = assertThrows(IllegalStateException.class, () -> engine.detect(img, img));
        assertTrue(direct.getMessage().contains("No deep-learning output is produced or simulated"));
    }

    @Test
    @DisplayName("Model dir defaults to models/siamese_unet_levir and is overridable with -Dlandwatch.model.dir")
    void modelDirResolution() {
        String previous = System.getProperty(SiameseUNetEngine.MODEL_DIR_PROPERTY);
        try {
            System.clearProperty(SiameseUNetEngine.MODEL_DIR_PROPERTY);
            assertEquals(Path.of("models/siamese_unet_levir"), SiameseUNetEngine.resolveModelDir());
            assertEquals(Path.of("models/siamese_unet_levir"), new SiameseUNetEngine().getModelDir());

            System.setProperty(SiameseUNetEngine.MODEL_DIR_PROPERTY, "/tmp/some/other/model");
            assertEquals(Path.of("/tmp/some/other/model"), SiameseUNetEngine.resolveModelDir());

            System.setProperty(SiameseUNetEngine.MODEL_DIR_PROPERTY, "  ");
            assertEquals(Path.of("models/siamese_unet_levir"), SiameseUNetEngine.resolveModelDir());
        } finally {
            if (previous == null) {
                System.clearProperty(SiameseUNetEngine.MODEL_DIR_PROPERTY);
            } else {
                System.setProperty(SiameseUNetEngine.MODEL_DIR_PROPERTY, previous);
            }
        }
    }

    @Test
    @DisplayName("Incomplete model directories give precise reasons (no model.properties / no weights / a file)")
    void incompleteModelDirs(@TempDir Path tmp) throws Exception {
        Path empty = Files.createDirectories(tmp.resolve("empty"));
        assertEquals("model not found at " + empty + " (no model.properties)", SiameseUNetEngine.checkModelDir(empty));
        SiameseUNetEngine engine = new SiameseUNetEngine(empty);
        assertFalse(engine.load());
        assertEquals("model not found at " + empty + " (no model.properties)", engine.getStatusNote());

        Path noWeights = Files.createDirectories(tmp.resolve("noweights"));
        Files.writeString(noWeights.resolve("model.properties"), "widths=4,8,8,16\n", StandardCharsets.UTF_8);
        assertEquals("model not found at " + noWeights + " (no *.params weights file)",
                SiameseUNetEngine.checkModelDir(noWeights));

        Path file = Files.writeString(tmp.resolve("not-a-dir.txt"), "x", StandardCharsets.UTF_8);
        assertEquals("model path is not a directory: " + file, SiameseUNetEngine.checkModelDir(file));
    }

    @Test
    @DisplayName("Corrupt weights: unavailable with the load error, not a crash")
    void corruptWeightsReportLoadError(@TempDir Path tmp) throws Exception {
        Path dir = tmp.resolve("corrupt");
        UiTestModels.saveTinyModelDir(dir, 0.5f);
        try (var params = Files.newDirectoryStream(dir, "*.params")) {
            for (Path p : params) {
                Files.write(p, new byte[]{1, 2, 3, 4, 5});
            }
        }
        SiameseUNetEngine engine = new SiameseUNetEngine(dir);
        assertFalse(engine.load());
        assertEquals(SiameseUNetEngine.State.FAILED, engine.getState());
        assertTrue(engine.getStatusNote().startsWith("model failed to load from " + dir + ": "), engine.getStatusNote());
    }

    @Test
    @DisplayName("A real (tiny) trained model loads, describes itself from its files and predicts a full-size mask")
    void loadsAndDetectsWithRealModel(@TempDir Path tmp) throws Exception {
        Path dir = tmp.resolve("tiny");
        UiTestModels.saveTinyModelDir(dir, 0.5f);
        SiameseUNetEngine engine = new SiameseUNetEngine(dir);
        try {
            assertTrue(engine.load(), engine.getStatusNote());
            assertTrue(engine.isAvailable());
            assertEquals(SiameseUNetEngine.State.READY, engine.getState());
            assertTrue(engine.getStatusNote().contains("FC-Siam-diff"), engine.getStatusNote());
            assertTrue(engine.getStatusNote().contains("threshold 0.50"), engine.getStatusNote());
            assertTrue(engine.getDescription().contains("FC-Siam-diff"));
            assertEquals(0.5f, engine.getThreshold(), 1e-6f);
            assertTrue(engine.getDeviceName().startsWith("cpu"), engine.getDeviceName());
            assertEquals("0.5", engine.getModelProperty("decision_threshold"));

            BufferedImage a = random(45, 37, 2);
            BufferedImage b = random(45, 37, 3);
            EngineResult r = EngineRunner.run(engine, a, b);
            assertEquals(SiameseUNetEngine.ID, r.getEngineId());
            assertEquals(SiameseUNetEngine.DISPLAY_NAME, r.getEngineName());
            assertEquals(45, r.getWidth());
            assertEquals(37, r.getHeight());
            assertNull(r.getOtsuThreshold(), "the deep model has no Otsu threshold");
        } finally {
            engine.close();
        }
        assertEquals(SiameseUNetEngine.State.CLOSED, engine.getState());
        assertFalse(engine.isAvailable());
        BufferedImage img = random(8, 8, 4);
        assertThrows(IllegalStateException.class, () -> engine.detect(img, img));
    }

    @Test
    @DisplayName("Closing before loading prevents the load (shutdown during start-up)")
    void closeBeforeLoad(@TempDir Path tmp) {
        SiameseUNetEngine engine = new SiameseUNetEngine(tmp.resolve("whatever"));
        engine.close();
        engine.close();
        assertFalse(engine.load());
        assertEquals(SiameseUNetEngine.State.CLOSED, engine.getState());
    }
}
