package com.landwatch.ui.engine;

import com.landwatch.deeplearning.eurosat.EuroSatClassifier;
import com.landwatch.ui.data.AppPaths;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The Land use page's classifier engine: precise unavailability notes, device choice, and a real load +
 * inference with the committed EuroSAT model on the CPU (synthetic images only; the tests never read the
 * EuroSAT test split and never touch the GPU).
 */
class EuroSatEngineTest {

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
    @DisplayName("It is a scene classifier, not a change-detection engine, so it can never be offered for LEVIR-CD pairs")
    void notAChangeDetectionEngine() {
        assertFalse(ChangeDetectionEngine.class.isAssignableFrom(EuroSatEngine.class));
    }

    @Test
    @DisplayName("Missing model folder: unavailable with 'model not found at <dir>' and no prediction at all")
    void missingModelDir(@TempDir Path tmp) {
        Path dir = tmp.resolve("models").resolve("eurosat_resnet");
        EuroSatEngine engine = new EuroSatEngine(dir, "cpu");
        assertEquals(EuroSatEngine.State.NOT_LOADED, engine.getState());
        assertTrue(engine.needsLoad());
        assertEquals(EuroSatEngine.UNAVAILABLE_NOTE, engine.getStatusNote());

        engine.markLoading();
        assertTrue(engine.isLoading());
        assertFalse(engine.needsLoad());
        assertFalse(engine.load());
        assertEquals(EuroSatEngine.State.FAILED, engine.getState());
        assertEquals("model not found at " + dir, engine.getStatusNote());
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> engine.classify(random(64, 64, 1)));
        assertTrue(e.getMessage().contains("No prediction is produced or simulated"), e.getMessage());
        assertThrows(IllegalStateException.class, () -> engine.classifyAll(List.of(random(64, 64, 1))));
        assertTrue(engine.getClassNames().isEmpty());
        assertNull(engine.getDeviceName());
    }

    @Test
    @DisplayName("A folder without model.properties or without weights is reported precisely")
    void incompleteModelDir(@TempDir Path tmp) throws Exception {
        assertEquals("model not found at " + tmp + " (no model.properties)", EuroSatEngine.checkModelDir(tmp));
        Files.writeString(tmp.resolve("model.properties"), "num_classes=10\n", StandardCharsets.UTF_8);
        assertEquals("model not found at " + tmp + " (no *.params weights file)", EuroSatEngine.checkModelDir(tmp));
        Path file = Files.writeString(tmp.resolve("file.txt"), "x", StandardCharsets.UTF_8);
        assertEquals("model path is not a directory: " + file, EuroSatEngine.checkModelDir(file));
    }

    @Test
    @DisplayName("Device: the EuroSAT property wins, then the dashboard's deep-model device, then auto")
    void deviceChoice() {
        assertEquals("cpu", EuroSatEngine.resolveDevice("cpu", "mps"));
        assertEquals("mps", EuroSatEngine.resolveDevice(null, "mps"));
        assertEquals("mps", EuroSatEngine.resolveDevice("  ", " mps "));
        assertEquals("auto", EuroSatEngine.resolveDevice(null, null));
        assertEquals("auto", new EuroSatEngine(Path.of("x"), " ").getRequestedDevice());
        assertThrows(IllegalArgumentException.class, () -> new EuroSatEngine(null, "cpu"));
    }

    @Test
    @DisplayName("Committed model on the CPU: 10 classes, probabilities sum to 1, other sizes carry the resize note")
    void committedModelClassifies() {
        Path dir = Path.of(AppPaths.DEFAULT_EUROSAT_MODEL_DIR);
        Assumptions.assumeTrue(EuroSatEngine.checkModelDir(dir) == null, "EuroSAT model not present");
        EuroSatEngine engine = new EuroSatEngine(dir, "cpu");
        try {
            assertTrue(engine.load(), engine.getStatusNote());
            assertTrue(engine.isAvailable());
            assertEquals("cpu()", engine.getDeviceName());
            assertEquals(10, engine.getClassNames().size());
            assertTrue(engine.getSummary().contains("10 classes"), engine.getSummary());
            assertTrue(engine.load(), "loading twice is a no-op");

            EuroSatEngine.Classified tile = engine.classify(random(64, 64, 3));
            EuroSatClassifier.Prediction p = tile.prediction();
            float sum = 0;
            for (float v : p.probabilities()) {
                assertTrue(v >= 0f && v <= 1f);
                sum += v;
            }
            assertEquals(1f, sum, 1e-4f);
            assertEquals(engine.getClassNames().get(p.classIndex()), p.className());
            assertFalse(p.wasResized());
            assertTrue(tile.millis() >= 0);

            EuroSatClassifier.Prediction other = engine.classify(random(100, 80, 4)).prediction();
            assertTrue(other.wasResized());
            assertTrue(other.note().contains("100x80") && other.note().contains("less reliable"), other.note());

            List<EuroSatClassifier.Prediction> batch = engine.classifyAll(List.of(random(64, 64, 5), random(64, 64, 6)));
            assertEquals(2, batch.size());
        } finally {
            engine.close();
        }
        assertEquals(EuroSatEngine.State.CLOSED, engine.getState());
        assertFalse(engine.load(), "a closed engine stays closed");
        assertThrows(IllegalStateException.class, () -> engine.classify(random(64, 64, 7)));
    }
}
