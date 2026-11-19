package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.tracker.Tracker;
import com.landwatch.deeplearning.ImageTensors;
import com.landwatch.deeplearning.Normalization;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Save / load round trip and the public inference API (tiny model, CPU, synthetic images). */
class EuroSatClassifierTest {

    private static final int[] TINY_WIDTHS = {4, 8};
    private static final List<String> NAMES = List.of("Forest", "River", "SeaLake");
    private static final Normalization NORM =
            new Normalization(new float[] {0.34f, 0.38f, 0.41f}, new float[] {0.20f, 0.14f, 0.12f});

    @TempDir
    Path dir;

    static BufferedImage randomImage(int w, int h, long seed) {
        Random rnd = new Random(seed);
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.setRGB(x, y, rnd.nextInt(0x1000000));
            }
        }
        return img;
    }

    private static float[] batchOf(List<BufferedImage> images) {
        int per = 3 * 64 * 64;
        float[] out = new float[images.size() * per];
        for (int i = 0; i < images.size(); i++) {
            System.arraycopy(NORM.applyInPlace(ImageTensors.toChw01(images.get(i))), 0, out, i * per, per);
        }
        return out;
    }

    /**
     * Trains a tiny model for a few steps (so weights and BatchNorm running statistics differ from
     * their initial values), saves it like the training CLI does, and returns the probabilities the
     * in-memory model gives for {@code images}.
     */
    private float[] trainAndSave(List<BufferedImage> images) throws IOException {
        Engine.getInstance().setRandomSeed(11);
        int n = images.size();
        float[] x = batchOf(images);
        long[] y = new long[n];
        for (int i = 0; i < n; i++) {
            y[i] = i % NAMES.size();
        }
        float[] expected;
        try (Model model = EuroSatTraining.newModel(TINY_WIDTHS, 1, NAMES.size(), Device.cpu());
             Trainer trainer = EuroSatTraining.newTrainer(model, Tracker.fixed(1e-3f), 0f, n, 64)) {
            for (int s = 0; s < 3; s++) {
                EuroSatTraining.trainStep(trainer, x, y, n, 64);
            }
            EuroSatTraining.saveCheckpoint(model, dir, 3);
            expected = EuroSatTraining.predictProbabilities(model.getBlock(),
                    new ParameterStore(model.getNDManager(), false), model.getNDManager(), x, n, 64);
        }
        new EuroSatModelConfig()
                .setWidths(TINY_WIDTHS)
                .set(EuroSatModelConfig.BLOCKS_PER_STAGE, 1)
                .setClassNames(NAMES)
                .set(EuroSatModelConfig.INPUT_SIZE, 64)
                .setNormalization(NORM)
                .set(EuroSatModelConfig.PARAMS_NAME, EuroSatTraining.MODEL_NAME)
                .set(EuroSatModelConfig.ARCHITECTURE, "tiny test model")
                .save(dir);
        return expected;
    }

    @Test
    void savedAndReloadedModelGivesIdenticalOutputs() throws Exception {
        List<BufferedImage> images = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            images.add(randomImage(64, 64, 100 + i));
        }
        float[] expected = trainAndSave(images);
        assertTrue(Files.exists(dir.resolve("eurosat_resnet-0003.params")));

        try (EuroSatClassifier classifier = EuroSatClassifier.load(dir, "cpu")) {
            assertEquals(NAMES, classifier.getClassNames());
            assertEquals(Device.cpu(), classifier.getDevice());
            List<EuroSatClassifier.Prediction> predictions = classifier.predict(images);
            float[] actual = new float[expected.length];
            for (int i = 0; i < predictions.size(); i++) {
                System.arraycopy(predictions.get(i).probabilities(), 0, actual, i * NAMES.size(), NAMES.size());
            }
            assertArrayEquals(expected, actual, 0f);
            // the single-image call gives the same answer as the batched call (up to float rounding:
            // libtorch may pick a different convolution kernel for batch size 1 than for 4)
            assertArrayEquals(predictions.get(2).probabilities(), classifier.predict(images.get(2)).probabilities(), 1e-6f);
        }
        // a second, independent load agrees as well
        try (EuroSatClassifier again = EuroSatClassifier.load(dir, "cpu")) {
            assertArrayEquals(expected, flatten(again.predict(images)), 0f);
        }
    }

    private static float[] flatten(List<EuroSatClassifier.Prediction> predictions) {
        int k = predictions.get(0).probabilities().length;
        float[] out = new float[predictions.size() * k];
        for (int i = 0; i < predictions.size(); i++) {
            System.arraycopy(predictions.get(i).probabilities(), 0, out, i * k, k);
        }
        return out;
    }

    @Test
    void predictionIsArgMaxOfProbabilities() throws Exception {
        BufferedImage img = randomImage(64, 64, 5);
        trainAndSave(List.of(img, randomImage(64, 64, 6), randomImage(64, 64, 7)));
        try (EuroSatClassifier classifier = EuroSatClassifier.load(dir, "cpu")) {
            EuroSatClassifier.Prediction p = classifier.predict(img);
            float[] probs = p.probabilities();
            assertEquals(NAMES.size(), probs.length);
            float sum = 0;
            for (float v : probs) {
                assertTrue(v >= 0f && v <= 1f);
                sum += v;
            }
            assertEquals(1f, sum, 1e-5f);
            assertEquals(EuroSatTraining.argMax(probs, 0, probs.length), p.classIndex());
            assertEquals(NAMES.get(p.classIndex()), p.className());
            assertEquals(probs[p.classIndex()], p.probability());
            assertEquals(p.classIndex(), p.topIndices(2)[0]);
            assertFalse(p.wasResized());
            assertEquals("", p.note());
            assertTrue(classifier.describe().contains("Forest, River, SeaLake"));
        }
    }

    @Test
    void otherSizesAreResizedWithANote() throws Exception {
        trainAndSave(List.of(randomImage(64, 64, 1), randomImage(64, 64, 2), randomImage(64, 64, 3)));
        BufferedImage big = randomImage(300, 200, 9);
        BufferedImage small = randomImage(20, 30, 10);
        try (EuroSatClassifier classifier = EuroSatClassifier.load(dir, "cpu")) {
            EuroSatClassifier.Prediction p = classifier.predict(big);
            assertTrue(p.wasResized());
            assertTrue(p.note().contains("300x200") && p.note().contains("resized to 64x64"), p.note());
            // same result as resizing first and passing a 64x64 image
            EuroSatClassifier.Prediction manual = classifier.predict(EuroSatClassifier.resize(big, 64));
            assertFalse(manual.wasResized());
            assertArrayEquals(manual.probabilities(), p.probabilities(), 0f);
            assertTrue(classifier.predict(small).note().contains("20x30"));
        }
        BufferedImage resized = EuroSatClassifier.resize(big, 64);
        assertEquals(64, resized.getWidth());
        assertEquals(64, resized.getHeight());
    }

    @Test
    void checkpointSavingKeepsOneParamsFileAndBadFoldersFail() throws Exception {
        try (Model model = EuroSatTraining.newModel(TINY_WIDTHS, 1, 3, Device.cpu());
             Trainer ignored = EuroSatTraining.newTrainer(model, Tracker.fixed(1e-3f), 0f, 1, 64)) {
            EuroSatTraining.saveCheckpoint(model, dir, 2);
            EuroSatTraining.saveCheckpoint(model, dir, 7);
        }
        List<String> params;
        try (var files = Files.list(dir)) {
            params = files.map(f -> f.getFileName().toString()).filter(n -> n.endsWith(".params")).toList();
        }
        assertEquals(List.of("eurosat_resnet-0007.params"), params);
        assertThrows(IOException.class, () -> EuroSatClassifier.load(dir.resolve("missing"), "cpu"));
        assertThrows(IOException.class, () -> EuroSatClassifier.load(dir, "cpu")); // no model.properties
    }
}
