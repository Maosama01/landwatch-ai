package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Sanity check (c): saving and loading parameters reproduces the model's outputs exactly. */
class SaveLoadRoundTripTest {

    @TempDir
    Path tempDir;

    @Test
    void savedAndReloadedModelGivesIdenticalOutputs() throws Exception {
        int size = 32;
        DeviceBenchmark.SyntheticBatch data = DeviceBenchmark.SyntheticBatch.create(2, size, 11L);
        float[] expected;
        try (Model model = TrainingSupport.newModel(TestModels.TINY_WIDTHS, Device.cpu());
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(2f, 1f), 1e-3f, 2, size)) {
            NDManager m = trainer.getManager();
            NDList x = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            NDList y = new NDList(m.create(data.y, data.labelShape()));
            for (int i = 0; i < 3; i++) {
                TrainingSupport.trainStep(trainer, x, y); // changes weights and BatchNorm running stats
            }
            TrainChangeDetector.saveCheckpoint(model, tempDir, 3);
            expected = TrainingSupport.predictProbabilities(trainer, x);
        }
        assertTrue(Files.exists(tempDir.resolve("fc_siam_diff-0003.params")));

        try (Model loaded = TrainingSupport.newModel(TestModels.TINY_WIDTHS, Device.cpu())) {
            loaded.load(tempDir, TrainingSupport.MODEL_NAME);
            NDManager m = loaded.getNDManager();
            NDList x = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            float[] actual = TrainingSupport.predictProbabilities(loaded.getBlock(), new ParameterStore(m, false), x);
            assertArrayEquals(expected, actual, 0f);
        }
    }

    @Test
    void checkpointSavingKeepsOnlyOneParamsFile() throws Exception {
        try (Model model = TrainingSupport.newModel(TestModels.TINY_WIDTHS, Device.cpu());
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(2f, 1f), 1e-3f, 1, 32)) {
            TrainChangeDetector.saveCheckpoint(model, tempDir, 2);
            TrainChangeDetector.saveCheckpoint(model, tempDir, 5);
        }
        java.util.List<String> paramsFiles;
        try (var files = Files.list(tempDir)) {
            paramsFiles = files.map(f -> f.getFileName().toString()).filter(n -> n.endsWith(".params")).toList();
        }
        assertEquals(java.util.List.of("fc_siam_diff-0005.params"), paramsFiles);
    }

    @Test
    void detectorLoadedTwiceFromSameDirectoryAgrees() throws Exception {
        TestModels.saveModelDir(tempDir, TestModels.TINY_WIDTHS, 32, 0.4f);
        BufferedImage a = TestModels.randomImage(40, 30, 1);
        BufferedImage b = TestModels.randomImage(40, 30, 2);
        float[][] first;
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            assertEquals(0.4f, d.getThreshold());
            first = d.predictProbabilities(a, b);
        }
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            float[][] second = d.predictProbabilities(a, b);
            for (int yy = 0; yy < first.length; yy++) {
                assertArrayEquals(first[yy], second[yy], 0f);
            }
        }
    }
}
