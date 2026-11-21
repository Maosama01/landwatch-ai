package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The bottleneck Transformer and the FC-Siam-diff + Transformer model built from a config. */
class BitemporalTransformerTest {

    private static ModelConfig transformerConfig(int[] widths, int layers, int heads, int mlp) {
        return new ModelConfig()
                .setWidths(widths)
                .set(ModelConfig.MODEL_TYPE, ModelConfig.TYPE_TRANSFORMER)
                .set(ModelConfig.TF_LAYERS, layers)
                .set(ModelConfig.TF_HEADS, heads)
                .set(ModelConfig.TF_MLP, mlp);
    }

    @Test
    void keepsTheShapeOfTheStackedBottleneck() {
        BitemporalTransformer t = new BitemporalTransformer(16, 2, 4, 32);
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            Shape in = new Shape(4, 16, 4, 4); // N = 2 pairs, A and B stacked
            t.initialize(m, DataType.FLOAT32, in);
            NDArray out = t.forward(new ParameterStore(m, false), new NDList(m.randomNormal(in)), false).singletonOrThrow();
            assertEquals(in, out.getShape());
            for (float v : out.toFloatArray()) {
                assertTrue(Float.isFinite(v));
            }
        }
    }

    @Test
    void parameterCountMatchesTheFormula() {
        // per layer: qkv C*3C+3C, projection C*C+C, MLP C*m+m + m*C+C, two LayerNorms 4C; plus a final LayerNorm 2C
        int c = 128;
        int m = 256;
        long perLayer = (long) c * 3 * c + 3L * c + (long) c * c + c + (long) c * m + m + (long) m * c + c + 4L * c;
        long expected = 2 * perLayer + 2L * c;
        assertEquals(265_216L, expected);
        try (Model model = TrainingSupport.newModel(transformerConfig(FcSiamDiff.DEFAULT_WIDTHS, 2, 4, 256), Device.cpu())) {
            Shape in = new Shape(1, 3, 64, 64);
            model.getBlock().initialize(model.getNDManager(), DataType.FLOAT32, in, in);
            assertEquals(1_348_785L + expected, TrainingSupport.countTrainableParameters(model.getBlock()));
            assertEquals(TrainingSupport.TRANSFORMER_MODEL_NAME, model.getName());
        }
    }

    @Test
    void swappingTheDatesSwapsTheOutputHalves() {
        // no date embedding: the block treats A and B symmetrically
        BitemporalTransformer t = new BitemporalTransformer(8, 1, 2, 16);
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            Shape half = new Shape(1, 8, 2, 2);
            t.initialize(m, DataType.FLOAT32, new Shape(2, 8, 2, 2));
            NDArray a = m.randomNormal(half);
            NDArray b = m.randomNormal(half);
            ParameterStore ps = new ParameterStore(m, false);
            NDList ab = t.forward(ps, new NDList(NDArrays.concat(new NDList(a, b), 0)), false).singletonOrThrow().split(2, 0);
            NDList ba = t.forward(ps, new NDList(NDArrays.concat(new NDList(b, a), 0)), false).singletonOrThrow().split(2, 0);
            assertArrayEquals(ab.get(0).toFloatArray(), ba.get(1).toFloatArray(), 1e-5f);
            assertArrayEquals(ab.get(1).toFloatArray(), ba.get(0).toFloatArray(), 1e-5f);
        }
    }

    @Test
    void wholeModelGivesTheSameMapForSwappedDates() {
        try (Model model = TrainingSupport.newModel(transformerConfig(TestModels.TINY_WIDTHS, 1, 2, 16), Device.cpu())) {
            NDManager m = model.getNDManager();
            Shape in = new Shape(2, 3, 32, 32);
            model.getBlock().initialize(m, DataType.FLOAT32, in, in);
            NDArray a = m.randomNormal(in);
            NDArray b = m.randomNormal(in);
            ParameterStore ps = new ParameterStore(m, false);
            float[] ab = model.getBlock().forward(ps, new NDList(a, b), false).singletonOrThrow().toFloatArray();
            float[] ba = model.getBlock().forward(ps, new NDList(b, a), false).singletonOrThrow().toFloatArray();
            assertArrayEquals(ab, ba, 1e-4f);
        }
    }

    @Test
    void positionCodeAlternatesSinAndCosForRowThenColumn() {
        float[] code = BitemporalTransformer.positionCode(2, 3, 8); // quarter = 2 frequencies: 1 and 0.01
        // token (y=1, x=2) starts at (1*3 + 2) * 8
        int base = 5 * 8;
        assertEquals((float) Math.sin(1), code[base], 1e-6);
        assertEquals((float) Math.cos(1), code[base + 1], 1e-6);
        assertEquals((float) Math.sin(0.01), code[base + 2], 1e-6);
        assertEquals((float) Math.sin(2), code[base + 4], 1e-6);
        assertEquals((float) Math.cos(2 * 0.01), code[base + 7], 1e-6);
        // token (0, 0): all sines 0, all cosines 1
        assertArrayEquals(new float[] {0, 1, 0, 1, 0, 1, 0, 1}, java.util.Arrays.copyOf(code, 8), 1e-6f);
    }

    @Test
    void rejectsInconsistentSizes() {
        assertThrows(IllegalArgumentException.class, () -> new BitemporalTransformer(10, 1, 3, 8));
        assertThrows(IllegalArgumentException.class, () -> new BitemporalTransformer(6, 1, 2, 8));
        assertThrows(IllegalArgumentException.class,
                () -> new FcSiamDiff(new int[] {4, 8, 8, 16}, new BitemporalTransformer(8, 1, 2, 8)));
        BitemporalTransformer t = new BitemporalTransformer(8, 1, 2, 8);
        assertThrows(IllegalArgumentException.class, () -> t.getOutputShapes(new Shape[] {new Shape(3, 8, 2, 2)}));
    }

    @Test
    void trainsSavesAndLoadsThroughTheDetectorApi(@TempDir Path dir) throws Exception {
        ModelConfig config = transformerConfig(TestModels.TINY_WIDTHS, 1, 2, 16)
                .set(ModelConfig.PATCH_SIZE, 32)
                .setNormalization(new Normalization(new float[] {0.4f, 0.39f, 0.33f}, new float[] {0.2f, 0.19f, 0.17f}))
                .set(ModelConfig.THRESHOLD, 0.5f)
                .set(ModelConfig.PARAMS_NAME, TrainingSupport.TRANSFORMER_MODEL_NAME);
        DeviceBenchmark.SyntheticBatch data = DeviceBenchmark.SyntheticBatch.create(2, 32, 7L);
        float first;
        float last = Float.NaN;
        try (Model model = TrainingSupport.newModel(config, Device.cpu());
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(2f, 1f), 1e-3f, 2, 32)) {
            NDManager m = trainer.getManager();
            NDList x = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            NDList y = new NDList(m.create(data.y, data.labelShape()));
            first = TrainingSupport.trainStep(trainer, x, y);
            for (int i = 0; i < 30; i++) {
                last = TrainingSupport.trainStep(trainer, x, y);
            }
            TrainChangeDetector.saveCheckpoint(model, dir, 1);
        }
        assertTrue(last < first, "loss should fall on a fixed batch: first " + first + ", last " + last);
        config.save(dir);
        assertTrue(dir.resolve(TrainingSupport.TRANSFORMER_MODEL_NAME + "-0001.params").toFile().isFile());

        try (DeepChangeDetector detector = DeepChangeDetector.load(dir, "cpu")) {
            assertEquals("FC-Siam-diff + Transformer", detector.getConfig().getModelDisplayName());
            assertTrue(detector.describe().startsWith("FC-Siam-diff + Transformer"));
            float[][] p = detector.predictProbabilities(TestModels.randomImage(40, 36, 1), TestModels.randomImage(40, 36, 2));
            assertEquals(36, p.length);
            assertEquals(40, p[0].length);
        }
    }

    @Test
    void parsesBooleanOptionsStrictly() {
        assertTrue(TrainChangeDetector.parseBoolean("TRUE", "augment"));
        assertEquals(false, TrainChangeDetector.parseBoolean("false", "augment"));
        assertThrows(IllegalArgumentException.class, () -> TrainChangeDetector.parseBoolean("no", "augment"));
    }

    @Test
    void oldConfigsWithoutATypeArePlainFcSiamDiff() {
        ModelConfig plain = new ModelConfig().setWidths(TestModels.TINY_WIDTHS);
        assertEquals(ModelConfig.TYPE_FC_SIAM_DIFF, plain.getModelType());
        assertEquals("FC-Siam-diff", plain.getModelDisplayName());
        try (Model model = TrainingSupport.newModel(plain, Device.cpu())) {
            assertEquals(TrainingSupport.MODEL_NAME, model.getName());
        }
        ModelConfig bad = new ModelConfig().set(ModelConfig.MODEL_TYPE, "lstm");
        assertThrows(IllegalStateException.class, bad::getModelType);
    }
}
