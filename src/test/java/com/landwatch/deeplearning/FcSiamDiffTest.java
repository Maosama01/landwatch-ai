package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Sanity check (a): the model builds and maps (N,3,256,256) x 2 to (N,1,256,256). */
class FcSiamDiffTest {

    @Test
    void outputShapeIsOneChannelAtFullResolution() {
        try (Model model = TrainingSupport.newModel(FcSiamDiff.DEFAULT_WIDTHS, Device.cpu())) {
            NDManager m = model.getNDManager();
            Shape in = new Shape(2, 3, 256, 256);
            model.getBlock().initialize(m, DataType.FLOAT32, in, in);
            assertArrayEquals(new Shape[] {new Shape(2, 1, 256, 256)},
                    model.getBlock().getOutputShapes(new Shape[] {in, in}));

            NDList inputs = new NDList(m.randomNormal(in), m.randomNormal(in));
            NDArray out = model.getBlock().forward(new ParameterStore(m, false), inputs, false).singletonOrThrow();
            assertEquals(new Shape(2, 1, 256, 256), out.getShape());
            for (float v : out.toFloatArray()) {
                assertTrue(Float.isFinite(v));
            }
        }
    }

    @Test
    void defaultModelHasExpectedParameterCount() {
        try (Model model = TrainingSupport.newModel(FcSiamDiff.DEFAULT_WIDTHS, Device.cpu())) {
            Shape in = new Shape(1, 3, 64, 64);
            model.getBlock().initialize(model.getNDManager(), DataType.FLOAT32, in, in);
            // about 1.35 M, the same order as the FC-Siam-diff paper's 1.35 M
            assertEquals(1_348_785L, TrainingSupport.countTrainableParameters(model.getBlock()));
        }
    }

    @Test
    void swappingBeforeAndAfterGivesTheSameOutput() {
        // |enc(A) - enc(B)| is symmetric, so in inference mode FC-Siam-diff cannot tell the order apart
        try (Model model = TrainingSupport.newModel(TestModels.TINY_WIDTHS, Device.cpu())) {
            NDManager m = model.getNDManager();
            Shape in = new Shape(2, 3, 32, 32);
            model.getBlock().initialize(m, DataType.FLOAT32, in, in);
            NDArray a = m.randomNormal(in);
            NDArray b = m.randomNormal(in);
            ParameterStore ps = new ParameterStore(m, false);
            float[] ab = model.getBlock().forward(ps, new NDList(a, b), false).singletonOrThrow().toFloatArray();
            float[] ba = model.getBlock().forward(ps, new NDList(b, a), false).singletonOrThrow().toFloatArray();
            assertArrayEquals(ab, ba, 1e-5f);
        }
    }

    @Test
    void rejectsSizesThatAreNotMultiplesOf16() {
        FcSiamDiff block = new FcSiamDiff();
        Shape bad = new Shape(1, 3, 100, 100);
        assertThrows(IllegalArgumentException.class, () -> block.getOutputShapes(new Shape[] {bad, bad}));
        Shape gray = new Shape(1, 1, 64, 64);
        assertThrows(IllegalArgumentException.class, () -> block.getOutputShapes(new Shape[] {gray, gray}));
        assertThrows(IllegalArgumentException.class, () -> new FcSiamDiff(new int[] {16, 32, 64}));
    }

    @Test
    void nearestUpsamplingCopiesEachValueIntoA2x2Block() {
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            NDArray x = m.create(new float[] {1, 2, 3, 4}, new Shape(1, 1, 2, 2));
            float[] up = FcSiamDiff.upsampleNearest2x(x).toFloatArray();
            assertArrayEquals(new float[] {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}, up);
        }
    }
}
