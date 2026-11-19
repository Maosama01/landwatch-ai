package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import com.landwatch.deeplearning.TrainingSupport;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Shape and behaviour checks of the ResNet and its residual block (CPU, small, fast). */
class EuroSatResNetTest {

    @Test
    void outputShapeIsBatchByTenClasses() {
        try (Model model = EuroSatTraining.newModel(EuroSatResNet.DEFAULT_WIDTHS, 2, 10, Device.cpu())) {
            NDManager m = model.getNDManager();
            Shape in = new Shape(3, 3, 64, 64);
            model.getBlock().initialize(m, DataType.FLOAT32, in);
            assertArrayEquals(new Shape[] {new Shape(3, 10)}, model.getBlock().getOutputShapes(new Shape[] {in}));
            NDArray out = model.getBlock().forward(new ParameterStore(m, false), new NDList(m.randomNormal(in)), false)
                    .singletonOrThrow();
            assertEquals(new Shape(3, 10), out.getShape());
            for (float v : out.toFloatArray()) {
                assertTrue(Float.isFinite(v));
            }
        }
    }

    @Test
    void defaultModelHasExpectedParameterCount() {
        try (Model model = EuroSatTraining.newModel(EuroSatResNet.DEFAULT_WIDTHS, 2, 10, Device.cpu())) {
            model.getBlock().initialize(model.getNDManager(), DataType.FLOAT32, new Shape(1, 3, 64, 64));
            long expected = 0;
            expected += 9 * 3 * 32 + 2 * 32;                        // stem: 3x3 conv (no bias) + BN gamma/beta
            int in = 32;
            for (int w : new int[] {32, 64, 128, 256}) {
                for (int b = 0; b < 2; b++) {
                    expected += 9L * in * w + 9L * w * w + 4L * w;  // two 3x3 convs + two BNs
                    if (in != w) {
                        expected += (long) in * w + 2L * w;         // 1x1 projection + its BN
                    }
                    in = w;
                }
            }
            expected += 256 * 10 + 10;                               // linear head
            assertEquals(2_797_610L, expected);
            assertEquals(expected, TrainingSupport.countTrainableParameters(model.getBlock()));
        }
    }

    @Test
    void identityBlockPreservesShape() {
        ResidualBlock block = new ResidualBlock(8, 8, 1);
        assertFalse(block.hasProjectionShortcut());
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            Shape in = new Shape(2, 8, 16, 16);
            block.initialize(m, DataType.FLOAT32, in);
            NDArray y = block.forward(new ParameterStore(m, false), new NDList(m.randomNormal(in)), false).singletonOrThrow();
            assertEquals(in, y.getShape());
            assertArrayEquals(new Shape[] {in}, block.getOutputShapes(new Shape[] {in}));
        }
    }

    @Test
    void projectionShortcutHandlesStrideAndChannelChange() {
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            Shape in = new Shape(2, 8, 16, 16);
            ParameterStore ps = new ParameterStore(m, false);

            ResidualBlock down = new ResidualBlock(8, 16, 2);  // halves H and W, doubles channels
            assertTrue(down.hasProjectionShortcut());
            down.initialize(m, DataType.FLOAT32, in);
            assertEquals(new Shape(2, 16, 8, 8),
                    down.forward(ps, new NDList(m.randomNormal(in)), false).singletonOrThrow().getShape());

            ResidualBlock wider = new ResidualBlock(8, 16, 1); // channel change only
            assertTrue(wider.hasProjectionShortcut());
            wider.initialize(m, DataType.FLOAT32, in);
            assertEquals(new Shape(2, 16, 16, 16),
                    wider.forward(ps, new NDList(m.randomNormal(in)), true).singletonOrThrow().getShape());

            ResidualBlock strided = new ResidualBlock(8, 8, 2); // stride only
            assertTrue(strided.hasProjectionShortcut());
            Shape odd = new Shape(1, 8, 15, 15);
            strided.initialize(m, DataType.FLOAT32, odd);
            assertEquals(new Shape(1, 8, 8, 8),
                    strided.forward(ps, new NDList(m.randomNormal(odd)), false).singletonOrThrow().getShape());
            assertArrayEquals(new Shape[] {new Shape(1, 8, 8, 8)}, strided.getOutputShapes(new Shape[] {odd}));
        }
    }

    /**
     * y = ReLU(F(x) + x): with the last BatchNorm's scale and shift set to zero the residual branch
     * outputs F(x) = 0, so an identity block must return exactly ReLU(x) - the input carried by the
     * shortcut.
     */
    @Test
    void identityShortcutAddsTheInputBack() {
        ResidualBlock block = new ResidualBlock(4, 4, 1);
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            Shape in = new Shape(2, 4, 8, 8);
            block.initialize(m, DataType.FLOAT32, in);
            int zeroed = 0;
            for (Pair<String, Parameter> p : block.getParameters()) {
                String name = p.getKey();
                if (name.contains("bn2") && (name.endsWith("gamma") || name.endsWith("beta"))) {
                    p.getValue().getArray().muli(0f);
                    zeroed++;
                }
            }
            assertEquals(2, zeroed, "expected bn2 gamma and beta");
            NDArray x = m.randomNormal(in);
            float[] y = block.forward(new ParameterStore(m, false), new NDList(x), false).singletonOrThrow().toFloatArray();
            float[] relu = x.toFloatArray();
            for (int i = 0; i < relu.length; i++) {
                relu[i] = Math.max(0f, relu[i]);
            }
            assertArrayEquals(relu, y, 1e-6f);
        }
    }

    @Test
    void rejectsInvalidConfigurations() {
        assertThrows(IllegalArgumentException.class, () -> new ResidualBlock(8, 8, 3));
        assertThrows(IllegalArgumentException.class, () -> new ResidualBlock(0, 8, 1));
        assertThrows(IllegalArgumentException.class, () -> new EuroSatResNet(new int[] {}, 2, 10));
        assertThrows(IllegalArgumentException.class, () -> new EuroSatResNet(new int[] {8}, 0, 10));
        assertThrows(IllegalArgumentException.class, () -> new EuroSatResNet(new int[] {8}, 1, 1));
        EuroSatResNet net = new EuroSatResNet(10);
        assertThrows(IllegalArgumentException.class,
                () -> net.getOutputShapes(new Shape[] {new Shape(1, 1, 64, 64)}));
    }

    /**
     * If the network cannot fit a tiny labelled batch, something in the model, the loss (softmax
     * cross-entropy with integer labels) or the optimiser is broken.
     */
    @Test
    void tinyNetworkFitsTwelveSyntheticImages() {
        Engine.getInstance().setRandomSeed(3);
        int n = 12;
        int size = 64;
        int plane = size * size;
        float[] x = new float[n * 3 * plane];
        long[] y = new long[n];
        Random rnd = new Random(5);
        for (int i = 0; i < n; i++) {
            y[i] = i % 3;
            for (int c = 0; c < 3; c++) {
                float base = c == y[i] ? 1f : -1f; // class k = image whose channel k is bright
                for (int k = 0; k < plane; k++) {
                    x[i * 3 * plane + c * plane + k] = base + 0.5f * (float) rnd.nextGaussian();
                }
            }
        }
        try (Model model = EuroSatTraining.newModel(new int[] {8, 16}, 1, 3, Device.cpu());
             Trainer trainer = EuroSatTraining.newTrainer(model, Tracker.fixed(1e-2f), 0f, n, size)) {
            EuroSatTraining.StepResult first = EuroSatTraining.trainStep(trainer, x, y, n, size);
            EuroSatTraining.StepResult last = first;
            for (int s = 0; s < 30; s++) {
                last = EuroSatTraining.trainStep(trainer, x, y, n, size);
            }
            float[] lossOut = new float[1];
            float[] logits = EuroSatTraining.evaluateBatch(trainer, x, y, n, size, lossOut);
            int correct = 0;
            for (int i = 0; i < n; i++) {
                if (EuroSatTraining.argMax(logits, i, 3) == y[i]) {
                    correct++;
                }
            }
            System.out.printf("Tiny ResNet fit: loss %.4f -> %.4f (train mode), eval loss %.4f, %d/%d correct%n",
                    first.loss(), last.loss(), lossOut[0], correct, n);
            assertTrue(last.loss() < 0.5f * first.loss(), first.loss() + " -> " + last.loss());
            assertEquals(n, correct);
            assertEquals(n, last.correct());
        }
    }
}
