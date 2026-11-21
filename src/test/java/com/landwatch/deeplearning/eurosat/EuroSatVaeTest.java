package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.tracker.Tracker;
import com.landwatch.deeplearning.TrainingSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The EuroSAT variational autoencoder: shapes, loss terms, sampling and a short training run. */
class EuroSatVaeTest {

    @Test
    void returnsLogitsMeanAndLogVariance() {
        try (Model model = VaeSupport.newModel(16, Device.cpu())) {
            NDManager m = model.getNDManager();
            Shape in = new Shape(3, 3, 64, 64);
            model.getBlock().initialize(m, DataType.FLOAT32, in);
            NDList out = model.getBlock().forward(new ParameterStore(m, false), new NDList(m.randomUniform(0, 1, in)), false);
            assertEquals(new Shape(3, 3, 64, 64), out.get(0).getShape());
            assertEquals(new Shape(3, 16), out.get(1).getShape());
            assertEquals(new Shape(3, 16), out.get(2).getShape());
            assertThrows(IllegalArgumentException.class,
                    () -> model.getBlock().getOutputShapes(new Shape[] {new Shape(1, 3, 32, 32)}));
        }
    }

    @Test
    void inferenceIsDeterministicAndTrainingSamples() {
        try (Model model = VaeSupport.newModel(8, Device.cpu())) {
            NDManager m = model.getNDManager();
            Shape in = new Shape(2, 3, 64, 64);
            model.getBlock().initialize(m, DataType.FLOAT32, in);
            ParameterStore ps = new ParameterStore(m, false);
            NDArray x = m.randomUniform(0, 1, in);
            float[] a = model.getBlock().forward(ps, new NDList(x), false).get(0).toFloatArray();
            float[] b = model.getBlock().forward(ps, new NDList(x), false).get(0).toFloatArray();
            assertArrayEquals(a, b, 0f);
            float[] c = model.getBlock().forward(ps, new NDList(x), true).get(0).toFloatArray();
            assertNotEquals(a[0], c[0]); // z is sampled in training mode
        }
    }

    @Test
    void klIsZeroForTheStandardNormalAndMatchesTheFormula() {
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            NDArray zero = m.zeros(new Shape(2, 4));
            assertArrayEquals(new float[] {0, 0}, VaeLoss.klPerImage(zero, zero).toFloatArray(), 1e-6f);
            // one dimension with mu = 1, logvar = ln 4: -1/2 (1 + ln 4 - 1 - 4) = 2 - ln 2
            NDArray mu = m.create(new float[] {1f}, new Shape(1, 1));
            NDArray logvar = m.create(new float[] {(float) Math.log(4)}, new Shape(1, 1));
            assertEquals(2 - Math.log(2), VaeLoss.klPerImage(mu, logvar).getFloat(0), 1e-5);
        }
    }

    @Test
    void reconstructionLossIsBinaryCrossEntropySummedPerImage() {
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            float[] logits = {2f, -1f, 0f, 30f};
            float[] x = {1f, 0.25f, 0.5f, 0f};
            double expected = 0;
            for (int i = 0; i < 4; i++) {
                // -log(sigmoid(l)) = log(1 + e^-l) and -log(1 - sigmoid(l)) = log(1 + e^l), exact in double
                double l = logits[i];
                expected += x[i] * Math.log1p(Math.exp(-l)) + (1 - x[i]) * Math.log1p(Math.exp(l));
            }
            Shape s = new Shape(1, 1, 2, 2);
            float got = VaeLoss.reconstructionPerImage(m.create(logits, s), m.create(x, s)).getFloat(0);
            assertEquals(expected, got, 1e-4); // stable even for the large logit 30
        }
    }

    @Test
    void shortTrainingLowersTheLossOnAFixedBatch() {
        Random rnd = new Random(3);
        float[] xs = new float[4 * 3 * 64 * 64];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = (i / 4096 % 3) * 0.3f + 0.1f * rnd.nextFloat(); // smooth colour planes with a little noise
        }
        try (Model model = VaeSupport.newModel(8, Device.cpu());
             Trainer trainer = VaeSupport.newTrainer(model, 1f, Tracker.fixed(1e-3f), 4)) {
            NDManager m = trainer.getManager();
            Shape s = new Shape(4, 3, 64, 64);
            float first = TrainingSupport.trainStep(trainer, new NDList(m.create(xs, s)), new NDList(m.create(xs, s)));
            float last = first;
            for (int i = 0; i < 40; i++) {
                last = TrainingSupport.trainStep(trainer, new NDList(m.create(xs, s)), new NDList(m.create(xs, s)));
            }
            assertTrue(Float.isFinite(last) && last < first, "first " + first + ", last " + last);
        }
    }

    @Test
    void decodesPriorSamplesToImages() {
        try (Model model = VaeSupport.newModel(8, Device.cpu())) {
            NDManager m = model.getNDManager();
            model.getBlock().initialize(m, DataType.FLOAT32, new Shape(1, 3, 64, 64));
            NDArray img = ((EuroSatVae) model.getBlock()).decode(new ParameterStore(m, false),
                    m.randomNormal(new Shape(5, 8)), false, null);
            assertEquals(new Shape(5, 3, 64, 64), img.getShape());
        }
    }

    @Test
    void propertiesRoundTrip(@TempDir Path dir) throws Exception {
        VaeSupport.saveProperties(Map.of("latent", "64", "loss", "a = b: c"), dir);
        Map<String, String> back = VaeSupport.loadProperties(dir);
        assertEquals("64", back.get("latent"));
        assertEquals("a = b: c", back.get("loss"));
        assertEquals(2, back.size());
    }

    @Test
    void statsComputePsnrFromMse() {
        VaeSupport.Stats s = new VaeSupport.Stats(1, 10, 2, 0.01, new double[0], new int[0]);
        assertEquals(12, s.negativeElbo(), 1e-12);
        assertEquals(20, s.psnr(), 1e-9);
    }
}
