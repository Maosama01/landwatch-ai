package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.landwatch.deeplearning.Normalization;
import com.landwatch.deeplearning.TrainingSupport;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import java.util.function.Function;

/** Shared code for training and evaluating {@link EuroSatVae}. */
public final class VaeSupport {

    /** DJL model name and parameter-file prefix ({@code eurosat_vae-0012.params}). */
    public static final String MODEL_NAME = "eurosat_vae";
    /** Settings file stored next to the weights. */
    public static final String PROPERTIES_FILE = "vae.properties";

    private VaeSupport() {
    }

    public static Model newModel(int latent, Device device) {
        Model model = Model.newInstance(MODEL_NAME, device, "PyTorch");
        model.setBlock(new EuroSatVae(latent));
        return model;
    }

    /** Adam with a learning-rate schedule, He initialisation, parameters initialised for 64x64 batches. */
    public static Trainer newTrainer(Model model, float beta, Tracker learningRate, int batchSize) {
        DefaultTrainingConfig config = new DefaultTrainingConfig(new VaeLoss(beta))
                .optOptimizer(Optimizer.adam().optLearningRateTracker(learningRate).build())
                .optDevices(new Device[] {model.getNDManager().getDevice()})
                .optInitializer(new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN,
                        XavierInitializer.FactorType.IN, 2f), Parameter.Type.WEIGHT);
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(batchSize, 3, EuroSatVae.INPUT_SIZE, EuroSatVae.INPUT_SIZE));
        return trainer;
    }

    /** Averages over a set of images, computed with z = mu (deterministic reconstructions). */
    public record Stats(int images, double reconstruction, double kl, double mse, double[] classMse, int[] classCount) {

        /** Negative ELBO per image (nats) with beta = 1: reconstruction + KL. */
        public double negativeElbo() {
            return reconstruction + kl;
        }

        /** Peak signal-to-noise ratio of the reconstructions on the [0, 1] scale, from the mean squared error. */
        public double psnr() {
            return 10 * Math.log10(1.0 / mse);
        }
    }

    /**
     * Runs the model over all images (no augmentation, pixel values in [0, 1]) and averages the
     * reconstruction loss, the KL term and the mean squared error of {@code sigmoid(logits)}.
     *
     * @param forward inference-mode forward pass returning (logits, mu, logvar)
     * @param manager manager on the model's device; per-batch arrays are freed after each batch
     */
    public static Stats evaluate(Function<NDList, NDList> forward, NDManager manager, EuroSatImages images,
                                 int classes, int batchSize) {
        int[] order = new int[images.count()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        double recon = 0;
        double kl = 0;
        double mse = 0;
        double[] classMse = new double[classes];
        int[] classCount = new int[classes];
        long[] labels = new long[batchSize];
        int size = images.imageSize();
        for (int from = 0; from < order.length; from += batchSize) {
            int to = Math.min(order.length, from + batchSize);
            int n = to - from;
            float[] xs = images.batch(order, from, to, Normalization.none(), null, labels);
            try (NDManager scope = manager.newSubManager()) {
                NDArray x = scope.create(xs, new Shape(n, 3, size, size));
                NDList out = forward.apply(new NDList(x));
                float[] r = TrainingSupport.toCpuFloats(VaeLoss.reconstructionPerImage(out.get(0), x));
                float[] k = TrainingSupport.toCpuFloats(VaeLoss.klPerImage(out.get(1), out.get(2)));
                float[] e = TrainingSupport.toCpuFloats(
                        Activation.sigmoid(out.get(0)).sub(x).square().reshape(n, -1).mean(new int[] {1})); // DJL: one axis per mean
                for (int j = 0; j < n; j++) {
                    recon += r[j];
                    kl += k[j];
                    mse += e[j];
                    int c = (int) labels[j];
                    classMse[c] += e[j];
                    classCount[c]++;
                }
            }
        }
        int total = order.length;
        for (int c = 0; c < classes; c++) {
            classMse[c] = classCount[c] == 0 ? Double.NaN : classMse[c] / classCount[c];
        }
        return new Stats(total, recon / total, kl / total, mse / total, classMse, classCount);
    }

    /** Per-pixel mean of a set of images (CHW, [0, 1]); the "always predict the average TRAIN image" baseline. */
    public static float[] meanImage(EuroSatImages images) {
        double[] sum = null;
        for (int i = 0; i < images.count(); i++) {
            float[] x = images.chw01(i);
            if (sum == null) {
                sum = new double[x.length];
            }
            for (int k = 0; k < x.length; k++) {
                sum[k] += x[k];
            }
        }
        float[] mean = new float[sum.length];
        for (int k = 0; k < sum.length; k++) {
            mean[k] = (float) (sum[k] / images.count());
        }
        return mean;
    }

    /** Mean squared error of predicting {@code reference} for every image of the set. */
    public static double mseAgainst(EuroSatImages images, float[] reference) {
        double total = 0;
        for (int i = 0; i < images.count(); i++) {
            float[] x = images.chw01(i);
            double s = 0;
            for (int k = 0; k < x.length; k++) {
                double d = x[k] - reference[k];
                s += d * d;
            }
            total += s / x.length;
        }
        return total / images.count();
    }

    public static void saveProperties(Map<String, String> values, Path dir) throws IOException {
        Files.createDirectories(dir);
        StringBuilder sb = new StringBuilder("# LandWatch AI - EuroSAT variational autoencoder (trained with DJL)\n");
        for (Map.Entry<String, String> e : new TreeMap<>(values).entrySet()) {
            Properties one = new Properties();
            one.setProperty(e.getKey(), e.getValue());
            java.io.StringWriter w = new java.io.StringWriter();
            one.store(w, null);
            String[] lines = w.toString().split("\\R");
            sb.append(lines[lines.length - 1]).append('\n'); // skip the date comment Properties writes
        }
        try (Writer w = Files.newBufferedWriter(dir.resolve(PROPERTIES_FILE), StandardCharsets.UTF_8)) {
            w.write(sb.toString());
        }
    }

    public static Map<String, String> loadProperties(Path dir) throws IOException {
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(dir.resolve(PROPERTIES_FILE), StandardCharsets.UTF_8)) {
            p.load(r);
        }
        Map<String, String> map = new TreeMap<>();
        for (String k : p.stringPropertyNames()) {
            map.put(k, p.getProperty(k));
        }
        return map;
    }

    static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }
}
