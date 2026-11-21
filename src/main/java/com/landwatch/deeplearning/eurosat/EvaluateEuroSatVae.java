package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.ParameterStore;
import com.landwatch.deeplearning.DjlRuntime;
import com.landwatch.deeplearning.Normalization;
import com.landwatch.deeplearning.TrainingSupport;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Evaluates the trained {@link EuroSatVae} on one EuroSAT split and writes
 * {@code results/eurosat_vae/<split>_metrics.txt} and {@code <split>_per_class.csv}.
 *
 * <p>Usage: {@code EvaluateEuroSatVae <val|test> [--data dir] [--splits dir] [--model models/eurosat_vae]
 * [--results results/eurosat_vae] [--device auto] [--force false]}</p>
 *
 * <p>Reported per image (averaged): reconstruction loss and KL (nats), their sum (negative ELBO),
 * and the mean squared error / PSNR of the reconstruction {@code sigmoid(decoder(mu))}. For
 * reference, the same MSE is given for the trivial predictor "the mean TRAIN image". For the test
 * split it also saves {@code test_reconstructions.png} (first row: one test image per class; second
 * row: its reconstruction) and {@code prior_samples.png} (20 images decoded from random
 * {@code z ~ N(0, I)}, seed 7). Like the other models, TEST is evaluated once: an existing
 * {@code test_metrics.txt} is not overwritten without {@code --force true}.</p>
 */
public final class EvaluateEuroSatVae {

    private static final int BATCH = 128;
    private static final int SCALE = 2;

    private EvaluateEuroSatVae() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args[0].startsWith("--")) {
            System.err.println("Usage: EvaluateEuroSatVae <val|test> [--data dir] [--model dir] [--results dir] [--device auto] [--force true]");
            System.exit(2);
        }
        String splitName = args[0];
        Map<String, String> opt = EuroSatSplit.parseArgs(java.util.Arrays.copyOfRange(args, 1, args.length));
        Path data = opt.containsKey("data") ? Path.of(opt.get("data")) : EuroSatSplit.locateDatasetRoot();
        if (data == null) {
            throw new IllegalArgumentException("EuroSAT not found; pass --data <folder containing EuroSAT_RGB/>");
        }
        Path splitDir = Path.of(opt.getOrDefault("splits", data.resolve("splits").toString()));
        Path modelDir = Path.of(opt.getOrDefault("model", "models/eurosat_vae"));
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/eurosat_vae"));
        boolean force = Boolean.parseBoolean(opt.getOrDefault("force", "false"));
        Path report = resultsDir.resolve(splitName + "_metrics.txt");
        if ("test".equals(splitName) && Files.exists(report) && !force) {
            System.err.println("Refusing to re-evaluate TEST: " + report + " already exists (use --force true deliberately).");
            System.exit(3);
        }

        EuroSatSplit.Split split = EuroSatSplit.read(splitDir);
        List<String> classNames = split.classNames();
        EuroSatImages images = EuroSatImages.load(data, split.get(splitName), EuroSatVae.INPUT_SIZE, 4);
        EuroSatImages train = EuroSatImages.load(data, split.train(), EuroSatVae.INPUT_SIZE, 4);
        float[] meanTrain = VaeSupport.meanImage(train);
        double baselineMse = VaeSupport.mseAgainst(images, meanTrain);

        Map<String, String> props = VaeSupport.loadProperties(modelDir);
        int latent = Integer.parseInt(props.get("latent"));
        Device device = DjlRuntime.resolveDevice(opt.getOrDefault("device", "auto"));
        Engine.getInstance().setRandomSeed(7);
        long start = System.nanoTime();
        VaeSupport.Stats s;
        try (Model model = VaeSupport.newModel(latent, device)) {
            try {
                model.load(modelDir, VaeSupport.MODEL_NAME);
            } catch (MalformedModelException e) {
                throw new IOException("Cannot load " + modelDir, e);
            }
            EuroSatVae vae = (EuroSatVae) model.getBlock();
            ParameterStore ps = new ParameterStore(model.getNDManager(), false);
            s = VaeSupport.evaluate(in -> vae.forward(ps, in, false), model.getNDManager(), images, classNames.size(), BATCH);
            if ("test".equals(splitName)) {
                writeReconstructions(vae, ps, model.getNDManager(), images, classNames.size(),
                        resultsDir.resolve("test_reconstructions.png"));
                writePriorSamples(vae, ps, model.getNDManager(), latent, resultsDir.resolve("prior_samples.png"));
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;

        StringBuilder sb = new StringBuilder();
        sb.append("LandWatch AI - EuroSAT variational autoencoder, evaluation on the ").append(splitName.toUpperCase(Locale.ROOT))
                .append(" split\n");
        sb.append("Generated: ").append(LocalDateTime.now()).append('\n');
        sb.append("Model: ").append(TrainingSupport.displayPath(modelDir)).append(" (latent ").append(latent)
                .append(", best epoch ").append(props.getOrDefault("training.best_epoch", "?")).append(", chosen on VAL)\n");
        sb.append("Images: ").append(s.images()).append(" of 64x64 RGB; device ").append(device).append(", ").append(DjlRuntime.describe()).append('\n');
        sb.append("Reconstructions use z = mu (no sampling). Losses are per image, in nats, summed over 3x64x64 values.\n\n");
        sb.append(String.format(Locale.ROOT, "Reconstruction loss (Bernoulli BCE): %.2f%n", s.reconstruction()));
        sb.append(String.format(Locale.ROOT, "KL to N(0, I):                       %.2f%n", s.kl()));
        sb.append(String.format(Locale.ROOT, "Negative ELBO (sum):                 %.2f%n", s.negativeElbo()));
        sb.append(String.format(Locale.ROOT, "Reconstruction MSE (pixels 0..1):    %.6f  (PSNR %.2f dB)%n", s.mse(), s.psnr()));
        sb.append(String.format(Locale.ROOT, "Baseline MSE, mean TRAIN image:      %.6f  (PSNR %.2f dB)%n", baselineMse, 10 * Math.log10(1 / baselineMse)));
        sb.append(String.format(Locale.ROOT, "MSE reduction vs the baseline:       %.1f %%%n%n", 100 * (1 - s.mse() / baselineMse)));
        sb.append("Reconstruction MSE per class (lower = easier to compress into ").append(latent).append(" numbers):\n");
        StringBuilder csv = new StringBuilder("class,images,mse,psnr_db\n");
        for (int c = 0; c < classNames.size(); c++) {
            sb.append(String.format(Locale.ROOT, "  %-22s %5d images  MSE %.6f  PSNR %.2f dB%n", classNames.get(c), s.classCount()[c],
                    s.classMse()[c], 10 * Math.log10(1 / s.classMse()[c])));
            csv.append(String.format(Locale.ROOT, "%s,%d,%.6f,%.2f%n", classNames.get(c), s.classCount()[c], s.classMse()[c],
                    10 * Math.log10(1 / s.classMse()[c])));
        }
        sb.append(String.format(Locale.ROOT, "%nEvaluation time (model part): %.1f s%n", seconds));
        Files.createDirectories(resultsDir);
        Files.writeString(report, sb.toString(), StandardCharsets.UTF_8);
        Files.writeString(resultsDir.resolve(splitName + "_per_class.csv"), csv.toString(), StandardCharsets.UTF_8);
        System.out.println(sb);
        System.out.println("Wrote " + TrainingSupport.displayPath(report));
    }

    /** First row: the first image of each class in split order; second row: the reconstructions. */
    private static void writeReconstructions(EuroSatVae vae, ParameterStore ps, NDManager manager, EuroSatImages images,
                                             int classes, Path file) throws IOException {
        List<Integer> picks = new ArrayList<>();
        for (int c = 0; c < classes; c++) {
            for (int i = 0; i < images.count(); i++) {
                if (images.label(i) == c) {
                    picks.add(i);
                    break;
                }
            }
        }
        int n = picks.size();
        int[] order = picks.stream().mapToInt(Integer::intValue).toArray();
        float[] xs = images.batch(order, 0, n, Normalization.none(), null, new long[n]);
        float[] rec;
        try (NDManager scope = manager.newSubManager()) {
            NDArray x = scope.create(xs, new Shape(n, 3, 64, 64));
            rec = TrainingSupport.toCpuFloats(Activation.sigmoid(vae.forward(ps, new NDList(x), false).get(0)));
        }
        BufferedImage grid = new BufferedImage(n * 64 * SCALE, 2 * 64 * SCALE, BufferedImage.TYPE_INT_RGB);
        for (int j = 0; j < n; j++) {
            paint(grid, xs, j, j * 64 * SCALE, 0);
            paint(grid, rec, j, j * 64 * SCALE, 64 * SCALE);
        }
        ImageIO.write(grid, "png", file.toFile());
    }

    /** Two rows of ten images decoded from z ~ N(0, I). */
    private static void writePriorSamples(EuroSatVae vae, ParameterStore ps, NDManager manager, int latent, Path file)
            throws IOException {
        int n = 20;
        float[] img;
        try (NDManager scope = manager.newSubManager()) {
            java.util.Random rnd = new java.util.Random(7);
            float[] z = new float[n * latent];
            for (int i = 0; i < z.length; i++) {
                z[i] = (float) rnd.nextGaussian();
            }
            NDArray zs = scope.create(z, new Shape(n, latent));
            img = TrainingSupport.toCpuFloats(Activation.sigmoid(vae.decode(ps, zs, false, null)));
        }
        BufferedImage grid = new BufferedImage(10 * 64 * SCALE, 2 * 64 * SCALE, BufferedImage.TYPE_INT_RGB);
        for (int j = 0; j < n; j++) {
            paint(grid, img, j, (j % 10) * 64 * SCALE, (j / 10) * 64 * SCALE);
        }
        ImageIO.write(grid, "png", file.toFile());
    }

    /** Paints image j of a flat (N, 3, 64, 64) array in [0, 1], enlarged SCALE times, at (ox, oy). */
    static void paint(BufferedImage dst, float[] chw, int j, int ox, int oy) {
        int plane = 64 * 64;
        int base = j * 3 * plane;
        for (int y = 0; y < 64; y++) {
            for (int x = 0; x < 64; x++) {
                int r = clamp(chw[base + y * 64 + x]);
                int g = clamp(chw[base + plane + y * 64 + x]);
                int b = clamp(chw[base + 2 * plane + y * 64 + x]);
                int rgb = (r << 16) | (g << 8) | b;
                for (int dy = 0; dy < SCALE; dy++) {
                    for (int dx = 0; dx < SCALE; dx++) {
                        dst.setRGB(ox + x * SCALE + dx, oy + y * SCALE + dy, rgb);
                    }
                }
            }
        }
    }

    private static int clamp(float v) {
        return Math.max(0, Math.min(255, Math.round(v * 255f)));
    }
}
