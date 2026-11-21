package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.Trainer;
import ai.djl.training.tracker.Tracker;
import com.landwatch.deeplearning.DjlRuntime;
import com.landwatch.deeplearning.Normalization;
import com.landwatch.deeplearning.TrainingSupport;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

/**
 * Trains the {@link EuroSatVae} generative model on the EuroSAT TRAIN split.
 *
 * <p>Protocol: the same stratified 70/15/15 split as the classifier. The VAE trains on TRAIN images
 * (random D4 rotations/flips, pixel values in [0, 1], no class labels used), with Adam and a cosine
 * learning-rate decay. After every epoch it is evaluated on VAL with {@code z = mu}; the epoch with
 * the lowest VAL negative ELBO is kept. TEST images are not read here; evaluate them once with
 * {@link EvaluateEuroSatVae}.</p>
 *
 * <p>Options: {@code --data <folder containing EuroSAT_RGB/> --splits <data>/splits
 * --out models/eurosat_vae --results results/eurosat_vae --latent 64 --beta 1 --epochs 25
 * --batch 64 --lr 1e-3 --device auto --seed 42 --workers 4 --train-limit 0 --val-limit 0}</p>
 */
public final class TrainEuroSatVae {

    private TrainEuroSatVae() {
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = EuroSatSplit.parseArgs(args);
        Path data = opt.containsKey("data") ? Path.of(opt.get("data")) : EuroSatSplit.locateDatasetRoot();
        if (data == null) {
            throw new IllegalArgumentException("EuroSAT not found; pass --data <folder containing EuroSAT_RGB/>");
        }
        Path splitDir = Path.of(opt.getOrDefault("splits", data.resolve("splits").toString()));
        Path outDir = Path.of(opt.getOrDefault("out", "models/eurosat_vae"));
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/eurosat_vae"));
        int latent = Integer.parseInt(opt.getOrDefault("latent", String.valueOf(EuroSatVae.DEFAULT_LATENT)));
        float beta = Float.parseFloat(opt.getOrDefault("beta", "1"));
        int epochs = Integer.parseInt(opt.getOrDefault("epochs", "25"));
        int batchSize = Integer.parseInt(opt.getOrDefault("batch", "64"));
        float lr = Float.parseFloat(opt.getOrDefault("lr", "1e-3"));
        int seed = Integer.parseInt(opt.getOrDefault("seed", "42"));
        int workers = Integer.parseInt(opt.getOrDefault("workers", "4"));
        int trainLimit = Integer.parseInt(opt.getOrDefault("train-limit", "0"));
        int valLimit = Integer.parseInt(opt.getOrDefault("val-limit", "0"));
        Device device = DjlRuntime.resolveDevice(opt.getOrDefault("device", "auto"));

        Engine.getInstance().setRandomSeed(seed);
        System.out.println("=== LandWatch AI - EuroSAT VAE training (" + LocalDateTime.now() + ") ===");
        System.out.println(DjlRuntime.describe() + ", device " + device + ", Java " + System.getProperty("java.version"));
        EuroSatSplit.Split split = EuroSatSplit.read(splitDir);
        List<EuroSatSample> trainSamples = TrainEuroSatClassifier.subset(split.train(), trainLimit, seed);
        List<EuroSatSample> valSamples = TrainEuroSatClassifier.subset(split.val(), valLimit, seed + 1L);
        int classes = split.classNames().size();
        System.out.printf("Images: train %d, val %d (test %d held out, not read); class labels are not used for training%n",
                trainSamples.size(), valSamples.size(), split.test().size());
        EuroSatImages train = EuroSatImages.load(data, trainSamples, EuroSatVae.INPUT_SIZE, workers);
        EuroSatImages val = EuroSatImages.load(data, valSamples, EuroSatVae.INPUT_SIZE, workers);

        int steps = (train.count() + batchSize - 1) / batchSize;
        Tracker schedule = Tracker.cosine().setBaseValue(lr).optFinalValue(lr * 0.01f)
                .setMaxUpdates(Math.max(1, epochs * steps)).build();
        Map<String, String> props = new TreeMap<>();
        props.put("architecture", "convolutional VAE (Kingma & Welling 2014): encoder 4 x [4x4 conv stride 2 + ReLU] (32-64-128-256),"
                + " Linear -> mu, logvar; decoder Linear + 4 x [2x nearest upsample, 3x3 conv + ReLU] (128-64-32-32), 3x3 conv -> logits");
        props.put("latent", String.valueOf(latent));
        props.put("input", "64x64 RGB, pixel values / 255 (no normalisation), sigmoid output");
        props.put("loss", "negative ELBO: Bernoulli reconstruction (BCE summed over pixels) + " + beta + " x KL to N(0, I)");
        props.put("training.beta", String.valueOf(beta));
        props.put("training.epochs_planned", String.valueOf(epochs));
        props.put("training.batch_size", String.valueOf(batchSize));
        props.put("training.optimizer", "Adam");
        props.put("training.lr_schedule", "cosine decay from " + lr + " to " + (lr * 0.01f));
        props.put("training.augmentation", "TRAIN only: random D4 transform (rotation by 0/90/180/270 deg, optional horizontal flip)");
        props.put("training.device", device.toString());
        props.put("training.seed", String.valueOf(seed));
        props.put("checkpoint.selection", "epoch with the lowest VAL negative ELBO (z = mu)");
        props.put("data.train_images", String.valueOf(train.count()));
        props.put("data.val_images", String.valueOf(val.count()));
        props.put("data.test_images", String.valueOf(split.test().size()));
        props.put("djl_version", DjlRuntime.djlVersion());
        props.put("engine_version", DjlRuntime.engineVersion());

        Files.createDirectories(outDir);
        Files.createDirectories(resultsDir);
        Random rng = new Random(seed);
        int[] order = new int[train.count()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        long[] labels = new long[batchSize];
        double best = Double.POSITIVE_INFINITY;
        long start = System.nanoTime();
        try (Model model = VaeSupport.newModel(latent, device);
             Trainer trainer = VaeSupport.newTrainer(model, beta, schedule, batchSize);
             PrintWriter log = new PrintWriter(Files.newBufferedWriter(resultsDir.resolve("training_log.csv"), StandardCharsets.UTF_8))) {
            long params = TrainingSupport.countTrainableParameters(model.getBlock());
            props.put("trainable_parameters", String.valueOf(params));
            System.out.printf("Model %s, %,d trainable parameters; %d steps/epoch%n", model.getBlock(), params, steps);
            log.println("epoch,train_loss,val_reconstruction,val_kl,val_negative_elbo,val_mse,val_psnr_db,seconds,checkpoint_saved");
            for (int epoch = 1; epoch <= epochs; epoch++) {
                long t0 = System.nanoTime();
                shuffle(order, rng);
                double lossSum = 0;
                for (int from = 0; from < order.length; from += batchSize) {
                    int to = Math.min(order.length, from + batchSize);
                    int n = to - from;
                    float[] xs = train.batch(order, from, to, Normalization.none(), rng, labels);
                    try (NDManager step = trainer.getManager().newSubManager()) {
                        Shape shape = new Shape(n, 3, EuroSatVae.INPUT_SIZE, EuroSatVae.INPUT_SIZE);
                        NDArray x = step.create(xs, shape);
                        NDArray target = step.create(xs, shape);
                        float l = TrainingSupport.trainStep(trainer, new NDList(x), new NDList(target));
                        if (!Float.isFinite(l)) {
                            throw new IllegalStateException("Non-finite loss at epoch " + epoch);
                        }
                        lossSum += l * n;
                    }
                }
                VaeSupport.Stats v = VaeSupport.evaluate(trainer::evaluate, trainer.getManager(), val, classes, batchSize);
                boolean improved = v.negativeElbo() < best;
                double seconds = (System.nanoTime() - t0) / 1e9;
                if (improved) {
                    best = v.negativeElbo();
                    saveCheckpoint(model, outDir, epoch);
                    props.put("training.best_epoch", String.valueOf(epoch));
                    props.put("metrics.val.reconstruction", VaeSupport.fmt(v.reconstruction()));
                    props.put("metrics.val.kl", VaeSupport.fmt(v.kl()));
                    props.put("metrics.val.negative_elbo", VaeSupport.fmt(v.negativeElbo()));
                    props.put("metrics.val.mse", String.format(Locale.ROOT, "%.6f", v.mse()));
                    props.put("metrics.val.psnr_db", String.format(Locale.ROOT, "%.2f", v.psnr()));
                    VaeSupport.saveProperties(props, outDir);
                }
                log.printf(Locale.ROOT, "%d,%.3f,%.3f,%.3f,%.3f,%.6f,%.2f,%.1f,%s%n", epoch, lossSum / order.length,
                        v.reconstruction(), v.kl(), v.negativeElbo(), v.mse(), v.psnr(), seconds, improved ? "yes" : "no");
                log.flush();
                System.out.printf(Locale.ROOT,
                        "Epoch %d/%d: train loss %.2f | val recon %.2f KL %.2f -ELBO %.2f MSE %.5f PSNR %.2f dB | %.0f s%s%n",
                        epoch, epochs, lossSum / order.length, v.reconstruction(), v.kl(), v.negativeElbo(), v.mse(), v.psnr(),
                        seconds, improved ? "  [checkpoint saved]" : "");
            }
        }
        props.put("training.time_seconds", String.format(Locale.ROOT, "%.0f", (System.nanoTime() - start) / 1e9));
        props.put("training.epochs_completed", String.valueOf(epochs));
        VaeSupport.saveProperties(props, outDir);
        System.out.println("Done; best val negative ELBO " + VaeSupport.fmt(best) + " at epoch " + props.get("training.best_epoch")
                + ". Model in " + TrainingSupport.displayPath(outDir));
    }

    static void saveCheckpoint(Model model, Path outDir, int epoch) throws IOException {
        try (java.util.stream.Stream<Path> files = Files.list(outDir)) {
            for (Path f : (Iterable<Path>) files::iterator) {
                if (f.getFileName().toString().matches(VaeSupport.MODEL_NAME + "-\\d{4}\\.params")) {
                    Files.delete(f);
                }
            }
        }
        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(outDir, VaeSupport.MODEL_NAME);
    }

    private static void shuffle(int[] a, Random rng) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}
