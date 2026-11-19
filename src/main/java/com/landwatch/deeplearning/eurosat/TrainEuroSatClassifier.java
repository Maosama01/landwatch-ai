package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.training.Trainer;
import ai.djl.training.tracker.Tracker;
import com.landwatch.deeplearning.DjlRuntime;
import com.landwatch.deeplearning.Normalization;
import com.landwatch.deeplearning.TrainingSupport;
import com.landwatch.evaluation.MulticlassMetrics;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Trains the {@link EuroSatResNet} land-use / land-cover classifier on the EuroSAT TRAIN split.
 *
 * <p>Protocol (the three splits are kept strictly apart):</p>
 * <ol>
 *   <li>The per-channel mean and standard deviation are computed on TRAIN images only.</li>
 *   <li>The network is trained on TRAIN with softmax cross-entropy, Adam and a cosine learning-rate
 *       decay. Every training image gets a random D4 transform each time it is used (rotation by
 *       0/90/180/270 degrees, optionally followed by a horizontal flip): a satellite tile has no
 *       "up", so these are label-preserving.</li>
 *   <li>After every epoch the model is evaluated on VAL (accuracy, macro F1, loss). The weights of
 *       the epoch with the highest VAL accuracy (ties: higher VAL macro F1, then the earlier epoch)
 *       are saved.</li>
 *   <li>TEST images are not read here (only the size of the split is recorded). Evaluate TEST once
 *       with {@link EvaluateEuroSatClassifier}.</li>
 * </ol>
 *
 * <p>Options (all optional): {@code --data <EuroSAT folder containing EuroSAT_RGB/>
 * --splits <data>/splits --out models/eurosat_resnet --results results/eurosat --epochs 15
 * --batch 64 --lr 1e-3 --final-lr-factor 0.01 --weight-decay 0 --device auto --seed 42
 * --max-minutes 30 --workers 4 --widths 32,64,128,256 --blocks 2 --train-limit 0 --val-limit 0}.
 * {@code --train-limit/--val-limit} use a random subset (for quick smoke tests only).</p>
 */
public final class TrainEuroSatClassifier {

    private TrainEuroSatClassifier() {
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = EuroSatSplit.parseArgs(args);
        Path data = opt.containsKey("data") ? Path.of(opt.get("data")) : EuroSatSplit.locateDatasetRoot();
        if (data == null) {
            throw new IllegalArgumentException("EuroSAT not found; pass --data <folder containing EuroSAT_RGB/>");
        }
        Path splitDir = Path.of(opt.getOrDefault("splits", data.resolve("splits").toString()));
        Path outDir = Path.of(opt.getOrDefault("out", "models/eurosat_resnet"));
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/eurosat"));
        int epochs = Integer.parseInt(opt.getOrDefault("epochs", "15"));
        int batchSize = Integer.parseInt(opt.getOrDefault("batch", "64"));
        float lr = Float.parseFloat(opt.getOrDefault("lr", "1e-3"));
        float finalLrFactor = Float.parseFloat(opt.getOrDefault("final-lr-factor", "0.01"));
        float weightDecay = Float.parseFloat(opt.getOrDefault("weight-decay", "0"));
        int seed = Integer.parseInt(opt.getOrDefault("seed", "42"));
        double maxMinutes = Double.parseDouble(opt.getOrDefault("max-minutes", "30"));
        int workers = Integer.parseInt(opt.getOrDefault("workers", "4"));
        int[] widths = parseInts(opt.getOrDefault("widths", "32,64,128,256"));
        int blocks = Integer.parseInt(opt.getOrDefault("blocks", "2"));
        int trainLimit = Integer.parseInt(opt.getOrDefault("train-limit", "0"));
        int valLimit = Integer.parseInt(opt.getOrDefault("val-limit", "0"));
        Device device = DjlRuntime.resolveDevice(opt.getOrDefault("device", "auto"));
        int size = EuroSatResNet.INPUT_SIZE;

        long processStart = System.nanoTime();
        Engine.getInstance().setRandomSeed(seed);
        System.out.println("=== LandWatch AI - EuroSAT ResNet classifier training (" + LocalDateTime.now() + ") ===");
        System.out.println(DjlRuntime.describe() + ", device " + device + ", Java " + System.getProperty("java.version"));
        System.out.println("Images: " + data.toAbsolutePath() + "\nSplits: " + splitDir.toAbsolutePath());

        EuroSatSplit.Split split = EuroSatSplit.read(splitDir);
        List<String> classNames = split.classNames();
        List<EuroSatSample> trainSamples = subset(split.train(), trainLimit, seed);
        List<EuroSatSample> valSamples = subset(split.val(), valLimit, seed + 1L);
        int testSize = split.test().size(); // size only: TEST images are not read during training
        System.out.printf("Classes (%d, index order): %s%n", classNames.size(), String.join(", ", classNames));
        System.out.printf("Images: train %d, val %d (test %d held out, not read)%n",
                trainSamples.size(), valSamples.size(), testSize);

        long tLoad = System.nanoTime();
        EuroSatImages train = EuroSatImages.load(data, trainSamples, size, workers);
        EuroSatImages val = EuroSatImages.load(data, valSamples, size, workers);
        Normalization norm = train.computeNormalization();
        double loadSeconds = (System.nanoTime() - tLoad) / 1e9;
        System.out.printf(Locale.ROOT, "Decoded train + val JPEGs into memory and computed normalisation in %.1f s%n", loadSeconds);
        System.out.println("Normalisation (TRAIN only): " + norm);

        int stepsPerEpoch = (train.count() + batchSize - 1) / batchSize;
        Tracker schedule = Tracker.cosine().setBaseValue(lr).optFinalValue(lr * finalLrFactor)
                .setMaxUpdates(Math.max(1, epochs * stepsPerEpoch)).build();

        EuroSatModelConfig config = baseConfig(widths, blocks, classNames, norm, epochs, batchSize, lr, finalLrFactor,
                weightDecay, device, seed, data, splitDir, train.count(), val.count(), testSize,
                trainLimit > 0 || valLimit > 0);
        Files.createDirectories(outDir);
        Files.createDirectories(resultsDir);
        Path logCsv = resultsDir.resolve("training_log.csv");

        Random rng = new Random(seed);
        int[] order = new int[train.count()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        int[] valOrder = new int[val.count()];
        for (int i = 0; i < valOrder.length; i++) {
            valOrder[i] = i;
        }

        double bestAcc = -1;
        double bestF1 = -1;
        int bestEpoch = -1;
        int epochsCompleted = 0;
        String stopReason = "planned epochs completed";
        long trainingStart = System.nanoTime();
        try (Model model = EuroSatTraining.newModel(widths, blocks, classNames.size(), device);
             Trainer trainer = EuroSatTraining.newTrainer(model, schedule, weightDecay, batchSize, size);
             PrintWriter log = new PrintWriter(Files.newBufferedWriter(logCsv, StandardCharsets.UTF_8))) {
            long params = TrainingSupport.countTrainableParameters(model.getBlock());
            config.set("model.trainable_parameters", params);
            System.out.printf("Model %s, %,d trainable parameters; batch %d, %d steps/epoch, %d epochs planned%n",
                    model.getBlock(), params, batchSize, stepsPerEpoch, epochs);
            log.println("epoch,learning_rate_end,train_loss,train_accuracy,val_loss,val_accuracy,val_macro_f1,"
                    + "train_seconds,val_seconds,epoch_seconds,checkpoint_saved");
            log.flush();

            double lastEpochSeconds = 0;
            int globalStep = 0;
            long[] labels = new long[batchSize];
            for (int epoch = 1; epoch <= epochs; epoch++) {
                double elapsed = (System.nanoTime() - trainingStart) / 1e9;
                if (epoch > 1 && elapsed + lastEpochSeconds > maxMinutes * 60) {
                    stopReason = String.format(Locale.ROOT,
                            "time budget: %.1f min elapsed, next epoch (~%.1f min) would exceed %.0f min",
                            elapsed / 60, lastEpochSeconds / 60, maxMinutes);
                    System.out.println("Stopping early - " + stopReason);
                    break;
                }
                long epochStart = System.nanoTime();

                // ---- train ----
                shuffle(order, rng);
                double lossSum = 0;
                long correct = 0;
                int seen = 0;
                int step = 0;
                for (int from = 0; from < order.length; from += batchSize) {
                    int to = Math.min(order.length, from + batchSize);
                    int n = to - from;
                    float[] x = train.batch(order, from, to, norm, rng, labels);
                    EuroSatTraining.StepResult r = EuroSatTraining.trainStep(trainer, x, labels, n, size);
                    if (!Float.isFinite(r.loss())) {
                        throw new IllegalStateException("Non-finite loss at epoch " + epoch + " step " + step);
                    }
                    lossSum += (double) r.loss() * n;
                    correct += r.correct();
                    seen += n;
                    step++;
                    globalStep++;
                    if (step % 100 == 0) {
                        System.out.printf(Locale.ROOT, "  epoch %d step %d/%d  loss %.4f  acc %.4f  lr %.2e  %.0f s%n",
                                epoch, step, stepsPerEpoch, lossSum / seen, correct / (double) seen,
                                schedule.getNewValue(globalStep), (System.nanoTime() - epochStart) / 1e9);
                    }
                }
                double trainSeconds = (System.nanoTime() - epochStart) / 1e9;
                double trainLoss = lossSum / seen;
                double trainAcc = correct / (double) seen;

                // ---- validate ----
                long valStart = System.nanoTime();
                double[] valLoss = new double[1];
                MulticlassMetrics valMetrics = evaluate(trainer, val, valOrder, norm, batchSize, classNames.size(), valLoss);
                double valSeconds = (System.nanoTime() - valStart) / 1e9;
                double valAcc = valMetrics.accuracy();
                double valF1 = valMetrics.macroF1();
                boolean improved = valAcc > bestAcc || (valAcc == bestAcc && valF1 > bestF1);
                epochsCompleted = epoch;
                lastEpochSeconds = (System.nanoTime() - epochStart) / 1e9;

                if (improved) {
                    bestAcc = valAcc;
                    bestF1 = valF1;
                    bestEpoch = epoch;
                    EuroSatTraining.saveCheckpoint(model, outDir, epoch);
                    config.set("training.best_epoch", epoch);
                    config.setMetric("training.best_epoch.val_accuracy", valAcc);
                    config.setMetric("training.best_epoch.val_macro_f1", valF1);
                    config.setMetric("training.best_epoch.val_loss", valLoss[0]);
                    config.setMetric("training.best_epoch.train_accuracy", trainAcc);
                    config.set("training.best_epoch.val_correct", valMetrics.getCorrect());
                    config.set("training.epochs_completed", epochsCompleted);
                    config.set("training.time_seconds", String.format(Locale.ROOT, "%.0f",
                            (System.nanoTime() - trainingStart) / 1e9));
                    config.save(outDir);
                }
                double lrEnd = schedule.getNewValue(globalStep);
                log.printf(Locale.ROOT, "%d,%.6g,%.5f,%.5f,%.5f,%.5f,%.5f,%.1f,%.1f,%.1f,%s%n",
                        epoch, lrEnd, trainLoss, trainAcc, valLoss[0], valAcc, valF1, trainSeconds, valSeconds,
                        lastEpochSeconds, improved ? "yes" : "no");
                log.flush();
                System.out.printf(Locale.ROOT,
                        "Epoch %2d/%d: lr %.2e | train loss %.4f acc %.4f | val loss %.4f acc %.4f macro-F1 %.4f "
                                + "(%d/%d) | %.1f s train + %.1f s val%s%n",
                        epoch, epochs, lrEnd, trainLoss, trainAcc, valLoss[0], valAcc, valF1, valMetrics.getCorrect(),
                        valMetrics.getTotal(), trainSeconds, valSeconds, improved ? "  [checkpoint saved]" : "");
            }
        }

        double trainingSeconds = (System.nanoTime() - trainingStart) / 1e9;
        double totalSeconds = (System.nanoTime() - processStart) / 1e9;
        config.set("training.epochs_completed", epochsCompleted);
        config.set("training.stop_reason", stopReason);
        config.set("training.time_seconds", String.format(Locale.ROOT, "%.0f", trainingSeconds));
        config.set("training.data_loading_seconds", String.format(Locale.ROOT, "%.0f", loadSeconds));
        config.set("training.total_wall_seconds", String.format(Locale.ROOT, "%.0f", totalSeconds));
        config.save(outDir);
        System.out.printf(Locale.ROOT, "Done: %d epochs, training loop %.1f min (total incl. data loading %.1f min); "
                        + "selected epoch %d with val accuracy %.4f, val macro F1 %.4f. Model in %s%n",
                epochsCompleted, trainingSeconds / 60, totalSeconds / 60, bestEpoch, bestAcc, bestF1,
                outDir.toAbsolutePath());
    }

    /** Runs the model (inference mode) over a split; returns the confusion matrix and the mean loss. */
    static MulticlassMetrics evaluate(Trainer trainer, EuroSatImages images, int[] order, Normalization norm,
                                      int batchSize, int numClasses, double[] meanLossOut) {
        MulticlassMetrics metrics = new MulticlassMetrics(numClasses);
        long[] labels = new long[batchSize];
        float[] loss = new float[1];
        double lossSum = 0;
        int size = images.imageSize();
        for (int from = 0; from < order.length; from += batchSize) {
            int to = Math.min(order.length, from + batchSize);
            int n = to - from;
            float[] x = images.batch(order, from, to, norm, null, labels);
            float[] logits = EuroSatTraining.evaluateBatch(trainer, x, labels, n, size, loss);
            lossSum += (double) loss[0] * n;
            for (int i = 0; i < n; i++) {
                metrics.add((int) labels[i], EuroSatTraining.argMax(logits, i, numClasses));
            }
        }
        meanLossOut[0] = lossSum / order.length;
        return metrics;
    }

    private static EuroSatModelConfig baseConfig(int[] widths, int blocks, List<String> classNames, Normalization norm,
                                                 int epochs, int batchSize, float lr, float finalLrFactor,
                                                 float weightDecay, Device device, int seed, Path data, Path splitDir,
                                                 int trainSize, int valSize, int testSize, boolean subset) {
        EuroSatModelConfig c = new EuroSatModelConfig();
        c.set(EuroSatModelConfig.ARCHITECTURE, "EuroSatResNet: ResNet-18-style CNN for 64x64 input; stem Conv3x3(stride 1)"
                + "+BN+ReLU without max-pool; 4 stages of BasicBlocks (y = ReLU(F(x) + shortcut(x)), identity or "
                + "1x1-conv projection shortcut), stride 2 at the start of stages 2-4; global average pooling; Linear head");
        c.setWidths(widths);
        c.set(EuroSatModelConfig.BLOCKS_PER_STAGE, blocks);
        c.setClassNames(classNames);
        c.set(EuroSatModelConfig.INPUT_SIZE, EuroSatResNet.INPUT_SIZE);
        c.setNormalization(norm);
        c.set("normalization.method", "RGB/255, then (x - mean) / std per channel; mean/std over all TRAIN pixels only");
        c.set(EuroSatModelConfig.PARAMS_NAME, EuroSatTraining.MODEL_NAME);
        c.set("output", "logits (N, num_classes); softmax gives class probabilities; prediction = arg-max");
        c.set("checkpoint.selection", "epoch with the highest VAL accuracy (ties: higher VAL macro F1, then earlier epoch)");
        c.set("training.epochs_planned", epochs);
        c.set("training.batch_size", batchSize);
        c.set("training.loss", "softmax cross-entropy (DJL SoftmaxCrossEntropyLoss)");
        c.set("training.optimizer", "Adam (beta1 0.9, beta2 0.999, eps 1e-8)");
        c.set("training.learning_rate", lr);
        c.set("training.lr_schedule", "cosine decay per step from " + lr + " to " + (lr * finalLrFactor)
                + " over the planned epochs");
        c.set("training.weight_decay", weightDecay);
        c.set("training.initialization", "He/Kaiming normal (DJL XavierInitializer GAUSSIAN, IN, magnitude 2) for weights");
        c.set("training.augmentation", "TRAIN only: random D4 transform per image per epoch (rotation by 0/90/180/270 deg, "
                + "optional horizontal flip; covers vertical flips)");
        c.set("training.device", device.toString());
        c.set("training.seed", seed);
        c.set("djl_version", DjlRuntime.djlVersion());
        c.set("engine", DjlRuntime.engineName());
        c.set("engine_version", DjlRuntime.engineVersion());
        c.set("java_version", System.getProperty("java.version"));
        c.set("data.dataset", "EuroSAT RGB (Helber et al., IEEE JSTARS 2019; Zenodo 7711810, MIT licence): "
                + "Sentinel-2 RGB, 64x64 px at 10 m GSD, 10 classes");
        c.set("data.images", "<EuroSAT folder>/EuroSAT_RGB/<Class>/<Class>_<n>.jpg (folder used: " + data.getFileName() + ")");
        c.set("data.splits", splitDir.toString() + " (stratified 70/15/15, see dataset_summary.txt)");
        c.set("data.train_images", trainSize);
        c.set("data.val_images", valSize);
        c.set("data.test_images", testSize);
        if (subset) {
            c.set("data.subset_warning", "trained on a random SUBSET (--train-limit/--val-limit): smoke test only");
        }
        return c;
    }

    /** Fisher-Yates shuffle with the given generator (deterministic for a fixed seed). */
    static void shuffle(int[] a, Random rng) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    private static List<EuroSatSample> subset(List<EuroSatSample> all, int limit, long seed) {
        if (limit <= 0 || limit >= all.size()) {
            return all;
        }
        List<EuroSatSample> copy = new ArrayList<>(all);
        Collections.shuffle(copy, new Random(seed));
        return List.copyOf(copy.subList(0, limit));
    }

    static int[] parseInts(String s) {
        String[] parts = s.split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Integer.parseInt(parts[i].trim());
        }
        return out;
    }
}
