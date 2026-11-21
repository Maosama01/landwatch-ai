package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.tracker.Tracker;
import com.landwatch.evaluation.BinaryChangeMetrics;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Trains FC-Siam-diff on LEVIR-CD with DJL.
 *
 * <p>Protocol (the three splits are kept strictly apart):</p>
 * <ol>
 *   <li>Normalisation constants are computed on TRAIN images only.</li>
 *   <li>The network is trained on TRAIN patches with random flips / 90-degree rotations, Adam and a
 *       cosine learning-rate decay, using weighted BCE + Dice loss ({@link BceDiceLoss}).</li>
 *   <li>After every epoch the model is run on the VAL split; precision / recall / F1 / IoU are
 *       logged at threshold 0.5 and at the best threshold of a sweep. The checkpoint with the best
 *       val F1 (at its tuned threshold) is saved together with that threshold.</li>
 *   <li>The TEST split is not touched here; evaluate it once with {@link EvaluateChangeDetector}.</li>
 * </ol>
 *
 * <p>Options (all optional): {@code --data <processed_256 dir> --out models/siamese_unet_levir
 * --results results/deeplearning --epochs 14 --batch 8 --lr 1e-3 --device auto --workers 4
 * --max-minutes 72 --train-limit 0 --val-limit 0 --seed 42 --pos-weight auto --dice-weight 1
 * --augment true --arch fc-siam-diff --tf-layers 2 --tf-heads 4 --tf-mlp 256}</p>
 *
 * <p>{@code --arch fc-siam-diff-transformer} trains FC-Siam-diff with a {@link BitemporalTransformer}
 * at the bottleneck (layers, heads and MLP size from the {@code --tf-*} options).</p>
 *
 * <p>For ablation runs, {@code --dice-weight 0} trains with weighted BCE alone and
 * {@code --augment false} switches off the random D4 augmentation. Ablations are compared on the
 * VAL split only.</p>
 */
public final class TrainChangeDetector {

    private static final double SWEEP_FROM = 0.05;
    private static final double SWEEP_TO = 0.95;
    private static final double SWEEP_STEP = 0.05;

    private TrainChangeDetector() {
    }

    public static void main(String[] args) throws Exception {
        Map<String, String> opt = parseArgs(args);
        Path data = opt.containsKey("data") ? Path.of(opt.get("data")) : LevirCdPatch.locateDatasetRoot();
        if (data == null) {
            throw new IllegalArgumentException("Dataset not found; pass --data <.../LEVIR-CD/processed_256>");
        }
        Path outDir = Path.of(opt.getOrDefault("out", "models/siamese_unet_levir"));
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/deeplearning"));
        int epochs = Integer.parseInt(opt.getOrDefault("epochs", "14"));
        int batchSize = Integer.parseInt(opt.getOrDefault("batch", "8"));
        float lr = Float.parseFloat(opt.getOrDefault("lr", "1e-3"));
        int workers = Integer.parseInt(opt.getOrDefault("workers", "4"));
        double maxMinutes = Double.parseDouble(opt.getOrDefault("max-minutes", "72"));
        int trainLimit = Integer.parseInt(opt.getOrDefault("train-limit", "0"));
        int valLimit = Integer.parseInt(opt.getOrDefault("val-limit", "0"));
        int seed = Integer.parseInt(opt.getOrDefault("seed", "42"));
        float diceWeight = Float.parseFloat(opt.getOrDefault("dice-weight", "1"));
        boolean augment = parseBoolean(opt.getOrDefault("augment", "true"), "augment");
        String arch = opt.getOrDefault("arch", ModelConfig.TYPE_FC_SIAM_DIFF);
        if (!ModelConfig.TYPE_FC_SIAM_DIFF.equals(arch) && !ModelConfig.TYPE_TRANSFORMER.equals(arch)) {
            throw new IllegalArgumentException("--arch must be " + ModelConfig.TYPE_FC_SIAM_DIFF + " or "
                    + ModelConfig.TYPE_TRANSFORMER + ", got " + arch);
        }
        int tfLayers = Integer.parseInt(opt.getOrDefault("tf-layers", String.valueOf(BitemporalTransformer.DEFAULT_LAYERS)));
        int tfHeads = Integer.parseInt(opt.getOrDefault("tf-heads", String.valueOf(BitemporalTransformer.DEFAULT_HEADS)));
        int tfMlp = Integer.parseInt(opt.getOrDefault("tf-mlp", String.valueOf(BitemporalTransformer.DEFAULT_MLP_HIDDEN)));
        int[] widths = FcSiamDiff.DEFAULT_WIDTHS;
        int patchSize = 256;
        Device device = DjlRuntime.resolveDevice(opt.getOrDefault("device", "auto"));

        Engine.getInstance().setRandomSeed(seed);
        String archName = ModelConfig.TYPE_TRANSFORMER.equals(arch) ? "FC-Siam-diff + Transformer" : "FC-Siam-diff";
        System.out.println("=== LandWatch AI - " + archName + " training (" + LocalDateTime.now() + ") ===");
        System.out.println(DjlRuntime.describe() + ", device " + device);
        System.out.println("Dataset: " + data);

        List<LevirCdPatch> train = subset(LevirCdPatch.readSplit(data, "train"), trainLimit, seed);
        List<LevirCdPatch> val = subset(LevirCdPatch.readSplit(data, "val"), valLimit, seed + 1);
        int testSize = LevirCdPatch.readSplit(data, "test").size(); // size only; test images are not read
        System.out.printf("Patches: train %d, val %d (test %d held out)%n", train.size(), val.size(), testSize);

        long changePx = 0;
        long allPx = 0;
        for (LevirCdPatch p : train) {
            changePx += p.changePixelCount();
            allPx += p.pixelCount();
        }
        double changeFraction = changePx / (double) allPx;
        float posWeight = "auto".equals(opt.getOrDefault("pos-weight", "auto"))
                ? (float) Math.sqrt((1 - changeFraction) / changeFraction)
                : Float.parseFloat(opt.get("pos-weight"));
        System.out.printf(Locale.ROOT, "Train change-pixel fraction %.5f -> BCE pos_weight %.4f (sqrt of neg/pos ratio)%n",
                changeFraction, posWeight);

        long tNorm = System.nanoTime();
        Normalization norm = Normalization.computeFromTrain(train, workers);
        System.out.printf(Locale.ROOT, "Normalisation (train only, A+B pooled): %s  [%.1f s]%n", norm, (System.nanoTime() - tNorm) / 1e9);

        LevirCdDataset trainSet = LevirCdDataset.builder().setPatches(train).optNormalization(norm)
                .optAugment(augment).setSampling(batchSize, true, true).optPrefetchNumber(workers * 2).build();
        LevirCdDataset valSet = LevirCdDataset.builder().setPatches(val).optNormalization(norm)
                .optAugment(false).setSampling(batchSize, false, false).optPrefetchNumber(workers * 2).build();

        int stepsPerEpoch = train.size() / batchSize;
        Tracker schedule = Tracker.cosine().setBaseValue(lr).optFinalValue(lr * 0.01f)
                .setMaxUpdates(Math.max(1, epochs * stepsPerEpoch)).build();
        BceDiceLoss loss = new BceDiceLoss(posWeight, diceWeight);

        Files.createDirectories(resultsDir);
        Files.createDirectories(outDir);
        Path logCsv = resultsDir.resolve("training_log.csv");
        ExecutorService loaders = Executors.newFixedThreadPool(workers);
        long trainingStart = System.nanoTime();
        double bestF1 = -1;
        int bestEpoch = -1;
        int epochsCompleted = 0;
        String stopReason = "planned epochs completed";
        ModelConfig config = baseConfig(widths, patchSize, norm, epochs, batchSize, lr, posWeight, diceWeight,
                augment, device, train.size(), val.size(), testSize, data, seed);
        if (ModelConfig.TYPE_TRANSFORMER.equals(arch)) {
            config.set(ModelConfig.MODEL_TYPE, arch);
            config.set(ModelConfig.TF_LAYERS, tfLayers);
            config.set(ModelConfig.TF_HEADS, tfHeads);
            config.set(ModelConfig.TF_MLP, tfMlp);
            config.set(ModelConfig.PARAMS_NAME, TrainingSupport.TRANSFORMER_MODEL_NAME);
            config.set(ModelConfig.ARCHITECTURE, "FC-Siam-diff (Daudt, Le Saux & Boulch, ICIP 2018) with a Transformer"
                    + " encoder at the bottleneck (" + tfLayers + " pre-norm layers, " + tfHeads + " heads, MLP " + tfMlp
                    + ", fixed 2-D sinusoidal position code) over the tokens of both dates, after BIT (Chen, Qi & Shi, TGRS 2022);"
                    + " decoder skips = |enc(A) - enc(B)|");
        }

        try (Model model = TrainingSupport.newModel(config, device);
             Trainer trainer = TrainingSupport.newTrainer(model, loss, schedule, batchSize, patchSize);
             NDManager cpu = NDManager.newBaseManager(Device.cpu());
             PrintWriter log = new PrintWriter(Files.newBufferedWriter(logCsv, StandardCharsets.UTF_8))) {
            long params = TrainingSupport.countTrainableParameters(model.getBlock());
            config.set("model.trainable_parameters", params);
            System.out.printf("Model %s, %,d trainable parameters; %d steps/epoch, batch %d%n",
                    model.getBlock(), params, stepsPerEpoch, batchSize);
            log.println("epoch,learning_rate_end,train_loss,train_seconds,val_seconds,"
                    + "val_precision@0.50,val_recall@0.50,val_f1@0.50,val_iou@0.50,"
                    + "val_best_threshold,val_precision@best,val_recall@best,val_f1@best,val_iou@best,checkpoint_saved");
            log.flush();

            double lastEpochSeconds = 0;
            int globalStep = 0;
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
                double lossSum = 0;
                int steps = 0;
                for (Batch batch : trainSet.getData(cpu, loaders)) {
                    try (NDManager step = trainer.getManager().newSubManager()) {
                        NDList x = TrainingSupport.copyTo(batch.getData(), step);
                        NDList y = TrainingSupport.copyTo(batch.getLabels(), step);
                        float l = TrainingSupport.trainStep(trainer, x, y);
                        if (!Float.isFinite(l)) {
                            throw new IllegalStateException("Non-finite loss at epoch " + epoch + " step " + steps);
                        }
                        lossSum += l;
                    } finally {
                        batch.close();
                    }
                    steps++;
                    globalStep++;
                    if (steps % 100 == 0) {
                        System.out.printf(Locale.ROOT, "  epoch %d step %d/%d  mean loss %.4f  lr %.2e  %.0f s%n",
                                epoch, steps, stepsPerEpoch, lossSum / steps, schedule.getNewValue(globalStep),
                                (System.nanoTime() - epochStart) / 1e9);
                    }
                }
                double trainSeconds = (System.nanoTime() - epochStart) / 1e9;

                // ---- validate ----
                long valStart = System.nanoTime();
                ProbabilityHistogram hist = validate(trainer, valSet, cpu, loaders);
                double valSeconds = (System.nanoTime() - valStart) / 1e9;
                BinaryChangeMetrics at05 = hist.metricsAt(0.5);
                double bestT = hist.bestF1Threshold(SWEEP_FROM, SWEEP_TO, SWEEP_STEP);
                BinaryChangeMetrics atBest = hist.metricsAt(bestT);
                boolean improved = atBest.f1() > bestF1;
                epochsCompleted = epoch;
                lastEpochSeconds = (System.nanoTime() - epochStart) / 1e9;

                if (improved) {
                    bestF1 = atBest.f1();
                    bestEpoch = epoch;
                    saveCheckpoint(model, outDir, epoch);
                    config.set(ModelConfig.THRESHOLD, String.format(Locale.ROOT, "%.3f", bestT));
                    config.set("training.best_epoch", epoch);
                    putMetrics(config, "metrics.val", atBest);
                    config.set("metrics.val.f1_at_0.50", fmt(at05.f1()));
                    config.set("training.epochs_completed", epochsCompleted);
                    config.set("training.time_seconds", String.format(Locale.ROOT, "%.0f", (System.nanoTime() - trainingStart) / 1e9));
                    config.save(outDir);
                    writeSweep(resultsDir.resolve("val_threshold_sweep.csv"), hist, epoch);
                }
                log.printf(Locale.ROOT, "%d,%.6g,%.5f,%.1f,%.1f,%.4f,%.4f,%.4f,%.4f,%.2f,%.4f,%.4f,%.4f,%.4f,%s%n",
                        epoch, schedule.getNewValue(globalStep), lossSum / Math.max(1, steps), trainSeconds, valSeconds,
                        at05.precision(), at05.recall(), at05.f1(), at05.iou(),
                        bestT, atBest.precision(), atBest.recall(), atBest.f1(), atBest.iou(), improved ? "yes" : "no");
                log.flush();
                System.out.printf(Locale.ROOT,
                        "Epoch %d/%d: train loss %.4f | val@0.50 P %.4f R %.4f F1 %.4f IoU %.4f | val@%.2f F1 %.4f IoU %.4f | %.0f s train + %.0f s val%s%n",
                        epoch, epochs, lossSum / Math.max(1, steps), at05.precision(), at05.recall(), at05.f1(), at05.iou(),
                        bestT, atBest.f1(), atBest.iou(), trainSeconds, valSeconds, improved ? "  [checkpoint saved]" : "");
            }
        } finally {
            loaders.shutdownNow();
        }

        double totalSeconds = (System.nanoTime() - trainingStart) / 1e9;
        config.set("training.epochs_completed", epochsCompleted);
        config.set("training.stop_reason", stopReason);
        config.set("training.time_seconds", String.format(Locale.ROOT, "%.0f", totalSeconds));
        config.save(outDir);
        System.out.printf(Locale.ROOT, "Done: %d epochs in %.1f min; best val F1 %.4f at epoch %d (threshold %s). Model in %s%n",
                epochsCompleted, totalSeconds / 60, bestF1, bestEpoch, config.get(ModelConfig.THRESHOLD, "?"), TrainingSupport.displayPath(outDir));
    }

    /**
     * Saves the weights as {@code fc_siam_diff-<epoch>.params}, removing older checkpoint files so
     * the directory always holds exactly one set of weights (DJL loads the highest epoch number).
     */
    static void saveCheckpoint(Model model, Path outDir, int epoch) throws IOException {
        try (java.util.stream.Stream<Path> files = Files.list(outDir)) {
            for (Path f : (Iterable<Path>) files::iterator) {
                if (f.getFileName().toString().matches(java.util.regex.Pattern.quote(model.getName())
                        + "-\\d{4}\\.params")) {
                    Files.delete(f);
                }
            }
        }
        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(outDir, model.getName());
    }

    /** Runs the model over a (non-augmented) split and returns the probability histogram. */
    static ProbabilityHistogram validate(Trainer trainer, LevirCdDataset set, NDManager cpu, ExecutorService loaders)
            throws Exception {
        ProbabilityHistogram hist = new ProbabilityHistogram();
        for (Batch batch : set.getData(cpu, loaders)) {
            try (NDManager step = trainer.getManager().newSubManager()) {
                NDList x = TrainingSupport.copyTo(batch.getData(), step);
                float[] probs = TrainingSupport.predictProbabilities(trainer, x);
                float[] mask = batch.getLabels().head().toFloatArray();
                hist.add(probs, mask);
            } finally {
                batch.close();
            }
        }
        return hist;
    }

    private static ModelConfig baseConfig(int[] widths, int patchSize, Normalization norm, int epochs, int batchSize,
                                          float lr, float posWeight, float diceWeight, boolean augment, Device device,
                                          int trainSize, int valSize, int testSize, Path data, int seed) {
        ModelConfig c = new ModelConfig();
        c.set(ModelConfig.ARCHITECTURE, "FC-Siam-diff (Daudt, Le Saux & Boulch, ICIP 2018); shared encoder, "
                + "decoder skips = |enc(A) - enc(B)|; Conv3x3+BatchNorm+ReLU, 2x2 max-pool, nearest 2x upsample + Conv3x3");
        c.setWidths(widths);
        c.set(ModelConfig.PATCH_SIZE, patchSize);
        c.setNormalization(norm);
        c.set("normalization.method", "x/255 then (x - mean) / std per RGB channel; mean/std from TRAIN A+B images only");
        c.set(ModelConfig.PARAMS_NAME, TrainingSupport.MODEL_NAME);
        c.set(ModelConfig.THRESHOLD, "0.5");
        c.set("decision_threshold.selection", String.format(Locale.ROOT,
                "max change-F1 on VAL over thresholds %.2f..%.2f step %.2f, for the selected checkpoint", SWEEP_FROM, SWEEP_TO, SWEEP_STEP));
        c.set("checkpoint.selection", "epoch with the highest VAL change-F1 (at its tuned threshold)");
        c.set("training.epochs_planned", epochs);
        c.set("training.batch_size", batchSize);
        c.set("training.optimizer", "Adam");
        c.set("training.learning_rate", lr);
        c.set("training.lr_schedule", "cosine decay from " + lr + " to " + (lr * 0.01f) + " over the planned epochs");
        c.set("training.loss", "weighted BCE-with-logits + " + diceWeight + " x soft Dice (batch-level)");
        c.set("training.loss.pos_weight", posWeight);
        c.set("training.loss.dice_weight", diceWeight);
        c.set("training.augmentation", augment
                ? "TRAIN only: random D4 transform (rotation by 0/90/180/270 deg, optional horizontal flip;"
                        + " covers vertical flips), identical for A, B and label"
                : "none (switched off with --augment false)");
        c.set("training.device", device.toString());
        c.set("training.seed", seed);
        c.set("djl_version", DjlRuntime.djlVersion());
        c.set("engine", DjlRuntime.engineName());
        c.set("engine_version", DjlRuntime.engineVersion());
        c.set("java_version", System.getProperty("java.version"));
        c.set("data.dataset", "LEVIR-CD, 256x256 non-overlapping patches (" + data.getFileName() + ")");
        c.set("data.train_patches", trainSize);
        c.set("data.val_patches", valSize);
        c.set("data.test_patches", testSize);
        c.set("data.label_rule", "label value > 0 = change");
        return c;
    }

    static void putMetrics(ModelConfig c, String prefix, BinaryChangeMetrics m) {
        c.set(prefix + ".precision", fmt(m.precision()));
        c.set(prefix + ".recall", fmt(m.recall()));
        c.set(prefix + ".f1", fmt(m.f1()));
        c.set(prefix + ".iou", fmt(m.iou()));
        c.set(prefix + ".kappa", fmt(m.kappa()));
        c.set(prefix + ".accuracy", fmt(m.accuracy()));
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.4f", v);
    }

    private static void writeSweep(Path file, ProbabilityHistogram hist, int epoch) throws IOException {
        try (BufferedWriter w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            w.write("# VAL threshold sweep for the checkpoint of epoch " + epoch + "\n");
            w.write("threshold,precision,recall,f1,iou\n");
            int n = (int) Math.round((SWEEP_TO - SWEEP_FROM) / SWEEP_STEP);
            for (int i = 0; i <= n; i++) {
                double t = SWEEP_FROM + i * SWEEP_STEP;
                BinaryChangeMetrics m = hist.metricsAt(t);
                w.write(String.format(Locale.ROOT, "%.2f,%.4f,%.4f,%.4f,%.4f%n", t, m.precision(), m.recall(), m.f1(), m.iou()));
            }
        }
    }

    private static List<LevirCdPatch> subset(List<LevirCdPatch> all, int limit, long seed) {
        if (limit <= 0 || limit >= all.size()) {
            return all;
        }
        List<LevirCdPatch> copy = new ArrayList<>(all);
        Collections.shuffle(copy, new Random(seed));
        return List.copyOf(copy.subList(0, limit));
    }

    /** Parses "true" or "false" (any case); anything else is an error rather than a silent false. */
    static boolean parseBoolean(String value, String option) {
        if ("true".equalsIgnoreCase(value)) {
            return true;
        }
        if ("false".equalsIgnoreCase(value)) {
            return false;
        }
        throw new IllegalArgumentException("--" + option + " must be true or false, got: " + value);
    }

    static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i++) {
            if (!args[i].startsWith("--") || i + 1 >= args.length) {
                throw new IllegalArgumentException("Expected --key value pairs, got: " + String.join(" ", args));
            }
            map.put(args[i].substring(2), args[++i]);
        }
        return map;
    }
}
