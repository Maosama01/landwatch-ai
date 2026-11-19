package com.landwatch.deeplearning.eurosat;

import com.landwatch.deeplearning.DjlRuntime;
import com.landwatch.deeplearning.ImageTensors;
import com.landwatch.evaluation.MulticlassMetrics;

import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Evaluates a trained EuroSAT classifier on one split through the public {@link EuroSatClassifier}
 * API (JPEG decoding, normalisation and the network exactly as any caller would use them).
 *
 * <p>Usage: {@code EvaluateEuroSatClassifier <val|test> [--model models/eurosat_resnet]
 * [--data <EuroSAT folder containing EuroSAT_RGB/>] [--splits <data>/splits]
 * [--results results/eurosat] [--device auto|cpu|mps] [--force false]}</p>
 *
 * <p>Writes {@code <split>_metrics.txt} (overall accuracy, macro F1, per-class precision / recall /
 * F1, confusion matrix), {@code confusion_matrix_<split>.csv} and {@code <split>_predictions.csv}
 * (one line per image), and records the headline numbers in the model's {@code model.properties}.
 * The TEST split is meant to be evaluated exactly once, with the final selected model: if
 * {@code test_metrics.txt} already exists the program refuses to run on TEST unless
 * {@code --force true} is given.</p>
 */
public final class EvaluateEuroSatClassifier {

    private static final int BATCH = 64;

    private EvaluateEuroSatClassifier() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1 || args[0].startsWith("--")) {
            System.err.println("Usage: EvaluateEuroSatClassifier <val|test> [--model dir] [--data dir] [--splits dir]"
                    + " [--results dir] [--device auto|cpu|mps] [--force true]");
            System.exit(2);
        }
        String split = args[0];
        if (!split.equals("val") && !split.equals("test") && !split.equals("train")) {
            throw new IllegalArgumentException("Split must be val, test or train: " + split);
        }
        Map<String, String> opt = EuroSatSplit.parseArgs(Arrays.copyOfRange(args, 1, args.length));
        Path modelDir = Path.of(opt.getOrDefault("model", "models/eurosat_resnet"));
        Path data = opt.containsKey("data") ? Path.of(opt.get("data")) : EuroSatSplit.locateDatasetRoot();
        if (data == null) {
            throw new IllegalArgumentException("EuroSAT not found; pass --data <folder containing EuroSAT_RGB/>");
        }
        Path splitDir = Path.of(opt.getOrDefault("splits", data.resolve("splits").toString()));
        Path resultsDir = Path.of(opt.getOrDefault("results", "results/eurosat"));
        boolean force = Boolean.parseBoolean(opt.getOrDefault("force", "false"));
        Path report = resultsDir.resolve(split + "_metrics.txt");
        if ("test".equals(split) && Files.exists(report) && !force) {
            System.err.println("Refusing to re-evaluate TEST: " + report + " already exists (use --force true deliberately).");
            System.exit(3);
        }

        EuroSatSplit.Split splits = EuroSatSplit.read(splitDir);
        List<EuroSatSample> samples = splits.get(split);
        List<String> names = splits.classNames();

        MulticlassMetrics metrics = new MulticlassMetrics(names.size());
        List<String> predictionRows = new ArrayList<>(samples.size());
        int resized = 0;
        String device;
        double inferenceSeconds = 0;
        long start = System.nanoTime();
        try (EuroSatClassifier classifier = EuroSatClassifier.load(modelDir, opt.getOrDefault("device", "auto"))) {
            if (!classifier.getClassNames().equals(names)) {
                throw new IllegalStateException("Model classes " + classifier.getClassNames()
                        + " differ from split classes " + names);
            }
            device = classifier.getDevice().toString();
            System.out.println("Evaluating " + split + " (" + samples.size() + " images) on " + device);
            for (int first = 0; first < samples.size(); first += BATCH) {
                int n = Math.min(BATCH, samples.size() - first);
                List<BufferedImage> images = new ArrayList<>(n);
                for (int i = 0; i < n; i++) {
                    images.add(ImageTensors.read(samples.get(first + i).resolve(data)));
                }
                long t0 = System.nanoTime();
                List<EuroSatClassifier.Prediction> predictions = classifier.predict(images);
                inferenceSeconds += (System.nanoTime() - t0) / 1e9;
                for (int i = 0; i < n; i++) {
                    EuroSatSample s = samples.get(first + i);
                    EuroSatClassifier.Prediction p = predictions.get(i);
                    metrics.add(s.classIndex(), p.classIndex());
                    if (p.wasResized()) {
                        resized++;
                    }
                    predictionRows.add(String.format(Locale.ROOT, "%s,%s,%d,%s,%d,%.4f,%s", s.relativePath(),
                            s.className(), s.classIndex(), p.className(), p.classIndex(), p.probability(),
                            s.classIndex() == p.classIndex() ? "yes" : "no"));
                }
                if ((first / BATCH + 1) % 20 == 0) {
                    System.out.printf(Locale.ROOT, "  %d/%d  running accuracy %.4f%n", first + n, samples.size(),
                            metrics.accuracy());
                }
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        EuroSatModelConfig config = EuroSatModelConfig.load(modelDir);

        StringBuilder sb = new StringBuilder();
        sb.append("LandWatch AI - EuroSAT land-use / land-cover classification (ResNet), evaluation on the ")
                .append(split.toUpperCase(Locale.ROOT)).append(" split\n");
        sb.append("Generated: ").append(LocalDateTime.now()).append('\n');
        sb.append("Model: ").append(modelDir).append(" (EuroSatResNet, widths ").append(config.get(EuroSatModelConfig.WIDTHS))
                .append(", ").append(config.get("model.trainable_parameters", "?")).append(" trainable parameters; selected epoch ")
                .append(config.get("training.best_epoch", "?")).append(" of ").append(config.get("training.epochs_completed", "?"))
                .append(" by VAL accuracy)\n");
        sb.append("Split: ").append(splitDir.resolve(split + ".csv")).append(" - ").append(samples.size())
                .append(" images (stratified 70/15/15 split of EuroSAT RGB; see dataset_summary.txt)\n");
        if ("test".equals(split)) {
            sb.append("Protocol: TEST evaluated once, with the final model selected on VAL; TEST was not used for training or selection.\n");
        }
        sb.append("Inference device: ").append(device).append(", ").append(DjlRuntime.describe())
                .append(" (training device: ").append(config.get("training.device", "?")).append(")\n");
        sb.append("Data: Sentinel-2 RGB tiles, 64x64 px at 10 m GSD. Task: scene classification into ").append(names.size())
                .append(" classes (not change detection).\n");
        sb.append("Metrics: com.landwatch.evaluation.MulticlassMetrics (rows = actual, columns = predicted)\n\n");
        sb.append(metrics.toReport(names));
        sb.append("\nMost frequent confusions (actual -> predicted: count):\n");
        for (long[] c : metrics.topConfusions(8)) {
            sb.append(String.format(Locale.ROOT, "  %s -> %s: %d%n", names.get((int) c[0]), names.get((int) c[1]), c[2]));
        }
        if ("val".equals(split)) {
            String trainingAcc = config.get("training.best_epoch.val_accuracy", null);
            if (trainingAcc != null) {
                sb.append(String.format(Locale.ROOT,
                        "%nConsistency check: VAL accuracy recorded during training for the selected epoch (on %s) = %s "
                                + "(%s correct); recomputed here from the saved weights through the public API on %s = %.4f "
                                + "(%d correct).%n",
                        config.get("training.device", "?"), trainingAcc, config.get("training.best_epoch.val_correct", "?"),
                        device, metrics.accuracy(), metrics.getCorrect()));
            }
        }
        if (resized > 0) {
            sb.append(resized).append(" images were not 64x64 and were resized.\n");
        }
        sb.append(String.format(Locale.ROOT, "%nInference: %.1f s for %d images in batches of %d on %s (%.2f ms/image); "
                        + "total wall time incl. JPEG decoding %.1f s%n",
                inferenceSeconds, samples.size(), BATCH, device, 1000 * inferenceSeconds / samples.size(), seconds));

        Files.createDirectories(resultsDir);
        Files.writeString(report, sb.toString(), StandardCharsets.UTF_8);
        Files.writeString(resultsDir.resolve("confusion_matrix_" + split + ".csv"), metrics.toConfusionMatrixCsv(names),
                StandardCharsets.UTF_8);
        try (BufferedWriter w = Files.newBufferedWriter(resultsDir.resolve(split + "_predictions.csv"), StandardCharsets.UTF_8)) {
            w.write("path,actual_class,actual_index,predicted_class,predicted_index,predicted_probability,correct\n");
            for (String row : predictionRows) {
                w.write(row);
                w.write('\n');
            }
        }
        System.out.println(sb);
        System.out.println("Wrote " + report.toAbsolutePath());

        config.setMetric("metrics." + split + ".accuracy", metrics.accuracy());
        config.setMetric("metrics." + split + ".macro_f1", metrics.macroF1());
        config.setMetric("metrics." + split + ".macro_precision", metrics.macroPrecision());
        config.setMetric("metrics." + split + ".macro_recall", metrics.macroRecall());
        config.set("metrics." + split + ".images", metrics.getTotal());
        config.set("metrics." + split + ".correct", metrics.getCorrect());
        for (int c = 0; c < names.size(); c++) {
            config.setMetric("metrics." + split + ".f1." + names.get(c), metrics.f1(c));
        }
        config.set("metrics." + split + ".device", device);
        config.set("metrics." + split + ".source", "results/eurosat/" + split + "_metrics.txt");
        config.save(modelDir);
        System.out.println("Updated " + modelDir.resolve(EuroSatModelConfig.FILE_NAME));
    }
}
