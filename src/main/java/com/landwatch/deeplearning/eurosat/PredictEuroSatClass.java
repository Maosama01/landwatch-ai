package com.landwatch.deeplearning.eurosat;

import com.landwatch.deeplearning.ImageTensors;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Command-line land-use / land-cover classification of one or more image files with
 * {@link EuroSatClassifier}.
 *
 * <p>Usage: {@code PredictEuroSatClass <image>... [--model models/eurosat_resnet]
 * [--device auto|cpu|mps] [--top 3]}</p>
 *
 * <p>Prints the predicted class, its probability and the next most likely classes for every
 * image. The model expects 64 x 64 Sentinel-2 RGB tiles at 10 m; other sizes are resized and
 * flagged with a note.</p>
 */
public final class PredictEuroSatClass {

    private PredictEuroSatClass() {
    }

    public static void main(String[] args) throws Exception {
        List<String> files = new ArrayList<>();
        List<String> options = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--") && i + 1 < args.length) {
                options.add(args[i]);
                options.add(args[++i]);
            } else {
                files.add(args[i]);
            }
        }
        if (files.isEmpty()) {
            System.err.println("Usage: PredictEuroSatClass <image>... [--model dir] [--device auto|cpu|mps] [--top 3]");
            System.exit(2);
        }
        Map<String, String> opt = EuroSatSplit.parseArgs(options.toArray(new String[0]));
        Path modelDir = Path.of(opt.getOrDefault("model", "models/eurosat_resnet"));
        int top = Integer.parseInt(opt.getOrDefault("top", "3"));
        try (EuroSatClassifier classifier = EuroSatClassifier.load(modelDir, opt.getOrDefault("device", "auto"))) {
            System.out.println(classifier.describe());
            System.out.println();
            List<String> names = classifier.getClassNames();
            for (String f : files) {
                BufferedImage img = ImageTensors.read(Path.of(f));
                long t0 = System.nanoTime();
                EuroSatClassifier.Prediction p = classifier.predict(img);
                double ms = (System.nanoTime() - t0) / 1e6;
                StringBuilder sb = new StringBuilder();
                sb.append(String.format(Locale.ROOT, "%s: %s (p=%.4f)", f, p.className(), p.probability()));
                int[] idx = p.topIndices(top);
                sb.append("  top-").append(idx.length).append(':');
                float[] probs = p.probabilities();
                for (int c : idx) {
                    sb.append(String.format(Locale.ROOT, " %s %.4f;", names.get(c), probs[c]));
                }
                sb.append(String.format(Locale.ROOT, "  [%.1f ms on %s]", ms, classifier.getDevice()));
                System.out.println(sb);
                if (p.wasResized()) {
                    System.out.println("  note: " + p.note());
                }
            }
        }
    }
}
