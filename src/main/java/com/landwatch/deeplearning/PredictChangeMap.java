package com.landwatch.deeplearning;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;

/**
 * Command-line inference on one before/after image pair of any size, using {@link DeepChangeDetector}.
 *
 * <p>Usage: {@code PredictChangeMap <before.png> <after.png> <mask-out.png> [--model models/siamese_unet_levir]
 * [--device auto|cpu|mps] [--probability-out prob.png]}</p>
 *
 * <p>Writes the binary change mask (white = change) and optionally an 8-bit probability map
 * (brightness = probability of change).</p>
 */
public final class PredictChangeMap {

    private PredictChangeMap() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: PredictChangeMap <before> <after> <mask-out.png> [--model dir] [--device auto|cpu|mps]"
                    + " [--probability-out file.png]");
            System.exit(2);
        }
        Map<String, String> opt = TrainChangeDetector.parseArgs(Arrays.copyOfRange(args, 3, args.length));
        BufferedImage before = ImageTensors.read(Path.of(args[0]));
        BufferedImage after = ImageTensors.read(Path.of(args[1]));
        Path modelDir = Path.of(opt.getOrDefault("model", "models/siamese_unet_levir"));
        try (DeepChangeDetector detector = DeepChangeDetector.load(modelDir, opt.getOrDefault("device", "auto"))) {
            long t0 = System.nanoTime();
            float[][] prob = detector.predictProbabilities(before, after);
            double ms = (System.nanoTime() - t0) / 1e6;
            boolean[][] mask = DeepChangeDetector.applyThreshold(prob, detector.getThreshold());
            int h = prob.length;
            int w = prob[0].length;
            BufferedImage maskImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            BufferedImage probImg = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            long changed = 0;
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    maskImg.getRaster().setSample(x, y, 0, mask[y][x] ? 255 : 0);
                    probImg.getRaster().setSample(x, y, 0, Math.round(prob[y][x] * 255f));
                    if (mask[y][x]) {
                        changed++;
                    }
                }
            }
            ImageIO.write(maskImg, "png", Path.of(args[2]).toFile());
            if (opt.containsKey("probability-out")) {
                ImageIO.write(probImg, "png", Path.of(opt.get("probability-out")).toFile());
            }
            System.out.println(detector.describe());
            System.out.printf(Locale.ROOT, "%nImage %dx%d, %d tile(s) of %d px, inference %.0f ms on %s; "
                            + "changed pixels %d (%.2f%%) at threshold %.3f -> %s%n",
                    w, h, ((w + detector.getTileSize() - 1) / detector.getTileSize())
                            * ((h + detector.getTileSize() - 1) / detector.getTileSize()),
                    detector.getTileSize(), ms, detector.getDevice(), changed, 100.0 * changed / ((double) w * h),
                    detector.getThreshold(), args[2]);
        }
    }
}
