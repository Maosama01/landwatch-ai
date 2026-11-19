package com.landwatch.ui;

import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.geospatial.ChangeVisualizer;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Per-image comparison of an engine's change mask with a ground-truth label.
 *
 * <p>The confusion matrix and every metric (F1, IoU, precision, recall, ...) come from
 * {@link BinaryChangeMetrics}, exactly as in the dataset benchmarks, but here they describe ONE image
 * pair only. Two images are rendered for the dashboard: the label as a black-and-white mask and an
 * error map.</p>
 *
 * <p>Error-map colours are three hues of the Okabe-Ito colour-universal-design palette, chosen so that
 * they stay distinguishable with the common forms of colour-vision deficiency (and differ in brightness
 * too): TP yellow, FP vermillion, FN sky blue. Correct no-change pixels (TN) show the post-event image as
 * dimmed grayscale context.</p>
 */
public final class GroundTruthComparison {

    /** Error map: change predicted and present in the label (true positive), Okabe-Ito yellow. */
    public static final int TP_RGB = 0xF0E442;
    /** Error map: change predicted but not in the label (false positive / false alarm), Okabe-Ito vermillion. */
    public static final int FP_RGB = 0xD55E00;
    /** Error map: change in the label but not predicted (false negative / missed), Okabe-Ito sky blue. */
    public static final int FN_RGB = 0x56B4E9;
    /** Error map: brightness factor applied to the grayscale post-event image for TN pixels. */
    public static final double TN_CONTEXT_DIM = 0.35;
    /** Representative swatch colour for TN pixels in the legend (mid-gray after dimming). */
    public static final int TN_LEGEND_RGB = 0x2D2D2D;

    private final BinaryChangeMetrics metrics;

    private GroundTruthComparison(BinaryChangeMetrics metrics) {
        this.metrics = metrics;
    }

    /**
     * Compares a predicted mask with the ground truth (both {@code [y][x]}, same size).
     *
     * @throws IllegalArgumentException if a mask is missing, empty or the sizes differ
     */
    public static GroundTruthComparison compare(boolean[][] predicted, boolean[][] groundTruth) {
        validate(predicted, groundTruth);
        BinaryChangeMetrics m = new BinaryChangeMetrics();
        m.accumulate(predicted, groundTruth);
        return new GroundTruthComparison(m);
    }

    /** The per-image confusion matrix and metrics. */
    public BinaryChangeMetrics getMetrics() {
        return metrics;
    }

    /** The label as a black (no change) / white (change) image. */
    public static BufferedImage renderGroundTruth(boolean[][] groundTruth) {
        return ChangeVisualizer.createBlackWhiteMask(groundTruth);
    }

    /**
     * Error map: TP, FP and FN pixels in their colours, TN pixels as dimmed grayscale of {@code context}.
     *
     * @param context post-event image (same size as the masks) used for the TN context
     */
    public static BufferedImage renderErrors(boolean[][] predicted, boolean[][] groundTruth, BufferedImage context) {
        validate(predicted, groundTruth);
        Objects.requireNonNull(context, "context");
        int height = groundTruth.length;
        int width = groundTruth[0].length;
        if (context.getWidth() != width || context.getHeight() != height) {
            throw new IllegalArgumentException(String.format("Context image is %dx%d, masks are %dx%d.",
                    context.getWidth(), context.getHeight(), width, height));
        }
        BufferedImage out = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                out.setRGB(x, y, errorColour(predicted[y][x], groundTruth[y][x], context.getRGB(x, y)));
            }
        }
        return out;
    }

    /** Colour (0xRRGGBB) of one error-map pixel. */
    public static int errorColour(boolean predictedChange, boolean actualChange, int contextRgb) {
        if (predictedChange && actualChange) {
            return TP_RGB;
        }
        if (predictedChange) {
            return FP_RGB;
        }
        if (actualChange) {
            return FN_RGB;
        }
        return dimmedGray(contextRgb);
    }

    /** BT.601 luminance scaled by {@link #TN_CONTEXT_DIM}, as R = G = B. */
    static int dimmedGray(int rgb) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        int v = (int) Math.round((0.299 * r + 0.587 * g + 0.114 * b) * TN_CONTEXT_DIM);
        v = Math.max(0, Math.min(255, v));
        return (v << 16) | (v << 8) | v;
    }

    private static void validate(boolean[][] predicted, boolean[][] groundTruth) {
        if (predicted == null || groundTruth == null) {
            throw new IllegalArgumentException("Both masks are required.");
        }
        if (groundTruth.length == 0 || groundTruth[0] == null || groundTruth[0].length == 0) {
            throw new IllegalArgumentException("Ground-truth mask must not be empty.");
        }
        int width = groundTruth[0].length;
        if (predicted.length != groundTruth.length) {
            throw new IllegalArgumentException(String.format("Mask heights differ: predicted %d, ground truth %d.",
                    predicted.length, groundTruth.length));
        }
        for (int y = 0; y < groundTruth.length; y++) {
            if (groundTruth[y] == null || groundTruth[y].length != width
                    || predicted[y] == null || predicted[y].length != width) {
                throw new IllegalArgumentException("Masks are not the same rectangular size (row " + y + ").");
            }
        }
    }
}
