package com.landwatch.evaluation;

import java.util.Locale;

/**
 * Pixel-level evaluation of binary change detection.
 *
 * <p>"Change" is the positive class. Predictions are compared with ground truth
 * pixel by pixel and accumulated into one global 2x2 confusion matrix over any
 * number of images or patches. All metrics are then computed from that global
 * matrix, which is the usual protocol for LEVIR-CD.</p>
 *
 * <pre>
 *                      actual change   actual no-change
 *   predicted change        TP               FP
 *   predicted no-change     FN               TN
 * </pre>
 *
 * <p>Pixel accuracy alone is misleading for change detection: in LEVIR-CD only
 * about 5% of pixels are changed, so a model that never predicts change is still
 * about 95% "accurate". Change-class F1 and IoU are the primary metrics.</p>
 *
 * <p>Convention: when a ratio's denominator is zero (for example, precision when
 * nothing was predicted as change), the metric is reported as 0.0.</p>
 */
public final class BinaryChangeMetrics {

    private long truePositives;
    private long falsePositives;
    private long falseNegatives;
    private long trueNegatives;

    /** Creates an empty confusion matrix. */
    public BinaryChangeMetrics() {
    }

    /** Creates a confusion matrix from known counts. */
    public static BinaryChangeMetrics of(long truePositives, long falsePositives,
                                         long falseNegatives, long trueNegatives) {
        if (truePositives < 0 || falsePositives < 0 || falseNegatives < 0 || trueNegatives < 0) {
            throw new IllegalArgumentException("Confusion matrix counts must be non-negative");
        }
        BinaryChangeMetrics m = new BinaryChangeMetrics();
        m.truePositives = truePositives;
        m.falsePositives = falsePositives;
        m.falseNegatives = falseNegatives;
        m.trueNegatives = trueNegatives;
        return m;
    }

    /** Adds a single pixel decision. */
    public void add(boolean predictedChange, boolean actualChange) {
        if (predictedChange && actualChange) {
            truePositives++;
        } else if (predictedChange) {
            falsePositives++;
        } else if (actualChange) {
            falseNegatives++;
        } else {
            trueNegatives++;
        }
    }

    /**
     * Adds every pixel of a predicted mask and its ground-truth mask.
     * Both masks are indexed as {@code mask[y][x]} and must have identical dimensions.
     */
    public void accumulate(boolean[][] predicted, boolean[][] groundTruth) {
        if (predicted == null || groundTruth == null) {
            throw new IllegalArgumentException("Masks must not be null");
        }
        if (predicted.length != groundTruth.length) {
            throw new IllegalArgumentException("Mask heights differ: " + predicted.length + " vs " + groundTruth.length);
        }
        for (int y = 0; y < predicted.length; y++) {
            if (predicted[y].length != groundTruth[y].length) {
                throw new IllegalArgumentException("Mask widths differ at row " + y + ": "
                        + predicted[y].length + " vs " + groundTruth[y].length);
            }
            for (int x = 0; x < predicted[y].length; x++) {
                add(predicted[y][x], groundTruth[y][x]);
            }
        }
    }

    /** Adds the counts of another confusion matrix to this one. */
    public void merge(BinaryChangeMetrics other) {
        truePositives += other.truePositives;
        falsePositives += other.falsePositives;
        falseNegatives += other.falseNegatives;
        trueNegatives += other.trueNegatives;
    }

    public long getTruePositives() { return truePositives; }
    public long getFalsePositives() { return falsePositives; }
    public long getFalseNegatives() { return falseNegatives; }
    public long getTrueNegatives() { return trueNegatives; }

    public long getTotal() {
        return truePositives + falsePositives + falseNegatives + trueNegatives;
    }

    /** Change precision: TP / (TP + FP). */
    public double precision() {
        return ratio(truePositives, truePositives + falsePositives);
    }

    /** Change recall: TP / (TP + FN). */
    public double recall() {
        return ratio(truePositives, truePositives + falseNegatives);
    }

    /** Change-class F1: 2TP / (2TP + FP + FN), the harmonic mean of precision and recall. */
    public double f1() {
        return ratio(2 * truePositives, 2 * truePositives + falsePositives + falseNegatives);
    }

    /** Change-class Intersection over Union: TP / (TP + FP + FN). */
    public double iou() {
        return ratio(truePositives, truePositives + falsePositives + falseNegatives);
    }

    /** No-change-class F1: 2TN / (2TN + FN + FP). */
    public double noChangeF1() {
        return ratio(2 * trueNegatives, 2 * trueNegatives + falseNegatives + falsePositives);
    }

    /** No-change-class IoU: TN / (TN + FN + FP). */
    public double noChangeIoU() {
        return ratio(trueNegatives, trueNegatives + falseNegatives + falsePositives);
    }

    /** Macro F1: unweighted mean of the change and no-change F1 scores. */
    public double macroF1() {
        return (f1() + noChangeF1()) / 2.0;
    }

    /** Mean IoU: unweighted mean of the change and no-change IoU scores. */
    public double meanIoU() {
        return (iou() + noChangeIoU()) / 2.0;
    }

    /** Overall pixel accuracy: (TP + TN) / total. Not reliable on its own for imbalanced data. */
    public double accuracy() {
        return ratio(truePositives + trueNegatives, getTotal());
    }

    /** Cohen's kappa: agreement corrected for the agreement expected by chance. */
    public double kappa() {
        double total = getTotal();
        if (total == 0) {
            return 0.0;
        }
        double observed = (truePositives + trueNegatives) / total;
        double expected = ((double) (truePositives + falsePositives) * (truePositives + falseNegatives)
                + (double) (falseNegatives + trueNegatives) * (falsePositives + trueNegatives)) / (total * total);
        return expected == 1.0 ? 0.0 : (observed - expected) / (1.0 - expected);
    }

    /** Multi-line, human-readable summary of the confusion matrix and all metrics. */
    /** Pixels the method marks as change: TP + FP. */
    public long predictedChangePixels() {
        return truePositives + falsePositives;
    }

    /** Pixels that truly changed: TP + FN. */
    public long trueChangePixels() {
        return truePositives + falseNegatives;
    }

    /**
     * Area error of the change class, {@code |predicted - true| = |FP - FN|} pixels. False alarms and
     * misses cancel here, so a small area error does not mean the changes are in the right places
     * (that is what IoU and F1 measure).
     */
    public long absoluteAreaErrorPixels() {
        return Math.abs(falsePositives - falseNegatives);
    }

    /**
     * Signed relative area error {@code (predicted - true) / true}: +0.10 means 10 % too much change
     * is reported, -0.10 means 10 % too little. Returns NaN when nothing truly changed, because the
     * ratio is then undefined; use {@link #absoluteAreaErrorPixels()} in that case.
     */
    public double relativeAreaError() {
        long t = trueChangePixels();
        return t == 0 ? Double.NaN : (predictedChangePixels() - t) / (double) t;
    }

    /** The area-error line of {@link #toReport()}; it handles the no-change case explicitly. */
    public String areaErrorLine() {
        double r = relativeAreaError();
        if (Double.isNaN(r)) {
            return String.format(Locale.ROOT, "Area error       : undefined (no true change); %d pixels predicted as change%n",
                    predictedChangePixels());
        }
        return String.format(Locale.ROOT, "Area error       : %+.2f %%  (predicted %d vs true %d change pixels; |FP - FN| = %d)%n",
                100 * r, predictedChangePixels(), trueChangePixels(), absoluteAreaErrorPixels());
    }

    public String toReport() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Confusion matrix (pixels): TP=%d  FP=%d  FN=%d  TN=%d  (total %d)%n",
                truePositives, falsePositives, falseNegatives, trueNegatives, getTotal()));
        sb.append(String.format("Change precision : %.4f%n", precision()));
        sb.append(String.format("Change recall    : %.4f%n", recall()));
        sb.append(String.format("Change F1        : %.4f%n", f1()));
        sb.append(String.format("Change IoU       : %.4f%n", iou()));
        sb.append(String.format("Macro F1         : %.4f%n", macroF1()));
        sb.append(String.format("Mean IoU         : %.4f%n", meanIoU()));
        sb.append(String.format("Cohen's kappa    : %.4f%n", kappa()));
        sb.append(areaErrorLine());
        sb.append(String.format("Pixel accuracy   : %.4f  (misleading alone: classes are imbalanced)%n", accuracy()));
        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("BinaryChangeMetrics[TP=%d, FP=%d, FN=%d, TN=%d, F1=%.4f, IoU=%.4f]",
                truePositives, falsePositives, falseNegatives, trueNegatives, f1(), iou());
    }

    private static double ratio(double numerator, double denominator) {
        return denominator == 0 ? 0.0 : numerator / denominator;
    }
}
