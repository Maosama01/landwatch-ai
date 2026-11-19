package com.landwatch.deeplearning;

import com.landwatch.evaluation.BinaryChangeMetrics;

/**
 * Histogram of predicted change probabilities, kept separately for truly-changed and truly-unchanged
 * pixels. It lets the decision threshold be swept cheaply: for a threshold {@code t = k / BINS}, every
 * pixel in bins {@code >= k} is predicted "change", so the confusion matrix for any such threshold
 * is a pair of cumulative sums. The metrics themselves are computed by
 * {@link BinaryChangeMetrics} (built with {@link BinaryChangeMetrics#of}).
 *
 * <p>Used during training for per-epoch validation and for tuning the decision threshold on the
 * validation split. Final reported metrics are computed pixel by pixel with
 * {@link BinaryChangeMetrics#accumulate} on the thresholded masks.</p>
 */
public final class ProbabilityHistogram {

    /** Number of bins; thresholds are multiples of 1 / BINS. */
    public static final int BINS = 1000;

    private final long[] positives = new long[BINS];
    private final long[] negatives = new long[BINS];

    /**
     * Adds pixels.
     *
     * @param probabilities change probabilities in [0, 1]
     * @param mask          ground truth, 1 = change, 0 = no change (same length)
     */
    public void add(float[] probabilities, float[] mask) {
        if (probabilities.length != mask.length) {
            throw new IllegalArgumentException("Length mismatch: " + probabilities.length + " vs " + mask.length);
        }
        for (int i = 0; i < probabilities.length; i++) {
            int bin = (int) (probabilities[i] * BINS);
            if (bin >= BINS) {
                bin = BINS - 1;
            } else if (bin < 0) {
                bin = 0;
            }
            if (mask[i] > 0.5f) {
                positives[bin]++;
            } else {
                negatives[bin]++;
            }
        }
    }

    /** Confusion matrix when pixels with probability {@code >= binIndex / BINS} are called change. */
    public BinaryChangeMetrics metricsAtBin(int binIndex) {
        if (binIndex < 0 || binIndex > BINS) {
            throw new IllegalArgumentException("Bin index out of range: " + binIndex);
        }
        long tp = 0;
        long fp = 0;
        long fn = 0;
        long tn = 0;
        for (int b = 0; b < BINS; b++) {
            if (b >= binIndex) {
                tp += positives[b];
                fp += negatives[b];
            } else {
                fn += positives[b];
                tn += negatives[b];
            }
        }
        return BinaryChangeMetrics.of(tp, fp, fn, tn);
    }

    /** Confusion matrix at a threshold, rounded to the nearest multiple of 1 / BINS. */
    public BinaryChangeMetrics metricsAt(double threshold) {
        return metricsAtBin((int) Math.round(threshold * BINS));
    }

    /**
     * Finds the threshold with the highest change-class F1 among {@code from, from+step, ..., to}.
     * Ties keep the first (lowest) threshold.
     *
     * @return the best threshold
     */
    public double bestF1Threshold(double from, double to, double step) {
        double best = from;
        double bestF1 = -1;
        int n = (int) Math.round((to - from) / step);
        for (int i = 0; i <= n; i++) {
            double t = from + i * step;
            double f1 = metricsAt(t).f1();
            if (f1 > bestF1) {
                bestF1 = f1;
                best = t;
            }
        }
        return Math.round(best * BINS) / (double) BINS;
    }
}
