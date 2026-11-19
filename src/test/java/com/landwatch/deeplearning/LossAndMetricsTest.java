package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import com.landwatch.evaluation.BinaryChangeMetrics;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Tests for the training loss and the validation threshold sweep. */
class LossAndMetricsTest {

    @Test
    void weightedBceMatchesHandComputedValues() {
        float[] logits = {2f, -1f, 0.5f, -3f};
        float[] y = {1f, 0f, 0f, 1f};
        float pw = 3f;
        double expected = 0;
        for (int i = 0; i < logits.length; i++) {
            double p = 1 / (1 + Math.exp(-logits[i]));
            expected += -(pw * y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
        }
        expected /= logits.length;
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            BceDiceLoss loss = new BceDiceLoss(pw, 0f);
            float bce = loss.weightedBce(m.create(logits), m.create(y)).getFloat();
            assertEquals(expected, bce, 1e-5);
            float total = loss.evaluate(new NDList(m.create(y)), new NDList(m.create(logits))).getFloat();
            assertEquals(expected, total, 1e-5);
        }
    }

    @Test
    void bceIsStableForExtremeLogitsAndDiceRewardsOverlap() {
        try (NDManager m = NDManager.newBaseManager(Device.cpu())) {
            BceDiceLoss loss = new BceDiceLoss(4f, 1f);
            NDArray y = m.create(new float[] {1f, 0f, 1f, 0f});
            NDArray perfect = m.create(new float[] {100f, -100f, 100f, -100f});
            NDArray wrong = m.create(new float[] {-100f, 100f, -100f, 100f});
            float good = loss.evaluate(new NDList(y), new NDList(perfect)).getFloat();
            float bad = loss.evaluate(new NDList(y), new NDList(wrong)).getFloat();
            assertTrue(Float.isFinite(good) && Float.isFinite(bad));
            assertTrue(good < 0.01f, "perfect prediction should have ~0 loss, got " + good);
            assertTrue(bad > 100f, "confidently wrong prediction should have a large loss, got " + bad);
            assertEquals(0f, loss.dice(perfect, y).getFloat(), 1e-3f);
        }
    }

    @Test
    void histogramSweepAgreesWithDirectPixelCounting() {
        Random rnd = new Random(5);
        int n = 20_000;
        float[] p = new float[n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            y[i] = rnd.nextDouble() < 0.1 ? 1f : 0f;
            p[i] = (float) Math.min(1.0, Math.max(0.0, rnd.nextGaussian() * 0.2 + (y[i] > 0 ? 0.65 : 0.3)));
        }
        ProbabilityHistogram hist = new ProbabilityHistogram();
        hist.add(p, y);
        for (int k = 1; k <= 19; k++) {
            double t = k * 0.05;
            BinaryChangeMetrics direct = new BinaryChangeMetrics();
            for (int i = 0; i < n; i++) {
                direct.add(p[i] >= (float) t, y[i] > 0.5f);
            }
            BinaryChangeMetrics swept = hist.metricsAt(t);
            assertEquals(direct.getTruePositives(), swept.getTruePositives(), "TP at " + t);
            assertEquals(direct.getFalsePositives(), swept.getFalsePositives(), "FP at " + t);
            assertEquals(direct.getTotal(), swept.getTotal());
        }
        double best = hist.bestF1Threshold(0.05, 0.95, 0.05);
        double bestF1 = hist.metricsAt(best).f1();
        for (int k = 1; k <= 19; k++) {
            assertTrue(bestF1 >= hist.metricsAt(k * 0.05).f1());
        }
    }
}
