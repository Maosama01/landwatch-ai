package com.landwatch.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for pixel-level change-detection metrics, checked against hand-computed values.
 */
class BinaryChangeMetricsTest {

    private static final double EPS = 1e-6;

    @Test
    @DisplayName("Metrics match hand-computed values for TP=40, FP=10, FN=20, TN=930")
    void testKnownConfusionMatrix() {
        BinaryChangeMetrics m = BinaryChangeMetrics.of(40, 10, 20, 930);

        assertEquals(1000, m.getTotal());
        assertEquals(0.8, m.precision(), EPS);                    // 40 / 50
        assertEquals(40.0 / 60.0, m.recall(), EPS);               // 40 / 60
        assertEquals(80.0 / 110.0, m.f1(), EPS);                  // 2*40 / (80 + 10 + 20)
        assertEquals(40.0 / 70.0, m.iou(), EPS);                  // 40 / (40 + 10 + 20)
        assertEquals(0.97, m.accuracy(), EPS);                    // 970 / 1000
        assertEquals(1860.0 / 1890.0, m.noChangeF1(), EPS);       // 2*930 / (1860 + 20 + 10)
        assertEquals(930.0 / 960.0, m.noChangeIoU(), EPS);        // 930 / (930 + 20 + 10)
        assertEquals((80.0 / 110.0 + 1860.0 / 1890.0) / 2, m.macroF1(), EPS);
        assertEquals((40.0 / 70.0 + 930.0 / 960.0) / 2, m.meanIoU(), EPS);
        // kappa: po = 0.97, pe = (50*60 + 950*940) / 1000^2 = 0.896
        assertEquals((0.97 - 0.896) / (1 - 0.896), m.kappa(), EPS);
    }

    @Test
    @DisplayName("Accumulating masks counts every pixel into the right cell")
    void testAccumulateMasks() {
        boolean[][] predicted = {
                {true, true, false},
                {false, false, true}
        };
        boolean[][] truth = {
                {true, false, false},
                {true, false, true}
        };
        BinaryChangeMetrics m = new BinaryChangeMetrics();
        m.accumulate(predicted, truth);

        assertEquals(2, m.getTruePositives());   // (0,0) and (1,2)
        assertEquals(1, m.getFalsePositives());  // (0,1)
        assertEquals(1, m.getFalseNegatives());  // (1,0)
        assertEquals(2, m.getTrueNegatives());   // (0,2) and (1,1)
    }

    @Test
    @DisplayName("Masks with different dimensions are rejected")
    void testDimensionMismatchRejected() {
        BinaryChangeMetrics m = new BinaryChangeMetrics();
        assertThrows(IllegalArgumentException.class,
                () -> m.accumulate(new boolean[2][3], new boolean[3][3]));
        assertThrows(IllegalArgumentException.class,
                () -> m.accumulate(new boolean[2][3], new boolean[2][4]));
    }

    @Test
    @DisplayName("Merging two matrices equals accumulating everything into one")
    void testMerge() {
        BinaryChangeMetrics a = BinaryChangeMetrics.of(1, 2, 3, 4);
        BinaryChangeMetrics b = BinaryChangeMetrics.of(10, 20, 30, 40);
        a.merge(b);
        assertEquals(11, a.getTruePositives());
        assertEquals(22, a.getFalsePositives());
        assertEquals(33, a.getFalseNegatives());
        assertEquals(44, a.getTrueNegatives());
    }

    @Test
    @DisplayName("A perfect prediction scores 1.0 on every metric")
    void testPerfectPrediction() {
        BinaryChangeMetrics m = BinaryChangeMetrics.of(50, 0, 0, 950);
        assertEquals(1.0, m.precision(), EPS);
        assertEquals(1.0, m.recall(), EPS);
        assertEquals(1.0, m.f1(), EPS);
        assertEquals(1.0, m.iou(), EPS);
        assertEquals(1.0, m.macroF1(), EPS);
        assertEquals(1.0, m.kappa(), EPS);
    }

    @Test
    @DisplayName("Predicting 'no change' everywhere looks accurate but scores zero F1 and IoU")
    void testAllBackgroundPredictionIsExposed() {
        // Same imbalance as LEVIR-CD: about 5% of pixels changed.
        BinaryChangeMetrics m = BinaryChangeMetrics.of(0, 0, 50, 950);
        assertEquals(0.95, m.accuracy(), EPS);
        assertEquals(0.0, m.f1(), EPS);
        assertEquals(0.0, m.iou(), EPS);
        assertEquals(0.0, m.kappa(), EPS);
    }

    @Test
    @DisplayName("Zero denominators give 0.0, never NaN")
    void testEmptyMatrixHasNoNaN() {
        BinaryChangeMetrics m = new BinaryChangeMetrics();
        assertEquals(0.0, m.precision());
        assertEquals(0.0, m.recall());
        assertEquals(0.0, m.f1());
        assertEquals(0.0, m.iou());
        assertEquals(0.0, m.accuracy());
        assertEquals(0.0, m.kappa());
        assertFalse(m.toReport().contains("NaN"));
    }

    @Test
    void areaErrorComparesPredictedWithTrueChangedArea() {
        // predicted = TP + FP = 110, true = TP + FN = 100 -> +10 %, |FP - FN| = 10
        BinaryChangeMetrics m = BinaryChangeMetrics.of(90, 20, 10, 880);
        assertEquals(110, m.predictedChangePixels());
        assertEquals(100, m.trueChangePixels());
        assertEquals(10, m.absoluteAreaErrorPixels());
        assertEquals(0.10, m.relativeAreaError(), 1e-12);
        assertTrue(m.toReport().contains("Area error       : +10.00 %"));
        // misses and false alarms cancel: perfect area, imperfect map
        BinaryChangeMetrics cancel = BinaryChangeMetrics.of(80, 20, 20, 880);
        assertEquals(0.0, cancel.relativeAreaError(), 0);
        assertTrue(cancel.iou() < 1);
    }

    @Test
    void areaErrorIsUndefinedNotInfiniteWithoutTrueChange() {
        BinaryChangeMetrics m = BinaryChangeMetrics.of(0, 7, 0, 993);
        assertTrue(Double.isNaN(m.relativeAreaError()));
        assertEquals(7, m.absoluteAreaErrorPixels());
        assertTrue(m.toReport().contains("undefined (no true change); 7 pixels predicted as change"));
        assertFalse(m.toReport().contains("NaN"));
        assertFalse(m.toReport().contains("Infinity"));
    }
}
