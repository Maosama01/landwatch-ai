package com.landwatch.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the K x K multi-class metrics, checked against values computed by hand.
 *
 * <pre>
 * Worked example (rows = actual, columns = predicted):
 *               pred 0  pred 1  pred 2   | support
 *   actual 0       5       1       0     |   6
 *   actual 1       2       3       1     |   6
 *   actual 2       0       1       4     |   5
 *   predicted      7       5       5     |  17
 *
 *   accuracy = (5 + 3 + 4) / 17 = 12/17
 *   class 0: TP 5, FP 2, FN 1 -> P 5/7, R 5/6, F1 10/13
 *   class 1: TP 3, FP 2, FN 3 -> P 3/5, R 3/6, F1 6/11
 *   class 2: TP 4, FP 1, FN 1 -> P 4/5, R 4/5, F1 8/10
 * </pre>
 */
class MulticlassMetricsTest {

    private static final double EPS = 1e-9;
    private static final List<String> NAMES = List.of("Forest", "River", "SeaLake");

    private static MulticlassMetrics worked() {
        return MulticlassMetrics.fromConfusionMatrix(new long[][] {
                {5, 1, 0},
                {2, 3, 1},
                {0, 1, 4}
        });
    }

    @Test
    @DisplayName("Accuracy, per-class P/R/F1 and macro scores match hand-computed values")
    void handComputedExample() {
        MulticlassMetrics m = worked();
        assertEquals(17, m.getTotal());
        assertEquals(12, m.getCorrect());
        assertEquals(12.0 / 17.0, m.accuracy(), EPS);

        assertEquals(5.0 / 7.0, m.precision(0), EPS);
        assertEquals(5.0 / 6.0, m.recall(0), EPS);
        assertEquals(10.0 / 13.0, m.f1(0), EPS);

        assertEquals(3.0 / 5.0, m.precision(1), EPS);
        assertEquals(3.0 / 6.0, m.recall(1), EPS);
        assertEquals(6.0 / 11.0, m.f1(1), EPS);

        assertEquals(4.0 / 5.0, m.precision(2), EPS);
        assertEquals(4.0 / 5.0, m.recall(2), EPS);
        assertEquals(8.0 / 10.0, m.f1(2), EPS);

        assertEquals((5.0 / 7.0 + 3.0 / 5.0 + 4.0 / 5.0) / 3.0, m.macroPrecision(), EPS);
        assertEquals((5.0 / 6.0 + 3.0 / 6.0 + 4.0 / 5.0) / 3.0, m.macroRecall(), EPS);
        assertEquals((10.0 / 13.0 + 6.0 / 11.0 + 8.0 / 10.0) / 3.0, m.macroF1(), EPS);
        // the same numbers rounded, as a reader would compute them on paper
        assertEquals(0.7059, m.accuracy(), 5e-5);
        assertEquals(0.7049, m.macroF1(), 5e-5);
    }

    @Test
    @DisplayName("TP / FP / FN, support and column counts come from rows and columns")
    void countsFromRowsAndColumns() {
        MulticlassMetrics m = worked();
        assertEquals(6, m.support(0));
        assertEquals(6, m.support(1));
        assertEquals(5, m.support(2));
        assertEquals(7, m.predictedCount(0));
        assertEquals(5, m.predictedCount(1));
        assertEquals(5, m.predictedCount(2));
        assertEquals(2, m.falsePositives(0));
        assertEquals(1, m.falseNegatives(0));
        assertEquals(2, m.falsePositives(1));
        assertEquals(3, m.falseNegatives(1));
        assertEquals(1, m.falsePositives(2));
        assertEquals(1, m.falseNegatives(2));
    }

    @Test
    @DisplayName("Adding predictions one by one builds the matrix as [actual][predicted]")
    void addBuildsMatrix() {
        MulticlassMetrics m = new MulticlassMetrics(3);
        int[] actual = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2};
        int[] predicted = {0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 2, 1, 2, 2, 2, 2};
        m.addAll(actual, predicted);
        assertArrayEquals(worked().getConfusionMatrix(), m.getConfusionMatrix());
        assertEquals(1, m.count(1, 2));
        assertEquals(2, m.count(1, 0));

        MulticlassMetrics merged = new MulticlassMetrics(3);
        merged.merge(m);
        merged.merge(m);
        assertEquals(34, merged.getTotal());
        assertEquals(m.accuracy(), merged.accuracy(), EPS);
    }

    @Test
    @DisplayName("Perfect predictions give 1.0 everywhere")
    void perfectPredictions() {
        MulticlassMetrics m = MulticlassMetrics.fromConfusionMatrix(new long[][] {{4, 0}, {0, 6}});
        assertEquals(1.0, m.accuracy(), EPS);
        assertEquals(1.0, m.macroF1(), EPS);
        assertEquals(1.0, m.precision(1), EPS);
        assertEquals(1.0, m.recall(0), EPS);
    }

    @Test
    @DisplayName("A class that is never predicted has precision 0 (zero-denominator convention)")
    void zeroDenominators() {
        // class 2 never occurs and is never predicted; class 1 is never predicted
        MulticlassMetrics m = MulticlassMetrics.fromConfusionMatrix(new long[][] {
                {3, 0, 0},
                {2, 0, 0},
                {0, 0, 0}
        });
        assertEquals(0.0, m.precision(1), EPS);
        assertEquals(0.0, m.recall(1), EPS);
        assertEquals(0.0, m.f1(1), EPS);
        assertEquals(0.0, m.precision(2), EPS);
        assertEquals(0.0, m.recall(2), EPS);
        assertEquals(0.0, m.f1(2), EPS);
        assertEquals(3.0 / 5.0, m.precision(0), EPS);
        assertEquals(1.0, m.recall(0), EPS);
        assertEquals(6.0 / 8.0, m.f1(0), EPS);                 // 2*3 / (6 + 2 + 0)
        assertEquals((6.0 / 8.0) / 3.0, m.macroF1(), EPS);     // macro average over all K classes
        assertEquals(0.6, m.accuracy(), EPS);
        assertEquals(0.0, new MulticlassMetrics(4).accuracy(), EPS);
    }

    @Test
    @DisplayName("Invalid input is rejected")
    void rejectsInvalidInput() {
        MulticlassMetrics m = new MulticlassMetrics(3);
        assertThrows(IllegalArgumentException.class, () -> m.add(3, 0));
        assertThrows(IllegalArgumentException.class, () -> m.add(0, -1));
        assertThrows(IllegalArgumentException.class, () -> m.addAll(new int[] {0, 1}, new int[] {0}));
        assertThrows(IllegalArgumentException.class, () -> new MulticlassMetrics(1));
        assertThrows(IllegalArgumentException.class,
                () -> MulticlassMetrics.fromConfusionMatrix(new long[][] {{1, 2}, {3}}));
        assertThrows(IllegalArgumentException.class,
                () -> MulticlassMetrics.fromConfusionMatrix(new long[][] {{1, -2}, {3, 4}}));
        assertThrows(IllegalArgumentException.class, () -> m.merge(new MulticlassMetrics(2)));
        assertThrows(IllegalArgumentException.class, () -> m.toReport(List.of("a", "b")));
    }

    @Test
    @DisplayName("Report has parseable 'Overall accuracy' and 'Macro F1' lines and the per-class table")
    void reportFormat() {
        String report = worked().toReport(NAMES);
        List<String> lines = report.lines().toList();
        assertTrue(lines.contains("Overall accuracy : 0.7059"), report);
        assertTrue(lines.contains("Macro F1         : 0.7049"), report);
        assertTrue(lines.stream().anyMatch(l -> l.matches("\\s*1\\s+River\\s+0\\.6000\\s+0\\.5000\\s+0\\.5455\\s+6")), report);
        assertTrue(lines.stream().anyMatch(l -> l.matches("\\s*2\\s+SeaLake\\s+0\\s+1\\s+4")), report);
    }

    @Test
    @DisplayName("Confusion-matrix CSV has a header and one row per actual class")
    void confusionCsv() {
        String csv = worked().toConfusionMatrixCsv(NAMES);
        assertEquals("actual\\predicted,Forest,River,SeaLake\n"
                + "Forest,5,1,0\n"
                + "River,2,3,1\n"
                + "SeaLake,0,1,4\n", csv);
    }

    @Test
    @DisplayName("Top confusions are the largest off-diagonal cells")
    void topConfusions() {
        List<long[]> top = worked().topConfusions(2);
        assertEquals(2, top.size());
        assertArrayEquals(new long[] {1, 0, 2}, top.get(0)); // River predicted as Forest, twice
        assertArrayEquals(new long[] {0, 1, 1}, top.get(1));
    }
}
