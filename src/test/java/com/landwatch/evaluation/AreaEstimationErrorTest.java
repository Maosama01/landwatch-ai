package com.landwatch.evaluation;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AreaEstimationErrorTest {

    @Test
    void convertsPixelsToSquareMetresAndComparesAreas() {
        long[] t = {4000, 2000, 0, 10};
        long[] p = {4400, 1000, 80, 30};
        AreaEstimationError.Summary s = AreaEstimationError.summarize(t, p, 0.5);
        assertEquals(6010 * 0.25, s.trueArea(), 1e-9);
        assertEquals(5510 * 0.25, s.predictedArea(), 1e-9);
        assertEquals((5510 - 6010) / 6010.0, s.totalRelativeError(), 1e-12);
        // absolute errors 400, 1000, 80, 20 px -> 100, 250, 20, 5 m^2
        assertEquals((100 + 250 + 20 + 5) / 4.0, s.meanAbsoluteError(), 1e-9);
        assertEquals((20 + 100) / 2.0, s.medianAbsoluteError(), 1e-9);
        // relative errors only for images with >= 1000 true pixels: 0.10 and 0.50
        assertEquals(2, s.relativeImages());
        assertEquals(0.30, s.medianRelative(), 1e-12);
        assertEquals(0.5, s.within10(), 1e-12);
        assertEquals(0.5, s.within25(), 1e-12);
        assertEquals(1, s.noChangeImages());
        assertEquals(20, s.meanFalseAreaOnNoChange(), 1e-9);
    }

    @Test
    void perfectPredictionsHaveNoErrorAndFullCorrelation() {
        long[] t = {1000, 5000, 0, 2500};
        AreaEstimationError.Summary s = AreaEstimationError.summarize(t, t.clone(), 0.5);
        assertEquals(0, s.totalRelativeError(), 0);
        assertEquals(0, s.meanAbsoluteError(), 0);
        assertEquals(1.0, s.correlation(), 1e-12);
        assertEquals(1.0, s.within10(), 0);
    }

    @Test
    void quantileInterpolates() {
        assertEquals(2.5, AreaEstimationError.quantile(new double[] {4, 1, 3, 2}, 0.5), 1e-12);
        assertEquals(1.75, AreaEstimationError.quantile(new double[] {1, 2, 3, 4}, 0.25), 1e-12);
        assertTrue(Double.isNaN(AreaEstimationError.quantile(new double[0], 0.5)));
    }

    @Test
    void readsCountColumnsByName(@TempDir Path dir) throws Exception {
        Path csv = dir.resolve("per_image.csv");
        Files.writeString(csv, "image,width,true_change_px,x,pred_change_px\na.png,4,10,9,12\nb.png,4,0,9,3\n");
        long[][] c = AreaEstimationError.readCounts(csv);
        assertArrayEquals(new long[] {10, 0}, c[0]);
        assertArrayEquals(new long[] {12, 3}, c[1]);
        Path bad = dir.resolve("bad.csv");
        Files.writeString(bad, "image,f1\na,0.5\n");
        assertThrows(java.io.IOException.class, () -> AreaEstimationError.readCounts(bad));
    }

    @Test
    void reportNamesEachMethodAndTheCaveat() {
        AreaEstimationError.Summary s = AreaEstimationError.summarize(new long[] {2000, 0}, new long[] {1800, 0}, 0.5);
        String text = AreaEstimationError.report(List.of("Model"), List.of(Path.of("x.csv")), List.of(s));
        assertTrue(text.contains("== Model"));
        assertTrue(text.contains("-10.00 %"));
        assertTrue(text.contains("Caveat"));
        assertThrows(IllegalArgumentException.class, () -> AreaEstimationError.summarize(new long[] {1}, new long[] {1}, 0));
    }
}
