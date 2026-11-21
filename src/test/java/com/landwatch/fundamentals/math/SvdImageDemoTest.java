package com.landwatch.fundamentals.math;

import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SvdImageDemoTest {

    @Test
    void rankTwoImageIsExactAtRankTwo() {
        // X = outer(a, b) + outer(c, d) has rank 2
        int m = 12;
        int n = 10;
        double[][] x = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                x[i][j] = 0.3 * Math.sin(i + 1) * Math.cos(j) + 0.2 * (i % 3) * (j % 4) / 6.0;
            }
        }
        Matrix mat = new Matrix(x);
        SingularValueDecomposition svd = new SingularValueDecomposition(mat);
        SvdImageDemo.RankResult[] t = SvdImageDemo.rankTable(mat, svd, new int[] {1, 2, 4, 64});
        assertEquals(3, t.length); // 64 > min(m, n) is skipped
        assertTrue(t[0].relativeError() > 1e-3);
        assertEquals(0, t[1].relativeError(), 1e-10);
        assertEquals(1, t[1].energy(), 1e-12);
        assertEquals(2 * (m + n + 1), t[1].numbers());
    }

    @Test
    void errorFallsAndEnergyRisesWithRank() {
        java.util.Random rnd = new java.util.Random(4);
        Matrix mat = Matrix.random(20, 16, 0, 1, rnd);
        SingularValueDecomposition svd = new SingularValueDecomposition(mat);
        SvdImageDemo.RankResult[] t = SvdImageDemo.rankTable(mat, svd, new int[] {1, 2, 4, 8, 16});
        for (int i = 1; i < t.length; i++) {
            assertTrue(t[i].relativeError() <= t[i - 1].relativeError());
            assertTrue(t[i].energy() >= t[i - 1].energy());
        }
        assertEquals(0, t[t.length - 1].relativeError(), 1e-10);
    }

    @Test
    void greyLevelUsesBt601Weights() {
        BufferedImage img = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, 0xFF0000);
        img.setRGB(1, 0, 0xFFFFFF);
        Matrix g = SvdImageDemo.toGrey(img);
        assertEquals(0.299, g.get(0, 0), 1e-9);
        assertEquals(1.0, g.get(0, 1), 1e-9);
    }
}
