package com.landwatch.deeplearning;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Sanity check (d): the inference API handles arbitrary image sizes (tiling, padding, stitching). */
class DeepChangeDetectorTest {

    @TempDir
    Path tempDir;

    @Test
    void handlesImagesThatAreNotMultiplesOf256WithTheFullSizeModel() throws Exception {
        TestModels.saveModelDir(tempDir, FcSiamDiff.DEFAULT_WIDTHS, 256, 0.5f);
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            assertEquals(256, d.getTileSize());
            BufferedImage a = TestModels.randomImage(300, 200, 1);
            BufferedImage b = TestModels.randomImage(300, 200, 2);
            float[][] p = d.predictProbabilities(a, b);
            assertEquals(200, p.length);
            for (float[] row : p) {
                assertEquals(300, row.length);
                for (float v : row) {
                    assertTrue(v >= 0f && v <= 1f && Float.isFinite(v), "probability out of range: " + v);
                }
            }
            boolean[][] mask = d.predictMask(a, b);
            assertEquals(200, mask.length);
            assertEquals(300, mask[0].length);
            assertTrue(d.describe().contains("FC-Siam-diff"));
        }
    }

    @Test
    void stitchedTilesMatchPredictingEachTileOnItsOwn() throws Exception {
        TestModels.saveModelDir(tempDir, TestModels.TINY_WIDTHS, 32, 0.5f);
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            BufferedImage a = TestModels.randomImage(64, 32, 3);
            BufferedImage b = TestModels.randomImage(64, 32, 4);
            float[][] whole = d.predictProbabilities(a, b);
            float[][] left = d.predictProbabilities(TestModels.crop(a, 0, 0, 32, 32), TestModels.crop(b, 0, 0, 32, 32));
            float[][] right = d.predictProbabilities(TestModels.crop(a, 32, 0, 32, 32), TestModels.crop(b, 32, 0, 32, 32));
            for (int y = 0; y < 32; y++) {
                for (int x = 0; x < 32; x++) {
                    assertEquals(left[y][x], whole[y][x], 1e-5f);
                    assertEquals(right[y][x], whole[y][x + 32], 1e-5f);
                }
            }
        }
    }

    @Test
    void handlesImagesSmallerThanOneTileAndOddSizes() throws Exception {
        TestModels.saveModelDir(tempDir, TestModels.TINY_WIDTHS, 32, 0.5f);
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            for (int[] wh : new int[][] {{1, 1}, {13, 20}, {75, 50}, {33, 97}}) {
                float[][] p = d.predictProbabilities(TestModels.randomImage(wh[0], wh[1], 5), TestModels.randomImage(wh[0], wh[1], 6));
                assertEquals(wh[1], p.length);
                assertEquals(wh[0], p[0].length);
            }
        }
    }

    @Test
    void maskUsesTheStoredThreshold() throws Exception {
        TestModels.saveModelDir(tempDir, TestModels.TINY_WIDTHS, 32, 0.37f);
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            BufferedImage a = TestModels.randomImage(50, 40, 7);
            BufferedImage b = TestModels.randomImage(50, 40, 8);
            float[][] p = d.predictProbabilities(a, b);
            boolean[][] mask = d.predictMask(a, b);
            for (int y = 0; y < p.length; y++) {
                for (int x = 0; x < p[y].length; x++) {
                    assertEquals(p[y][x] >= 0.37f, mask[y][x]);
                }
            }
        }
    }

    @Test
    void rejectsMismatchedSizesAndMissingModels() throws Exception {
        TestModels.saveModelDir(tempDir, TestModels.TINY_WIDTHS, 32, 0.5f);
        try (DeepChangeDetector d = DeepChangeDetector.load(tempDir, "cpu")) {
            assertThrows(IllegalArgumentException.class,
                    () -> d.predictProbabilities(TestModels.randomImage(10, 10, 1), TestModels.randomImage(11, 10, 1)));
        }
        assertThrows(IOException.class, () -> DeepChangeDetector.load(tempDir.resolve("missing")));
    }

    @Test
    void reflectIndexMirrorsWithoutRepeatingTheEdge() {
        int[] expected = {0, 1, 2, 3, 4, 3, 2, 1, 0, 1};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], DeepChangeDetector.reflect(i, 5));
        }
        assertEquals(0, DeepChangeDetector.reflect(300, 1));
    }

    @Test
    void applyThresholdIsInclusive() {
        boolean[][] m = DeepChangeDetector.applyThreshold(new float[][] {{0.2f, 0.5f, 0.9f}}, 0.5f);
        assertArrayEquals(new boolean[] {false, true, true}, m[0]);
    }
}
