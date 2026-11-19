package com.landwatch.ui;

import com.landwatch.evaluation.BinaryChangeMetrics;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** Per-image ground-truth comparison: metrics on small synthetic masks and the error-map colours. */
class GroundTruthComparisonTest {

    /** 4x4 masks with TP = 2, FP = 1, FN = 3, TN = 10 (rows are y). */
    private static final boolean T = true;
    private static final boolean F = false;
    private static final boolean[][] PREDICTED = {
            {T, T, F, F},
            {F, F, F, T},
            {F, F, F, F},
            {F, F, F, F}};
    private static final boolean[][] TRUTH = {
            {T, T, T, F},
            {F, F, F, F},
            {F, T, F, F},
            {F, F, F, T}};

    @Test
    @DisplayName("Metrics on a small synthetic pair match hand-computed values (via BinaryChangeMetrics)")
    void metricsOnSyntheticMasks() {
        BinaryChangeMetrics m = GroundTruthComparison.compare(PREDICTED, TRUTH).getMetrics();
        assertEquals(2, m.getTruePositives());
        assertEquals(1, m.getFalsePositives());
        assertEquals(3, m.getFalseNegatives());
        assertEquals(10, m.getTrueNegatives());
        assertEquals(2.0 / 3.0, m.precision(), 1e-12);
        assertEquals(2.0 / 5.0, m.recall(), 1e-12);
        assertEquals(4.0 / 8.0, m.f1(), 1e-12);    // 2TP / (2TP + FP + FN)
        assertEquals(2.0 / 6.0, m.iou(), 1e-12);   // TP / (TP + FP + FN)
    }

    @Test
    @DisplayName("Perfect prediction gives F1 = IoU = 1; an all-empty pair gives 0 by convention")
    void edgeCases() {
        BinaryChangeMetrics perfect = GroundTruthComparison.compare(TRUTH, TRUTH).getMetrics();
        assertEquals(1.0, perfect.f1(), 1e-12);
        assertEquals(1.0, perfect.iou(), 1e-12);

        boolean[][] none = new boolean[3][5];
        BinaryChangeMetrics empty = GroundTruthComparison.compare(none, new boolean[3][5]).getMetrics();
        assertEquals(15, empty.getTrueNegatives());
        assertEquals(0.0, empty.f1());
        assertEquals(0.0, empty.precision());
    }

    @Test
    @DisplayName("Masks of different sizes are rejected")
    void sizeMismatch() {
        assertThrows(IllegalArgumentException.class, () -> GroundTruthComparison.compare(new boolean[4][4], new boolean[4][5]));
        assertThrows(IllegalArgumentException.class, () -> GroundTruthComparison.compare(new boolean[3][4], new boolean[4][4]));
        assertThrows(IllegalArgumentException.class, () -> GroundTruthComparison.compare(null, new boolean[4][4]));
        assertThrows(IllegalArgumentException.class, () -> GroundTruthComparison.compare(new boolean[0][0], new boolean[0][0]));
        assertThrows(IllegalArgumentException.class,
                () -> GroundTruthComparison.renderErrors(PREDICTED, TRUTH, new BufferedImage(5, 4, BufferedImage.TYPE_INT_RGB)));
    }

    @Test
    @DisplayName("Errors view: TP, FP, FN in their colours; TN is dimmed grayscale of the post-event image")
    void errorViewColours() {
        BufferedImage context = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                context.setRGB(x, y, 0xFFFFFF);  // brightest possible context
            }
        }
        context.setRGB(2, 2, 0x226E30);           // a green TN pixel
        BufferedImage errors = GroundTruthComparison.renderErrors(PREDICTED, TRUTH, context);
        assertEquals(4, errors.getWidth());
        assertEquals(4, errors.getHeight());

        assertEquals(GroundTruthComparison.TP_RGB, errors.getRGB(0, 0) & 0xFFFFFF, "TP");
        assertEquals(GroundTruthComparison.TP_RGB, errors.getRGB(1, 0) & 0xFFFFFF, "TP");
        assertEquals(GroundTruthComparison.FP_RGB, errors.getRGB(3, 1) & 0xFFFFFF, "FP");
        assertEquals(GroundTruthComparison.FN_RGB, errors.getRGB(2, 0) & 0xFFFFFF, "FN");
        assertEquals(GroundTruthComparison.FN_RGB, errors.getRGB(1, 2) & 0xFFFFFF, "FN");
        assertEquals(GroundTruthComparison.FN_RGB, errors.getRGB(3, 3) & 0xFFFFFF, "FN");

        int whiteTn = errors.getRGB(0, 3) & 0xFFFFFF;
        int expected = (int) Math.round(255 * GroundTruthComparison.TN_CONTEXT_DIM);
        assertEquals((expected << 16) | (expected << 8) | expected, whiteTn, "white context dimmed to gray");

        int greenTn = errors.getRGB(2, 2) & 0xFFFFFF;
        int r = (greenTn >> 16) & 0xFF;
        assertEquals(r, (greenTn >> 8) & 0xFF, "TN is grayscale");
        assertEquals(r, greenTn & 0xFF, "TN is grayscale");
        int lum = (int) Math.round((0.299 * 0x22 + 0.587 * 0x6E + 0.114 * 0x30) * GroundTruthComparison.TN_CONTEXT_DIM);
        assertEquals(lum, r);

        // Every pixel is exactly one of the four classes.
        assertEquals(GroundTruthComparison.errorColour(true, true, 0), GroundTruthComparison.TP_RGB);
        assertEquals(GroundTruthComparison.errorColour(true, false, 0), GroundTruthComparison.FP_RGB);
        assertEquals(GroundTruthComparison.errorColour(false, true, 0), GroundTruthComparison.FN_RGB);
        assertEquals(0, GroundTruthComparison.errorColour(false, false, 0));
    }

    @Test
    @DisplayName("Error colours are distinct hues, separated in luminance (readable without hue) and brighter than any TN pixel")
    void errorColoursAreColourBlindFriendly() {
        int[] colours = {GroundTruthComparison.TP_RGB, GroundTruthComparison.FP_RGB, GroundTruthComparison.FN_RGB};
        assertEquals(3, Set.of(colours[0], colours[1], colours[2]).size());
        for (int c : colours) {
            assertFalse(((c >> 16) & 0xFF) == ((c >> 8) & 0xFF) && ((c >> 8) & 0xFF) == (c & 0xFF), "not a gray");
        }
        double[] lum = new double[3];
        for (int i = 0; i < 3; i++) {
            lum[i] = relativeLuminance(colours[i]);
        }
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                assertTrue(Math.abs(lum[i] - lum[j]) >= 0.15,
                        String.format("luminance of %06x and %06x too close", colours[i], colours[j]));
            }
        }
        double brightestTn = relativeLuminance(GroundTruthComparison.errorColour(false, false, 0xFFFFFF));
        for (double l : lum) {
            assertTrue(l > brightestTn, "error colours must stand out from the dimmed TN context");
        }
    }

    @Test
    @DisplayName("Ground-truth view is black/white and the legends use the renderer's colours")
    void groundTruthViewAndLegends() {
        BufferedImage gt = GroundTruthComparison.renderGroundTruth(TRUTH);
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                assertEquals(TRUTH[y][x] ? 0xFFFFFF : 0x000000, gt.getRGB(x, y) & 0xFFFFFF);
            }
        }
        List<String> errorLegend = GroundTruthView.ERRORS.getLegend().stream().map(VisualizationMode.LegendEntry::cssColor).toList();
        assertTrue(errorLegend.contains(String.format("#%06x", GroundTruthComparison.TP_RGB)));
        assertTrue(errorLegend.contains(String.format("#%06x", GroundTruthComparison.FP_RGB)));
        assertTrue(errorLegend.contains(String.format("#%06x", GroundTruthComparison.FN_RGB)));
        assertEquals(List.of("#000000", "#ffffff"),
                GroundTruthView.GROUND_TRUTH.getLegend().stream().map(VisualizationMode.LegendEntry::cssColor).toList());
        for (GroundTruthView v : GroundTruthView.values()) {
            assertFalse(v.getDisplayName().isBlank());
            assertFalse(v.getDescription().isBlank());
        }
    }

    /** WCAG relative luminance of an sRGB colour. */
    private static double relativeLuminance(int rgb) {
        double r = lin(((rgb >> 16) & 0xFF) / 255.0);
        double g = lin(((rgb >> 8) & 0xFF) / 255.0);
        double b = lin((rgb & 0xFF) / 255.0);
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    private static double lin(double c) {
        return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    }
}
