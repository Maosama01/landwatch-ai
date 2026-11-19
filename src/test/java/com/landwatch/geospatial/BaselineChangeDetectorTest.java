package com.landwatch.geospatial;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.net.URL;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Rigorous JUnit 5 test suite for the Baseline Geospatial Change Detection Engine.
 */
class BaselineChangeDetectorTest {

    private static BufferedImage createSolidImage(int width, int height, Color color) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, width, height);
        g.dispose();
        return img;
    }

    @Test
    @DisplayName("Identical satellite images must produce exactly zero change")
    void testIdenticalImagesZeroChange() {
        BufferedImage pre = createSolidImage(100, 100, new Color(34, 110, 48));
        BufferedImage post = createSolidImage(100, 100, new Color(34, 110, 48));

        ChangeDetectionResult result = BaselineChangeDetector.analyze(pre, post, 5);

        assertNotNull(result);
        assertEquals(0, result.getFilteredChangedPixels(), "Filtered changed pixels must be zero for identical images");
        assertEquals(0, result.getRawChangedPixels(), "Raw changed pixels must be zero for identical images");
        assertEquals(0.0, result.getChangePercentage(), 1e-6, "Change percentage must be 0.0%");
        assertEquals(0, result.getNumConnectedRegions(), "Connected regions must be zero");
        assertEquals("NO CHANGE", result.getRiskCategory());
        assertEquals(10000, result.getUnchangedPixels());
    }

    @Test
    @DisplayName("Known rectangular changed region detection and percentage calculation")
    void testKnownRectangularChangeRegion() {
        int width = 100;
        int height = 100;
        // Background: Forest green
        BufferedImage pre = createSolidImage(width, height, new Color(34, 110, 48));

        // Post: Identical background, but with a 20x20 patch of cleared brown earth at (20, 20)
        BufferedImage post = createSolidImage(width, height, new Color(34, 110, 48));
        Graphics2D g = post.createGraphics();
        g.setColor(new Color(190, 120, 70));
        g.fillRect(20, 20, 20, 20); // exactly 400 pixels
        g.dispose();

        // 400 pixels out of 10,000 = exactly 4.0% change
        ChangeDetectionResult result = BaselineChangeDetector.analyze(pre, post, 10);

        assertNotNull(result);
        assertEquals(400, result.getRawChangedPixels(), "Raw change count must exactly match 20x20 rectangle");
        assertTrue(result.getFilteredChangedPixels() > 0, "Filtered change count must be positive");
        assertTrue(result.getChangePercentage() > 2.0 && result.getChangePercentage() <= 4.0,
                "Change percentage should reflect the 4.0% rectangle area (with morphology)");

        assertEquals(1, result.getNumConnectedRegions(), "There should be exactly 1 connected encroachment cluster");
        List<Rectangle> boxes = result.getBoundingBoxes();
        assertEquals(1, boxes.size());
        Rectangle box = boxes.get(0);
        assertTrue(box.x >= 19 && box.x <= 21);
        assertTrue(box.y >= 19 && box.y <= 21);
    }

    @Test
    @DisplayName("Otsu threshold calculation on bimodal histogram")
    void testOtsuThresholdCalculation() {
        int[] histogram = new int[256];
        int totalPixels = 1000;

        // Bimodal distribution: 600 background pixels centered at 15, 400 foreground pixels centered at 180
        histogram[10] = 200;
        histogram[15] = 200;
        histogram[20] = 200;

        histogram[170] = 100;
        histogram[180] = 200;
        histogram[190] = 100;

        int threshold = BaselineChangeDetector.calculateOtsuThreshold(histogram, totalPixels);

        // Optimal threshold must separate the two peaks.
        // Under the convention (diff > threshold), any threshold in [20, 169] achieves the maximal
        // between-class variance (6,534,000,000) and perfectly partitions background (<= t) from foreground (> t).
        // The implementation selects the first index that achieves maximum variance, making t = 20 mathematically optimal.
        assertTrue(threshold >= 20 && threshold <= 169,
                "Otsu threshold must fall within the optimal inter-modal plateau [20, 169]. Calculated: " + threshold);
        assertEquals(20, threshold, "First threshold achieving global maximum variance must be 20");
    }

    @Test
    @DisplayName("Image dimension mismatch must throw IllegalArgumentException")
    void testDimensionMismatchThrowsException() {
        BufferedImage pre = createSolidImage(100, 100, Color.GREEN);
        BufferedImage post = createSolidImage(120, 100, Color.GREEN);

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> {
            BaselineChangeDetector.analyze(pre, post);
        });

        assertTrue(ex.getMessage().contains("Image dimension mismatch"),
                "Exception message must specify dimension mismatch");
    }

    @Test
    @DisplayName("Null image inputs must throw IllegalArgumentException")
    void testNullInputsThrowException() {
        BufferedImage valid = createSolidImage(50, 50, Color.GREEN);

        assertThrows(IllegalArgumentException.class, () -> BaselineChangeDetector.analyze(null, valid));
        assertThrows(IllegalArgumentException.class, () -> BaselineChangeDetector.analyze(valid, null));
    }

    @Test
    @DisplayName("Visual change map and overlay generation")
    void testVisualChangeMapsGenerated() {
        BufferedImage pre = createSolidImage(80, 60, Color.DARK_GRAY);
        BufferedImage post = createSolidImage(80, 60, Color.DARK_GRAY);
        Graphics2D g = post.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(10, 10, 15, 15);
        g.dispose();

        ChangeDetectionResult result = BaselineChangeDetector.analyze(pre, post, 5);

        assertNotNull(result.getBinaryChangeMap(), "Binary change map must not be null");
        assertEquals(80, result.getBinaryChangeMap().getWidth());
        assertEquals(60, result.getBinaryChangeMap().getHeight());

        assertNotNull(result.getOverlayChangeMap(), "Overlay change map must not be null");
        assertEquals(80, result.getOverlayChangeMap().getWidth());
        assertEquals(60, result.getOverlayChangeMap().getHeight());
    }

    @Test
    @DisplayName("End-to-end verification with the bundled synthetic demo pair")
    void testRealDemoPairExecution() throws Exception {
        URL preUrl = getClass().getResource("/com/landwatch/sample_data/siteA_pre.bmp");
        URL postUrl = getClass().getResource("/com/landwatch/sample_data/siteA_post.bmp");

        assertNotNull(preUrl, "Bundled demo siteA_pre.bmp must exist");
        assertNotNull(postUrl, "Bundled demo siteA_post.bmp must exist");

        BufferedImage pre = SatelliteImageLoader.load(preUrl);
        BufferedImage post = SatelliteImageLoader.load(postUrl);

        assertEquals(400, pre.getWidth());
        assertEquals(300, pre.getHeight());
        assertEquals(400, post.getWidth());
        assertEquals(300, post.getHeight());

        ChangeDetectionResult result = BaselineChangeDetector.analyze(pre, post);

        assertNotNull(result);
        assertEquals(120000, result.getTotalPixels());
        assertTrue(result.getRawChangedPixels() > 0, "Raw change pixels must be detected");
        assertTrue(result.getFilteredChangedPixels() > 0, "Filtered change pixels must be detected");
        assertTrue(result.getChangePercentage() > 0.0, "Change percentage must be greater than 0");
        assertTrue(result.getNumConnectedRegions() >= 1, "Encroachment clusters must be detected");
        assertTrue(result.getOtsuThreshold() > 0, "Otsu threshold must be positive");
        assertNotNull(result.getBinaryChangeMap());
        assertNotNull(result.getOverlayChangeMap());
    }
}
