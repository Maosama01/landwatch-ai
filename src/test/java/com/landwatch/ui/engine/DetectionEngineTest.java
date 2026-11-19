package com.landwatch.ui.engine;

import com.landwatch.geospatial.BaselineChangeDetector;
import com.landwatch.geospatial.ChangeDetectionResult;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the pluggable engine layer used by the dashboard.
 */
class DetectionEngineTest {

    private static BufferedImage solid(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    private static BufferedImage withPatch(BufferedImage base, int x, int y, int w, int h, Color c) {
        BufferedImage copy = solid(base.getWidth(), base.getHeight(), Color.BLACK);
        Graphics2D g = copy.createGraphics();
        g.drawImage(base, 0, 0, null);
        g.setColor(c);
        g.fillRect(x, y, w, h);
        g.dispose();
        return copy;
    }

    // ---------------------------------------------------------------- EngineResult

    @Test
    @DisplayName("EngineResult derives pixel counts, percentage and regions from the mask")
    void engineResultStatsFromMask() {
        boolean[][] mask = new boolean[10][10];
        // region A: 2x2 at (1,1); region B: 3x1 at (6,7)
        mask[1][1] = mask[1][2] = mask[2][1] = mask[2][2] = true;
        mask[7][6] = mask[7][7] = mask[7][8] = true;

        EngineResult r = EngineResult.fromMask("test", "Test engine", mask, 5);
        assertEquals(10, r.getWidth());
        assertEquals(10, r.getHeight());
        assertEquals(100, r.getTotalPixels());
        assertEquals(7, r.getChangedPixels());
        assertEquals(93, r.getUnchangedPixels());
        assertEquals(7.0, r.getChangePercentage(), 1e-9);
        assertEquals(2, r.getRegionCount());
        assertTrue(r.getRegions().contains(new Rectangle(1, 1, 2, 2)));
        assertTrue(r.getRegions().contains(new Rectangle(6, 7, 3, 1)));
        assertNull(r.getOtsuThreshold(), "generic engines have no Otsu threshold");
        assertNull(r.getRawChangedPixels());
        assertEquals("Test engine", r.getEngineName());
        assertEquals(5, r.getLatencyMs());
        assertEquals(EngineResult.ChangeLevel.HIGH, r.getChangeLevel());
    }

    @Test
    @DisplayName("EngineResult copies the mask defensively")
    void engineResultIsImmutable() {
        boolean[][] mask = new boolean[2][2];
        EngineResult r = EngineResult.fromMask("t", "T", mask, 0);
        mask[0][0] = true;
        assertFalse(r.isChanged(0, 0));
        r.getMask()[1][1] = true;
        assertFalse(r.isChanged(1, 1));
    }

    @Test
    @DisplayName("EngineResult rejects invalid masks and metadata")
    void engineResultValidation() {
        assertThrows(IllegalArgumentException.class, () -> EngineResult.fromMask("t", "T", null, 0));
        assertThrows(IllegalArgumentException.class, () -> EngineResult.fromMask("t", "T", new boolean[0][0], 0));
        assertThrows(IllegalArgumentException.class,
                () -> EngineResult.fromMask("t", "T", new boolean[][]{new boolean[3], new boolean[2]}, 0));
        assertThrows(IllegalArgumentException.class, () -> EngineResult.fromMask("", "T", new boolean[1][1], 0));
        assertThrows(IllegalArgumentException.class, () -> EngineResult.fromMask("t", " ", new boolean[1][1], 0));
        assertThrows(IllegalArgumentException.class, () -> EngineResult.fromMask("t", "T", new boolean[1][1], -1));
    }

    @Test
    @DisplayName("Change level thresholds match the existing risk categories")
    void changeLevelThresholds() {
        assertEquals(EngineResult.ChangeLevel.NONE, EngineResult.ChangeLevel.classify(0, 0.0));
        assertEquals(EngineResult.ChangeLevel.LOW, EngineResult.ChangeLevel.classify(1, 0.99));
        assertEquals(EngineResult.ChangeLevel.MODERATE, EngineResult.ChangeLevel.classify(1, 1.0));
        assertEquals(EngineResult.ChangeLevel.HIGH, EngineResult.ChangeLevel.classify(1, 5.0));
        assertEquals(EngineResult.ChangeLevel.SEVERE, EngineResult.ChangeLevel.classify(1, 15.0));
    }

    // ---------------------------------------------------------------- engines

    @Test
    @DisplayName("Classical baseline engine wraps BaselineChangeDetector faithfully")
    void classicalBaselineEngineMatchesDetector() throws Exception {
        BufferedImage pre = solid(100, 100, new Color(34, 110, 48));
        BufferedImage post = withPatch(pre, 20, 20, 20, 20, new Color(190, 120, 70));

        ClassicalBaselineEngine engine = new ClassicalBaselineEngine();
        assertTrue(engine.isAvailable());
        assertEquals("Classical baseline (Otsu differencing)", engine.getDisplayName());

        EngineResult r = engine.detect(pre, post);
        ChangeDetectionResult expected = BaselineChangeDetector.analyze(pre, post);

        assertEquals(ClassicalBaselineEngine.ID, r.getEngineId());
        assertEquals(engine.getDisplayName(), r.getEngineName());
        assertEquals(100, r.getWidth());
        assertEquals(100, r.getHeight());
        assertEquals(expected.getFilteredChangedPixels(), r.getChangedPixels());
        assertEquals(expected.getNumConnectedRegions(), r.getRegionCount());
        assertEquals(expected.getOtsuThreshold(), r.getOtsuThreshold());
        assertEquals(expected.getRawChangedPixels(), r.getRawChangedPixels());
        assertEquals(1, r.getRegionCount());
        assertTrue(r.isChanged(30, 30), "centre of the changed patch must be in the mask");
        assertFalse(r.isChanged(80, 80), "untouched area must not be in the mask");
    }

    @Test
    @DisplayName("Siamese U-Net placeholder is unavailable and never produces output")
    void deepEnginePlaceholderRefusesToRun() {
        SiameseUNetEngine engine = new SiameseUNetEngine();
        assertFalse(engine.isAvailable());
        assertEquals("Deep model – Siamese U-Net", engine.getDisplayName());
        assertEquals("no trained model loaded", engine.getStatusNote());

        BufferedImage img = solid(8, 8, Color.GREEN);
        assertThrows(IllegalStateException.class, () -> engine.detect(img, img));
        IllegalStateException ex = assertThrows(IllegalStateException.class, () -> EngineRunner.run(engine, img, img));
        assertTrue(ex.getMessage().contains("no trained model loaded"));
    }

    @Test
    @DisplayName("EngineRunner validates inputs and engine output size")
    void engineRunnerValidation() throws Exception {
        BufferedImage a = solid(10, 10, Color.GREEN);
        BufferedImage b = solid(12, 10, Color.GREEN);
        assertThrows(IllegalStateException.class, () -> EngineRunner.run(null, a, a));
        assertThrows(IllegalArgumentException.class, () -> EngineRunner.run(new ClassicalBaselineEngine(), a, b));

        ChangeDetectionEngine wrongSize = new ChangeDetectionEngine() {
            @Override public String getId() { return "bad"; }
            @Override public String getDisplayName() { return "Bad engine"; }
            @Override public String getShortLabel() { return "BAD"; }
            @Override public boolean isAvailable() { return true; }
            @Override public String getStatusNote() { return "test"; }
            @Override public EngineResult detect(BufferedImage before, BufferedImage after) {
                return EngineResult.fromMask("bad", "Bad engine", new boolean[3][3], 1);
            }
        };
        assertThrows(IllegalStateException.class, () -> EngineRunner.run(wrongSize, a, a));

        EngineResult ok = EngineRunner.run(new ClassicalBaselineEngine(), a, a);
        assertEquals(0, ok.getChangedPixels());
        assertEquals(EngineResult.ChangeLevel.NONE, ok.getChangeLevel());
    }

    @Test
    @DisplayName("A mask-only engine plugs in without UI changes (regions derived from mask)")
    void pluggableMaskOnlyEngine() throws Exception {
        ChangeDetectionEngine maskOnly = new ChangeDetectionEngine() {
            @Override public String getId() { return "mask-only"; }
            @Override public String getDisplayName() { return "Mask-only test engine"; }
            @Override public String getShortLabel() { return "TEST"; }
            @Override public boolean isAvailable() { return true; }
            @Override public String getStatusNote() { return "test"; }
            @Override public EngineResult detect(BufferedImage before, BufferedImage after) {
                boolean[][] m = new boolean[before.getHeight()][before.getWidth()];
                m[0][0] = true;
                m[4][4] = true;
                return EngineResult.fromMask(getId(), getDisplayName(), m, 3);
            }
        };
        BufferedImage img = solid(6, 6, Color.BLUE);
        EngineResult r = EngineRunner.run(maskOnly, img, img);
        assertEquals("Mask-only test engine", r.getEngineName());
        assertEquals(2, r.getChangedPixels());
        assertEquals(2, r.getRegionCount());
        assertEquals(List.of(new Rectangle(0, 0, 1, 1), new Rectangle(4, 4, 1, 1)), r.getRegions());
    }
}
