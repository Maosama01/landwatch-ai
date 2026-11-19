package com.landwatch.ui;

import com.landwatch.geospatial.ChangeVisualizer;
import com.landwatch.ui.engine.EngineResult;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that the dashboard's three visualisations are built from the engine mask and are distinct.
 */
class ChangeVisualizationSetTest {

    private static BufferedImage solid(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    private static EngineResult resultWithBlock() {
        boolean[][] mask = new boolean[12][16];
        for (int y = 4; y < 8; y++) {
            for (int x = 5; x < 10; x++) {
                mask[y][x] = true;
            }
        }
        return EngineResult.fromMask("test", "Test engine", mask, 1);
    }

    @Test
    @DisplayName("All three modes are built from the mask at the input size")
    void buildsAllModes() {
        EngineResult r = resultWithBlock();
        BufferedImage after = solid(16, 12, new Color(34, 110, 48));
        ChangeVisualizationSet set = ChangeVisualizationSet.build(r, after);

        for (VisualizationMode mode : VisualizationMode.values()) {
            BufferedImage img = set.get(mode);
            assertNotNull(img, mode + " image");
            assertEquals(16, img.getWidth());
            assertEquals(12, img.getHeight());
        }
        assertSame(set.getBinaryMask(), set.get(VisualizationMode.BINARY_MASK));
        assertSame(set.getRedOverlay(), set.get(VisualizationMode.RED_OVERLAY));
        assertSame(set.getThematicMap(), set.get(VisualizationMode.CHANGE_MAP));

        // Binary mask is exactly the engine mask
        for (int y = 0; y < 12; y++) {
            for (int x = 0; x < 16; x++) {
                int px = set.getBinaryMask().getRGB(x, y) & 0xFFFFFF;
                assertEquals(r.isChanged(x, y) ? 0xFFFFFF : 0x000000, px);
            }
        }
        // Thematic fill on a changed pixel
        assertEquals(ChangeVisualizer.THEMATIC_CHANGE_RGB, set.getThematicMap().getRGB(7, 5) & 0xFFFFFF);
    }

    @Test
    @DisplayName("Size mismatch between engine mask and post-event image is rejected")
    void rejectsSizeMismatch() {
        EngineResult r = resultWithBlock();
        assertThrows(IllegalArgumentException.class,
                () -> ChangeVisualizationSet.build(r, solid(10, 10, Color.GRAY)));
    }

    @Test
    @DisplayName("Every mode has a display name, description and legend whose colours match the renderer")
    void legendsMatchRenderer() {
        for (VisualizationMode mode : VisualizationMode.values()) {
            assertFalse(mode.getDisplayName().isBlank());
            assertFalse(mode.getDescription().isBlank());
            assertFalse(mode.getLegend().isEmpty());
        }
        assertEquals("#000000", VisualizationMode.BINARY_MASK.getLegend().get(0).cssColor());
        assertEquals("#ffffff", VisualizationMode.BINARY_MASK.getLegend().get(1).cssColor());
        assertTrue(VisualizationMode.CHANGE_MAP.getLegend().stream()
                .anyMatch(e -> String.format("#%06x", ChangeVisualizer.THEMATIC_CHANGE_RGB).equals(e.cssColor())));
        assertTrue(VisualizationMode.RED_OVERLAY.getLegend().stream()
                .anyMatch(e -> String.format("#%06x", ChangeVisualizer.OVERLAY_BOX_RGB).equals(e.cssColor())));
    }
}
