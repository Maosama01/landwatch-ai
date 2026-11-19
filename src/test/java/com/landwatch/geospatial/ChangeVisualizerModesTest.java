package com.landwatch.geospatial;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Pixel-level tests for the dashboard visualisation modes added to {@link ChangeVisualizer}:
 * black/white binary mask and thematic change map (plus file export).
 */
class ChangeVisualizerModesTest {

    private static int rgb(BufferedImage img, int x, int y) {
        return img.getRGB(x, y) & 0xFFFFFF;
    }

    private static BufferedImage solid(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    /** 7x7 mask with a 3x3 changed block at x,y in [2..4]. */
    private static boolean[][] centreBlockMask() {
        boolean[][] mask = new boolean[7][7];
        for (int y = 2; y <= 4; y++) {
            for (int x = 2; x <= 4; x++) {
                mask[y][x] = true;
            }
        }
        return mask;
    }

    // ---------------------------------------------------------------- B/W mask

    @Test
    @DisplayName("B/W mask: changed = pure white, unchanged = pure black")
    void blackWhiteMaskPixelColours() {
        boolean[][] mask = {
                {true, false, false, true},
                {false, true, false, false},
                {false, false, false, true}
        };
        BufferedImage img = ChangeVisualizer.createBlackWhiteMask(mask, 4, 3);

        assertEquals(4, img.getWidth());
        assertEquals(3, img.getHeight());
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 4; x++) {
                int expected = mask[y][x] ? 0xFFFFFF : 0x000000;
                assertEquals(expected, rgb(img, x, y), "pixel (" + x + "," + y + ")");
            }
        }
        assertEquals(255, (img.getRGB(0, 0) >>> 24) & 0xFF, "mask must be fully opaque");
    }

    @Test
    @DisplayName("B/W mask: all-unchanged is all black, all-changed is all white")
    void blackWhiteMaskUniform() {
        boolean[][] none = new boolean[5][6];
        boolean[][] all = new boolean[5][6];
        for (boolean[] row : all) java.util.Arrays.fill(row, true);

        BufferedImage black = ChangeVisualizer.createBlackWhiteMask(none);
        BufferedImage white = ChangeVisualizer.createBlackWhiteMask(all);
        assertEquals(6, black.getWidth());
        assertEquals(5, black.getHeight());
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 6; x++) {
                assertEquals(0x000000, rgb(black, x, y));
                assertEquals(0xFFFFFF, rgb(white, x, y));
            }
        }
    }

    @Test
    @DisplayName("B/W mask: mask/dimension mismatch is rejected")
    void blackWhiteMaskRejectsBadInput() {
        boolean[][] mask = new boolean[3][4];
        assertThrows(IllegalArgumentException.class, () -> ChangeVisualizer.createBlackWhiteMask(mask, 5, 3));
        assertThrows(IllegalArgumentException.class, () -> ChangeVisualizer.createBlackWhiteMask(mask, 4, 2));
        assertThrows(IllegalArgumentException.class, () -> ChangeVisualizer.createBlackWhiteMask(null, 4, 3));
        boolean[][] ragged = {new boolean[4], new boolean[3]};
        assertThrows(IllegalArgumentException.class, () -> ChangeVisualizer.createBlackWhiteMask(ragged, 4, 2));
    }

    // ---------------------------------------------------------------- thematic map

    @Test
    @DisplayName("Thematic map: cyan fill on changed pixels, yellow outline around, dimmed gray elsewhere")
    void thematicMapPixelColours() {
        Color base = new Color(200, 100, 50);
        BufferedImage after = solid(7, 7, base);
        boolean[][] mask = centreBlockMask();

        BufferedImage map = ChangeVisualizer.createThematicChangeMap(after, mask);
        assertEquals(7, map.getWidth());
        assertEquals(7, map.getHeight());

        double lum = 0.299 * 200 + 0.587 * 100 + 0.114 * 50;
        int g = (int) Math.round(lum * ChangeVisualizer.THEMATIC_CONTEXT_DIM);
        int expectedGray = (g << 16) | (g << 8) | g;

        for (int y = 0; y < 7; y++) {
            for (int x = 0; x < 7; x++) {
                boolean changed = mask[y][x];
                boolean ring = !changed && x >= 1 && x <= 5 && y >= 1 && y <= 5;
                int expected = changed ? ChangeVisualizer.THEMATIC_CHANGE_RGB
                        : ring ? ChangeVisualizer.THEMATIC_OUTLINE_RGB
                        : expectedGray;
                assertEquals(expected, rgb(map, x, y), "pixel (" + x + "," + y + ")");
            }
        }
        // Context really is grayscale and darker than the original
        int ctx = rgb(map, 0, 0);
        assertEquals((ctx >> 16) & 0xFF, (ctx >> 8) & 0xFF);
        assertEquals((ctx >> 8) & 0xFF, ctx & 0xFF);
        assertTrue((ctx & 0xFF) < lum, "context must be dimmed");
    }

    @Test
    @DisplayName("Thematic map: empty mask gives pure dimmed grayscale context (no fill, no outline)")
    void thematicMapNoChange() {
        BufferedImage after = solid(5, 4, Color.WHITE);
        BufferedImage map = ChangeVisualizer.createThematicChangeMap(after, new boolean[4][5]);
        int g = (int) Math.round(255 * ChangeVisualizer.THEMATIC_CONTEXT_DIM);
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 5; x++) {
                assertEquals((g << 16) | (g << 8) | g, rgb(map, x, y));
            }
        }
    }

    @Test
    @DisplayName("Thematic map: region touching the image border is handled without errors")
    void thematicMapRegionAtBorder() {
        BufferedImage after = solid(4, 4, Color.GRAY);
        boolean[][] mask = new boolean[4][4];
        mask[0][0] = true;
        BufferedImage map = ChangeVisualizer.createThematicChangeMap(after, mask);
        assertEquals(ChangeVisualizer.THEMATIC_CHANGE_RGB, rgb(map, 0, 0));
        assertEquals(ChangeVisualizer.THEMATIC_OUTLINE_RGB, rgb(map, 1, 0));
        assertEquals(ChangeVisualizer.THEMATIC_OUTLINE_RGB, rgb(map, 0, 1));
        assertEquals(ChangeVisualizer.THEMATIC_OUTLINE_RGB, rgb(map, 1, 1));
        assertNotEquals(ChangeVisualizer.THEMATIC_OUTLINE_RGB, rgb(map, 2, 2));
    }

    @Test
    @DisplayName("Thematic map: null image or size mismatch is rejected")
    void thematicMapRejectsBadInput() {
        assertThrows(IllegalArgumentException.class,
                () -> ChangeVisualizer.createThematicChangeMap(null, new boolean[2][2]));
        assertThrows(IllegalArgumentException.class,
                () -> ChangeVisualizer.createThematicChangeMap(solid(3, 3, Color.RED), new boolean[2][3]));
    }

    // ---------------------------------------------------------------- distinctness

    @Test
    @DisplayName("The three modes render the same changed pixel in three different colours")
    void modesAreDistinct() {
        BufferedImage after = solid(7, 7, new Color(34, 110, 48));
        boolean[][] mask = centreBlockMask();
        BufferedImage bw = ChangeVisualizer.createBlackWhiteMask(mask, 7, 7);
        BufferedImage overlay = ChangeVisualizer.createOverlayChangeMap(after, mask, List.of());
        BufferedImage thematic = ChangeVisualizer.createThematicChangeMap(after, mask);

        int bwPx = rgb(bw, 3, 3);
        int ovPx = rgb(overlay, 3, 3);
        int thPx = rgb(thematic, 3, 3);
        assertEquals(0xFFFFFF, bwPx);
        assertEquals(ChangeVisualizer.THEMATIC_CHANGE_RGB, thPx);
        assertNotEquals(bwPx, ovPx);
        assertNotEquals(bwPx, thPx);
        assertNotEquals(ovPx, thPx);

        // Overlay changed pixel is red-dominant; unchanged overlay pixel keeps the post image colour.
        assertTrue(((ovPx >> 16) & 0xFF) > ((ovPx >> 8) & 0xFF) && ((ovPx >> 16) & 0xFF) > (ovPx & 0xFF));
        assertEquals(new Color(34, 110, 48).getRGB() & 0xFFFFFF, rgb(overlay, 0, 0));
    }

    @Test
    @DisplayName("Existing slate/red binary change map behaviour is unchanged")
    void legacyBinaryChangeMapUnchanged() {
        boolean[][] mask = {{true, false}};
        BufferedImage legacy = ChangeVisualizer.createBinaryChangeMap(mask, 2, 1);
        assertEquals(0xEF4444, rgb(legacy, 0, 0));
        assertEquals(0x0F172A, rgb(legacy, 1, 0));
    }

    @Test
    @DisplayName("Red overlay draws amber bounding boxes")
    void overlayDrawsAmberBox() {
        BufferedImage after = solid(20, 20, Color.BLACK);
        boolean[][] mask = new boolean[20][20];
        BufferedImage overlay = ChangeVisualizer.createOverlayChangeMap(after, mask, List.of(new Rectangle(5, 5, 8, 8)));
        int px = rgb(overlay, 5, 9);
        int r = (px >> 16) & 0xFF, g = (px >> 8) & 0xFF, b = px & 0xFF;
        assertTrue(r > 150 && g > 80 && b < 60, "box edge should be amber, got #" + Integer.toHexString(px));
    }

    // ---------------------------------------------------------------- export

    @Test
    @DisplayName("saveVisualizationSet writes the three PNGs with clear names and exact B/W pixels")
    void saveVisualizationSetWritesFiles(@TempDir Path tmp) throws Exception {
        BufferedImage after = solid(7, 7, Color.GRAY);
        boolean[][] mask = centreBlockMask();
        BufferedImage bw = ChangeVisualizer.createBlackWhiteMask(mask, 7, 7);
        BufferedImage overlay = ChangeVisualizer.createOverlayChangeMap(after, mask, List.of());
        BufferedImage thematic = ChangeVisualizer.createThematicChangeMap(after, mask);

        File outDir = tmp.resolve("results").toFile();
        List<File> files = ChangeVisualizer.saveVisualizationSet(outDir, "run_x", bw, overlay, thematic);

        assertEquals(3, files.size());
        assertEquals("run_x_binary_mask_bw.png", files.get(0).getName());
        assertEquals("run_x_red_overlay.png", files.get(1).getName());
        assertEquals("run_x_thematic_change_map.png", files.get(2).getName());
        for (File f : files) {
            assertTrue(f.isFile() && f.length() > 0, f + " must exist");
        }
        BufferedImage reread = ImageIO.read(files.get(0));
        assertEquals(0xFFFFFF, rgb(reread, 3, 3));
        assertEquals(0x000000, rgb(reread, 0, 0));
        assertEquals(ChangeVisualizer.THEMATIC_CHANGE_RGB, rgb(ImageIO.read(files.get(2)), 3, 3));
    }
}
