package com.landwatch.ui;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class ImageFitTest {

    @Test
    @DisplayName("A square 1024 px image fits a wide frame by its height, a tall frame by its width")
    void fitKeepsAspectRatio() {
        assertArrayEquals(new double[] {380, 380}, ImageFit.fit(1024, 1024, 460, 380), 1e-9);
        assertArrayEquals(new double[] {200, 200}, ImageFit.fit(1024, 1024, 200, 300), 1e-9);
        // The 400 x 300 demo image is scaled up to fill a 460 x 380 frame by its width.
        assertArrayEquals(new double[] {460, 345}, ImageFit.fit(400, 300, 460, 380), 1e-9);
    }

    @Test
    @DisplayName("Nothing to fit gives a zero size instead of NaN or infinity")
    void fitDegenerateSizes() {
        assertArrayEquals(new double[] {0, 0}, ImageFit.fit(0, 300, 460, 380), 0);
        assertArrayEquals(new double[] {0, 0}, ImageFit.fit(400, 300, 0, 380), 0);
        assertArrayEquals(new double[] {0, 0}, ImageFit.fit(400, 300, 460, -1), 0);
    }

    @Test
    @DisplayName("The swipe divider follows the pointer across the image and stops at its edges")
    void dividerPosition() {
        assertEquals(0.0, ImageFit.positionAt(40, 40, 380), 1e-12);
        assertEquals(0.5, ImageFit.positionAt(230, 40, 380), 1e-12);
        assertEquals(1.0, ImageFit.positionAt(420, 40, 380), 1e-12);
        assertEquals(0.0, ImageFit.positionAt(-50, 40, 380), 1e-12);
        assertEquals(1.0, ImageFit.positionAt(900, 40, 380), 1e-12);
        assertEquals(0.5, ImageFit.positionAt(100, 40, 0), 1e-12);
    }

    @Test
    @DisplayName("Small tiles are enlarged by a whole factor, each source pixel becoming an exact block")
    void nearestEnlargement() {
        assertEquals(8, ImageFit.nearestScale(64, 64, 512));
        assertEquals(2, ImageFit.nearestScale(64, 64, 128));
        assertEquals(1, ImageFit.nearestScale(1024, 1024, 512));
        assertEquals(1, ImageFit.nearestScale(0, 0, 512));

        java.awt.image.BufferedImage src = new java.awt.image.BufferedImage(2, 2, java.awt.image.BufferedImage.TYPE_INT_RGB);
        src.setRGB(0, 0, 0xff0000);
        src.setRGB(1, 0, 0x00ff00);
        src.setRGB(0, 1, 0x0000ff);
        src.setRGB(1, 1, 0xffffff);
        java.awt.image.BufferedImage big = ImageFit.enlargeNearest(src, 3);
        assertEquals(6, big.getWidth());
        assertEquals(6, big.getHeight());
        for (int y = 0; y < 6; y++) {
            for (int x = 0; x < 6; x++) {
                assertEquals(src.getRGB(x / 3, y / 3), big.getRGB(x, y), "pixel " + x + "," + y);
            }
        }
        org.junit.jupiter.api.Assertions.assertSame(src, ImageFit.enlargeNearest(src, 1));
    }
}
