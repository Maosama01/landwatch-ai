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
}
