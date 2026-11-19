package com.landwatch.ui;

/** Geometry shared by the image frames and the swipe comparison (no JavaFX types, so it is unit-testable). */
public final class ImageFit {

    private ImageFit() {
    }

    /**
     * Displayed size of an image of {@code imgW x imgH} fitted inside {@code boxW x boxH}, keeping its
     * aspect ratio (scaled up or down). Returns {0, 0} when either size is not positive.
     */
    public static double[] fit(double imgW, double imgH, double boxW, double boxH) {
        if (imgW <= 0 || imgH <= 0 || boxW <= 0 || boxH <= 0) {
            return new double[] {0, 0};
        }
        double scale = Math.min(boxW / imgW, boxH / imgH);
        return new double[] {imgW * scale, imgH * scale};
    }

    /**
     * Swipe-divider position, clamped to 0..1, for a pointer at {@code x} over an image whose left edge is
     * at {@code left} and whose displayed width is {@code width}; 0.5 when the width is not positive.
     */
    public static double positionAt(double x, double left, double width) {
        if (width <= 0) {
            return 0.5;
        }
        return Math.max(0, Math.min(1, (x - left) / width));
    }
}
