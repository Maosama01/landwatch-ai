package com.landwatch.ui;

import java.awt.image.BufferedImage;

/**
 * Geometry shared by the image frames and the swipe comparison, plus the nearest-neighbour enlargement of
 * small tiles (no JavaFX types, so it is unit-testable).
 */
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

    /**
     * Largest whole-number factor that keeps an image of {@code w x h} at most {@code target} px on its
     * longer side; at least 1 (images already that large are not enlarged).
     */
    public static int nearestScale(int w, int h, int target) {
        int longer = Math.max(w, h);
        return longer <= 0 ? 1 : Math.max(1, target / longer);
    }

    /**
     * Enlarges an image by a whole-number factor with nearest-neighbour sampling: every source pixel
     * becomes a {@code factor x factor} block, so a 64 x 64 EuroSAT tile shown large keeps its real
     * pixels instead of being blurred (JavaFX's own {@code ImageView.setSmooth(false)} is not honoured by
     * every rendering pipeline). Returns {@code src} itself when {@code factor <= 1}.
     */
    public static BufferedImage enlargeNearest(BufferedImage src, int factor) {
        if (factor <= 1) {
            return src;
        }
        int w = src.getWidth();
        int h = src.getHeight();
        int[] in = src.getRGB(0, 0, w, h, null, 0, w);
        int outW = w * factor;
        int outH = h * factor;
        int[] out = new int[outW * outH];
        for (int y = 0; y < outH; y++) {
            int row = (y / factor) * w;
            for (int x = 0; x < outW; x++) {
                out[y * outW + x] = in[row + x / factor];
            }
        }
        BufferedImage img = new BufferedImage(outW, outH, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, outW, outH, out, 0, outW);
        return img;
    }
}
