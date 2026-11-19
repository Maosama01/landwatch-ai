package com.landwatch.geospatial;

import java.awt.image.BufferedImage;

/**
 * Image normalization and photometric conversion utilities for satellite imagery.
 */
public class ImageNormalizer {

    // Standard ITU-R BT.601 luma coefficients
    public static final double LUMA_RED = 0.299;
    public static final double LUMA_GREEN = 0.587;
    public static final double LUMA_BLUE = 0.114;

    /**
     * Converts a BufferedImage into a 2D grayscale luminance matrix.
     * Formula: Y = 0.299 * R + 0.587 * G + 0.114 * B
     *
     * @param image Input image
     * @return 2D double array of dimensions [height][width] with luminance values in range [0.0, 255.0]
     */
    public static double[][] toGrayscaleMatrix(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("Input image must not be null.");
        }

        int width = image.getWidth();
        int height = image.getHeight();
        double[][] grayMatrix = new double[height][width];

        int[] rgbBuffer = new int[width * height];
        image.getRGB(0, 0, width, height, rgbBuffer, 0, width);

        for (int y = 0; y < height; y++) {
            int rowOffset = y * width;
            for (int x = 0; x < width; x++) {
                int rgb = rgbBuffer[rowOffset + x];
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                double luma = (LUMA_RED * r) + (LUMA_GREEN * g) + (LUMA_BLUE * b);
                grayMatrix[y][x] = luma;
            }
        }

        return grayMatrix;
    }

    /**
     * Converts a 2D grayscale matrix back into a grayscale BufferedImage.
     *
     * @param grayMatrix Luminance matrix with values in [0, 255]
     * @return Grayscale BufferedImage (TYPE_BYTE_GRAY)
     */
    public static BufferedImage toGrayscaleImage(double[][] grayMatrix) {
        if (grayMatrix == null || grayMatrix.length == 0 || grayMatrix[0].length == 0) {
            throw new IllegalArgumentException("Grayscale matrix must not be empty.");
        }

        int height = grayMatrix.length;
        int width = grayMatrix[0].length;
        BufferedImage grayImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int val = Math.max(0, Math.min(255, (int) Math.round(grayMatrix[y][x])));
                int rgb = (val << 16) | (val << 8) | val;
                grayImage.setRGB(x, y, rgb);
            }
        }

        return grayImage;
    }

    /**
     * Normalizes a matrix to the range [0.0, 1.0].
     */
    public static double[][] normalizeToUnit(double[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("Matrix must not be empty.");
        }

        int height = matrix.length;
        int width = matrix[0].length;
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double val = matrix[y][x];
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }

        double[][] normalized = new double[height][width];
        double range = max - min;
        if (range < 1e-9) {
            return normalized; // All zeros if constant
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                normalized[y][x] = (matrix[y][x] - min) / range;
            }
        }

        return normalized;
    }
}
