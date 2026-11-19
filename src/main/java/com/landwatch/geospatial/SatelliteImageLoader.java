package com.landwatch.geospatial;

import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

/**
 * Robust loader and validator for satellite imagery.
 * Supports standard satellite image formats (PNG, JPG, BMP, TIFF where ImageIO provider is present).
 */
public class SatelliteImageLoader {

    /**
     * Loads an image from a local file and converts it to a standard RGB BufferedImage.
     *
     * @param file image file to load
     * @return standardized BufferedImage (TYPE_INT_RGB)
     * @throws IOException if file cannot be read or decoded
     */
    public static BufferedImage load(File file) throws IOException {
        if (file == null || !file.exists()) {
            throw new IllegalArgumentException("Satellite image file does not exist or is null: " + file);
        }
        BufferedImage raw = ImageIO.read(file);
        if (raw == null) {
            throw new IOException("Unsupported image format or corrupt file: " + file.getName());
        }
        return toStandardRgb(raw);
    }

    /**
     * Loads an image from an InputStream.
     */
    public static BufferedImage load(InputStream inputStream) throws IOException {
        if (inputStream == null) {
            throw new IllegalArgumentException("InputStream must not be null.");
        }
        BufferedImage raw = ImageIO.read(inputStream);
        if (raw == null) {
            throw new IOException("Could not decode image from input stream.");
        }
        return toStandardRgb(raw);
    }

    /**
     * Loads an image from a URL (e.g. classpath resource).
     */
    public static BufferedImage load(URL url) throws IOException {
        if (url == null) {
            throw new IllegalArgumentException("URL must not be null.");
        }
        BufferedImage raw = ImageIO.read(url);
        if (raw == null) {
            throw new IOException("Could not decode image from URL: " + url);
        }
        return toStandardRgb(raw);
    }

    /**
     * Validates that both images are non-null and share identical dimensions.
     *
     * @param img1 Pre-event image
     * @param img2 Post-event image
     * @throws IllegalArgumentException if dimensions differ or either image is null
     */
    public static void validateDimensions(BufferedImage img1, BufferedImage img2) {
        if (img1 == null || img2 == null) {
            throw new IllegalArgumentException("Both satellite images must be provided.");
        }
        if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
            throw new IllegalArgumentException(String.format(
                    "Image dimension mismatch: Pre-event is %dx%d, but Post-event is %dx%d. " +
                    "Bi-temporal change detection requires spatially co-registered, identical dimensions.",
                    img1.getWidth(), img1.getHeight(), img2.getWidth(), img2.getHeight()));
        }
    }

    /**
     * Resizes an image to target dimensions using high-quality bilinear interpolation.
     */
    public static BufferedImage resize(BufferedImage source, int targetWidth, int targetHeight) {
        if (source == null) {
            throw new IllegalArgumentException("Source image must not be null.");
        }
        if (source.getWidth() == targetWidth && source.getHeight() == targetHeight) {
            return source;
        }

        BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resized.createGraphics();
        try {
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.drawImage(source, 0, 0, targetWidth, targetHeight, null);
        } finally {
            g2d.dispose();
        }
        return resized;
    }

    /**
     * Converts any BufferedImage format (indexed, CMYK, ARGB, etc.) to a standardized TYPE_INT_RGB.
     */
    public static BufferedImage toStandardRgb(BufferedImage source) {
        if (source.getType() == BufferedImage.TYPE_INT_RGB) {
            return source;
        }
        BufferedImage rgbImage = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = rgbImage.createGraphics();
        try {
            g.drawImage(source, 0, 0, null);
        } finally {
            g.dispose();
        }
        return rgbImage;
    }
}
