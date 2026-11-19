package com.landwatch.geospatial;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Generates and exports visual change maps, binary masks, and annotated overlays.
 */
public class ChangeVisualizer {

    private static final Color COLOR_UNCHANGED = new Color(15, 23, 42, 255);       // Deep slate (#0f172a)
    private static final Color COLOR_CHANGED = new Color(239, 68, 68, 255);        // Vibrant red (#ef4444)
    private static final Color COLOR_OVERLAY_RED = new Color(239, 68, 68, 170);    // Semi-transparent red
    private static final Color COLOR_BOUNDING_BOX = new Color(245, 158, 11, 230);  // Amber (#f59e0b)


    // ---------------------------------------------------------------------
    // Public colour constants (0xRRGGBB) shared with the dashboard legend.
    // ---------------------------------------------------------------------

    /** Binary mask: unchanged pixel = pure black. */
    public static final int BW_UNCHANGED_RGB = 0x000000;
    /** Binary mask: changed pixel = pure white. */
    public static final int BW_CHANGED_RGB = 0xFFFFFF;

    /** Red overlay: tint colour blended over changed pixels (same as the existing overlay). */
    public static final int OVERLAY_TINT_RGB = 0xEF4444;
    /** Red overlay: amber bounding-box colour (same as the existing overlay). */
    public static final int OVERLAY_BOX_RGB = 0xF59E0B;

    /** Thematic change map: fill colour of changed pixels (cyan). */
    public static final int THEMATIC_CHANGE_RGB = 0x22D3EE;
    /** Thematic change map: 1-px outline drawn on unchanged pixels that touch a change region (yellow). */
    public static final int THEMATIC_OUTLINE_RGB = 0xFDE047;
    /** Thematic change map: brightness factor applied to the grayscale post-event context. */
    public static final double THEMATIC_CONTEXT_DIM = 0.6;

    /** File-name suffixes used by {@link #saveVisualizationSet}. */
    public static final String SUFFIX_BINARY_MASK_BW = "_binary_mask_bw.png";
    public static final String SUFFIX_RED_OVERLAY = "_red_overlay.png";
    public static final String SUFFIX_THEMATIC_MAP = "_thematic_change_map.png";

    /**
     * Creates a high-contrast binary change map.
     * Unchanged pixels = dark slate; Changed pixels = vibrant red.
     */
    public static BufferedImage createBinaryChangeMap(boolean[][] mask, int width, int height) {
        BufferedImage map = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int rgbUnchanged = COLOR_UNCHANGED.getRGB();
        int rgbChanged = COLOR_CHANGED.getRGB();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                map.setRGB(x, y, mask[y][x] ? rgbChanged : rgbUnchanged);
            }
        }
        return map;
    }

    /**
     * Creates an annotated overlay on the post-event (AFTER) image.
     * Changed pixels are tinted in translucent red, and significant encroachment clusters
     * are outlined with yellow/amber bounding boxes.
     */
    public static BufferedImage createOverlayChangeMap(BufferedImage baseAfterImage,
                                                       boolean[][] mask,
                                                       List<Rectangle> boundingBoxes) {
        int width = baseAfterImage.getWidth();
        int height = baseAfterImage.getHeight();

        BufferedImage overlay = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D g = overlay.createGraphics();
        try {
            // Draw base AFTER image
            g.drawImage(baseAfterImage, 0, 0, null);

            // Draw tinted changed pixels
            int overlayRgb = COLOR_OVERLAY_RED.getRGB();
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    if (mask[y][x]) {
                        // Blend red tint over base pixel
                        int baseRgb = baseAfterImage.getRGB(x, y);
                        int blended = blendColors(baseRgb, overlayRgb, 0.65);
                        overlay.setRGB(x, y, blended);
                    }
                }
            }

            // Draw bounding boxes around significant clusters
            if (boundingBoxes != null && !boundingBoxes.isEmpty()) {
                g.setColor(COLOR_BOUNDING_BOX);
                g.setStroke(new BasicStroke(2.0f));
                for (Rectangle box : boundingBoxes) {
                    g.drawRect(box.x, box.y, box.width, box.height);
                }
            }
        } finally {
            g.dispose();
        }

        return overlay;
    }

    /**
     * Blends a foreground color over a background color with given foreground alpha weight.
     */
    private static int blendColors(int bgRgb, int fgRgb, double fgWeight) {
        int bgR = (bgRgb >> 16) & 0xFF;
        int bgG = (bgRgb >> 8) & 0xFF;
        int bgB = bgRgb & 0xFF;

        int fgR = (fgRgb >> 16) & 0xFF;
        int fgG = (fgRgb >> 8) & 0xFF;
        int fgB = fgRgb & 0xFF;

        int r = (int) Math.round((1 - fgWeight) * bgR + fgWeight * fgR);
        int g = (int) Math.round((1 - fgWeight) * bgG + fgWeight * fgG);
        int b = (int) Math.round((1 - fgWeight) * bgB + fgWeight * fgB);

        return (0xFF << 24) | (r << 16) | (g << 8) | b;
    }

    // ---------------------------------------------------------------------
    // Additional visualisation modes (dashboard: Binary Mask / Change Map)
    // ---------------------------------------------------------------------

    /**
     * Creates a strict black-and-white binary mask.
     * Unchanged pixels are pure black (0,0,0); changed pixels are pure white (255,255,255).
     *
     * @param mask   change mask indexed as {@code mask[y][x]}
     * @param width  image width (must match mask row length)
     * @param height image height (must match mask row count)
     * @return TYPE_INT_RGB image of size width x height
     */
    public static BufferedImage createBlackWhiteMask(boolean[][] mask, int width, int height) {
        validateMask(mask, width, height);
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                img.setRGB(x, y, mask[y][x] ? BW_CHANGED_RGB : BW_UNCHANGED_RGB);
            }
        }
        return img;
    }

    /**
     * Convenience overload that takes the dimensions from the mask itself.
     */
    public static BufferedImage createBlackWhiteMask(boolean[][] mask) {
        if (mask == null || mask.length == 0 || mask[0] == null) {
            throw new IllegalArgumentException("Mask must be a non-empty boolean[height][width] array.");
        }
        return createBlackWhiteMask(mask, mask[0].length, mask.length);
    }

    /**
     * Creates a thematic change map that looks clearly different from the binary mask and red overlay:
     * <ul>
     *   <li>Unchanged pixels: the post-event image as dimmed grayscale context
     *       (gray = round(luminance x {@value #THEMATIC_CONTEXT_DIM}), so R = G = B).</li>
     *   <li>Changed pixels: solid cyan fill ({@link #THEMATIC_CHANGE_RGB}).</li>
     *   <li>Region outline: unchanged pixels that touch a changed pixel (8-neighbourhood) are drawn in
     *       yellow ({@link #THEMATIC_OUTLINE_RGB}), so every change region gets a closed 1-px contour
     *       without hiding any changed pixel.</li>
     * </ul>
     *
     * @param baseAfterImage post-event image used as context
     * @param mask           change mask indexed as {@code mask[y][x]}, same size as the image
     * @return TYPE_INT_RGB thematic map
     */
    public static BufferedImage createThematicChangeMap(BufferedImage baseAfterImage, boolean[][] mask) {
        if (baseAfterImage == null) {
            throw new IllegalArgumentException("Post-event image must not be null.");
        }
        int width = baseAfterImage.getWidth();
        int height = baseAfterImage.getHeight();
        validateMask(mask, width, height);

        BufferedImage map = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb;
                if (mask[y][x]) {
                    rgb = THEMATIC_CHANGE_RGB;
                } else if (touchesChange(mask, x, y, width, height)) {
                    rgb = THEMATIC_OUTLINE_RGB;
                } else {
                    rgb = dimmedGray(baseAfterImage.getRGB(x, y));
                }
                map.setRGB(x, y, rgb);
            }
        }
        return map;
    }

    /**
     * Writes the three dashboard visualisations to {@code outputDirectory} using clear file names:
     * {@code <baseName>_binary_mask_bw.png}, {@code <baseName>_red_overlay.png} and
     * {@code <baseName>_thematic_change_map.png}. Null images are skipped.
     *
     * @return the files that were written, in the order mask, overlay, thematic map
     */
    public static List<File> saveVisualizationSet(File outputDirectory, String baseName,
                                                  BufferedImage blackWhiteMask,
                                                  BufferedImage redOverlay,
                                                  BufferedImage thematicMap) throws IOException {
        if (outputDirectory == null) {
            throw new IllegalArgumentException("Output directory must not be null.");
        }
        if (baseName == null || baseName.isBlank()) {
            throw new IllegalArgumentException("Base file name must not be blank.");
        }
        if (!outputDirectory.exists() && !outputDirectory.mkdirs()) {
            throw new IOException("Could not create output directory: " + outputDirectory);
        }
        String safeBase = baseName.replaceAll("[^A-Za-z0-9._-]", "_");

        List<File> written = new ArrayList<>();
        writePng(blackWhiteMask, new File(outputDirectory, safeBase + SUFFIX_BINARY_MASK_BW), written);
        writePng(redOverlay, new File(outputDirectory, safeBase + SUFFIX_RED_OVERLAY), written);
        writePng(thematicMap, new File(outputDirectory, safeBase + SUFFIX_THEMATIC_MAP), written);
        return written;
    }

    private static void writePng(BufferedImage image, File file, List<File> written) throws IOException {
        if (image == null) {
            return;
        }
        if (!ImageIO.write(image, "png", file)) {
            throw new IOException("No PNG writer available for " + file);
        }
        written.add(file);
    }

    /** True if any 8-neighbour of (x, y) inside the image is a changed pixel. */
    private static boolean touchesChange(boolean[][] mask, int x, int y, int width, int height) {
        for (int dy = -1; dy <= 1; dy++) {
            int ny = y + dy;
            if (ny < 0 || ny >= height) continue;
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                if (nx < 0 || nx >= width) continue;
                if (mask[ny][nx]) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Converts an RGB pixel to dimmed grayscale using ITU-R BT.601 luminance. */
    private static int dimmedGray(int rgb) {
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        double luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        int v = (int) Math.round(luminance * THEMATIC_CONTEXT_DIM);
        v = Math.max(0, Math.min(255, v));
        return (v << 16) | (v << 8) | v;
    }

    private static void validateMask(boolean[][] mask, int width, int height) {
        if (mask == null) {
            throw new IllegalArgumentException("Mask must not be null.");
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Image dimensions must be positive: " + width + "x" + height);
        }
        if (mask.length != height) {
            throw new IllegalArgumentException("Mask has " + mask.length + " rows, expected " + height + ".");
        }
        for (int y = 0; y < height; y++) {
            if (mask[y] == null || mask[y].length != width) {
                throw new IllegalArgumentException("Mask row " + y + " does not have width " + width + ".");
            }
        }
    }
}
