package com.landwatch.deeplearning;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Conversions between {@link BufferedImage}s and the flat float arrays fed to the network.
 *
 * <p>Layout convention: images become channel-first ("CHW") arrays, i.e. all red values row by
 * row, then all green, then all blue: {@code index = c * H * W + y * W + x}. Values are scaled to
 * [0, 1] by dividing by 255; per-channel standardisation is applied separately by
 * {@link Normalization}.</p>
 */
public final class ImageTensors {

    private ImageTensors() {
    }

    /** Reads an image file, failing loudly if ImageIO cannot decode it. */
    public static BufferedImage read(Path file) throws IOException {
        BufferedImage img = ImageIO.read(file.toFile());
        if (img == null) {
            throw new IOException("Cannot decode image " + file);
        }
        return img;
    }

    /**
     * RGB image to a CHW float array with values in [0, 1]. Works for any image type (alpha is
     * ignored, grayscale images are replicated into three channels by Java's colour model).
     */
    public static float[] toChw01(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        int plane = w * h;
        int[] argb = img.getRGB(0, 0, w, h, null, 0, w);
        float[] out = new float[3 * plane];
        for (int i = 0; i < plane; i++) {
            int p = argb[i];
            out[i] = ((p >> 16) & 0xFF) / 255f;
            out[plane + i] = ((p >> 8) & 0xFF) / 255f;
            out[2 * plane + i] = (p & 0xFF) / 255f;
        }
        return out;
    }

    /**
     * Change mask from a label image: 1 where the label value is &gt; 0, else 0 (row-major, length
     * H*W). Single-band (grayscale) labels are read from the raster directly so that no colour
     * conversion can alter the values; for other image types any non-zero RGB value counts.
     */
    public static float[] toBinaryMask(BufferedImage label) {
        int w = label.getWidth();
        int h = label.getHeight();
        float[] out = new float[w * h];
        Raster raster = label.getRaster();
        if (raster.getNumBands() == 1) {
            int[] samples = raster.getSamples(0, 0, w, h, 0, (int[]) null);
            for (int i = 0; i < samples.length; i++) {
                out[i] = samples[i] > 0 ? 1f : 0f;
            }
        } else {
            int[] argb = label.getRGB(0, 0, w, h, null, 0, w);
            for (int i = 0; i < argb.length; i++) {
                out[i] = (argb[i] & 0xFFFFFF) != 0 ? 1f : 0f;
            }
        }
        return out;
    }

    /** Converts a row-major 0/1 float mask to {@code boolean[y][x]}. */
    public static boolean[][] toBooleanMask(float[] mask, int width, int height) {
        boolean[][] out = new boolean[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                out[y][x] = mask[y * width + x] > 0.5f;
            }
        }
        return out;
    }
}
