package com.landwatch.deeplearning.eurosat;

import com.landwatch.deeplearning.D4Augmentation;
import com.landwatch.deeplearning.ImageTensors;
import com.landwatch.deeplearning.Normalization;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The images of one split, decoded once and kept in memory as 8-bit RGB.
 *
 * <p>EuroSAT tiles are tiny (64 x 64 x 3 = 12,288 bytes), so the whole training split (18,900
 * images) needs only about 230 MB as bytes. Decoding every JPEG once up front is much faster than
 * decoding it again in every epoch. Bytes are converted to normalised floats only when a batch is
 * built.</p>
 *
 * <p>Pre-processing is exactly the one used at inference time by {@link EuroSatClassifier}: the
 * JPEG is decoded by {@code ImageIO}, each 8-bit value is divided by 255
 * ({@link ImageTensors#toChw01}) and then standardised per channel with the TRAIN mean and
 * standard deviation ({@link Normalization#applyInPlace}).</p>
 */
public final class EuroSatImages {

    private final List<EuroSatSample> samples;
    private final byte[][] pixels;
    private final int size;

    private EuroSatImages(List<EuroSatSample> samples, byte[][] pixels, int size) {
        this.samples = samples;
        this.pixels = pixels;
        this.size = size;
    }

    /**
     * Decodes the images of {@code samples} (paths relative to {@code root}) in parallel.
     *
     * @param expectedSize every image must be {@code expectedSize x expectedSize} pixels
     */
    public static EuroSatImages load(Path root, List<EuroSatSample> samples, int expectedSize, int threads)
            throws IOException {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("No samples to load");
        }
        byte[][] pixels = new byte[samples.size()][];
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
        try {
            List<Future<?>> parts = new ArrayList<>();
            int chunk = Math.max(64, samples.size() / (Math.max(1, threads) * 8));
            for (int start = 0; start < samples.size(); start += chunk) {
                final int from = start;
                final int to = Math.min(samples.size(), start + chunk);
                parts.add(pool.submit(() -> {
                    for (int i = from; i < to; i++) {
                        Path file = samples.get(i).resolve(root);
                        BufferedImage img = ImageTensors.read(file);
                        if (img.getWidth() != expectedSize || img.getHeight() != expectedSize) {
                            throw new IOException("Expected " + expectedSize + "x" + expectedSize + " image, got "
                                    + img.getWidth() + "x" + img.getHeight() + ": " + file);
                        }
                        pixels[i] = toChwBytes(img);
                    }
                    return null;
                }));
            }
            for (Future<?> f : parts) {
                f.get();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while decoding images", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof IOException io) {
                throw io;
            }
            throw new IOException("Failed to decode images", cause);
        } finally {
            pool.shutdownNow();
        }
        return new EuroSatImages(List.copyOf(samples), pixels, expectedSize);
    }

    /** RGB image to channel-first 8-bit values (same channel order and layout as {@link ImageTensors#toChw01}). */
    static byte[] toChwBytes(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        int plane = w * h;
        int[] argb = img.getRGB(0, 0, w, h, null, 0, w);
        byte[] out = new byte[3 * plane];
        for (int i = 0; i < plane; i++) {
            int p = argb[i];
            out[i] = (byte) ((p >> 16) & 0xFF);
            out[plane + i] = (byte) ((p >> 8) & 0xFF);
            out[2 * plane + i] = (byte) (p & 0xFF);
        }
        return out;
    }

    public int count() {
        return samples.size();
    }

    public int imageSize() {
        return size;
    }

    public EuroSatSample sample(int i) {
        return samples.get(i);
    }

    public int label(int i) {
        return samples.get(i).classIndex();
    }

    /** Image {@code i} as CHW floats in [0, 1] (identical to {@link ImageTensors#toChw01} of the decoded JPEG). */
    public float[] chw01(int i) {
        byte[] b = pixels[i];
        float[] out = new float[b.length];
        for (int k = 0; k < b.length; k++) {
            out[k] = (b[k] & 0xFF) / 255f;
        }
        return out;
    }

    /**
     * Per-channel mean and standard deviation of all pixels of these images (call it on the TRAIN
     * split only), on the [0, 1] scale.
     */
    public Normalization computeNormalization() {
        double[] sum = new double[3];
        double[] sumSq = new double[3];
        int plane = size * size;
        for (byte[] img : pixels) {
            for (int c = 0; c < 3; c++) {
                double s = 0;
                double sq = 0;
                for (int k = c * plane; k < (c + 1) * plane; k++) {
                    double v = (img[k] & 0xFF) / 255.0;
                    s += v;
                    sq += v * v;
                }
                sum[c] += s;
                sumSq[c] += sq;
            }
        }
        double n = (double) pixels.length * plane;
        float[] mean = new float[3];
        float[] std = new float[3];
        for (int c = 0; c < 3; c++) {
            double m = sum[c] / n;
            mean[c] = (float) m;
            std[c] = (float) Math.sqrt(Math.max(sumSq[c] / n - m * m, 1e-12));
        }
        return new Normalization(mean, std);
    }

    /**
     * Builds one batch in NCHW order.
     *
     * @param order     image indices, the batch is {@code order[from..to)}
     * @param norm      TRAIN normalisation
     * @param augmenter random source for D4 augmentation (a random rotation by a multiple of 90
     *                  degrees, optionally followed by a horizontal flip), or {@code null} for none
     * @param labelsOut receives the class indices of the batch (length at least to - from)
     * @return normalised images, length (to - from) * 3 * size * size
     */
    public float[] batch(int[] order, int from, int to, Normalization norm, Random augmenter, long[] labelsOut) {
        int n = to - from;
        int per = 3 * size * size;
        float[] out = new float[n * per];
        for (int j = 0; j < n; j++) {
            int i = order[from + j];
            float[] chw = norm.applyInPlace(chw01(i));
            if (augmenter != null) {
                chw = D4Augmentation.apply(chw, 3, size, augmenter.nextInt(D4Augmentation.COUNT));
            }
            System.arraycopy(chw, 0, out, j * per, per);
            labelsOut[j] = label(i);
        }
        return out;
    }
}
