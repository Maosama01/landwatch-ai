package com.landwatch.deeplearning;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Per-channel standardisation of [0, 1] RGB inputs: {@code x' = (x - mean[c]) / std[c]}.
 *
 * <p>The constants are computed once on the TRAIN split only (pixels of both the before and the
 * after images pooled) and stored with the model, so validation, test and dashboard images are
 * normalised with exactly the same numbers the network was trained with.</p>
 */
public final class Normalization {

    private final float[] mean;
    private final float[] std;

    public Normalization(float[] mean, float[] std) {
        if (mean.length != 3 || std.length != 3) {
            throw new IllegalArgumentException("Need 3 means and 3 standard deviations");
        }
        for (float s : std) {
            if (!(s > 0)) {
                throw new IllegalArgumentException("Standard deviations must be positive: " + Arrays.toString(std));
            }
        }
        this.mean = mean.clone();
        this.std = std.clone();
    }

    /** Identity-like normalisation (mean 0, std 1): inputs stay in [0, 1]. */
    public static Normalization none() {
        return new Normalization(new float[] {0f, 0f, 0f}, new float[] {1f, 1f, 1f});
    }

    public float[] getMean() {
        return mean.clone();
    }

    public float[] getStd() {
        return std.clone();
    }

    /** Standardises a CHW [0, 1] array in place and returns it. */
    public float[] applyInPlace(float[] chw) {
        int plane = chw.length / 3;
        for (int c = 0; c < 3; c++) {
            float m = mean[c];
            float inv = 1f / std[c];
            int start = c * plane;
            for (int i = start; i < start + plane; i++) {
                chw[i] = (chw[i] - m) * inv;
            }
        }
        return chw;
    }

    /**
     * Computes per-channel mean and standard deviation over the A and B images of the given
     * patches (use TRAIN patches only), decoding images in parallel.
     */
    public static Normalization computeFromTrain(List<LevirCdPatch> trainPatches, int threads)
            throws IOException {
        ExecutorService pool = Executors.newFixedThreadPool(Math.max(1, threads));
        try {
            List<Future<double[]>> parts = new ArrayList<>();
            int chunk = Math.max(1, trainPatches.size() / (threads * 4));
            for (int start = 0; start < trainPatches.size(); start += chunk) {
                List<LevirCdPatch> slice = trainPatches.subList(start, Math.min(trainPatches.size(), start + chunk));
                parts.add(pool.submit(() -> sums(slice)));
            }
            double[] total = new double[7]; // sum[3], sumSq[3], count
            for (Future<double[]> f : parts) {
                double[] s = f.get();
                for (int i = 0; i < total.length; i++) {
                    total[i] += s[i];
                }
            }
            float[] mean = new float[3];
            float[] std = new float[3];
            for (int c = 0; c < 3; c++) {
                double m = total[c] / total[6];
                double var = total[3 + c] / total[6] - m * m;
                mean[c] = (float) m;
                std[c] = (float) Math.sqrt(Math.max(var, 1e-12));
            }
            return new Normalization(mean, std);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted while computing normalisation", e);
        } catch (ExecutionException e) {
            throw new IOException("Failed while computing normalisation", e.getCause());
        } finally {
            pool.shutdownNow();
        }
    }

    private static double[] sums(List<LevirCdPatch> patches) throws IOException {
        double[] s = new double[7];
        for (LevirCdPatch p : patches) {
            for (Path path : new Path[] {p.imageA(), p.imageB()}) {
                float[] chw = ImageTensors.toChw01(ImageTensors.read(path));
                int plane = chw.length / 3;
                for (int c = 0; c < 3; c++) {
                    double sum = 0;
                    double sq = 0;
                    for (int i = c * plane; i < (c + 1) * plane; i++) {
                        sum += chw[i];
                        sq += (double) chw[i] * chw[i];
                    }
                    s[c] += sum;
                    s[3 + c] += sq;
                }
                s[6] += plane;
            }
        }
        return s;
    }

    @Override
    public String toString() {
        return "Normalization[mean=" + Arrays.toString(mean) + ", std=" + Arrays.toString(std) + "]";
    }
}
