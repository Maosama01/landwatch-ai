package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Small synthetic data sets with analytically known answers, used to demonstrate and test
 * the learning rules in this package.
 * <p>
 * Every generator takes a seeded {@link Random}, so the same seed always produces exactly
 * the same data.
 */
public final class SyntheticDatasets {

    private SyntheticDatasets() {
        // Static factory class; not instantiable.
    }

    /**
     * Unit vector at the given angle: u = (cos(angle), sin(angle)).
     */
    public static Vector unitDirection(double angleRadians) {
        return Vector.of(Math.cos(angleRadians), Math.sin(angleRadians));
    }

    /**
     * Zero-mean, correlated 2-D Gaussian data whose principal direction is known in advance.
     * <p>
     * Each sample is built as
     * <pre>
     *   x = a * u1 + c * u2,   a ~ N(0, sigmaMajor^2),  c ~ N(0, sigmaMinor^2)
     *   u1 = ( cos(angle), sin(angle) )      (major axis)
     *   u2 = (-sin(angle), cos(angle) )      (minor axis, perpendicular to u1)
     * </pre>
     * so the population covariance matrix is
     * <pre>
     *   C = sigmaMajor^2 * u1 u1^T + sigmaMinor^2 * u2 u2^T
     * </pre>
     * whose eigenvectors are u1 and u2 with eigenvalues sigmaMajor^2 and sigmaMinor^2.
     * When sigmaMajor &gt; sigmaMinor the first principal component is therefore exactly u1.
     * <p>
     * After generation the sample mean is subtracted, so the returned data is exactly zero-mean
     * (Oja's rule assumes zero-mean input).
     *
     * @param n            number of samples (at least 2)
     * @param sigmaMajor   standard deviation along the principal direction u1
     * @param sigmaMinor   standard deviation along the perpendicular direction u2
     * @param angleRadians angle of u1 measured from the x-axis
     * @param rng          seeded random number generator
     * @return n zero-mean samples
     */
    public static Vector[] correlatedGaussian2D(int n, double sigmaMajor, double sigmaMinor,
                                               double angleRadians, Random rng) {
        if (n < 2) {
            throw new IllegalArgumentException("Need at least 2 samples, got: " + n);
        }
        if (sigmaMajor <= 0.0 || sigmaMinor <= 0.0) {
            throw new IllegalArgumentException("Standard deviations must be positive.");
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random number generator must not be null.");
        }
        Vector u1 = unitDirection(angleRadians);
        Vector u2 = Vector.of(-Math.sin(angleRadians), Math.cos(angleRadians));

        Vector[] data = new Vector[n];
        Vector mean = Vector.zeros(2);
        for (int k = 0; k < n; k++) {
            double a = sigmaMajor * rng.nextGaussian();
            double c = sigmaMinor * rng.nextGaussian();
            data[k] = u1.multiply(a).add(u2.multiply(c));
            mean = mean.add(data[k]);
        }
        mean = mean.multiply(1.0 / n);

        // Centre the data so it is exactly zero-mean.
        for (int k = 0; k < n; k++) {
            data[k] = data[k].subtract(mean);
        }
        return data;
    }

    /**
     * Isotropic Gaussian clusters around the given centres.
     * <p>
     * Sample p belongs to cluster {@code p / pointsPerCluster}: the first
     * {@code pointsPerCluster} samples come from centre 0, the next block from centre 1, and so on.
     *
     * @param centres          cluster centres (all of the same dimension)
     * @param pointsPerCluster number of samples per cluster
     * @param sigma            standard deviation of every coordinate around its centre
     * @param rng              seeded random number generator
     * @return centres.length * pointsPerCluster samples
     */
    public static Vector[] gaussianClusters(Vector[] centres, int pointsPerCluster, double sigma, Random rng) {
        LearningRuleUtils.requireNonEmpty(centres, "Cluster centres");
        if (pointsPerCluster <= 0) {
            throw new IllegalArgumentException("Points per cluster must be positive, got: " + pointsPerCluster);
        }
        if (sigma < 0.0) {
            throw new IllegalArgumentException("Sigma must be non-negative, got: " + sigma);
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random number generator must not be null.");
        }
        int dim = centres[0].size();
        Vector[] data = new Vector[centres.length * pointsPerCluster];
        int idx = 0;
        for (Vector centre : centres) {
            LearningRuleUtils.requireSize(centre, dim, "Cluster centre");
            for (int p = 0; p < pointsPerCluster; p++) {
                Vector point = centre.copy();
                for (int d = 0; d < dim; d++) {
                    point.set(d, point.get(d) + sigma * rng.nextGaussian());
                }
                data[idx++] = point;
            }
        }
        return data;
    }

    /**
     * The four binary input patterns of a 2-input logic gate, in truth-table order:
     * (0,0), (0,1), (1,0), (1,1).
     */
    public static Vector[] logicGateInputs() {
        return new Vector[]{
                Vector.of(0.0, 0.0),
                Vector.of(0.0, 1.0),
                Vector.of(1.0, 0.0),
                Vector.of(1.0, 1.0)
        };
    }

    /** AND targets for {@link #logicGateInputs()}: 0, 0, 0, 1 (linearly separable). */
    public static int[] andTargets() {
        return new int[]{0, 0, 0, 1};
    }

    /** OR targets for {@link #logicGateInputs()}: 0, 1, 1, 1 (linearly separable). */
    public static int[] orTargets() {
        return new int[]{0, 1, 1, 1};
    }

    /** XOR targets for {@link #logicGateInputs()}: 0, 1, 1, 0 (NOT linearly separable). */
    public static int[] xorTargets() {
        return new int[]{0, 1, 1, 0};
    }
}
