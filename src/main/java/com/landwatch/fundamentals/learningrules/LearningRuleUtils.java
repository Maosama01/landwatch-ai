package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Small helper functions shared by the learning-rule classes in this package.
 * <p>
 * Nothing here is a learning rule itself. These are the bookkeeping pieces
 * (shuffling the presentation order, measuring angles and distances, validating arguments)
 * that every rule needs.
 */
public final class LearningRuleUtils {

    private LearningRuleUtils() {
        // Static utility class; not instantiable.
    }

    /**
     * Returns the indices 0..n-1 in a random order (Fisher-Yates shuffle).
     * <p>
     * Presenting training samples in a fresh random order each epoch avoids the learner
     * being biased by a fixed ordering of the data set. Passing a seeded {@link Random}
     * makes the order (and therefore the whole training run) reproducible.
     *
     * @param n   number of indices (must be positive)
     * @param rng seeded random number generator
     * @return a random permutation of 0..n-1
     */
    public static int[] shuffledIndices(int n, Random rng) {
        if (n <= 0) {
            throw new IllegalArgumentException("Number of indices must be positive, got: " + n);
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random number generator must not be null.");
        }
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int tmp = order[i];
            order[i] = order[j];
            order[j] = tmp;
        }
        return order;
    }

    /**
     * Cosine of the angle between two vectors:
     * <pre>
     *   cos(angle) = (a . b) / (|a| * |b|)
     * </pre>
     * A value of +1 or -1 means the vectors point along the same line.
     *
     * @param a first vector (non-zero)
     * @param b second vector (non-zero, same size)
     * @return cosine similarity in [-1, 1]
     */
    public static double cosineSimilarity(Vector a, Vector b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Vectors must not be null.");
        }
        double normProduct = a.norm() * b.norm();
        if (normProduct < 1e-15) {
            throw new ArithmeticException("Cosine similarity is undefined for a zero vector.");
        }
        return a.dot(b) / normProduct;
    }

    /**
     * Euclidean distance between two points:
     * <pre>
     *   d(a, b) = |a - b| = sqrt( sum_i (a_i - b_i)^2 )
     * </pre>
     *
     * @param a first point
     * @param b second point (same size)
     * @return Euclidean distance
     */
    public static double euclideanDistance(Vector a, Vector b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Vectors must not be null.");
        }
        return a.subtract(b).norm();
    }

    /**
     * Logistic sigmoid: sigma(z) = 1 / (1 + exp(-z)).
     */
    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    // --- Package-private validation helpers ---

    static void requireValidLearningRate(double learningRate) {
        if (!(learningRate > 0.0) || Double.isInfinite(learningRate)) {
            throw new IllegalArgumentException(
                    "Learning rate must be strictly positive and finite, got: " + learningRate);
        }
    }

    static void requireNonEmpty(Vector[] data, String name) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException(name + " must not be null or empty.");
        }
        for (int i = 0; i < data.length; i++) {
            if (data[i] == null) {
                throw new IllegalArgumentException(name + " contains a null vector at index " + i);
            }
        }
    }

    static void requireSize(Vector v, int expectedSize, String name) {
        if (v == null) {
            throw new IllegalArgumentException(name + " must not be null.");
        }
        if (v.size() != expectedSize) {
            throw new IllegalArgumentException(String.format(
                    "%s dimension mismatch: expected %d, got %d", name, expectedSize, v.size()));
        }
    }

    static void requireNonNegativeEpochs(int epochs) {
        if (epochs < 0) {
            throw new IllegalArgumentException("Number of epochs must be non-negative, got: " + epochs);
        }
    }
}
