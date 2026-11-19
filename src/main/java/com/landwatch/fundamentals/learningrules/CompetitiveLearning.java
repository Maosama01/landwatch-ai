package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Competitive ("winner-take-all") learning with K prototype weight vectors w_1 ... w_K.
 * <p>
 * For each input x:
 * <ol>
 *   <li><b>Competition.</b> The prototype closest to x wins:
 *   <pre>
 *   k* = argmin_k |x - w_k|        (Euclidean distance)
 *   </pre></li>
 *   <li><b>Update (winner only).</b>
 *   <pre>
 *   Delta w_k* = eta * (x - w_k*)     (winner moves a fraction eta of the way towards x)
 *   Delta w_k  = 0                    for every loser k != k*
 *   </pre></li>
 * </ol>
 * Over many presentations each prototype drifts towards the mean of the inputs it wins,
 * so the prototypes become cluster centres. This is online k-means, also called vector quantisation.
 * <p>
 * <b>Distance vs. dot product.</b> Since |x - w|^2 = |x|^2 - 2 (w . x) + |w|^2, when every
 * prototype has the same length (e.g. normalised weights) the minimum-distance winner is also
 * the maximum-dot-product winner, which is the "largest net input" form used in many textbooks.
 * This class uses the minimum Euclidean distance, which does not need normalised weights.
 * <p>
 * <b>Dead units.</b> A prototype that starts far from all data may never win and therefore never
 * learns. Starting the prototypes near the centre of the data (small random values for
 * zero-centred data) avoids this in the demonstrations here.
 * <p>
 * Ties are broken in favour of the lowest prototype index.
 */
public class CompetitiveLearning {

    private final Vector[] prototypes;
    private final double learningRate;
    private final int dimension;

    /**
     * Creates a competitive layer from explicit initial prototypes.
     *
     * @param initialPrototypes starting prototype vectors (copied; all same dimension)
     * @param learningRate      eta in (0, 1]
     */
    public CompetitiveLearning(Vector[] initialPrototypes, double learningRate) {
        LearningRuleUtils.requireNonEmpty(initialPrototypes, "Initial prototypes");
        LearningRuleUtils.requireValidLearningRate(learningRate);
        if (learningRate > 1.0) {
            throw new IllegalArgumentException(
                    "Competitive learning rate must be in (0, 1] so the winner does not overshoot x, got: "
                            + learningRate);
        }
        this.dimension = initialPrototypes[0].size();
        this.prototypes = new Vector[initialPrototypes.length];
        for (int k = 0; k < initialPrototypes.length; k++) {
            LearningRuleUtils.requireSize(initialPrototypes[k], dimension, "Prototype " + k);
            this.prototypes[k] = initialPrototypes[k].copy();
        }
        this.learningRate = learningRate;
    }

    /**
     * Creates a competitive layer whose prototypes are drawn uniformly from [min, max] in every coordinate.
     *
     * @param numPrototypes number of competing units K
     * @param dimension     input dimension
     * @param min           lower bound of initial coordinates
     * @param max           upper bound of initial coordinates
     * @param learningRate  eta in (0, 1]
     * @param rng           seeded random number generator
     * @return a new competitive layer
     */
    public static CompetitiveLearning withRandomPrototypes(int numPrototypes, int dimension, double min,
                                                           double max, double learningRate, Random rng) {
        if (numPrototypes <= 0) {
            throw new IllegalArgumentException("Number of prototypes must be positive, got: " + numPrototypes);
        }
        Vector[] init = new Vector[numPrototypes];
        for (int k = 0; k < numPrototypes; k++) {
            init[k] = Vector.random(dimension, min, max, rng);
        }
        return new CompetitiveLearning(init, learningRate);
    }

    /**
     * Competition step: index of the prototype nearest to x (minimum Euclidean distance).
     */
    public int findWinner(Vector x) {
        LearningRuleUtils.requireSize(x, dimension, "Input");
        int winner = 0;
        double bestDistance = Double.POSITIVE_INFINITY;
        for (int k = 0; k < prototypes.length; k++) {
            double distance = LearningRuleUtils.euclideanDistance(x, prototypes[k]);
            if (distance < bestDistance) {   // strict '<' keeps the lowest index on ties
                bestDistance = distance;
                winner = k;
            }
        }
        return winner;
    }

    /**
     * Applies one winner-take-all step: w_k* &lt;- w_k* + eta * (x - w_k*). Losers are unchanged.
     *
     * @param x input pattern
     * @return index of the winning prototype
     */
    public int update(Vector x) {
        int winner = findWinner(x);
        Vector delta = x.subtract(prototypes[winner]).multiply(learningRate);
        prototypes[winner] = prototypes[winner].add(delta);
        return winner;
    }

    /**
     * Trains for a number of epochs with a fresh random presentation order each epoch.
     *
     * @param data   input patterns
     * @param epochs number of passes through the data
     * @param rng    seeded generator for the presentation order
     * @return array of length epochs + 1 with the quantisation error (see
     *         {@link #quantizationError(Vector[])}) before training and after each epoch
     */
    public double[] train(Vector[] data, int epochs, Random rng) {
        LearningRuleUtils.requireNonEmpty(data, "Training data");
        LearningRuleUtils.requireNonNegativeEpochs(epochs);
        double[] errors = new double[epochs + 1];
        errors[0] = quantizationError(data);
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int idx : LearningRuleUtils.shuffledIndices(data.length, rng)) {
                update(data[idx]);
            }
            errors[epoch] = quantizationError(data);
        }
        return errors;
    }

    /**
     * Mean distance from each input to its winning prototype:
     * <pre>
     *   QE = (1/N) * sum_n |x_n - w_k*(x_n)|
     * </pre>
     * Lower values mean the prototypes represent the data better.
     */
    public double quantizationError(Vector[] data) {
        LearningRuleUtils.requireNonEmpty(data, "Data");
        double total = 0.0;
        for (Vector x : data) {
            total += LearningRuleUtils.euclideanDistance(x, prototypes[findWinner(x)]);
        }
        return total / data.length;
    }

    /** Returns a copy of prototype k. */
    public Vector getPrototype(int k) {
        if (k < 0 || k >= prototypes.length) {
            throw new IndexOutOfBoundsException("Prototype index out of range: " + k);
        }
        return prototypes[k].copy();
    }

    /** Returns copies of all prototypes. */
    public Vector[] getPrototypes() {
        Vector[] copy = new Vector[prototypes.length];
        for (int k = 0; k < prototypes.length; k++) {
            copy[k] = prototypes[k].copy();
        }
        return copy;
    }

    public int getNumPrototypes() {
        return prototypes.length;
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public String toString() {
        return String.format("CompetitiveLearning(K=%d, dim=%d, eta=%.4f)",
                prototypes.length, dimension, learningRate);
    }
}
