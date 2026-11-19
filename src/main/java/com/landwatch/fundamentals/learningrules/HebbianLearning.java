package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Hebbian learning for a single linear neuron (Donald Hebb, 1949).
 * <p>
 * Hebb's postulate, often summarised as "cells that fire together, wire together": a synapse
 * is strengthened whenever its input x_i and the neuron's output y are active at the same time.
 * <p>
 * <b>Neuron output</b> (linear, no bias):
 * <pre>
 *   y = w . x = sum_i w_i * x_i
 * </pre>
 * <b>Unsupervised Hebbian update</b> (the neuron's own output drives learning):
 * <pre>
 *   Delta w   = eta * y * x
 *   Delta w_i = eta * y * x_i        (component form)
 * </pre>
 * <b>Supervised (forced) Hebbian update</b> (a teacher clamps the output to the target t):
 * <pre>
 *   Delta w = eta * t * x
 * </pre>
 * <b>Why plain Hebbian weights grow without bound.</b> Substitute y = w . x into |w + Delta w|^2:
 * <pre>
 *   |w + eta*y*x|^2 = |w|^2 + 2*eta*y*(w . x) + eta^2 * y^2 * |x|^2
 *                   = |w|^2 + 2*eta*y^2       + eta^2 * y^2 * |x|^2   &gt;=  |w|^2
 * </pre>
 * Both added terms are non-negative, so the weight norm can never shrink and it strictly grows
 * whenever y != 0. Averaged over the data, w(t+1) ~ (I + eta*C) w(t), where C = E[x x^T] is the
 * input correlation matrix, so |w| grows exponentially, fastest along the principal eigenvector
 * of C. The direction of w becomes meaningful but its length explodes.
 * {@link OjaLearning} fixes this by adding a forgetting term.
 * <p>
 * Note: if w starts exactly at the zero vector then y = 0 forever and nothing is learned,
 * so the unsupervised rule needs non-zero initial weights.
 */
public class HebbianLearning {

    private Vector weights;
    private final double learningRate;

    /**
     * Creates a Hebbian neuron with the given initial weights.
     *
     * @param initialWeights starting weight vector (copied)
     * @param learningRate   eta &gt; 0
     */
    public HebbianLearning(Vector initialWeights, double learningRate) {
        if (initialWeights == null) {
            throw new IllegalArgumentException("Initial weights must not be null.");
        }
        LearningRuleUtils.requireValidLearningRate(learningRate);
        this.weights = initialWeights.copy();
        this.learningRate = learningRate;
    }

    /**
     * Creates a Hebbian neuron with small random weights drawn uniformly from [-0.5, 0.5].
     *
     * @param numInputs    number of inputs
     * @param learningRate eta &gt; 0
     * @param rng          seeded random number generator
     */
    public HebbianLearning(int numInputs, double learningRate, Random rng) {
        this(Vector.random(numInputs, -0.5, 0.5, rng), learningRate);
    }

    /**
     * Neuron output y = w . x.
     */
    public double output(Vector x) {
        LearningRuleUtils.requireSize(x, weights.size(), "Input");
        return weights.dot(x);
    }

    /**
     * Computes (without applying) the unsupervised Hebbian change Delta w = eta * y * x.
     */
    public Vector computeDelta(Vector x) {
        double y = output(x);
        return x.multiply(learningRate * y);
    }

    /**
     * Applies one unsupervised Hebbian step w &lt;- w + eta * y * x.
     *
     * @param x input pattern
     * @return the weight change Delta w that was applied
     */
    public Vector update(Vector x) {
        Vector delta = computeDelta(x);
        weights = weights.add(delta);
        return delta;
    }

    /**
     * Computes (without applying) the supervised Hebbian change Delta w = eta * t * x,
     * where the output has been clamped to the target t.
     */
    public Vector computeSupervisedDelta(Vector x, double target) {
        LearningRuleUtils.requireSize(x, weights.size(), "Input");
        return x.multiply(learningRate * target);
    }

    /**
     * Applies one supervised Hebbian step w &lt;- w + eta * t * x.
     *
     * @param x      input pattern
     * @param target teacher-forced output t
     * @return the weight change Delta w that was applied
     */
    public Vector updateSupervised(Vector x, double target) {
        Vector delta = computeSupervisedDelta(x, target);
        weights = weights.add(delta);
        return delta;
    }

    /**
     * Repeatedly presents the data set (unsupervised rule) and records |w| after each epoch.
     * <p>
     * Because growth is exponential, long runs or large eta eventually overflow to Infinity.
     * That overflow is the instability this method demonstrates.
     *
     * @param data   input patterns
     * @param epochs number of passes through the data
     * @param rng    seeded generator for the presentation order
     * @return array of length epochs + 1: element 0 is the initial norm, element e the norm after epoch e
     */
    public double[] train(Vector[] data, int epochs, Random rng) {
        LearningRuleUtils.requireNonEmpty(data, "Training data");
        LearningRuleUtils.requireNonNegativeEpochs(epochs);
        double[] norms = new double[epochs + 1];
        norms[0] = weights.norm();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int idx : LearningRuleUtils.shuffledIndices(data.length, rng)) {
                update(data[idx]);
            }
            norms[epoch] = weights.norm();
        }
        return norms;
    }

    /** Current Euclidean norm |w|. */
    public double weightNorm() {
        return weights.norm();
    }

    /** Returns a copy of the current weight vector. */
    public Vector getWeights() {
        return weights.copy();
    }

    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public String toString() {
        return String.format("HebbianLearning(eta=%.4f, w=%s)", learningRate, weights);
    }
}
