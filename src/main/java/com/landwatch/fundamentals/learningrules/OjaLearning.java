package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Oja's rule (Erkki Oja, 1982): the classic "modified Hebbian" learning rule.
 * <p>
 * <b>Neuron output</b> (linear):
 * <pre>
 *   y = w . x
 * </pre>
 * <b>Update rule</b>:
 * <pre>
 *   Delta w = eta * y * (x - y * w)
 *           = eta * y * x   -   eta * y^2 * w
 *             (Hebbian term)    (forgetting / decay term)
 * </pre>
 * The first term is exactly Hebb's rule ({@link HebbianLearning}). The second term shrinks w in
 * proportion to y^2, which stops the weights from growing without bound.
 * <p>
 * <b>Self-normalisation.</b> To first order in eta:
 * <pre>
 *   |w + Delta w|^2 ~ |w|^2 + 2 * eta * y^2 * (1 - |w|^2)
 * </pre>
 * If |w| &gt; 1 the norm shrinks, and if |w| &lt; 1 it grows, so |w| is driven to 1.
 * <p>
 * <b>Convergence to the first principal component.</b> For zero-mean inputs with covariance
 * C = E[x x^T], the averaged dynamics are
 * <pre>
 *   dw/dt = C w - (w^T C w) w
 * </pre>
 * When the largest eigenvalue lambda1 of C is strictly larger than the others, the only stable fixed
 * points are w = +e1 or w = -e1, where e1 is the unit-length eigenvector belonging to lambda1. A single Oja neuron therefore performs Principal Component
 * Analysis: its output y is the projection of x onto the first principal component.
 * <p>
 * Practical requirements: the data must be zero-mean, and eta must be small (eta * lambda1 &lt;&lt; 1).
 */
public class OjaLearning {

    private Vector weights;
    private final double learningRate;

    /**
     * Creates an Oja neuron with the given initial weights.
     *
     * @param initialWeights starting weight vector (copied; must be non-zero)
     * @param learningRate   eta &gt; 0
     */
    public OjaLearning(Vector initialWeights, double learningRate) {
        if (initialWeights == null) {
            throw new IllegalArgumentException("Initial weights must not be null.");
        }
        LearningRuleUtils.requireValidLearningRate(learningRate);
        this.weights = initialWeights.copy();
        this.learningRate = learningRate;
    }

    /**
     * Creates an Oja neuron with small random weights drawn uniformly from [-0.5, 0.5].
     *
     * @param numInputs    number of inputs
     * @param learningRate eta &gt; 0
     * @param rng          seeded random number generator
     */
    public OjaLearning(int numInputs, double learningRate, Random rng) {
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
     * Computes (without applying) the Oja change Delta w = eta * y * (x - y * w).
     */
    public Vector computeDelta(Vector x) {
        double y = output(x);
        Vector residual = x.subtract(weights.multiply(y));   // x - y*w
        return residual.multiply(learningRate * y);           // eta * y * (x - y*w)
    }

    /**
     * Applies one Oja step w &lt;- w + eta * y * (x - y * w).
     *
     * @param x input pattern (zero-mean data expected)
     * @return the weight change Delta w that was applied
     */
    public Vector update(Vector x) {
        Vector delta = computeDelta(x);
        weights = weights.add(delta);
        return delta;
    }

    /**
     * Repeatedly presents the data set and records |w| after each epoch.
     *
     * @param data   zero-mean input patterns
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

    /** Current Euclidean norm |w| (converges to about 1). */
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
        return String.format("OjaLearning(eta=%.4f, w=%s)", learningRate, weights);
    }
}
