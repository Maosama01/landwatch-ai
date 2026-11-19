package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Widrow-Hoff LMS (Least Mean Squares) rule, also called the delta rule or ADALINE rule
 * (Bernard Widrow and Ted Hoff, 1960). It is the error-correction rule for a linear neuron.
 * <p>
 * <b>Forward pass</b> (linear output, no threshold while learning):
 * <pre>
 *   y = w . x + b
 * </pre>
 * <b>Update</b>:
 * <pre>
 *   e       = t - y = t - (w . x + b)
 *   Delta w = eta * e * x
 *   Delta b = eta * e
 * </pre>
 * <b>Where it comes from.</b> It is stochastic gradient descent on the instantaneous squared error:
 * <pre>
 *   J(w)  = 1/2 * e^2
 *   dJ/dw = -e * x      =&gt;   Delta w = -eta * dJ/dw = eta * e * x
 * </pre>
 * <b>Difference from the perceptron.</b> The perceptron's error t - step(v) is 0 as soon as
 * the sign is right. The LMS error is measured on the continuous output, so learning continues
 * until the squared error is minimised. For noiseless linear data LMS recovers the exact
 * coefficients, and for noisy or non-separable data it converges (for small eta) close to the
 * least-squares solution.
 * <p>
 * <b>Stability.</b> The mean weights converge when 0 &lt; eta &lt; 2 / lambda_max, where lambda_max is the
 * largest eigenvalue of the input correlation matrix R = E[x x^T] (with the bias treated as an
 * extra input fixed at 1). In practice a small eta is used.
 */
public class LmsLearning {

    private Vector weights;
    private double bias;
    private final double learningRate;

    /**
     * Creates an LMS neuron with explicit initial weights and bias.
     *
     * @param initialWeights starting weights (copied)
     * @param initialBias    starting bias b
     * @param learningRate   eta &gt; 0
     */
    public LmsLearning(Vector initialWeights, double initialBias, double learningRate) {
        if (initialWeights == null) {
            throw new IllegalArgumentException("Initial weights must not be null.");
        }
        LearningRuleUtils.requireValidLearningRate(learningRate);
        this.weights = initialWeights.copy();
        this.bias = initialBias;
        this.learningRate = learningRate;
    }

    /**
     * Creates an LMS neuron with small random weights and bias drawn uniformly from [-0.5, 0.5].
     *
     * @param numInputs    number of inputs
     * @param learningRate eta &gt; 0
     * @param rng          seeded random number generator
     */
    public LmsLearning(int numInputs, double learningRate, Random rng) {
        this(Vector.random(numInputs, -0.5, 0.5, rng), -0.5 + rng.nextDouble(), learningRate);
    }

    /** Linear output y = w . x + b. */
    public double output(Vector x) {
        LearningRuleUtils.requireSize(x, weights.size(), "Input");
        return weights.dot(x) + bias;
    }

    /**
     * Applies one LMS step: Delta w = eta * (t - y) * x, Delta b = eta * (t - y).
     *
     * @param x      input pattern
     * @param target desired real-valued output t
     * @return the error e = t - y measured before the update
     */
    public double update(Vector x, double target) {
        double error = target - output(x);
        weights = weights.add(x.multiply(learningRate * error));
        bias += learningRate * error;
        return error;
    }

    /**
     * Trains for a fixed number of epochs, presenting the samples in their given order.
     *
     * @param inputs  input patterns
     * @param targets desired outputs
     * @param epochs  number of passes through the data
     * @return array of length epochs + 1 with the mean squared error before training and after each epoch
     */
    public double[] train(Vector[] inputs, double[] targets, int epochs) {
        validateDataset(inputs, targets);
        LearningRuleUtils.requireNonNegativeEpochs(epochs);
        double[] mseHistory = new double[epochs + 1];
        mseHistory[0] = meanSquaredError(inputs, targets);
        for (int epoch = 1; epoch <= epochs; epoch++) {
            for (int n = 0; n < inputs.length; n++) {
                update(inputs[n], targets[n]);
            }
            mseHistory[epoch] = meanSquaredError(inputs, targets);
        }
        return mseHistory;
    }

    /**
     * Mean squared error MSE = (1/N) * sum_n (t_n - y_n)^2 of the current weights.
     */
    public double meanSquaredError(Vector[] inputs, double[] targets) {
        validateDataset(inputs, targets);
        double sum = 0.0;
        for (int n = 0; n < inputs.length; n++) {
            double e = targets[n] - output(inputs[n]);
            sum += e * e;
        }
        return sum / inputs.length;
    }

    /** Returns a copy of the weight vector w. */
    public Vector getWeights() {
        return weights.copy();
    }

    public double getBias() {
        return bias;
    }

    public double getLearningRate() {
        return learningRate;
    }

    private static void validateDataset(Vector[] inputs, double[] targets) {
        LearningRuleUtils.requireNonEmpty(inputs, "Inputs");
        if (targets == null || targets.length != inputs.length) {
            throw new IllegalArgumentException("Targets must be non-null and match the number of inputs.");
        }
    }

    @Override
    public String toString() {
        return String.format("LmsLearning(eta=%.4f, w=%s, b=%.4f)", learningRate, weights, bias);
    }
}
