package com.landwatch.fundamentals.neurons;

import com.landwatch.fundamentals.math.Vector;

import java.util.Objects;

/**
 * Sigmoid Artificial Neuron (Logistic Unit).
 * <p>
 * Implements a continuous, differentiable non-linear neuron with activation:
 * <pre>
 * z = sum(w_i * x_i) + b
 * a = sigma(z) = 1 / (1 + exp(-z))
 * </pre>
 * Output is strictly bounded in the open interval (0, 1).
 * Features numerically stable sigmoid evaluation to prevent floating-point overflow.
 */
public class SigmoidNeuron {

    private Vector weights;
    private double bias;
    private final int numInputs;

    /**
     * Constructs a Sigmoid Neuron with specified weight vector and bias.
     *
     * @param weights synaptic weight vector
     * @param bias    scalar bias term
     */
    public SigmoidNeuron(Vector weights, double bias) {
        Objects.requireNonNull(weights, "Weights vector must not be null.");
        if (weights.size() == 0) {
            throw new IllegalArgumentException("Weights vector must have at least one element.");
        }
        this.weights = weights.copy();
        this.bias = bias;
        this.numInputs = weights.size();
    }

    /**
     * Constructs a Sigmoid Neuron from double array weights and scalar bias.
     */
    public SigmoidNeuron(double[] weights, double bias) {
        this(new Vector(weights), bias);
    }

    /**
     * Constructs a Sigmoid Neuron with zero-initialized weights and zero bias.
     *
     * @param numInputs number of input connections
     */
    public SigmoidNeuron(int numInputs) {
        this(Vector.zeros(numInputs), 0.0);
    }

    // --- Core Computation ---

    /**
     * Computes the linear pre-activation (net input):
     * z = w . x + b
     *
     * @param input feature input vector
     * @return pre-activation scalar z
     */
    public double computeNetInput(Vector input) {
        validateInput(input);
        return weights.dot(input) + bias;
    }

    /**
     * Computes the linear pre-activation from double array inputs.
     */
    public double computeNetInput(double... input) {
        validateInputArray(input);
        double sum = bias;
        for (int i = 0; i < numInputs; i++) {
            sum += weights.get(i) * input[i];
        }
        return sum;
    }

    /**
     * Evaluates the non-linear activation:
     * a = sigma(w . x + b)
     *
     * @param input feature input vector
     * @return activated output in range (0, 1)
     */
    public double activate(Vector input) {
        double z = computeNetInput(input);
        return sigmoid(z);
    }

    /**
     * Evaluates the non-linear activation from double array inputs.
     */
    public double activate(double... input) {
        double z = computeNetInput(input);
        return sigmoid(z);
    }

    // --- Mathematical Functions ---

    /**
     * Numerically stable sigmoid function:
     * <pre>
     * For z >= 0: sigma(z) = 1 / (1 + exp(-z))
     * For z < 0:  sigma(z) = exp(z) / (1 + exp(z))
     * </pre>
     * Prevents exponential overflow for large negative values of z.
     *
     * @param z pre-activation scalar
     * @return sigma(z) in (0, 1)
     */
    public static double sigmoid(double z) {
        if (Double.isNaN(z)) {
            return Double.NaN;
        }
        // Clamping extreme values to avoid IEEE-754 denormalized underflow
        if (z > 45.0) {
            return 1.0;
        }
        if (z < -45.0) {
            return 0.0;
        }

        if (z >= 0.0) {
            return 1.0 / (1.0 + Math.exp(-z));
        } else {
            double expZ = Math.exp(z);
            return expZ / (1.0 + expZ);
        }
    }

    /**
     * Computes the first derivative of the sigmoid function given pre-activation z:
     * sigma'(z) = sigma(z) * (1 - sigma(z))
     */
    public static double sigmoidDerivative(double z) {
        double s = sigmoid(z);
        return s * (1.0 - s);
    }

    /**
     * Computes the derivative directly from already-computed activation a = sigma(z):
     * sigma'(a) = a * (1 - a)
     */
    public static double derivativeFromActivation(double activation) {
        return activation * (1.0 - activation);
    }

    // --- Validation Helpers ---

    private void validateInput(Vector input) {
        if (input == null) {
            throw new IllegalArgumentException("Input vector must not be null.");
        }
        if (input.size() != numInputs) {
            throw new IllegalArgumentException(String.format(
                    "Input dimension mismatch: expected %d inputs, got %d", numInputs, input.size()));
        }
    }

    private void validateInputArray(double... input) {
        if (input == null) {
            throw new IllegalArgumentException("Input array must not be null.");
        }
        if (input.length != numInputs) {
            throw new IllegalArgumentException(String.format(
                    "Input dimension mismatch: expected %d inputs, got %d", numInputs, input.length));
        }
    }

    // --- Accessors & Mutators ---

    public Vector getWeights() {
        return weights.copy();
    }

    public void setWeights(Vector weights) {
        Objects.requireNonNull(weights, "Weights cannot be set to null.");
        if (weights.size() != numInputs) {
            throw new IllegalArgumentException(String.format(
                    "New weights dimension %d does not match neuron input size %d", weights.size(), numInputs));
        }
        this.weights = weights.copy();
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public int getNumInputs() {
        return numInputs;
    }

    @Override
    public String toString() {
        return String.format("SigmoidNeuron(inputs=%d, weights=%s, bias=%.4f)",
                numInputs, weights, bias);
    }
}
