package com.landwatch.fundamentals.neurons;

import com.landwatch.fundamentals.math.Vector;

import java.util.Objects;

/**
 * McCulloch-Pitts (M-P) Artificial Neuron.
 * <p>
 * The earliest mathematical model of a biological neuron (Warren McCulloch & Walter Pitts, 1943).
 * Operates strictly with binary inputs {0, 1} and produces a binary output {0, 1} based on
 * whether the weighted sum of inputs meets or exceeds a specified threshold theta:
 * <pre>
 * y = 1 if sum(w_i * x_i) >= theta
 * y = 0 if sum(w_i * x_i) < theta
 * </pre>
 */
public class McCullochPittsNeuron {

    private final Vector weights;
    private final double threshold;
    private final int numInputs;

    /**
     * Constructs a McCulloch-Pitts neuron with custom weights and activation threshold.
     *
     * @param weights   synaptic weights (dimension determines expected input size)
     * @param threshold activation threshold theta
     */
    public McCullochPittsNeuron(Vector weights, double threshold) {
        Objects.requireNonNull(weights, "Weights vector must not be null.");
        if (weights.size() == 0) {
            throw new IllegalArgumentException("Weights vector must have at least one element.");
        }
        this.weights = weights.copy();
        this.threshold = threshold;
        this.numInputs = weights.size();
    }

    /**
     * Constructs a McCulloch-Pitts neuron from double array weights.
     */
    public McCullochPittsNeuron(double[] weights, double threshold) {
        this(new Vector(weights), threshold);
    }

    // --- Canonical Logic Gate Factories ---

    /**
     * Factory for an n-input AND gate.
     * Fires (outputs 1) if and only if ALL inputs are 1.
     * Weights = [1, 1, ..., 1], Threshold = n.
     */
    public static McCullochPittsNeuron andGate(int numInputs) {
        if (numInputs <= 0) {
            throw new IllegalArgumentException("Number of inputs must be positive: " + numInputs);
        }
        return new McCullochPittsNeuron(Vector.ones(numInputs), (double) numInputs);
    }

    /**
     * Factory for an n-input OR gate.
     * Fires (outputs 1) if AT LEAST ONE input is 1.
     * Weights = [1, 1, ..., 1], Threshold = 1.0.
     */
    public static McCullochPittsNeuron orGate(int numInputs) {
        if (numInputs <= 0) {
            throw new IllegalArgumentException("Number of inputs must be positive: " + numInputs);
        }
        return new McCullochPittsNeuron(Vector.ones(numInputs), 1.0);
    }

    /**
     * Factory for an n-input NOR gate.
     * Fires (outputs 1) if and only if ALL inputs are 0.
     * Weights = [-1, -1, ..., -1], Threshold = 0.0.
     */
    public static McCullochPittsNeuron norGate(int numInputs) {
        if (numInputs <= 0) {
            throw new IllegalArgumentException("Number of inputs must be positive: " + numInputs);
        }
        Vector w = Vector.ones(numInputs).multiply(-1.0);
        return new McCullochPittsNeuron(w, 0.0);
    }

    // --- Activation & Computation ---

    /**
     * Computes the linear weighted sum: sum(w_i * x_i).
     *
     * @param inputs integer array of binary inputs
     * @return weighted linear sum
     */
    public double computeWeightedSum(int... inputs) {
        validateInputs(inputs);
        double sum = 0.0;
        for (int i = 0; i < numInputs; i++) {
            sum += weights.get(i) * inputs[i];
        }
        return sum;
    }

    /**
     * Computes the linear weighted sum from a Vector of binary inputs.
     */
    public double computeWeightedSum(Vector inputs) {
        validateVectorInputs(inputs);
        return weights.dot(inputs);
    }

    /**
     * Evaluates the McCulloch-Pitts neuron activation function:
     * returns 1 if weighted sum >= threshold, else 0.
     *
     * @param inputs integer array of binary inputs
     * @return 1 (firing) or 0 (quiescent)
     */
    public int activate(int... inputs) {
        double weightedSum = computeWeightedSum(inputs);
        return weightedSum >= threshold ? 1 : 0;
    }

    /**
     * Evaluates activation using a Vector.
     */
    public int activate(Vector inputs) {
        double weightedSum = computeWeightedSum(inputs);
        return weightedSum >= threshold ? 1 : 0;
    }

    // --- Validation Helpers ---

    private void validateInputs(int... inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("Inputs array must not be null.");
        }
        if (inputs.length != numInputs) {
            throw new IllegalArgumentException(String.format(
                    "Input dimension mismatch: expected %d inputs, got %d", numInputs, inputs.length));
        }
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] != 0 && inputs[i] != 1) {
                throw new IllegalArgumentException(String.format(
                        "McCulloch-Pitts neuron requires binary inputs {0, 1}. Invalid value at index %d: %d",
                        i, inputs[i]));
            }
        }
    }

    private void validateVectorInputs(Vector inputs) {
        if (inputs == null) {
            throw new IllegalArgumentException("Inputs vector must not be null.");
        }
        if (inputs.size() != numInputs) {
            throw new IllegalArgumentException(String.format(
                    "Input dimension mismatch: expected %d inputs, got %d", numInputs, inputs.size()));
        }
        for (int i = 0; i < inputs.size(); i++) {
            double val = inputs.get(i);
            if (val != 0.0 && val != 1.0) {
                throw new IllegalArgumentException(String.format(
                        "McCulloch-Pitts neuron requires binary inputs {0.0, 1.0}. Invalid value at index %d: %f",
                        i, val));
            }
        }
    }

    // --- Accessors ---

    public Vector getWeights() {
        return weights.copy();
    }

    public double getThreshold() {
        return threshold;
    }

    public int getNumInputs() {
        return numInputs;
    }

    @Override
    public String toString() {
        return String.format("McCullochPittsNeuron(inputs=%d, weights=%s, threshold=%.2f)",
                numInputs, weights, threshold);
    }
}
