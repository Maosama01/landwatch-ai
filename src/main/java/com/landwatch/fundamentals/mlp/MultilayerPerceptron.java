package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Multilayer Perceptron (MLP) Feed-Forward Deep Neural Network.
 * <p>
 * Supports arbitrary network topologies with configurable input, hidden, and output layers,
 * non-linear activation functions, reproducible seeded initialization, and forward propagation.
 */
public class MultilayerPerceptron {

    private final int[] layerSizes;
    private final List<Layer> layers;
    private final ActivationFunction hiddenActivation;
    private final ActivationFunction outputActivation;

    /**
     * Constructs a Multilayer Perceptron with explicit layer dimensions and activations.
     *
     * @param layerSizes       array of layer neuron counts: [inputSize, hidden1, ..., outputSize]
     * @param hiddenActivation activation function for all hidden layers
     * @param outputActivation activation function for the output layer
     * @param rng              reproducible pseudo-random number generator
     */
    public MultilayerPerceptron(int[] layerSizes,
                                ActivationFunction hiddenActivation,
                                ActivationFunction outputActivation,
                                Random rng) {
        Objects.requireNonNull(layerSizes, "Layer sizes array must not be null.");
        if (layerSizes.length < 2) {
            throw new IllegalArgumentException("An MLP requires at least 2 layers (input and output), got: " + layerSizes.length);
        }
        for (int i = 0; i < layerSizes.length; i++) {
            if (layerSizes[i] <= 0) {
                throw new IllegalArgumentException(String.format(
                        "Layer %d size must be positive, got: %d", i, layerSizes[i]));
            }
        }

        this.layerSizes = Arrays.copyOf(layerSizes, layerSizes.length);
        this.hiddenActivation = Objects.requireNonNull(hiddenActivation, "Hidden activation must not be null.");
        this.outputActivation = Objects.requireNonNull(outputActivation, "Output activation must not be null.");

        this.layers = new ArrayList<>();
        int numTransitions = layerSizes.length - 1;
        for (int i = 0; i < numTransitions; i++) {
            int inSize = layerSizes[i];
            int outSize = layerSizes[i + 1];
            boolean isLast = (i == numTransitions - 1);
            ActivationFunction act = isLast ? outputActivation : hiddenActivation;
            layers.add(new Layer(inSize, outSize, act, rng));
        }
    }

    public MultilayerPerceptron(int[] layerSizes,
                                ActivationFunction hiddenActivation,
                                ActivationFunction outputActivation,
                                long seed) {
        this(layerSizes, hiddenActivation, outputActivation, new Random(seed));
    }

    /**
     * Constructs an MLP defaulting to Sigmoid activation throughout.
     */
    public MultilayerPerceptron(int[] layerSizes, long seed) {
        this(layerSizes, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, seed);
    }

    // --- Forward Propagation ---

    /**
     * Propagates input vector through the entire network:
     * a^[0] = input
     * a^[l] = f(W^[l] * a^[l-1] + b^[l]) for l = 1 .. L
     *
     * @param input input feature vector (dimension must match input layer size)
     * @return final output vector a^[L]
     */
    public Vector forward(Vector input) {
        if (input == null || input.size() != layerSizes[0]) {
            throw new IllegalArgumentException(String.format(
                    "Network input size mismatch: expected %d, got %s",
                    layerSizes[0], input == null ? "null" : input.size()));
        }

        Vector current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    /**
     * Evaluates inference prediction (alias for forward).
     */
    public Vector predict(Vector input) {
        return forward(input);
    }

    // --- Accessors ---

    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    public Layer getLayer(int index) {
        if (index < 0 || index >= layers.size()) {
            throw new IndexOutOfBoundsException(String.format("Layer index %d out of bounds (total layers: %d)", index, layers.size()));
        }
        return layers.get(index);
    }

    public int getNumLayers() {
        return layers.size();
    }

    public int[] getLayerSizes() {
        return Arrays.copyOf(layerSizes, layerSizes.length);
    }

    public ActivationFunction getHiddenActivation() {
        return hiddenActivation;
    }

    public ActivationFunction getOutputActivation() {
        return outputActivation;
    }

    @Override
    public String toString() {
        return String.format("MultilayerPerceptron(topology=%s, hiddenAct=%s, outputAct=%s)",
                Arrays.toString(layerSizes), hiddenActivation, outputActivation);
    }
}
