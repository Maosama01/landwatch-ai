package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;

import java.util.Objects;
import java.util.Random;

/**
 * Dense (Fully Connected) Neural Network Layer.
 * <p>
 * Computes:
 * <pre>
 * z = W * x + b
 * a = f(z)
 * </pre>
 * Holds weights of shape (outputSize x inputSize) and biases of shape (outputSize).
 * Caches pre- and post-activations during forward propagation to support exact backpropagation.
 */
public class Layer {

    private final int inputSize;
    private final int outputSize;
    private Matrix weights;
    private Vector biases;
    private final ActivationFunction activationFunction;

    // Cache stored during forward pass for backpropagation
    private Vector lastInput;
    private Vector lastZ;
    private Vector lastA;

    // Parameter gradients computed during backward pass
    private Matrix gradW;
    private Vector gradB;

    /**
     * Constructs a layer with explicit weights, biases, and activation function.
     */
    public Layer(Matrix weights, Vector biases, ActivationFunction activationFunction) {
        Objects.requireNonNull(weights, "Weights matrix must not be null.");
        Objects.requireNonNull(biases, "Biases vector must not be null.");
        Objects.requireNonNull(activationFunction, "Activation function must not be null.");

        if (weights.getRows() != biases.size()) {
            throw new IllegalArgumentException(String.format(
                    "Weights rows (%d) must match bias size (%d).", weights.getRows(), biases.size()));
        }

        this.outputSize = weights.getRows();
        this.inputSize = weights.getCols();
        this.weights = weights.copy();
        this.biases = biases.copy();
        this.activationFunction = activationFunction;

        this.gradW = Matrix.zeros(outputSize, inputSize);
        this.gradB = Vector.zeros(outputSize);
    }

    /**
     * Constructs a layer with Xavier/Glorot uniform initialization.
     *
     * @param inputSize          number of input connections
     * @param outputSize         number of neurons in this layer
     * @param activationFunction activation function
     * @param rng                random number generator for reproducibility
     */
    public Layer(int inputSize, int outputSize, ActivationFunction activationFunction, Random rng) {
        if (inputSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException(String.format(
                    "Layer dimensions must be positive, got: input=%d, output=%d", inputSize, outputSize));
        }
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activationFunction = Objects.requireNonNull(activationFunction, "Activation function must not be null.");

        // Xavier/Glorot uniform initialization for Sigmoid, He initialization for ReLU
        double limit;
        if (activationFunction == ActivationFunction.RELU) {
            limit = Math.sqrt(6.0 / inputSize);
        } else {
            limit = Math.sqrt(6.0 / (inputSize + outputSize));
        }

        this.weights = Matrix.random(outputSize, inputSize, -limit, limit, rng);
        this.biases = Vector.zeros(outputSize); // Biases initialized to zero

        this.gradW = Matrix.zeros(outputSize, inputSize);
        this.gradB = Vector.zeros(outputSize);
    }

    public Layer(int inputSize, int outputSize, ActivationFunction activationFunction, long seed) {
        this(inputSize, outputSize, activationFunction, new Random(seed));
    }

    // --- Forward Propagation ---

    /**
     * Executes forward propagation for this layer:
     * z = W * x + b
     * a = f(z)
     *
     * @param input feature vector of dimension inputSize
     * @return activated output vector of dimension outputSize
     */
    public Vector forward(Vector input) {
        if (input == null || input.size() != inputSize) {
            throw new IllegalArgumentException(String.format(
                    "Input dimension mismatch: expected %d, got %s",
                    inputSize, input == null ? "null" : input.size()));
        }

        this.lastInput = input.copy();

        // Linear transformation: z = W * x + b
        Vector z = weights.multiply(input).add(biases);
        this.lastZ = z;

        // Non-linear activation: a = f(z)
        Vector a = activationFunction.activate(z);
        this.lastA = a;

        return a;
    }

    // --- Backpropagation ---

    /**
     * Executes backward pass using the chain rule:
     * delta = upstreamGrad (x) f'(z)
     * gradW = delta (outer) (lastInput)^T
     * gradB = delta
     * downstreamGrad = W^T * delta
     *
     * @param upstreamGrad loss gradient with respect to this layer's activation a (dLoss/da)
     * @return loss gradient with respect to previous layer's activation (dLoss/da_{prev})
     */
    public Vector backward(Vector upstreamGrad) {
        if (upstreamGrad == null || upstreamGrad.size() != outputSize) {
            throw new IllegalArgumentException(String.format(
                    "Upstream gradient size mismatch: expected %d, got %s",
                    outputSize, upstreamGrad == null ? "null" : upstreamGrad.size()));
        }
        if (lastZ == null || lastInput == null) {
            throw new IllegalStateException("Forward pass must be executed before backward pass.");
        }

        // Local error: delta = dLoss/da * f'(z)
        Vector fPrime = activationFunction.derivative(lastZ);
        Vector delta = upstreamGrad.elementwiseMultiply(fPrime);

        // Weight gradient: dLoss/dW = delta * (lastInput)^T
        this.gradW = delta.outerProduct(lastInput);

        // Bias gradient: dLoss/db = delta
        this.gradB = delta.copy();

        // Downstream gradient to pass to layer l-1: dLoss/da_prev = W^T * delta
        return weights.transpose().multiply(delta);
    }

    /**
     * Updates weights and biases using computed gradients via Stochastic Gradient Descent (SGD):
     * W = W - learningRate * gradW
     * b = b - learningRate * gradB
     *
     * @param learningRate step size eta
     */
    public void updateParameters(double learningRate) {
        if (gradW == null || gradB == null) {
            throw new IllegalStateException("Gradients must be computed via backward() before parameter update.");
        }
        this.weights = this.weights.subtract(this.gradW.multiply(learningRate));
        this.biases = this.biases.subtract(this.gradB.multiply(learningRate));
    }

    // --- Accessors & Mutators ---

    public int getInputSize() { return inputSize; }
    public int getOutputSize() { return outputSize; }
    public Matrix getWeights() { return weights.copy(); }
    public void setWeights(Matrix weights) { this.weights = weights.copy(); }
    public Vector getBiases() { return biases.copy(); }
    public void setBiases(Vector biases) { this.biases = biases.copy(); }
    public ActivationFunction getActivationFunction() { return activationFunction; }

    public Vector getLastInput() { return lastInput != null ? lastInput.copy() : null; }
    public Vector getLastZ() { return lastZ != null ? lastZ.copy() : null; }
    public Vector getLastA() { return lastA != null ? lastA.copy() : null; }

    public Matrix getGradW() { return gradW != null ? gradW.copy() : null; }
    public void setGradW(Matrix gradW) { this.gradW = gradW.copy(); }
    public Vector getGradB() { return gradB != null ? gradB.copy() : null; }
    public void setGradB(Vector gradB) { this.gradB = gradB.copy(); }

    /**
     * Resets computed gradients to zero.
     */
    public void zeroGradients() {
        this.gradW = Matrix.zeros(outputSize, inputSize);
        this.gradB = Vector.zeros(outputSize);
    }

    /**
     * Accumulates given gradients into the layer's gradient buffers.
     */
    public void accumulateGradients(Matrix dW, Vector db) {
        if (this.gradW == null) {
            this.gradW = dW.copy();
        } else {
            this.gradW = this.gradW.add(dW);
        }
        if (this.gradB == null) {
            this.gradB = db.copy();
        } else {
            this.gradB = this.gradB.add(db);
        }
    }

    @Override
    public String toString() {
        return String.format("Layer(%d -> %d, %s)", inputSize, outputSize, activationFunction);
    }
}
