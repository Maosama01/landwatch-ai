package com.landwatch.fundamentals.optimizers;

import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.Layer;

/**
 * Standard Stochastic Gradient Descent (SGD) Optimizer.
 * <p>
 * Parameter update rule:
 * <pre>
 * W = W - eta * gradW
 * b = b - eta * gradB
 * </pre>
 * Performs immediate online parameter updates per sample or mini-batch without momentum buffers.
 */
public class StochasticGradientDescent implements Optimizer {

    private double learningRate;

    public StochasticGradientDescent(double learningRate) {
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be strictly positive, got: " + learningRate);
        }
        this.learningRate = learningRate;
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be strictly positive, got: " + learningRate);
        }
        this.learningRate = learningRate;
    }

    @Override
    public void update(Layer layer, int layerIndex) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer must not be null.");
        }
        Matrix gradW = layer.getGradW();
        Vector gradB = layer.getGradB();

        if (gradW == null || gradB == null) {
            throw new IllegalStateException("Layer gradients must be computed before optimizer update.");
        }

        Matrix newWeights = layer.getWeights().subtract(gradW.multiply(learningRate));
        Vector newBiases = layer.getBiases().subtract(gradB.multiply(learningRate));

        layer.setWeights(newWeights);
        layer.setBiases(newBiases);
    }

    @Override
    public void reset() {
        // Stateless optimizer; no buffers to clear.
    }

    @Override
    public String toString() {
        return String.format("SGD(learningRate=%.4f)", learningRate);
    }
}
