package com.landwatch.fundamentals.optimizers;

import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.Layer;

import java.util.HashMap;
import java.util.Map;

/**
 * Classical Momentum Optimizer (Polyak, 1964).
 * <p>
 * Incorporates past gradient velocity to accelerate convergence along shallow ravines
 * and dampen oscillations in high-curvature directions:
 * <pre>
 * V_W = beta * V_W + eta * gradW
 * V_b = beta * V_b + eta * gradB
 * W = W - V_W
 * b = b - V_b
 * </pre>
 */
public class MomentumOptimizer implements Optimizer {

    private double learningRate;
    private double beta; // Momentum coefficient (typically 0.9)

    // Layer-indexed velocity buffers
    private final Map<Integer, Matrix> velocityW;
    private final Map<Integer, Vector> velocityB;

    public MomentumOptimizer(double learningRate, double beta) {
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be strictly positive, got: " + learningRate);
        }
        if (beta < 0.0 || beta >= 1.0) {
            throw new IllegalArgumentException("Momentum beta must be in range [0.0, 1.0), got: " + beta);
        }
        this.learningRate = learningRate;
        this.beta = beta;
        this.velocityW = new HashMap<>();
        this.velocityB = new HashMap<>();
    }

    public MomentumOptimizer(double learningRate) {
        this(learningRate, 0.9);
    }

    @Override
    public double getLearningRate() {
        return learningRate;
    }

    @Override
    public void setLearningRate(double learningRate) {
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be positive, got: " + learningRate);
        }
        this.learningRate = learningRate;
    }

    public double getBeta() {
        return beta;
    }

    public void setBeta(double beta) {
        if (beta < 0.0 || beta >= 1.0) {
            throw new IllegalArgumentException("Momentum beta must be in range [0.0, 1.0), got: " + beta);
        }
        this.beta = beta;
    }

    @Override
    public void update(Layer layer, int layerIndex) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer must not be null.");
        }
        Matrix gradW = layer.getGradW();
        Vector gradB = layer.getGradB();

        if (gradW == null || gradB == null) {
            throw new IllegalStateException("Layer gradients must be present before optimizer update.");
        }

        // Initialize velocity buffers if first step for this layer
        Matrix vW = velocityW.computeIfAbsent(layerIndex, k -> Matrix.zeros(layer.getOutputSize(), layer.getInputSize()));
        Vector vB = velocityB.computeIfAbsent(layerIndex, k -> Vector.zeros(layer.getOutputSize()));

        // Update velocity: V = beta * V + eta * grad
        Matrix newVW = vW.multiply(beta).add(gradW.multiply(learningRate));
        Vector newVB = vB.multiply(beta).add(gradB.multiply(learningRate));

        velocityW.put(layerIndex, newVW);
        velocityB.put(layerIndex, newVB);

        // Apply parameter update: theta = theta - V
        Matrix newWeights = layer.getWeights().subtract(newVW);
        Vector newBiases = layer.getBiases().subtract(newVB);

        layer.setWeights(newWeights);
        layer.setBiases(newBiases);
    }

    public Matrix getVelocityW(int layerIndex) {
        Matrix v = velocityW.get(layerIndex);
        return v != null ? v.copy() : null;
    }

    public Vector getVelocityB(int layerIndex) {
        Vector v = velocityB.get(layerIndex);
        return v != null ? v.copy() : null;
    }

    @Override
    public void reset() {
        velocityW.clear();
        velocityB.clear();
    }

    @Override
    public String toString() {
        return String.format("MomentumOptimizer(learningRate=%.4f, beta=%.2f)", learningRate, beta);
    }
}
