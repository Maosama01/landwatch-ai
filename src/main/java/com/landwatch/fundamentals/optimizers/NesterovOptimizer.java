package com.landwatch.fundamentals.optimizers;

import com.landwatch.fundamentals.backprop.LossFunction;
import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.Layer;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Nesterov Accelerated Gradient (NAG) Optimizer (Nesterov, 1983).
 * <p>
 * Standard mathematical formulation evaluating gradients at the momentum lookahead parameters:
 * <pre>
 * 1. Compute Lookahead Parameters:
 *    lookahead = theta - beta * previousVelocity
 * 2. Evaluate Gradient AT Lookahead:
 *    gradient = grad_loss(lookahead)
 * 3. Update Velocity:
 *    velocity = beta * previousVelocity + learningRate * gradient
 * 4. Update Original Parameters:
 *    theta = theta - velocity
 * </pre>
 * By evaluating the loss gradient at the anticipated position (theta - beta * velocity)
 * rather than the current position, NAG anticipates the slope of the landscape ahead,
 * providing adaptive braking that prevents overshooting oscillations.
 */
public class NesterovOptimizer implements Optimizer {

    private double learningRate;
    private double beta; // Momentum lookahead factor in [0.0, 1.0)

    // Velocity state buffers per layer index
    private final Map<Integer, Matrix> velocityW;
    private final Map<Integer, Vector> velocityB;

    // Original parameter caches when lookahead is applied to network layers
    private final Map<Integer, Matrix> originalW;
    private final Map<Integer, Vector> originalB;
    private final Map<Integer, Boolean> lookaheadActive;

    public NesterovOptimizer(double learningRate, double beta) {
        if (learningRate <= 0.0) {
            throw new IllegalArgumentException("Learning rate must be strictly positive, got: " + learningRate);
        }
        if (beta < 0.0 || beta >= 1.0) {
            throw new IllegalArgumentException("Nesterov beta must be in range [0.0, 1.0), got: " + beta);
        }
        this.learningRate = learningRate;
        this.beta = beta;
        this.velocityW = new HashMap<>();
        this.velocityB = new HashMap<>();
        this.originalW = new HashMap<>();
        this.originalB = new HashMap<>();
        this.lookaheadActive = new HashMap<>();
    }

    public NesterovOptimizer(double learningRate) {
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
            throw new IllegalArgumentException("Nesterov beta must be in range [0.0, 1.0), got: " + beta);
        }
        this.beta = beta;
    }

    // --- Lookahead Phase ---

    /**
     * Shifts a layer's parameters to the lookahead point:
     * W_lookahead = W - beta * V_W
     * b_lookahead = b - beta * V_b
     * Stores original parameters to apply the final velocity update later.
     *
     * @param layer      the layer to shift
     * @param layerIndex index of the layer
     */
    public void applyLookahead(Layer layer, int layerIndex) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer must not be null.");
        }

        Matrix origW = layer.getWeights();
        Vector origB = layer.getBiases();

        originalW.put(layerIndex, origW);
        originalB.put(layerIndex, origB);

        Matrix vW = velocityW.get(layerIndex);
        Vector vB = velocityB.get(layerIndex);

        if (vW != null && vB != null) {
            // Shift to lookahead point: theta_lookahead = theta - beta * V
            Matrix lookaheadW = origW.subtract(vW.multiply(beta));
            Vector lookaheadB = origB.subtract(vB.multiply(beta));

            layer.setWeights(lookaheadW);
            layer.setBiases(lookaheadB);
        }

        lookaheadActive.put(layerIndex, true);
    }

    /**
     * Shifts all layers of an MLP to their respective lookahead points.
     */
    public void applyLookahead(MultilayerPerceptron mlp) {
        Objects.requireNonNull(mlp, "MLP must not be null.");
        for (int i = 0; i < mlp.getNumLayers(); i++) {
            applyLookahead(mlp.getLayer(i), i);
        }
    }

    // --- Update Phase ---

    /**
     * Updates the layer parameters using the gradient evaluated at the lookahead point:
     * velocity = beta * velocity + learningRate * gradAtLookahead
     * theta = theta_original - velocity
     */
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

        // Retrieve or initialize velocity buffers
        Matrix vW = velocityW.computeIfAbsent(layerIndex, k -> Matrix.zeros(layer.getOutputSize(), layer.getInputSize()));
        Vector vB = velocityB.computeIfAbsent(layerIndex, k -> Vector.zeros(layer.getOutputSize()));

        // Update velocity using gradient evaluated at lookahead: V = beta * V + eta * gradAtLookahead
        Matrix newVW = vW.multiply(beta).add(gradW.multiply(learningRate));
        Vector newVB = vB.multiply(beta).add(gradB.multiply(learningRate));

        velocityW.put(layerIndex, newVW);
        velocityB.put(layerIndex, newVB);

        // Retrieve original parameters (or current if lookahead was not pre-applied)
        Matrix origW = originalW.getOrDefault(layerIndex, layer.getWeights());
        Vector origB = originalB.getOrDefault(layerIndex, layer.getBiases());

        // Update parameters: theta = theta_original - V
        Matrix newWeights = origW.subtract(newVW);
        Vector newBiases = origB.subtract(newVB);

        layer.setWeights(newWeights);
        layer.setBiases(newBiases);

        // Reset lookahead state for this layer
        lookaheadActive.put(layerIndex, false);
        originalW.remove(layerIndex);
        originalB.remove(layerIndex);
    }

    // --- Convenience Pure Numerical Methods ---

    /**
     * Computes the lookahead vector for a given parameter vector:
     * lookahead = theta - beta * velocity
     */
    public Vector computeLookahead(Vector theta, int layerIndex) {
        Vector v = velocityB.get(layerIndex);
        if (v == null) {
            return theta.copy();
        }
        return theta.subtract(v.multiply(beta));
    }

    /**
     * Computes the lookahead matrix for a given parameter matrix.
     */
    public Matrix computeLookahead(Matrix theta, int layerIndex) {
        Matrix v = velocityW.get(layerIndex);
        if (v == null) {
            return theta.copy();
        }
        return theta.subtract(v.multiply(beta));
    }

    /**
     * Executes a direct Nesterov step on a parameter vector given its gradient evaluated at lookahead:
     * velocity = beta * velocity + eta * gradAtLookahead
     * theta_new = theta_orig - velocity
     */
    public Vector stepVector(Vector origTheta, Vector gradAtLookahead, int layerIndex) {
        Vector v = velocityB.computeIfAbsent(layerIndex, k -> Vector.zeros(origTheta.size()));
        Vector newV = v.multiply(beta).add(gradAtLookahead.multiply(learningRate));
        velocityB.put(layerIndex, newV);
        return origTheta.subtract(newV);
    }

    /**
     * Executes a direct Nesterov step on a parameter matrix given its gradient evaluated at lookahead.
     */
    public Matrix stepMatrix(Matrix origTheta, Matrix gradAtLookahead, int layerIndex) {
        Matrix v = velocityW.computeIfAbsent(layerIndex, k -> Matrix.zeros(origTheta.getRows(), origTheta.getCols()));
        Matrix newV = v.multiply(beta).add(gradAtLookahead.multiply(learningRate));
        velocityW.put(layerIndex, newV);
        return origTheta.subtract(newV);
    }

    /**
     * Self-contained complete Nesterov training step on an MLP:
     * 1. Applies lookahead to all layers
     * 2. Evaluates forward pass and backward pass at lookahead
     * 3. Updates velocities and restores/updates original parameters
     */
    public double step(MultilayerPerceptron mlp, Vector input, Vector target, LossFunction lossFunction) {
        applyLookahead(mlp);
        Vector output = mlp.forward(input);
        double loss = lossFunction.compute(output, target);
        Vector upstreamGrad = lossFunction.gradient(output, target);

        for (int l = mlp.getNumLayers() - 1; l >= 0; l--) {
            upstreamGrad = mlp.getLayer(l).backward(upstreamGrad);
        }

        update(mlp);
        return loss;
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
        originalW.clear();
        originalB.clear();
        lookaheadActive.clear();
    }

    @Override
    public String toString() {
        return String.format("NesterovOptimizer(learningRate=%.4f, beta=%.2f)", learningRate, beta);
    }
}
