package com.landwatch.fundamentals.optimizers;

import com.landwatch.fundamentals.backprop.LossFunction;
import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.Layer;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Batch Gradient Descent (BGD) Optimizer.
 * <p>
 * Accumulates exact analytical gradients across all N training samples in the batch,
 * averages the gradient tensors, and executes a single parameter update per epoch:
 * <pre>
 * W = W - eta * (1 / N) * sum(gradW_i)
 * b = b - eta * (1 / N) * sum(gradB_i)
 * </pre>
 * Guarantees a monotonic descent on convex surfaces without gradient noise.
 */
public class BatchGradientDescent implements Optimizer {

    private double learningRate;

    public BatchGradientDescent(double learningRate) {
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

    /**
     * Executes one complete batch step over all samples in the dataset:
     * 1. Accumulates gradients across all N samples
     * 2. Averages accumulated gradients by dividing by N
     * 3. Updates network weights and biases
     *
     * @param mlp          neural network model
     * @param inputs       batch feature vectors
     * @param targets      batch ground-truth vectors
     * @param lossFunction loss function
     * @return average batch loss
     */
    public double step(MultilayerPerceptron mlp,
                      List<Vector> inputs,
                      List<Vector> targets,
                      LossFunction lossFunction) {
        Objects.requireNonNull(mlp, "MLP must not be null.");
        Objects.requireNonNull(inputs, "Inputs must not be null.");
        Objects.requireNonNull(targets, "Targets must not be null.");
        Objects.requireNonNull(lossFunction, "Loss function must not be null.");

        int n = inputs.size();
        if (n == 0 || n != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must be non-empty and matching in size.");
        }

        int numLayers = mlp.getNumLayers();
        List<Matrix> accumulatedGradW = new ArrayList<>(numLayers);
        List<Vector> accumulatedGradB = new ArrayList<>(numLayers);

        for (int l = 0; l < numLayers; l++) {
            Layer layer = mlp.getLayer(l);
            accumulatedGradW.add(Matrix.zeros(layer.getOutputSize(), layer.getInputSize()));
            accumulatedGradB.add(Vector.zeros(layer.getOutputSize()));
        }

        double totalLoss = 0.0;

        // 1. Accumulate gradients over all samples
        for (int i = 0; i < n; i++) {
            Vector pred = mlp.forward(inputs.get(i));
            totalLoss += lossFunction.compute(pred, targets.get(i));
            Vector upstreamGrad = lossFunction.gradient(pred, targets.get(i));

            // Backpropagate
            Vector currentGrad = upstreamGrad;
            for (int l = numLayers - 1; l >= 0; l--) {
                Layer layer = mlp.getLayer(l);
                currentGrad = layer.backward(currentGrad);

                // Accumulate sample gradients
                accumulatedGradW.set(l, accumulatedGradW.get(l).add(layer.getGradW()));
                accumulatedGradB.set(l, accumulatedGradB.get(l).add(layer.getGradB()));
            }
        }

        // 2. Average gradients and update parameters
        double invN = 1.0 / n;
        for (int l = 0; l < numLayers; l++) {
            Layer layer = mlp.getLayer(l);
            Matrix avgGradW = accumulatedGradW.get(l).multiply(invN);
            Vector avgGradB = accumulatedGradB.get(l).multiply(invN);

            layer.setGradW(avgGradW);
            layer.setGradB(avgGradB);

            // Apply parameter update
            update(layer, l);
        }

        return totalLoss * invN;
    }

    @Override
    public void update(Layer layer, int layerIndex) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer must not be null.");
        }
        Matrix gradW = layer.getGradW();
        Vector gradB = layer.getGradB();

        if (gradW == null || gradB == null) {
            throw new IllegalStateException("Layer gradients must be present before update.");
        }

        layer.setWeights(layer.getWeights().subtract(gradW.multiply(learningRate)));
        layer.setBiases(layer.getBiases().subtract(gradB.multiply(learningRate)));
    }

    @Override
    public void reset() {
        // Stateless optimizer
    }

    @Override
    public String toString() {
        return String.format("BatchGradientDescent(learningRate=%.4f)", learningRate);
    }
}
