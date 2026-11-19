package com.landwatch.fundamentals.backprop;

import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.Layer;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;
import com.landwatch.fundamentals.optimizers.Optimizer;
import com.landwatch.fundamentals.optimizers.StochasticGradientDescent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Backpropagation Engine implementing the Generalized Delta Rule and Optimizer integration.
 * <p>
 * Evaluates exact analytical gradients using the chain rule:
 * <pre>
 * 1. Forward Pass: a^[l] = f(W^[l] * a^[l-1] + b^[l])
 * 2. Output Layer Error: delta^[L] = grad_a Loss (x) f'(z^[L])
 * 3. Hidden Layer Error Propagation: delta^[l] = ((W^[l+1])^T * delta^[l+1]) (x) f'(z^[l])
 * 4. Weight Gradient: dLoss / dW^[l] = delta^[l] * (a^[l-1])^T
 * 5. Bias Gradient: dLoss / db^[l] = delta^[l]
 * 6. Optimizer Update: delegated to Optimizer (SGD, Momentum, Nesterov, BGD)
 * </pre>
 */
public class BackpropagationEngine {

    /**
     * Executes a single forward and backward pass for a single training sample,
     * updates network parameters using the provided optimizer, and returns the sample loss.
     *
     * @param mlp          the multilayer perceptron model
     * @param input        sample input vector
     * @param target       sample target vector
     * @param optimizer    parameter optimizer (e.g. SGD, Momentum, Nesterov)
     * @param lossFunction loss function (e.g. MSE)
     * @return loss before parameter update
     */
    public static double trainStep(MultilayerPerceptron mlp,
                                  Vector input,
                                  Vector target,
                                  Optimizer optimizer,
                                  LossFunction lossFunction) {
        Objects.requireNonNull(mlp, "MLP must not be null.");
        Objects.requireNonNull(input, "Input vector must not be null.");
        Objects.requireNonNull(target, "Target vector must not be null.");
        Objects.requireNonNull(optimizer, "Optimizer must not be null.");
        Objects.requireNonNull(lossFunction, "Loss function must not be null.");

        // For Nesterov Accelerated Gradient: shift parameters to lookahead point prior to forward pass
        if (optimizer instanceof com.landwatch.fundamentals.optimizers.NesterovOptimizer nag) {
            nag.applyLookahead(mlp);
        }

        // 1. Forward pass (evaluates at lookahead if NAG, or current parameters if standard)
        Vector output = mlp.forward(input);

        // 2. Compute loss and output loss gradient dLoss / dOutput
        double loss = lossFunction.compute(output, target);
        Vector upstreamGrad = lossFunction.gradient(output, target);

        // 3. Backward propagation through layers in reverse order (L down to 1)
        List<Layer> layers = mlp.getLayers();
        Vector currentGrad = upstreamGrad;
        for (int l = layers.size() - 1; l >= 0; l--) {
            currentGrad = layers.get(l).backward(currentGrad);
        }

        // 4. Parameter update via configured Optimizer
        optimizer.update(mlp);

        return loss;
    }

    /**
     * Legacy/Convenience overload using default SGD with fixed learning rate.
     */
    public static double trainStep(MultilayerPerceptron mlp,
                                  Vector input,
                                  Vector target,
                                  double learningRate,
                                  LossFunction lossFunction) {
        return trainStep(mlp, input, target, new StochasticGradientDescent(learningRate), lossFunction);
    }

    /**
     * Computes the mean loss over a dataset without modifying weights.
     */
    public static double computeAverageLoss(MultilayerPerceptron mlp,
                                           List<Vector> inputs,
                                           List<Vector> targets,
                                           LossFunction lossFunction) {
        if (inputs.isEmpty() || inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must be non-empty and equal in length.");
        }
        double totalLoss = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            Vector pred = mlp.forward(inputs.get(i));
            totalLoss += lossFunction.compute(pred, targets.get(i));
        }
        return totalLoss / inputs.size();
    }

    /**
     * Trains an MLP over multiple epochs using the given Optimizer.
     */
    public static TrainingResult train(MultilayerPerceptron mlp,
                                       List<Vector> inputs,
                                       List<Vector> targets,
                                       int epochs,
                                       Optimizer optimizer,
                                       LossFunction lossFunction) {
        if (epochs <= 0) {
            throw new IllegalArgumentException("Epochs must be positive: " + epochs);
        }
        if (inputs.isEmpty() || inputs.size() != targets.size()) {
            throw new IllegalArgumentException("Inputs and targets must be non-empty and of matching size.");
        }

        double initialLoss = computeAverageLoss(mlp, inputs, targets, lossFunction);
        List<Double> lossHistory = new ArrayList<>(epochs + 1);
        lossHistory.add(initialLoss);

        for (int ep = 1; ep <= epochs; ep++) {
            double epochCumulativeLoss = 0.0;
            for (int i = 0; i < inputs.size(); i++) {
                double sampleLoss = trainStep(mlp, inputs.get(i), targets.get(i), optimizer, lossFunction);
                epochCumulativeLoss += sampleLoss;
            }
            double avgEpochLoss = epochCumulativeLoss / inputs.size();
            lossHistory.add(avgEpochLoss);
        }

        double finalLoss = computeAverageLoss(mlp, inputs, targets, lossFunction);
        return new TrainingResult(initialLoss, finalLoss, lossHistory, epochs);
    }

    /**
     * Overload for training using a scalar learning rate (defaults to SGD).
     */
    public static TrainingResult train(MultilayerPerceptron mlp,
                                       List<Vector> inputs,
                                       List<Vector> targets,
                                       int epochs,
                                       double learningRate,
                                       LossFunction lossFunction) {
        return train(mlp, inputs, targets, epochs, new StochasticGradientDescent(learningRate), lossFunction);
    }

    /**
     * Encapsulates the training history and metrics across epochs.
     */
    public static class TrainingResult {
        private final double initialLoss;
        private final double finalLoss;
        private final List<Double> lossHistory;
        private final int epochs;

        public TrainingResult(double initialLoss, double finalLoss, List<Double> lossHistory, int epochs) {
            this.initialLoss = initialLoss;
            this.finalLoss = finalLoss;
            this.lossHistory = new ArrayList<>(lossHistory);
            this.epochs = epochs;
        }

        public double getInitialLoss() { return initialLoss; }
        public double getFinalLoss() { return finalLoss; }
        public List<Double> getLossHistory() { return Collections.unmodifiableList(lossHistory); }
        public int getEpochs() { return epochs; }

        public double getLossReduction() {
            return initialLoss - finalLoss;
        }

        public double getLossReductionPercentage() {
            return initialLoss > 0 ? (getLossReduction() / initialLoss) * 100.0 : 0.0;
        }

        @Override
        public String toString() {
            return String.format("TrainingResult(epochs=%d, initialLoss=%.6f, finalLoss=%.6f, reduction=%.2f%%)",
                    epochs, initialLoss, finalLoss, getLossReductionPercentage());
        }
    }
}
