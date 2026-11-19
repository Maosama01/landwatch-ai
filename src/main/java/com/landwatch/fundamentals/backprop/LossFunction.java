package com.landwatch.fundamentals.backprop;

import com.landwatch.fundamentals.math.Vector;

import java.util.Objects;

/**
 * Loss function interface for supervised neural network training.
 */
public interface LossFunction {

    /**
     * Computes scalar loss between network predictions and ground-truth targets.
     *
     * @param predicted network output vector
     * @param target    ground-truth target vector
     * @return scalar loss
     */
    double compute(Vector predicted, Vector target);

    /**
     * Computes the gradient of the loss with respect to the network predictions: dLoss / dPredicted.
     *
     * @param predicted network output vector
     * @param target    ground-truth target vector
     * @return gradient vector
     */
    Vector gradient(Vector predicted, Vector target);

    /**
     * Mean Squared Error (Half-MSE):
     * L = 0.5 * sum((y_hat - y)^2)
     * dL/dy_hat = (y_hat - y)
     */
    LossFunction MSE = new LossFunction() {
        @Override
        public double compute(Vector predicted, Vector target) {
            validate(predicted, target);
            double sum = 0.0;
            for (int i = 0; i < predicted.size(); i++) {
                double diff = predicted.get(i) - target.get(i);
                sum += diff * diff;
            }
            return 0.5 * sum;
        }

        @Override
        public Vector gradient(Vector predicted, Vector target) {
            validate(predicted, target);
            Vector grad = new Vector(predicted.size());
            for (int i = 0; i < predicted.size(); i++) {
                grad.set(i, predicted.get(i) - target.get(i));
            }
            return grad;
        }

        private void validate(Vector p, Vector t) {
            Objects.requireNonNull(p, "Predicted vector must not be null.");
            Objects.requireNonNull(t, "Target vector must not be null.");
            if (p.size() != t.size()) {
                throw new IllegalArgumentException(String.format(
                        "Loss dimension mismatch: predicted has size %d, target has size %d", p.size(), t.size()));
            }
        }

        @Override
        public String toString() {
            return "MeanSquaredError(Half-MSE)";
        }
    };
}
