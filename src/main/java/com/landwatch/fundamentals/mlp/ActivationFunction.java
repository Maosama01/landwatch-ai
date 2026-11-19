package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.neurons.SigmoidNeuron;

import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

/**
 * Common activation functions and their analytical derivatives for neural networks.
 */
public enum ActivationFunction {

    /**
     * Logistic Sigmoid: f(z) = 1 / (1 + exp(-z)).
     * Derivative: f'(z) = f(z) * (1 - f(z)).
     */
    SIGMOID {
        @Override
        public double activate(double z) {
            return SigmoidNeuron.sigmoid(z);
        }

        @Override
        public double derivative(double z) {
            return SigmoidNeuron.sigmoidDerivative(z);
        }

        @Override
        public double derivativeFromActivation(double a) {
            return SigmoidNeuron.derivativeFromActivation(a);
        }
    },

    /**
     * Rectified Linear Unit: f(z) = max(0, z).
     * Derivative: f'(z) = 1 if z > 0 else 0.
     */
    RELU {
        @Override
        public double activate(double z) {
            return Math.max(0.0, z);
        }

        @Override
        public double derivative(double z) {
            return z > 0.0 ? 1.0 : 0.0;
        }

        @Override
        public double derivativeFromActivation(double a) {
            return a > 0.0 ? 1.0 : 0.0;
        }
    },

    /**
     * Linear (Identity) activation: f(z) = z.
     * Derivative: f'(z) = 1.
     */
    LINEAR {
        @Override
        public double activate(double z) {
            return z;
        }

        @Override
        public double derivative(double z) {
            return 1.0;
        }

        @Override
        public double derivativeFromActivation(double a) {
            return 1.0;
        }
    };

    /**
     * Computes scalar activation f(z).
     */
    public abstract double activate(double z);

    /**
     * Computes scalar derivative f'(z) given pre-activation z.
     */
    public abstract double derivative(double z);

    /**
     * Computes scalar derivative f'(a) directly from activated value a = f(z).
     */
    public abstract double derivativeFromActivation(double a);

    /**
     * Applies activation element-wise across a Vector.
     */
    public Vector activate(Vector z) {
        Objects.requireNonNull(z, "Vector must not be null.");
        return z.map(this::activate);
    }

    /**
     * Applies derivative element-wise across a pre-activation Vector z.
     */
    public Vector derivative(Vector z) {
        Objects.requireNonNull(z, "Vector must not be null.");
        return z.map(this::derivative);
    }

    /**
     * Applies derivative element-wise across an activation Vector a.
     */
    public Vector derivativeFromActivation(Vector a) {
        Objects.requireNonNull(a, "Vector must not be null.");
        return a.map(this::derivativeFromActivation);
    }
}
