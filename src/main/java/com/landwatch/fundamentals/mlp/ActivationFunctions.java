package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Vector;

/**
 * Utility helper class exposing static activation functions and derivatives.
 */
public final class ActivationFunctions {

    private ActivationFunctions() {}

    public static final ActivationFunction SIGMOID = ActivationFunction.SIGMOID;
    public static final ActivationFunction RELU = ActivationFunction.RELU;
    public static final ActivationFunction LINEAR = ActivationFunction.LINEAR;

    public static double sigmoid(double z) {
        return SIGMOID.activate(z);
    }

    public static double sigmoidDerivative(double z) {
        return SIGMOID.derivative(z);
    }

    public static Vector sigmoid(Vector z) {
        return SIGMOID.activate(z);
    }

    public static Vector sigmoidDerivative(Vector z) {
        return SIGMOID.derivative(z);
    }

    public static double relu(double z) {
        return RELU.activate(z);
    }

    public static double reluDerivative(double z) {
        return RELU.derivative(z);
    }

    public static Vector relu(Vector z) {
        return RELU.activate(z);
    }

    public static Vector reluDerivative(Vector z) {
        return RELU.derivative(z);
    }
}
