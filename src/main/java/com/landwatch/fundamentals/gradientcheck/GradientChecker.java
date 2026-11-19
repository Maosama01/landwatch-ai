package com.landwatch.fundamentals.gradientcheck;

import com.landwatch.fundamentals.backprop.LossFunction;
import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.Layer;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Numerical Gradient Checker using Central Finite Differences.
 * <p>
 * Verifies analytical backpropagation gradients against the two-sided central difference approximation:
 * <pre>
 * numericalGradient = (L(theta + epsilon) - L(theta - epsilon)) / (2 * epsilon)
 * </pre>
 * Evaluates absolute and relative errors for every weight and bias parameter across all layers.
 */
public class GradientChecker {

    public static final double DEFAULT_EPSILON = 1e-5;
    public static final double DEFAULT_TOLERANCE = 1e-5;

    private final double epsilon;
    private final double tolerance;

    public GradientChecker(double epsilon, double tolerance) {
        if (epsilon <= 0.0 || epsilon >= 0.1) {
            throw new IllegalArgumentException("Epsilon must be in range (0.0, 0.1), got: " + epsilon);
        }
        if (tolerance <= 0.0) {
            throw new IllegalArgumentException("Tolerance must be positive, got: " + tolerance);
        }
        this.epsilon = epsilon;
        this.tolerance = tolerance;
    }

    public GradientChecker() {
        this(DEFAULT_EPSILON, DEFAULT_TOLERANCE);
    }

    /**
     * Executes numerical gradient checking on a MultilayerPerceptron model given a test sample.
     *
     * @param mlp          neural network model
     * @param input        sample input vector
     * @param target       sample target vector
     * @param lossFunction loss function (e.g. LossFunction.MSE)
     * @return comprehensive verification result
     */
    public GradientCheckResult check(MultilayerPerceptron mlp,
                                     Vector input,
                                     Vector target,
                                     LossFunction lossFunction) {
        Objects.requireNonNull(mlp, "MLP must not be null.");
        Objects.requireNonNull(input, "Input must not be null.");
        Objects.requireNonNull(target, "Target must not be null.");
        Objects.requireNonNull(lossFunction, "Loss function must not be null.");

        // 1. Compute analytical gradients using Backpropagation
        Vector output = mlp.forward(input);
        Vector upstreamGrad = lossFunction.gradient(output, target);

        List<Layer> layers = mlp.getLayers();
        Vector currentGrad = upstreamGrad;
        for (int l = layers.size() - 1; l >= 0; l--) {
            currentGrad = layers.get(l).backward(currentGrad);
        }

        List<GradientCheckResult.ComparisonEntry> comparisons = new ArrayList<>();
        double maxRelError = 0.0;
        double maxAbsError = 0.0;
        int failedCount = 0;
        int totalParams = 0;

        // 2. Iterate through each layer and test both weights and biases
        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            Matrix weights = layer.getWeights();
            Vector biases = layer.getBiases();
            Matrix analyticalGradW = layer.getGradW();
            Vector analyticalGradB = layer.getGradB();

            // A. Check Weight Parameters
            for (int r = 0; r < weights.getRows(); r++) {
                for (int c = 0; c < weights.getCols(); c++) {
                    totalParams++;
                    double originalVal = weights.get(r, c);

                    // Perturb theta + epsilon
                    weights.set(r, c, originalVal + epsilon);
                    layer.setWeights(weights);
                    double lossPlus = lossFunction.compute(mlp.forward(input), target);

                    // Perturb theta - epsilon
                    weights.set(r, c, originalVal - epsilon);
                    layer.setWeights(weights);
                    double lossMinus = lossFunction.compute(mlp.forward(input), target);

                    // Restore original parameter
                    weights.set(r, c, originalVal);
                    layer.setWeights(weights);

                    // Central finite difference
                    double numGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
                    double anaGrad = analyticalGradW.get(r, c);

                    double absError = Math.abs(anaGrad - numGrad);
                    double divisor = Math.max(Math.max(Math.abs(anaGrad), Math.abs(numGrad)), 1e-8);
                    double relError = absError / divisor;

                    if (absError > maxAbsError) maxAbsError = absError;
                    if (relError > maxRelError) maxRelError = relError;

                    boolean ok = relError <= tolerance;
                    if (!ok) failedCount++;

                    comparisons.add(new GradientCheckResult.ComparisonEntry(
                            String.format("L%d_W[%d,%d]", l, r, c), anaGrad, numGrad, absError, relError, ok));
                }
            }

            // B. Check Bias Parameters
            for (int r = 0; r < biases.size(); r++) {
                totalParams++;
                double originalVal = biases.get(r);

                // Perturb theta + epsilon
                biases.set(r, originalVal + epsilon);
                layer.setBiases(biases);
                double lossPlus = lossFunction.compute(mlp.forward(input), target);

                // Perturb theta - epsilon
                biases.set(r, originalVal - epsilon);
                layer.setBiases(biases);
                double lossMinus = lossFunction.compute(mlp.forward(input), target);

                // Restore original parameter
                biases.set(r, originalVal);
                layer.setBiases(biases);

                double numGrad = (lossPlus - lossMinus) / (2.0 * epsilon);
                double anaGrad = analyticalGradB.get(r);

                double absError = Math.abs(anaGrad - numGrad);
                double divisor = Math.max(Math.max(Math.abs(anaGrad), Math.abs(numGrad)), 1e-8);
                double relError = absError / divisor;

                if (absError > maxAbsError) maxAbsError = absError;
                if (relError > maxRelError) maxRelError = relError;

                boolean ok = relError <= tolerance;
                if (!ok) failedCount++;

                comparisons.add(new GradientCheckResult.ComparisonEntry(
                        String.format("L%d_b[%d]", l, r), anaGrad, numGrad, absError, relError, ok));
            }
        }

        boolean overallPassed = (failedCount == 0);
        return new GradientCheckResult(
                overallPassed, maxRelError, maxAbsError, tolerance, epsilon, totalParams, failedCount, comparisons);
    }

    public double getEpsilon() { return epsilon; }
    public double getTolerance() { return tolerance; }
}
