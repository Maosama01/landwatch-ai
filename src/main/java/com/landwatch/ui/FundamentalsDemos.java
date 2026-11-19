package com.landwatch.ui;

import com.landwatch.fundamentals.backprop.BackpropagationEngine;
import com.landwatch.fundamentals.backprop.LossFunction;
import com.landwatch.fundamentals.gradientcheck.GradientCheckResult;
import com.landwatch.fundamentals.gradientcheck.GradientChecker;
import com.landwatch.fundamentals.learningrules.LearningRulesDemo;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.ActivationFunction;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;

import java.util.List;
import java.util.Locale;

/**
 * Runs the project's pure-Java neural-network fundamentals for the dashboard and returns their output
 * as text. Every number is computed live by {@code com.landwatch.fundamentals}; the configurations are
 * the seeded ones used in the unit tests, so the output is reproducible.
 */
public final class FundamentalsDemos {

    /** Seed used by BackpropagationEngineTest and GradientCheckerTest. */
    public static final long SEED = 42L;
    public static final int XOR_EPOCHS = 6000;
    public static final double XOR_LEARNING_RATE = 2.0;

    private FundamentalsDemos() {
    }

    /**
     * XOR with a 2-4-1 sigmoid MLP trained by backpropagation and plain SGD
     * (the configuration of {@code BackpropagationEngineTest#testXorLearning}).
     */
    public static String runXor() {
        long t0 = System.nanoTime();
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                new int[]{2, 4, 1}, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, SEED);
        List<Vector> inputs = List.of(Vector.of(0.0, 0.0), Vector.of(0.0, 1.0), Vector.of(1.0, 0.0), Vector.of(1.0, 1.0));
        List<Vector> targets = List.of(Vector.of(0.0), Vector.of(1.0), Vector.of(1.0), Vector.of(0.0));

        BackpropagationEngine.TrainingResult result = BackpropagationEngine.train(
                mlp, inputs, targets, XOR_EPOCHS, XOR_LEARNING_RATE, LossFunction.MSE);
        long ms = (System.nanoTime() - t0) / 1_000_000;

        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US, "XOR: 2-4-1 MLP (sigmoid), backpropagation + plain SGD, learning rate %.1f,%n",
                XOR_LEARNING_RATE));
        sb.append(String.format(Locale.US, "%d epochs x 4 patterns, half-MSE loss, seed %d "
                + "(as in BackpropagationEngineTest#testXorLearning)%n%n", XOR_EPOCHS, SEED));

        sb.append("Mean training loss per epoch (epoch 0 = before training):\n");
        List<Double> history = result.getLossHistory();
        int[] epochs = {0, 100, 500, 1000, 2000, 3000, 4000, 5000, XOR_EPOCHS};
        int shown = 0;
        for (int e : epochs) {
            if (e < history.size()) {
                sb.append(String.format(Locale.US, "  epoch %4d: %.6f", e, history.get(e)));
                if (++shown % 3 == 0) {
                    sb.append('\n');
                }
            }
        }
        if (shown % 3 != 0) {
            sb.append('\n');
        }
        sb.append('\n');
        sb.append(String.format(Locale.US, "Initial loss : %.6f%n", result.getInitialLoss()));
        sb.append(String.format(Locale.US, "Final loss   : %.6f   (%.2f %% lower)%n%n",
                result.getFinalLoss(), result.getLossReductionPercentage()));

        int correct = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Vector x = inputs.get(i);
            double t = targets.get(i).get(0);
            double p = mlp.predict(x).get(0);
            boolean ok = (p >= 0.5) == (t >= 0.5);
            if (ok) {
                correct++;
            }
            sb.append(String.format(Locale.US, "  [%.0f, %.0f] -> target %.0f, predicted %.4f  %s%n",
                    x.get(0), x.get(1), t, p, ok ? "correct" : "WRONG"));
        }
        sb.append(String.format(Locale.US, "Patterns classified correctly at threshold 0.5: %d / 4%n", correct));
        sb.append("XOR is not linearly separable, so a single perceptron cannot learn it; the hidden layer can.\n");
        sb.append(String.format(Locale.US, "Wall time: %d ms%n", ms));
        return sb.toString();
    }

    /**
     * Central finite-difference gradient check of backpropagation on a 2-3-1 sigmoid MLP
     * (the configuration of {@code GradientCheckerTest#testSmallNetworkGradientCheck}).
     */
    public static String runGradientCheck() {
        long t0 = System.nanoTime();
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                new int[]{2, 3, 1}, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, SEED);
        Vector input = Vector.of(0.4, -0.7);
        Vector target = Vector.of(0.85);
        GradientChecker checker = new GradientChecker(1e-5, 1e-5);
        GradientCheckResult result = checker.check(mlp, input, target, LossFunction.MSE);
        long ms = (System.nanoTime() - t0) / 1_000_000;

        StringBuilder sb = new StringBuilder();
        sb.append("Gradient check: analytical (backpropagation) vs numerical gradients\n");
        sb.append("numerical: (L(w + eps) - L(w - eps)) / (2 eps) for every weight and bias\n");
        sb.append(String.format(Locale.US, "2-3-1 sigmoid MLP, seed %d, input (0.4, -0.7), target 0.85, half-MSE loss%n", SEED));
        sb.append("(same configuration as GradientCheckerTest#testSmallNetworkGradientCheck)\n\n");
        sb.append(result.getSummary()).append("\n\n");
        sb.append("Per parameter:\n");
        for (GradientCheckResult.ComparisonEntry e : result.getComparisons()) {
            sb.append("  ").append(e).append('\n');
        }
        sb.append(String.format(Locale.US, "%nWall time: %d ms%n", ms));
        return sb.toString();
    }

    /** The learning-rules demonstration (Hebbian, Oja, competitive, perceptron, LMS, Boltzmann). */
    public static String runLearningRules() {
        long t0 = System.nanoTime();
        String out = LearningRulesDemo.runToString();
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return out + String.format(Locale.US, "%nWall time: %d ms%n", ms);
    }
}
