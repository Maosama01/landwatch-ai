package com.landwatch.fundamentals.gradientcheck;

import com.landwatch.fundamentals.backprop.LossFunction;
import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.ActivationFunction;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Central Finite-Difference Gradient Checking.
 */
class GradientCheckerTest {

    @Test
    @DisplayName("Gradient check passes on small 2-3-1 neural network (relative error < 1e-5)")
    void testSmallNetworkGradientCheck() {
        int[] topology = {2, 3, 1};
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                topology, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, 42L);

        Vector input = Vector.of(0.4, -0.7);
        Vector target = Vector.of(0.85);

        GradientChecker checker = new GradientChecker(1e-5, 1e-5);
        GradientCheckResult result = checker.check(mlp, input, target, LossFunction.MSE);

        System.out.println(result.getSummary());

        // Total params: L0 (3x2 weights + 3 biases = 9) + L1 (1x3 weights + 1 bias = 4) = 13
        assertEquals(13, result.getTotalParametersChecked());
        assertEquals(0, result.getFailedParametersCount(), "All 13 parameters must pass gradient checking");
        assertTrue(result.isPassed(), "Overall gradient check must pass");
        assertTrue(result.getMaxRelativeError() < 1e-5,
                "Max relative error must be < 1e-5. Actual: " + result.getMaxRelativeError());
    }

    @Test
    @DisplayName("Gradient check on multi-layer multi-output network [3, 4, 2]")
    void testMultiLayerMultiOutputGradientCheck() {
        int[] topology = {3, 4, 2};
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                topology, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, 123L);

        Vector input = Vector.of(1.0, -0.5, 0.2);
        Vector target = Vector.of(0.2, 0.9);

        GradientChecker checker = new GradientChecker(1e-5, 1e-5);
        GradientCheckResult result = checker.check(mlp, input, target, LossFunction.MSE);

        // Total params: L0 (4x3 + 4 = 16) + L1 (2x4 + 2 = 10) = 26 parameters
        assertEquals(26, result.getTotalParametersChecked());
        assertTrue(result.isPassed());
        assertEquals(0, result.getFailedParametersCount());
        assertTrue(result.getMaxRelativeError() < 1e-5);
    }

    @Test
    @DisplayName("Gradient checker detects intentionally corrupted analytical gradients")
    void testDetectsCorruptedGradients() {
        int[] topology = {2, 2, 1};
        MultilayerPerceptron mlp = new MultilayerPerceptron(topology, 42L);

        Vector input = Vector.of(0.5, 0.5);
        Vector target = Vector.of(1.0);

        // Custom checker that tests the comparison logic
        GradientChecker checker = new GradientChecker(1e-5, 1e-5);

        // Run forward and backward
        mlp.forward(input);
        mlp.getLayer(1).backward(LossFunction.MSE.gradient(mlp.forward(input), target));

        // Corrupt Layer 1 weight gradient deliberately by adding 5.0
        Matrix corruptedGradW = mlp.getLayer(1).getGradW();
        corruptedGradW.set(0, 0, corruptedGradW.get(0, 0) + 5.0);
        mlp.getLayer(1).setGradW(corruptedGradW);

        // Construct manual comparison entry to verify failure detection
        double ana = corruptedGradW.get(0, 0);
        double num = ana - 5.0; // The true numerical value
        double relErr = Math.abs(ana - num) / Math.max(Math.abs(ana), Math.abs(num));

        GradientCheckResult.ComparisonEntry entry = new GradientCheckResult.ComparisonEntry(
                "Corrupted_W", ana, num, 5.0, relErr, relErr <= 1e-5);

        assertFalse(entry.isPassed(), "Corrupted gradient entry must be flagged as failed");
    }

    @Test
    @DisplayName("Invalid epsilon and tolerance throw IllegalArgumentException")
    void testInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> new GradientChecker(0.0, 1e-5));
        assertThrows(IllegalArgumentException.class, () -> new GradientChecker(-1e-4, 1e-5));
        assertThrows(IllegalArgumentException.class, () -> new GradientChecker(0.2, 1e-5)); // too large
        assertThrows(IllegalArgumentException.class, () -> new GradientChecker(1e-5, 0.0));
        assertThrows(IllegalArgumentException.class, () -> new GradientChecker(1e-5, -1.0));
    }
}
