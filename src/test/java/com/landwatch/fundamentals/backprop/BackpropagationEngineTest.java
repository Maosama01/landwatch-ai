package com.landwatch.fundamentals.backprop;

import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.ActivationFunction;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Rigorous JUnit 5 tests for BackpropagationEngine and the canonical XOR problem.
 */
class BackpropagationEngineTest {

    private static final double EPSILON = 1e-6;

    @Test
    @DisplayName("LossFunction MSE value and analytical gradient")
    void testMseCalculation() {
        Vector pred = Vector.of(0.8, 0.2);
        Vector target = Vector.of(1.0, 0.0);

        // Half-MSE: 0.5 * [(0.8 - 1.0)^2 + (0.2 - 0.0)^2] = 0.5 * [0.04 + 0.04] = 0.04
        double loss = LossFunction.MSE.compute(pred, target);
        assertEquals(0.04, loss, EPSILON);

        // dL/dy_hat = pred - target = [0.8 - 1.0, 0.2 - 0.0] = [-0.2, 0.2]
        Vector grad = LossFunction.MSE.gradient(pred, target);
        assertEquals(2, grad.size());
        assertEquals(-0.2, grad.get(0), EPSILON);
        assertEquals(0.2, grad.get(1), EPSILON);
    }

    @Test
    @DisplayName("Single backpropagation step updates parameters and reduces sample loss")
    void testSingleStepUpdate() {
        MultilayerPerceptron mlp = new MultilayerPerceptron(new int[]{2, 3, 1}, 42L);
        Vector input = Vector.of(1.0, 0.0);
        Vector target = Vector.of(1.0);

        double initialLoss = LossFunction.MSE.compute(mlp.forward(input), target);

        // Execute one SGD step with moderate learning rate
        double stepLoss = BackpropagationEngine.trainStep(mlp, input, target, 0.5, LossFunction.MSE);
        assertEquals(initialLoss, stepLoss, EPSILON);

        // Post-update loss on the same sample must decrease
        double postLoss = LossFunction.MSE.compute(mlp.forward(input), target);
        assertTrue(postLoss < initialLoss,
                String.format("Loss must decrease after gradient step: initial=%.6f, post=%.6f", initialLoss, postLoss));
    }

    @Test
    @DisplayName("XOR problem demonstration: Multilayer Perceptron learns non-linear XOR function")
    void testXorLearning() {
        // Architecture: 2 inputs -> 4 hidden units (Sigmoid) -> 1 output (Sigmoid)
        int[] topology = {2, 4, 1};
        long seed = 42L;
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                topology, ActivationFunction.SIGMOID, ActivationFunction.SIGMOID, seed);

        // XOR Truth Table
        List<Vector> inputs = List.of(
                Vector.of(0.0, 0.0),
                Vector.of(0.0, 1.0),
                Vector.of(1.0, 0.0),
                Vector.of(1.0, 1.0)
        );
        List<Vector> targets = List.of(
                Vector.of(0.0),
                Vector.of(1.0),
                Vector.of(1.0),
                Vector.of(0.0)
        );

        int epochs = 6000;
        double learningRate = 2.0;

        // Execute actual backpropagation training loop
        BackpropagationEngine.TrainingResult result = BackpropagationEngine.train(
                mlp, inputs, targets, epochs, learningRate, LossFunction.MSE);

        // Print training summary for demonstration auditing
        System.out.println("=== XOR TRAINING DEMONSTRATION ===");
        System.out.println(String.format("Initial Loss: %.6f", result.getInitialLoss()));
        System.out.println(String.format("Final Loss:   %.6f", result.getFinalLoss()));
        System.out.println(String.format("Reduction:    %.2f%%", result.getLossReductionPercentage()));

        // Verification 1: Loss must drop significantly (greater than 90% reduction)
        assertTrue(result.getInitialLoss() > 0.10, "Initial loss should be significant");
        assertTrue(result.getFinalLoss() < 0.01, "Final loss must be < 0.01. Actual: " + result.getFinalLoss());
        assertTrue(result.getLossReductionPercentage() > 95.0, "Loss reduction must exceed 95%");

        // Verification 2: Check predictions for all 4 XOR patterns
        double pred00 = mlp.predict(inputs.get(0)).get(0);
        double pred01 = mlp.predict(inputs.get(1)).get(0);
        double pred10 = mlp.predict(inputs.get(2)).get(0);
        double pred11 = mlp.predict(inputs.get(3)).get(0);

        System.out.println(String.format("Pattern [0, 0] -> Target: 0.0, Predicted: %.4f", pred00));
        System.out.println(String.format("Pattern [0, 1] -> Target: 1.0, Predicted: %.4f", pred01));
        System.out.println(String.format("Pattern [1, 0] -> Target: 1.0, Predicted: %.4f", pred10));
        System.out.println(String.format("Pattern [1, 1] -> Target: 0.0, Predicted: %.4f", pred11));

        // Actual predictions must classify the non-linear XOR boundary
        assertTrue(pred00 < 0.15, "XOR(0,0) must predict near 0. Got: " + pred00);
        assertTrue(pred01 > 0.85, "XOR(0,1) must predict near 1. Got: " + pred01);
        assertTrue(pred10 > 0.85, "XOR(1,0) must predict near 1. Got: " + pred10);
        assertTrue(pred11 < 0.15, "XOR(1,1) must predict near 0. Got: " + pred11);
    }

    @Test
    @DisplayName("Invalid inputs and dimensions throw IllegalArgumentException")
    void testInvalidDimensions() {
        MultilayerPerceptron mlp = new MultilayerPerceptron(new int[]{2, 2, 1}, 1L);

        List<Vector> inputs = List.of(Vector.of(1.0, 0.0));
        List<Vector> mismatchedTargets = List.of(Vector.of(1.0, 2.0)); // target size 2 != output size 1

        assertThrows(IllegalArgumentException.class, () ->
                BackpropagationEngine.trainStep(mlp, inputs.get(0), mismatchedTargets.get(0), 0.1, LossFunction.MSE));

        assertThrows(IllegalArgumentException.class, () ->
                BackpropagationEngine.train(mlp, inputs, List.of(), 10, 0.1, LossFunction.MSE));

        assertThrows(IllegalArgumentException.class, () ->
                BackpropagationEngine.train(mlp, inputs, List.of(Vector.of(1.0)), -5, 0.1, LossFunction.MSE));
    }
}
