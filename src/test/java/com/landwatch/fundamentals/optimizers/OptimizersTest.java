package com.landwatch.fundamentals.optimizers;

import com.landwatch.fundamentals.backprop.LossFunction;
import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import com.landwatch.fundamentals.mlp.ActivationFunction;
import com.landwatch.fundamentals.mlp.Layer;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Optimizers: BGD, SGD, Momentum, and Nesterov Accelerated Gradient.
 */
class OptimizersTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("SGD updates weights and biases according to W = W - eta * grad")
    void testSgdUpdate() {
        Matrix w = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Vector b = Vector.of(0.5, -0.5);
        Layer layer = new Layer(w, b, ActivationFunction.LINEAR);

        // Mock gradients
        layer.setGradW(new Matrix(new double[][]{{0.1, 0.2}, {0.3, 0.4}}));
        layer.setGradB(Vector.of(0.05, -0.05));

        StochasticGradientDescent sgd = new StochasticGradientDescent(0.1);
        sgd.update(layer, 0);

        // Expected W: 1.0 - 0.1*0.1 = 0.99, 2.0 - 0.1*0.2 = 1.98, etc.
        assertEquals(0.99, layer.getWeights().get(0, 0), EPSILON);
        assertEquals(1.98, layer.getWeights().get(0, 1), EPSILON);
        assertEquals(2.97, layer.getWeights().get(1, 0), EPSILON);
        assertEquals(3.96, layer.getWeights().get(1, 1), EPSILON);

        // Expected b: 0.5 - 0.1*0.05 = 0.495, -0.5 - 0.1*(-0.05) = -0.495
        assertEquals(0.495, layer.getBiases().get(0), EPSILON);
        assertEquals(-0.495, layer.getBiases().get(1), EPSILON);
    }

    @Test
    @DisplayName("Batch Gradient Descent averages gradients across multiple samples")
    void testBatchGradientDescentAveraging() {
        MultilayerPerceptron mlp = new MultilayerPerceptron(new int[]{2, 2, 1}, 42L);

        List<Vector> inputs = List.of(
                Vector.of(1.0, 0.0),
                Vector.of(0.0, 1.0)
        );
        List<Vector> targets = List.of(
                Vector.of(1.0),
                Vector.of(0.0)
        );

        BatchGradientDescent bgd = new BatchGradientDescent(0.5);
        double loss1 = bgd.step(mlp, inputs, targets, LossFunction.MSE);
        double loss2 = bgd.step(mlp, inputs, targets, LossFunction.MSE);

        assertTrue(loss1 > 0.0, "Batch loss should be positive");
        assertTrue(loss2 < loss1, "Loss must decrease after batch gradient descent step");
    }

    @Test
    @DisplayName("Momentum optimizer accumulates velocity across multiple steps")
    void testMomentumAccumulation() {
        Matrix w = new Matrix(new double[][]{{1.0, 2.0}});
        Vector b = Vector.of(0.0);
        Layer layer = new Layer(w, b, ActivationFunction.LINEAR);

        double lr = 0.1;
        double beta = 0.9;
        MomentumOptimizer momentum = new MomentumOptimizer(lr, beta);

        // Step 1: grad = [1.0, 2.0]
        layer.setGradW(new Matrix(new double[][]{{1.0, 2.0}}));
        layer.setGradB(Vector.of(1.0));
        momentum.update(layer, 0);

        // V1 = 0.9*0 + 0.1*1.0 = 0.1 for W[0,0]
        // W1 = 1.0 - 0.1 = 0.9
        assertEquals(0.1, momentum.getVelocityW(0).get(0, 0), EPSILON);
        assertEquals(0.9, layer.getWeights().get(0, 0), EPSILON);

        // Step 2: same grad = [1.0, 2.0]
        layer.setGradW(new Matrix(new double[][]{{1.0, 2.0}}));
        layer.setGradB(Vector.of(1.0));
        momentum.update(layer, 0);

        // V2 = beta * V1 + lr * grad = 0.9 * 0.1 + 0.1 * 1.0 = 0.09 + 0.10 = 0.19
        // W2 = W1 - V2 = 0.9 - 0.19 = 0.71
        assertEquals(0.19, momentum.getVelocityW(0).get(0, 0), EPSILON);
        assertEquals(0.71, layer.getWeights().get(0, 0), EPSILON);
    }

    @Test
    @DisplayName("Nesterov optimizer evaluates gradients at lookahead and updates parameters")
    void testNesterovUpdates() {
        Matrix w = new Matrix(new double[][]{{1.0, 2.0}});
        Vector b = Vector.of(0.0);
        Layer layer = new Layer(w, b, ActivationFunction.LINEAR);

        double lr = 0.1;
        double beta = 0.9;
        NesterovOptimizer nag = new NesterovOptimizer(lr, beta);

        // Step 1: Initial velocity = 0, so lookahead == current parameter
        nag.applyLookahead(layer, 0);
        assertEquals(1.0, layer.getWeights().get(0, 0), EPSILON, "At step 1 with V=0, lookahead equals initial weight");

        // Evaluated gradient at lookahead: [1.0, 2.0]
        layer.setGradW(new Matrix(new double[][]{{1.0, 2.0}}));
        layer.setGradB(Vector.of(1.0));
        nag.update(layer, 0);

        // V1 = beta * 0 + lr * grad = 0.1 * 1.0 = 0.10
        // W1 = W_orig - V1 = 1.0 - 0.10 = 0.90
        assertEquals(0.1, nag.getVelocityW(0).get(0, 0), EPSILON);
        assertEquals(0.90, layer.getWeights().get(0, 0), EPSILON);

        // Step 2: Apply lookahead before gradient evaluation
        nag.applyLookahead(layer, 0);
        // W_lookahead = W1 - beta * V1 = 0.90 - 0.9 * 0.10 = 0.90 - 0.09 = 0.81
        assertEquals(0.81, layer.getWeights().get(0, 0), EPSILON, "Lookahead weight must be W - beta * V = 0.81");

        // Evaluated gradient at lookahead: [1.0, 2.0]
        layer.setGradW(new Matrix(new double[][]{{1.0, 2.0}}));
        layer.setGradB(Vector.of(1.0));
        nag.update(layer, 0);

        // V2 = beta * V1 + lr * grad = 0.9 * 0.10 + 0.1 * 1.0 = 0.09 + 0.10 = 0.19
        // W2 = W1_orig - V2 = 0.90 - 0.19 = 0.71
        assertEquals(0.19, nag.getVelocityW(0).get(0, 0), EPSILON);
        assertEquals(0.71, layer.getWeights().get(0, 0), EPSILON);
    }

    @Test
    @DisplayName("Deterministic distinction: Nesterov lookahead gradient vs Classical Momentum on quadratic objective")
    void testNesterovVsMomentumDistinction() {
        // Objective: f(theta) = theta^2, analytical gradient g(theta) = 2 * theta
        // Minimum at theta* = 0.0
        double initialTheta = 10.0;
        double lr = 0.1;
        double beta = 0.9;

        // 1. Classical Momentum Simulation
        MomentumOptimizer momentum = new MomentumOptimizer(lr, beta);
        Vector thetaMom = Vector.of(initialTheta);

        // Step 1: Momentum
        Vector gradMom1 = Vector.of(2.0 * thetaMom.get(0)); // g(10) = 20.0
        Layer layerMom = new Layer(new Matrix(new double[][]{{thetaMom.get(0)}}), Vector.of(0.0), ActivationFunction.LINEAR);
        layerMom.setGradW(new Matrix(new double[][]{{gradMom1.get(0)}}));
        layerMom.setGradB(Vector.of(0.0));
        momentum.update(layerMom, 0);
        double thetaMomStep1 = layerMom.getWeights().get(0, 0); // 10.0 - (0.9*0 + 0.1*20) = 8.0

        // Step 2: Momentum evaluates gradient at CURRENT theta_1 = 8.0
        double gradMomStep2 = 2.0 * thetaMomStep1; // g(8.0) = 16.0
        layerMom.setGradW(new Matrix(new double[][]{{gradMomStep2}}));
        momentum.update(layerMom, 0);
        // V2 = 0.9*2.0 + 0.1*16.0 = 1.8 + 1.6 = 3.40
        // theta2 = 8.0 - 3.40 = 4.60
        double thetaMomStep2 = layerMom.getWeights().get(0, 0);

        // 2. Standard Nesterov Accelerated Gradient Simulation
        NesterovOptimizer nag = new NesterovOptimizer(lr, beta);
        Layer layerNag = new Layer(new Matrix(new double[][]{{initialTheta}}), Vector.of(0.0), ActivationFunction.LINEAR);

        // Step 1: NAG (lookahead = 10.0 - 0 = 10.0)
        nag.applyLookahead(layerNag, 0);
        double gradNag1 = 2.0 * layerNag.getWeights().get(0, 0); // g(10) = 20.0
        layerNag.setGradW(new Matrix(new double[][]{{gradNag1}}));
        nag.update(layerNag, 0);
        double thetaNagStep1 = layerNag.getWeights().get(0, 0); // 8.0

        // Step 2: NAG shifts to LOOKAHEAD point before gradient evaluation
        nag.applyLookahead(layerNag, 0);
        // lookahead = 8.0 - 0.9 * 2.0 = 6.20!
        double lookaheadPoint = layerNag.getWeights().get(0, 0);
        assertEquals(6.20, lookaheadPoint, EPSILON, "Nesterov lookahead must evaluate to 6.20");

        // Evaluates gradient AT lookahead: g(6.20) = 12.40! (strictly less than Momentum's 16.0)
        double gradNagStep2 = 2.0 * lookaheadPoint;
        assertEquals(12.40, gradNagStep2, EPSILON, "Gradient at lookahead must be 2 * 6.20 = 12.40");

        layerNag.setGradW(new Matrix(new double[][]{{gradNagStep2}}));
        nag.update(layerNag, 0);
        // V2 = 0.9*2.0 + 0.1*12.40 = 1.80 + 1.24 = 3.04
        // theta2 = 8.0 - 3.04 = 4.96
        double thetaNagStep2 = layerNag.getWeights().get(0, 0);

        // 3. Verifications distinguishing Nesterov from Momentum
        assertEquals(8.0, thetaMomStep1, EPSILON);
        assertEquals(8.0, thetaNagStep1, EPSILON);

        assertEquals(16.00, gradMomStep2, EPSILON, "Momentum evaluates gradient at current position 8.0");
        assertEquals(12.40, gradNagStep2, EPSILON, "Nesterov evaluates gradient at lookahead position 6.20");

        assertEquals(3.40, momentum.getVelocityW(0).get(0, 0), EPSILON, "Momentum step 2 velocity = 3.40");
        assertEquals(3.04, nag.getVelocityW(0).get(0, 0), EPSILON, "Nesterov step 2 velocity = 3.04");

        assertEquals(4.60, thetaMomStep2, EPSILON, "Momentum position after step 2 = 4.60");
        assertEquals(4.96, thetaNagStep2, EPSILON, "Nesterov position after step 2 = 4.96");

        // Nesterov braked early because lookahead slope was shallower towards the minimum
        assertTrue(thetaNagStep2 > thetaMomStep2,
                "Nesterov must exhibit braking effect compared to classical Momentum");
    }

    @Test
    @DisplayName("Optimizer reset clears internal state buffers")
    void testOptimizerReset() {
        MomentumOptimizer momentum = new MomentumOptimizer(0.1, 0.9);
        Layer layer = new Layer(2, 2, ActivationFunction.LINEAR, 42L);
        layer.setGradW(Matrix.ones(2, 2));
        layer.setGradB(Vector.ones(2));

        momentum.update(layer, 0);
        assertNotNull(momentum.getVelocityW(0));

        momentum.reset();
        assertNull(momentum.getVelocityW(0), "Reset must clear velocity state");
    }

    @Test
    @DisplayName("Invalid optimizer parameters throw IllegalArgumentException")
    void testInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> new StochasticGradientDescent(0.0));
        assertThrows(IllegalArgumentException.class, () -> new StochasticGradientDescent(-0.5));

        assertThrows(IllegalArgumentException.class, () -> new MomentumOptimizer(0.1, 1.0));
        assertThrows(IllegalArgumentException.class, () -> new MomentumOptimizer(0.1, -0.1));

        assertThrows(IllegalArgumentException.class, () -> new NesterovOptimizer(0.1, 1.5));
    }
}
