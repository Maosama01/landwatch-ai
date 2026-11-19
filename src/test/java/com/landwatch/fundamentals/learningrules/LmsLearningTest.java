package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 tests for the Widrow-Hoff LMS / delta rule Delta w = eta * (t - w . x - b) * x.
 */
class LmsLearningTest {

    private static final double EPSILON = 1e-12;

    @Test
    @DisplayName("Single update matches the hand-computed delta-rule step")
    void testSingleUpdateMatchesHandComputation() {
        // w = (0,0), b = 0, eta = 0.1, x = (1,2), t = 3
        // y = 0, e = 3
        // Delta w = 0.1 * 3 * (1,2) = (0.3, 0.6), Delta b = 0.3
        LmsLearning lms = new LmsLearning(Vector.of(0.0, 0.0), 0.0, 0.1);

        double error = lms.update(Vector.of(1.0, 2.0), 3.0);

        assertEquals(3.0, error, EPSILON);
        assertTrue(lms.getWeights().equals(Vector.of(0.3, 0.6), EPSILON));
        assertEquals(0.3, lms.getBias(), EPSILON);
        // New output 0.3 + 1.2 + 0.3 = 1.8: the error shrank from 3 to 1.2.
        assertEquals(1.8, lms.output(Vector.of(1.0, 2.0)), EPSILON);
    }

    @Test
    @DisplayName("Unlike the perceptron, LMS keeps learning while the error is non-zero")
    void testUpdatesEvenWithCorrectSign() {
        // Output 1.0 already has the right sign for target 2.0, but the error is still 1.0.
        LmsLearning lms = new LmsLearning(Vector.of(1.0), 0.0, 0.5);
        double error = lms.update(Vector.of(1.0), 2.0);
        assertEquals(1.0, error, EPSILON);
        assertEquals(1.5, lms.getWeights().get(0), EPSILON);
    }

    @Test
    @DisplayName("Recovers the coefficients of a noiseless linear function")
    void testRecoversLinearCoefficients() {
        Vector trueWeights = Vector.of(2.0, -3.0, 0.5);
        double trueBias = 1.0;
        Random rng = new Random(42L);
        Vector[] inputs = new Vector[200];
        double[] targets = new double[200];
        for (int n = 0; n < inputs.length; n++) {
            inputs[n] = Vector.random(3, -1.0, 1.0, rng);
            targets[n] = trueWeights.dot(inputs[n]) + trueBias;
        }

        LmsLearning lms = new LmsLearning(3, 0.05, new Random(43L));
        double[] mse = lms.train(inputs, targets, 100);

        assertTrue(lms.getWeights().equals(trueWeights, 1e-6), "Learned " + lms.getWeights());
        assertEquals(trueBias, lms.getBias(), 1e-6);
        assertTrue(mse[0] > 1.0, "Initial MSE should be large, got " + mse[0]);
        assertTrue(mse[100] < 1e-12, "Final MSE should be ~0, got " + mse[100]);
    }

    @Test
    @DisplayName("MSE decreases over epochs on the linear problem")
    void testMseDecreases() {
        Random rng = new Random(9L);
        Vector[] inputs = new Vector[50];
        double[] targets = new double[50];
        for (int n = 0; n < inputs.length; n++) {
            inputs[n] = Vector.random(2, -1.0, 1.0, rng);
            targets[n] = -1.5 * inputs[n].get(0) + 0.75 * inputs[n].get(1) - 0.25;
        }
        LmsLearning lms = new LmsLearning(2, 0.02, new Random(10L));
        double[] mse = lms.train(inputs, targets, 50);
        for (int epoch = 1; epoch < mse.length; epoch++) {
            assertTrue(mse[epoch] < mse[epoch - 1], "MSE rose at epoch " + epoch);
        }
    }

    @Test
    @DisplayName("Input validation")
    void testValidation() {
        LmsLearning lms = new LmsLearning(Vector.of(0.0, 0.0), 0.0, 0.1);
        assertThrows(IllegalArgumentException.class, () -> lms.update(Vector.of(1.0), 1.0));
        assertThrows(IllegalArgumentException.class,
                () -> lms.train(new Vector[]{Vector.of(1.0, 1.0)}, new double[]{1.0, 2.0}, 1));
        assertThrows(IllegalArgumentException.class, () -> new LmsLearning(Vector.of(0.0), 0.0, -1.0));
    }
}
