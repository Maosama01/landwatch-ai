package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 tests for Rosenblatt's perceptron rule Delta w = eta * (t - y) * x.
 */
class PerceptronLearningTest {

    private static final double EPSILON = 1e-12;
    private static final Vector[] INPUTS = SyntheticDatasets.logicGateInputs();

    @Test
    @DisplayName("Single update matches the hand-computed error-correction step")
    void testSingleUpdateMatchesHandComputation() {
        // w = (0,0), b = 0, eta = 0.1, x = (1,1), t = 0
        // v = 0 -> y = step(0) = 1 (threshold uses >= 0), so e = t - y = -1
        // Delta w = 0.1 * (-1) * (1,1) = (-0.1, -0.1), Delta b = -0.1
        PerceptronLearning p = new PerceptronLearning(Vector.of(0.0, 0.0), 0.0, 0.1);

        int error = p.update(Vector.of(1.0, 1.0), 0);

        assertEquals(-1, error);
        assertTrue(p.getWeights().equals(Vector.of(-0.1, -0.1), EPSILON));
        assertEquals(-0.1, p.getBias(), EPSILON);
    }

    @Test
    @DisplayName("Correctly classified sample leaves the weights unchanged")
    void testNoUpdateWhenCorrect() {
        PerceptronLearning p = new PerceptronLearning(Vector.of(1.0, 1.0), -1.5, 0.1);
        // (1,1): v = 0.5 -> y = 1 = t
        assertEquals(0, p.update(Vector.of(1.0, 1.0), 1));
        assertEquals(Vector.of(1.0, 1.0), p.getWeights());
        assertEquals(-1.5, p.getBias(), 0.0);
    }

    @Test
    @DisplayName("AND gate: converges and classifies all four patterns perfectly")
    void testLearnsAnd() {
        int[] targets = SyntheticDatasets.andTargets();
        PerceptronLearning p = new PerceptronLearning(2, 0.1, new Random(42L));

        PerceptronLearning.TrainingResult result = p.train(INPUTS, targets, 100);

        assertTrue(result.converged(), "AND is linearly separable: " + result);
        assertEquals(0, result.errorsInLastEpoch());
        assertEquals(1.0, p.accuracy(INPUTS, targets), EPSILON);
        for (int n = 0; n < INPUTS.length; n++) {
            assertEquals(targets[n], p.predict(INPUTS[n]));
        }
    }

    @Test
    @DisplayName("OR gate: converges and classifies all four patterns perfectly")
    void testLearnsOr() {
        int[] targets = SyntheticDatasets.orTargets();
        PerceptronLearning p = new PerceptronLearning(2, 0.1, new Random(42L));

        PerceptronLearning.TrainingResult result = p.train(INPUTS, targets, 100);

        assertTrue(result.converged(), "OR is linearly separable: " + result);
        assertEquals(1.0, p.accuracy(INPUTS, targets), EPSILON);
    }

    @Test
    @DisplayName("AND and OR also converge from zero weights and from several random starts")
    void testConvergesFromManyStarts() {
        for (long seed = 0; seed < 20; seed++) {
            PerceptronLearning pAnd = new PerceptronLearning(2, 0.1, new Random(seed));
            PerceptronLearning pOr = new PerceptronLearning(2, 0.1, new Random(seed + 1000));
            assertTrue(pAnd.train(INPUTS, SyntheticDatasets.andTargets(), 100).converged(), "AND seed " + seed);
            assertTrue(pOr.train(INPUTS, SyntheticDatasets.orTargets(), 100).converged(), "OR seed " + seed);
        }
        PerceptronLearning zero = new PerceptronLearning(Vector.zeros(2), 0.0, 1.0);
        assertTrue(zero.train(INPUTS, SyntheticDatasets.andTargets(), 100).converged());
    }

    @Test
    @DisplayName("XOR: does NOT reach 100% within 1000 epochs (linear-separability limit)")
    void testCannotLearnXor() {
        int[] targets = SyntheticDatasets.xorTargets();
        PerceptronLearning p = new PerceptronLearning(2, 0.1, new Random(42L));

        PerceptronLearning.TrainingResult result = p.train(INPUTS, targets, 1000);

        assertFalse(result.converged(), "A single perceptron must not solve XOR");
        assertEquals(1000, result.epochsRun());
        for (int errors : result.errorsPerEpoch()) {
            assertTrue(errors > 0, "Every epoch must contain at least one misclassification");
        }
        double accuracy = p.accuracy(INPUTS, targets);
        assertTrue(accuracy < 1.0);
        assertTrue(accuracy <= 0.75 + EPSILON, "No line can get more than 3 of 4 XOR patterns right");
    }

    @Test
    @DisplayName("XOR accuracy never exceeds 75% at any epoch boundary, from several random starts")
    void testXorAccuracyBoundedAtEveryEpoch() {
        int[] targets = SyntheticDatasets.xorTargets();
        for (long seed = 0; seed < 5; seed++) {
            PerceptronLearning p = new PerceptronLearning(2, 0.1, new Random(seed));
            for (int epoch = 0; epoch < 200; epoch++) {
                assertFalse(p.train(INPUTS, targets, 1).converged());
                assertTrue(p.accuracy(INPUTS, targets) <= 0.75 + EPSILON);
            }
        }
    }

    @Test
    @DisplayName("Input validation: non-binary targets and mismatched lengths")
    void testValidation() {
        PerceptronLearning p = new PerceptronLearning(Vector.of(0.0, 0.0), 0.0, 0.1);
        assertThrows(IllegalArgumentException.class, () -> p.update(Vector.of(1.0, 1.0), 2));
        assertThrows(IllegalArgumentException.class, () -> p.train(INPUTS, new int[]{0, 1}, 10));
        assertThrows(IllegalArgumentException.class, () -> p.train(INPUTS, SyntheticDatasets.andTargets(), 0));
        assertThrows(IllegalArgumentException.class, () -> p.predict(Vector.of(1.0)));
    }
}
