package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 tests for the plain Hebbian learning rule Delta w = eta * y * x.
 */
class HebbianLearningTest {

    private static final double EPSILON = 1e-12;

    @Test
    @DisplayName("Single unsupervised update matches the hand-computed Delta w = eta * y * x")
    void testSingleUpdateMatchesHandComputation() {
        // w = (0.5, -0.2), x = (1, 2), eta = 0.1
        // y = 0.5*1 + (-0.2)*2 = 0.1
        // Delta w = 0.1 * 0.1 * (1, 2) = (0.01, 0.02)
        HebbianLearning hebb = new HebbianLearning(Vector.of(0.5, -0.2), 0.1);
        Vector x = Vector.of(1.0, 2.0);

        assertEquals(0.1, hebb.output(x), EPSILON);
        Vector delta = hebb.update(x);

        assertEquals(0.01, delta.get(0), EPSILON);
        assertEquals(0.02, delta.get(1), EPSILON);
        assertTrue(hebb.getWeights().equals(Vector.of(0.51, -0.18), EPSILON));
    }

    @Test
    @DisplayName("Supervised update uses the clamped target: Delta w = eta * t * x")
    void testSupervisedUpdate() {
        // Same w and x as above, but the teacher forces the output to t = 1.
        // Delta w = 0.1 * 1 * (1, 2) = (0.1, 0.2), independent of the neuron's own output 0.1.
        HebbianLearning hebb = new HebbianLearning(Vector.of(0.5, -0.2), 0.1);
        Vector delta = hebb.updateSupervised(Vector.of(1.0, 2.0), 1.0);

        assertTrue(delta.equals(Vector.of(0.1, 0.2), EPSILON));
        assertTrue(hebb.getWeights().equals(Vector.of(0.6, 0.0), EPSILON));
    }

    @Test
    @DisplayName("computeDelta does not modify the weights")
    void testComputeDeltaIsPure() {
        HebbianLearning hebb = new HebbianLearning(Vector.of(0.3, 0.4), 0.5);
        Vector before = hebb.getWeights();
        hebb.computeDelta(Vector.of(1.0, -1.0));
        hebb.computeSupervisedDelta(Vector.of(1.0, -1.0), 1.0);
        assertEquals(before, hebb.getWeights());
    }

    @Test
    @DisplayName("A single Hebbian step can never decrease |w| (|w + eta*y*x|^2 = |w|^2 + 2*eta*y^2 + eta^2*y^2*|x|^2)")
    void testSingleStepNeverDecreasesNorm() {
        Random rng = new Random(7L);
        for (int trial = 0; trial < 500; trial++) {
            Vector w = Vector.random(3, -1.0, 1.0, rng);
            Vector x = Vector.random(3, -2.0, 2.0, rng);
            HebbianLearning hebb = new HebbianLearning(w, 0.05);

            double y = hebb.output(x);
            double expectedSquaredNorm = w.dot(w) + 2 * 0.05 * y * y + 0.05 * 0.05 * y * y * x.dot(x);
            hebb.update(x);

            double newNorm = hebb.weightNorm();
            assertEquals(expectedSquaredNorm, newNorm * newNorm, 1e-9);
            assertTrue(newNorm >= w.norm() - 1e-12, "Norm decreased on trial " + trial);
        }
    }

    @Test
    @DisplayName("Weight norm grows without bound over repeated presentations")
    void testWeightNormGrowsWithoutBound() {
        Vector[] data = SyntheticDatasets.correlatedGaussian2D(500, 2.0, 0.5, Math.toRadians(30), new Random(42L));
        HebbianLearning hebb = new HebbianLearning(2, 0.001, new Random(43L));

        double[] norms = hebb.train(data, 8, new Random(44L));

        assertEquals(9, norms.length);
        for (int epoch = 1; epoch < norms.length; epoch++) {
            assertTrue(norms[epoch] > norms[epoch - 1],
                    "Norm must strictly increase every epoch, epoch " + epoch + ": " + norms[epoch]);
        }
        double growthFactor = norms[8] / norms[0];
        assertTrue(growthFactor > 1e4, "Expected explosive growth, got factor " + growthFactor);
        assertTrue(Double.isFinite(norms[8]));
    }

    @Test
    @DisplayName("Hebbian weights align with the principal direction even though their length explodes")
    void testDirectionAlignsWithPrincipalComponent() {
        double angle = Math.toRadians(30);
        Vector[] data = SyntheticDatasets.correlatedGaussian2D(500, 2.0, 0.5, angle, new Random(42L));
        HebbianLearning hebb = new HebbianLearning(2, 0.001, new Random(43L));
        hebb.train(data, 8, new Random(44L));

        double cos = Math.abs(LearningRuleUtils.cosineSimilarity(hebb.getWeights(), SyntheticDatasets.unitDirection(angle)));
        assertTrue(cos > 0.99, "|cos| with principal direction was " + cos);
    }

    @Test
    @DisplayName("Zero initial weights give zero output, so the unsupervised rule never learns")
    void testZeroWeightsNeverLearn() {
        HebbianLearning hebb = new HebbianLearning(Vector.zeros(2), 0.5);
        Vector delta = hebb.update(Vector.of(3.0, -4.0));
        assertEquals(0.0, delta.norm(), EPSILON);
        assertEquals(0.0, hebb.weightNorm(), EPSILON);
    }

    @Test
    @DisplayName("Input validation: dimension mismatch, bad learning rate, null weights")
    void testValidation() {
        HebbianLearning hebb = new HebbianLearning(Vector.of(1.0, 1.0), 0.1);
        assertThrows(IllegalArgumentException.class, () -> hebb.update(Vector.of(1.0, 2.0, 3.0)));
        assertThrows(IllegalArgumentException.class, () -> new HebbianLearning(Vector.of(1.0), 0.0));
        assertThrows(IllegalArgumentException.class, () -> new HebbianLearning(Vector.of(1.0), -0.1));
        assertThrows(IllegalArgumentException.class, () -> new HebbianLearning(null, 0.1));
    }
}
