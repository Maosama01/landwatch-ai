package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 tests for Oja's rule Delta w = eta * y * (x - y * w).
 */
class OjaLearningTest {

    private static final double EPSILON = 1e-12;

    @Test
    @DisplayName("Single update matches the hand-computed Delta w = eta * y * (x - y * w)")
    void testSingleUpdateMatchesHandComputation() {
        // w = (0.6, 0.8), x = (1, 0), eta = 0.1
        // y = 0.6
        // x - y*w = (1 - 0.36, 0 - 0.48) = (0.64, -0.48)
        // Delta w = 0.1 * 0.6 * (0.64, -0.48) = (0.0384, -0.0288)
        OjaLearning oja = new OjaLearning(Vector.of(0.6, 0.8), 0.1);
        Vector delta = oja.update(Vector.of(1.0, 0.0));

        assertEquals(0.0384, delta.get(0), EPSILON);
        assertEquals(-0.0288, delta.get(1), EPSILON);
        assertTrue(oja.getWeights().equals(Vector.of(0.6384, 0.7712), EPSILON));
    }

    @Test
    @DisplayName("Oja = Hebbian term minus forgetting term eta * y^2 * w")
    void testDecompositionIntoHebbianAndForgettingTerms() {
        Vector w = Vector.of(0.3, -0.7);
        Vector x = Vector.of(1.5, 0.5);
        double eta = 0.05;

        Vector ojaDelta = new OjaLearning(w, eta).computeDelta(x);
        Vector hebbDelta = new HebbianLearning(w, eta).computeDelta(x);
        double y = w.dot(x);
        Vector forgetting = w.multiply(eta * y * y);

        assertTrue(ojaDelta.equals(hebbDelta.subtract(forgetting), EPSILON));
    }

    @Test
    @DisplayName("Converges to a unit-length first principal component on correlated 2-D data")
    void testConvergesToUnitPrincipalComponent() {
        // Several known principal directions, each with its own data and starting weights.
        double[] anglesDegrees = {30.0, 60.0, 135.0};
        for (int k = 0; k < anglesDegrees.length; k++) {
            double angle = Math.toRadians(anglesDegrees[k]);
            Vector[] data = SyntheticDatasets.correlatedGaussian2D(1000, 2.0, 0.5, angle, new Random(100L + k));
            Vector principal = SyntheticDatasets.unitDirection(angle);   // analytic eigenvector of C

            OjaLearning oja = new OjaLearning(2, 0.005, new Random(200L + k));
            oja.train(data, 30, new Random(300L + k));

            double norm = oja.weightNorm();
            double absCos = Math.abs(LearningRuleUtils.cosineSimilarity(oja.getWeights(), principal));
            assertEquals(1.0, norm, 0.05, "Norm should self-normalise to 1 at angle " + anglesDegrees[k]);
            assertTrue(absCos > 0.99,
                    "|cos| with principal eigenvector must exceed 0.99 at angle " + anglesDegrees[k] + ", got " + absCos);
        }
    }

    @Test
    @DisplayName("Starting far from unit length (|w| = 3), the norm is pulled back to ~1")
    void testNormIsPulledBackFromLargeStart() {
        double angle = Math.toRadians(45.0);
        Vector[] data = SyntheticDatasets.correlatedGaussian2D(1000, 2.0, 0.5, angle, new Random(5L));
        OjaLearning oja = new OjaLearning(Vector.of(0.0, 3.0), 0.002);

        double[] norms = oja.train(data, 20, new Random(6L));

        assertEquals(3.0, norms[0], EPSILON);
        assertEquals(1.0, norms[20], 0.05);
    }

    @Test
    @DisplayName("Oja stays bounded where plain Hebbian explodes (same data, same start)")
    void testOjaBoundedWhileHebbianExplodes() {
        Vector[] data = SyntheticDatasets.correlatedGaussian2D(500, 2.0, 0.5, Math.toRadians(30), new Random(42L));
        Vector start = Vector.of(0.2, 0.3);

        double[] hebbNorms = new HebbianLearning(start, 0.001).train(data, 8, new Random(1L));
        double[] ojaNorms = new OjaLearning(start, 0.001).train(data, 8, new Random(1L));

        assertTrue(hebbNorms[8] > 1000.0, "Hebbian norm should explode, got " + hebbNorms[8]);
        for (double n : ojaNorms) {
            assertTrue(n < 1.1, "Oja norm should stay bounded near 1, got " + n);
        }
        assertEquals(1.0, ojaNorms[8], 0.05);
    }

    @Test
    @DisplayName("Input validation")
    void testValidation() {
        OjaLearning oja = new OjaLearning(Vector.of(1.0, 0.0), 0.1);
        assertThrows(IllegalArgumentException.class, () -> oja.update(Vector.of(1.0)));
        assertThrows(IllegalArgumentException.class, () -> new OjaLearning(Vector.of(1.0, 0.0), 0.0));
        assertThrows(IllegalArgumentException.class, () -> oja.train(new Vector[0], 1, new Random(1L)));
    }
}
