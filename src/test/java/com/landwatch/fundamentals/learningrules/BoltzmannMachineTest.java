package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 tests for the energy-based Boltzmann machine and its learning rule
 * Delta w_ij = eta * (rho_ij(data) - rho_ij(model)).
 */
class BoltzmannMachineTest {

    private static final double EPSILON = 1e-12;

    /** Training set: units 0 and 1 are always equal, unit 2 is independent. */
    private static final int[][] EQUAL_PAIR_PATTERNS = {{0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}};
    /** The complementary states, where units 0 and 1 disagree. */
    private static final int[][] ANTI_CORRELATED_PATTERNS = {{0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}};

    private static BoltzmannMachine fixedMachine() {
        BoltzmannMachine bm = new BoltzmannMachine(3, new Random(1L));
        bm.setWeight(0, 1, 2.0);
        bm.setWeight(0, 2, -1.0);
        bm.setWeight(1, 2, 0.5);
        bm.setBias(0, 0.1);
        bm.setBias(1, -0.2);
        bm.setBias(2, 0.3);
        return bm;
    }

    @Test
    @DisplayName("Energy matches the hand computation E(s) = -sum_{i<j} w_ij s_i s_j - sum_i b_i s_i")
    void testEnergyHandComputation() {
        BoltzmannMachine bm = fixedMachine();
        // s = (1,1,0): E = -(w01) - (b0 + b1) = -2.0 - (0.1 - 0.2) = -1.9
        assertEquals(-1.9, bm.energy(new int[]{1, 1, 0}), EPSILON);
        // s = (1,1,1): E = -(2.0 - 1.0 + 0.5) - (0.1 - 0.2 + 0.3) = -1.5 - 0.2 = -1.7
        assertEquals(-1.7, bm.energy(new int[]{1, 1, 1}), EPSILON);
        // s = (0,0,0): E = 0
        assertEquals(0.0, bm.energy(new int[]{0, 0, 0}), EPSILON);
    }

    @Test
    @DisplayName("Energy equals the matrix form -1/2 s^T W s - b^T s")
    void testEnergyMatchesMatrixForm() {
        BoltzmannMachine bm = fixedMachine();
        Matrix w = bm.getWeights();
        Vector b = bm.getBiases();
        for (int index = 0; index < 8; index++) {
            int[] s = BoltzmannMachine.stateFromIndex(index, 3);
            Vector sv = Vector.of(s[0], s[1], s[2]);
            double matrixForm = -0.5 * sv.dot(w.multiply(sv)) - b.dot(sv);
            assertEquals(matrixForm, bm.energy(s), EPSILON);
        }
    }

    @Test
    @DisplayName("W stays symmetric with zero diagonal; self-connections are rejected")
    void testSymmetricWeights() {
        BoltzmannMachine bm = fixedMachine();
        Matrix w = bm.getWeights();
        assertTrue(w.equals(w.transpose(), 0.0));
        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, w.get(i, i), 0.0);
        }
        assertThrows(IllegalArgumentException.class, () -> bm.setWeight(1, 1, 0.5));
    }

    @Test
    @DisplayName("Gibbs conditional P(s_i=1 | rest) = sigmoid(h_i / T) equals the Boltzmann ratio of energies")
    void testConditionalProbabilityMatchesEnergyGap() {
        BoltzmannMachine bm = fixedMachine();
        for (int index = 0; index < 8; index++) {
            int[] s = BoltzmannMachine.stateFromIndex(index, 3);
            for (int i = 0; i < 3; i++) {
                int[] on = s.clone();
                on[i] = 1;
                int[] off = s.clone();
                off[i] = 0;
                double expected = 1.0 / (1.0 + Math.exp(bm.energy(on) - bm.energy(off)));
                assertEquals(expected, bm.probabilityOn(s, i), 1e-12);
                assertEquals(bm.energy(off) - bm.energy(on), bm.localField(s, i), 1e-12);
            }
        }
    }

    @Test
    @DisplayName("Weight update matches the rule on a hand-checkable case")
    void testWeightUpdateHandCheckable() {
        // Untrained machine: W = 0, b = 0, so all 8 states are equally likely (P = 1/8).
        // Exact model statistics: rho_ij(model) = P(s_i = 1 and s_j = 1) = 1/4, m_i(model) = 1/2.
        BoltzmannMachine bm = new BoltzmannMachine(3, new Random(1L));
        BoltzmannMachine.Statistics model = bm.exactModelStatistics();
        assertEquals(0.25, model.pairwise().get(0, 1), EPSILON);
        assertEquals(0.5, model.means().get(2), EPSILON);

        // Data: patterns (1,1,0) and (0,0,1)
        // rho_01 = (1 + 0)/2 = 0.5, rho_02 = 0, rho_12 = 0, m = (0.5, 0.5, 0.5)
        BoltzmannMachine.Statistics data = BoltzmannMachine.dataStatistics(new int[][]{{1, 1, 0}, {0, 0, 1}});
        assertEquals(0.5, data.pairwise().get(0, 1), EPSILON);
        assertEquals(0.0, data.pairwise().get(0, 2), EPSILON);

        // Delta w_ij = eta * (rho_data - rho_model), eta = 0.1
        // Delta w_01 = 0.1 * (0.5 - 0.25) = +0.025; Delta w_02 = Delta w_12 = 0.1 * (0 - 0.25) = -0.025
        Matrix deltaW = BoltzmannMachine.weightUpdate(data, model, 0.1);
        assertEquals(0.025, deltaW.get(0, 1), EPSILON);
        assertEquals(0.025, deltaW.get(1, 0), EPSILON);
        assertEquals(-0.025, deltaW.get(0, 2), EPSILON);
        assertEquals(-0.025, deltaW.get(1, 2), EPSILON);
        for (int i = 0; i < 3; i++) {
            assertEquals(0.0, deltaW.get(i, i), 0.0, "Diagonal of Delta W must be zero");
        }

        // Delta b_i = eta * (m_data - m_model) = 0.1 * (0.5 - 0.5) = 0
        Vector deltaB = BoltzmannMachine.biasUpdate(data, model, 0.1);
        assertTrue(deltaB.equals(Vector.zeros(3), EPSILON));

        bm.applyUpdate(deltaW, deltaB);
        assertEquals(0.025, bm.getWeight(0, 1), EPSILON);
        assertEquals(0.025, bm.getWeight(1, 0), EPSILON);
        assertEquals(-0.025, bm.getWeight(1, 2), EPSILON);
    }

    @Test
    @DisplayName("Exact probabilities sum to 1 and follow P(s) proportional to exp(-E(s))")
    void testExactDistribution() {
        BoltzmannMachine bm = fixedMachine();
        double total = 0.0;
        for (int index = 0; index < 8; index++) {
            total += bm.probability(BoltzmannMachine.stateFromIndex(index, 3));
        }
        assertEquals(1.0, total, 1e-12);

        int[] a = {1, 1, 0};
        int[] b = {0, 0, 1};
        double ratio = bm.probability(a) / bm.probability(b);
        assertEquals(Math.exp(-(bm.energy(a) - bm.energy(b))), ratio, 1e-12);
    }

    @Test
    @DisplayName("Gibbs-sampling estimate of model statistics agrees with exact enumeration")
    void testGibbsSamplingMatchesExactStatistics() {
        BoltzmannMachine bm = fixedMachine();
        BoltzmannMachine.Statistics exact = bm.exactModelStatistics();
        BoltzmannMachine.Statistics estimated = bm.estimateModelStatistics(20000, 100);

        for (int i = 0; i < 3; i++) {
            assertEquals(exact.means().get(i), estimated.means().get(i), 0.02, "mean of unit " + i);
            for (int j = 0; j < 3; j++) {
                assertEquals(exact.pairwise().get(i, j), estimated.pairwise().get(i, j), 0.02,
                        "rho_" + i + j);
            }
        }
    }

    @Test
    @DisplayName("Training on 'units 0 and 1 always equal': positive w01, correlated samples, lower energy")
    void testLearnsEqualPairCorrelation() {
        BoltzmannMachine bm = new BoltzmannMachine(3, new Random(7L));
        double[] logLikelihood = bm.train(EQUAL_PAIR_PATTERNS, 300, 0.5, 1000, 20);

        // 1. The weight between the always-equal units becomes clearly positive.
        double w01 = bm.getWeight(0, 1);
        assertTrue(w01 > 2.0, "w01 should be strongly positive, got " + w01);
        assertTrue(w01 > Math.abs(bm.getWeight(0, 2)) && w01 > Math.abs(bm.getWeight(1, 2)),
                "w01 should dominate the weights to the independent unit");

        // 2. Free-running samples show s0 == s1 far more often than chance (50%).
        int[][] samples = bm.sample(5000, 100);
        int equal = 0;
        for (int[] s : samples) {
            if (s[0] == s[1]) {
                equal++;
            }
        }
        double fractionEqual = (double) equal / samples.length;
        assertTrue(fractionEqual > 0.85, "Fraction with s0 == s1 was " + fractionEqual);

        // 3. Every training pattern has lower energy than every anti-correlated pattern.
        double maxTrainingEnergy = Double.NEGATIVE_INFINITY;
        for (int[] s : EQUAL_PAIR_PATTERNS) {
            maxTrainingEnergy = Math.max(maxTrainingEnergy, bm.energy(s));
        }
        double minAntiEnergy = Double.POSITIVE_INFINITY;
        for (int[] s : ANTI_CORRELATED_PATTERNS) {
            minAntiEnergy = Math.min(minAntiEnergy, bm.energy(s));
        }
        assertTrue(maxTrainingEnergy < minAntiEnergy,
                "Training energies (max " + maxTrainingEnergy + ") must be below anti-correlated (min "
                        + minAntiEnergy + ")");

        // 4. The data log-likelihood improved from ln(1/8) towards the optimum ln(1/4).
        assertEquals(Math.log(1.0 / 8.0), logLikelihood[0], 1e-12);
        assertTrue(logLikelihood[300] > logLikelihood[0] + 0.5,
                "Log-likelihood should rise substantially: " + logLikelihood[0] + " -> " + logLikelihood[300]);
        assertTrue(logLikelihood[300] <= Math.log(0.25) + 1e-9, "Cannot exceed the data entropy bound");
    }

    @Test
    @DisplayName("Same seed gives an identical training run")
    void testDeterministic() {
        BoltzmannMachine a = new BoltzmannMachine(3, new Random(11L));
        BoltzmannMachine b = new BoltzmannMachine(3, new Random(11L));
        a.train(EQUAL_PAIR_PATTERNS, 20, 0.5, 200, 10);
        b.train(EQUAL_PAIR_PATTERNS, 20, 0.5, 200, 10);
        assertEquals(a.getWeights(), b.getWeights());
        assertEquals(a.getBiases(), b.getBiases());
    }

    @Test
    @DisplayName("stateFromIndex decodes binary indices with unit 0 as the most significant bit")
    void testStateFromIndex() {
        assertArrayEquals(new int[]{1, 1, 0}, BoltzmannMachine.stateFromIndex(6, 3));
        assertArrayEquals(new int[]{0, 0, 1}, BoltzmannMachine.stateFromIndex(1, 3));
        assertThrows(IllegalArgumentException.class, () -> BoltzmannMachine.stateFromIndex(8, 3));
    }

    @Test
    @DisplayName("Input validation: non-binary states and wrong lengths are rejected")
    void testValidation() {
        BoltzmannMachine bm = new BoltzmannMachine(3, new Random(1L));
        assertThrows(IllegalArgumentException.class, () -> bm.energy(new int[]{0, 2, 1}));
        assertThrows(IllegalArgumentException.class, () -> bm.energy(new int[]{0, 1}));
        assertThrows(IllegalArgumentException.class, () -> new BoltzmannMachine(1, new Random(1L)));
        assertThrows(IllegalArgumentException.class, () -> new BoltzmannMachine(3, 0.0, new Random(1L)));
        assertThrows(IllegalArgumentException.class,
                () -> bm.trainStep(new int[][]{{0, 1}}, 0.1, 10, 1));
    }
}
