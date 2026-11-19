package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * JUnit 5 tests for winner-take-all competitive learning.
 */
class CompetitiveLearningTest {

    private static final double EPSILON = 1e-12;

    @Test
    @DisplayName("Single update: winner moves by eta * (x - w_winner), losers are unchanged")
    void testSingleUpdateMovesOnlyWinner() {
        // Prototypes (0,0) and (10,10); x = (1,2) is nearest to prototype 0.
        // eta = 0.5: w_0 <- (0,0) + 0.5 * ((1,2) - (0,0)) = (0.5, 1.0)
        CompetitiveLearning net = new CompetitiveLearning(
                new Vector[]{Vector.of(0.0, 0.0), Vector.of(10.0, 10.0)}, 0.5);

        int winner = net.update(Vector.of(1.0, 2.0));

        assertEquals(0, winner);
        assertTrue(net.getPrototype(0).equals(Vector.of(0.5, 1.0), EPSILON));
        assertEquals(Vector.of(10.0, 10.0), net.getPrototype(1), "Loser must be exactly unchanged");
    }

    @Test
    @DisplayName("Winner is the prototype at minimum Euclidean distance; ties go to the lowest index")
    void testFindWinner() {
        CompetitiveLearning net = new CompetitiveLearning(
                new Vector[]{Vector.of(-1.0, 0.0), Vector.of(1.0, 0.0), Vector.of(0.0, 5.0)}, 0.1);

        assertEquals(0, net.findWinner(Vector.of(-2.0, 0.3)));
        assertEquals(1, net.findWinner(Vector.of(2.0, -0.3)));
        assertEquals(2, net.findWinner(Vector.of(0.0, 4.0)));
        assertEquals(0, net.findWinner(Vector.of(0.0, 0.0)), "Equidistant: lowest index wins");
    }

    @Test
    @DisplayName("Three well-separated clusters: each prototype ends near a different cluster centre")
    void testPrototypesFindThreeClusterCentres() {
        Vector[] centres = {Vector.of(0.0, 4.0), Vector.of(-4.0, -2.0), Vector.of(4.0, -2.0)};
        int perCluster = 100;
        Vector[] data = SyntheticDatasets.gaussianClusters(centres, perCluster, 0.5, new Random(42L));

        CompetitiveLearning net = CompetitiveLearning.withRandomPrototypes(3, 2, -0.5, 0.5, 0.1, new Random(43L));
        double[] qe = net.train(data, 20, new Random(44L));

        // Map each centre to its nearest prototype; the three must be distinct and close.
        int[] prototypeForCentre = new int[centres.length];
        Set<Integer> used = new HashSet<>();
        for (int c = 0; c < centres.length; c++) {
            int k = net.findWinner(centres[c]);
            prototypeForCentre[c] = k;
            used.add(k);
            double distance = LearningRuleUtils.euclideanDistance(net.getPrototype(k), centres[c]);
            assertTrue(distance < 0.5, "Prototype " + k + " is " + distance + " from centre " + centres[c]);
        }
        assertEquals(3, used.size(), "Each cluster must be captured by a different prototype");

        // Every training point is won by the prototype assigned to its own cluster (100% purity).
        for (int n = 0; n < data.length; n++) {
            int trueCluster = n / perCluster;
            assertEquals(prototypeForCentre[trueCluster], net.findWinner(data[n]), "Point " + n + " misassigned");
        }

        assertTrue(qe[20] < qe[0] / 3.0, "Quantisation error should drop sharply: " + qe[0] + " -> " + qe[20]);
    }

    @Test
    @DisplayName("Repeated presentation of one point pulls the winner geometrically towards it")
    void testRepeatedPresentationConverges() {
        // After n steps: w_n - x = (1 - eta)^n * (w_0 - x)
        CompetitiveLearning net = new CompetitiveLearning(new Vector[]{Vector.of(0.0, 0.0)}, 0.2);
        Vector x = Vector.of(5.0, -5.0);
        for (int i = 0; i < 10; i++) {
            net.update(x);
        }
        double expectedGap = Math.pow(0.8, 10) * x.norm();
        assertEquals(expectedGap, LearningRuleUtils.euclideanDistance(net.getPrototype(0), x), 1e-9);
    }

    @Test
    @DisplayName("Input validation")
    void testValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> new CompetitiveLearning(new Vector[]{Vector.of(0.0, 0.0)}, 1.5));
        assertThrows(IllegalArgumentException.class,
                () -> new CompetitiveLearning(new Vector[]{Vector.of(0.0, 0.0), Vector.of(1.0)}, 0.1));
        CompetitiveLearning net = new CompetitiveLearning(new Vector[]{Vector.of(0.0, 0.0)}, 0.1);
        assertThrows(IllegalArgumentException.class, () -> net.update(Vector.of(1.0, 2.0, 3.0)));
        assertThrows(IndexOutOfBoundsException.class, () -> net.getPrototype(1));
    }
}
