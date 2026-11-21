package com.landwatch.fundamentals.math;

import com.landwatch.fundamentals.learningrules.LearningRuleUtils;
import com.landwatch.fundamentals.learningrules.OjaLearning;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SingularValueDecompositionTest {

    @Test
    void knownTwoByTwoExample() {
        // A^T A = [[25, 20], [20, 25]] has eigenvalues 45 and 5
        SingularValueDecomposition svd = new SingularValueDecomposition(new Matrix(new double[][] {{3, 0}, {4, 5}}));
        assertArrayEquals(new double[] {Math.sqrt(45), Math.sqrt(5)}, svd.getSingularValues(), 1e-12);
        // first right singular vector is (1, 1)/sqrt(2) up to sign
        Vector v1 = svd.rightSingularVector(0);
        assertEquals(1.0, Math.abs(v1.get(0) + v1.get(1)) / Math.sqrt(2), 1e-12);
    }

    @Test
    void reconstructsRandomMatricesWithOrthonormalFactors() {
        Random rng = new Random(11);
        for (int[] size : new int[][] {{6, 4}, {4, 4}, {3, 5}}) {
            Matrix a = Matrix.random(size[0], size[1], -2, 2, rng);
            SingularValueDecomposition svd = new SingularValueDecomposition(a);
            assertTrue(svd.reconstruct().equals(a, 1e-10), "A = U S V^T for " + size[0] + "x" + size[1]);
            int k = Math.min(size[0], size[1]);
            Matrix v = svd.getV();
            assertTrue(v.transpose().multiply(v).equals(Matrix.identity(k), 1e-10), "V^T V = I");
            Matrix u = svd.getU();
            assertTrue(u.transpose().multiply(u).equals(Matrix.identity(k), 1e-10), "U^T U = I");
            double[] s = svd.getSingularValues();
            for (int i = 1; i < s.length; i++) {
                assertTrue(s[i - 1] >= s[i] && s[i] >= 0, "sorted and non-negative");
            }
        }
    }

    @Test
    void truncationErrorIsTheDiscardedSingularValues() {
        // Eckart-Young: ||A - A_k||_F^2 = s_{k+1}^2 + ... + s_n^2
        Matrix a = Matrix.random(7, 5, -1, 1, new Random(5));
        SingularValueDecomposition svd = new SingularValueDecomposition(a);
        double[] s = svd.getSingularValues();
        for (int k = 0; k <= s.length; k++) {
            double rest = 0;
            for (int i = k; i < s.length; i++) {
                rest += s[i] * s[i];
            }
            double err = a.subtract(svd.reconstruct(k)).frobeniusNorm();
            assertEquals(Math.sqrt(rest), err, 1e-10);
        }
        assertThrows(IllegalArgumentException.class, () -> svd.reconstruct(6));
    }

    @Test
    void rankOneMatrixHasOneNonZeroSingularValue() {
        Matrix a = new Matrix(new double[][] {{1, 2, 3}, {2, 4, 6}, {-1, -2, -3}});
        double[] s = new SingularValueDecomposition(a).getSingularValues();
        assertEquals(Math.sqrt(6) * Math.sqrt(14), s[0], 1e-10); // |(1,2,-1)| * |(1,2,3)|
        assertEquals(0, s[1], 1e-10);
        assertEquals(0, s[2], 1e-10);
    }

    @Test
    void ojasRuleLearnsTheTopRightSingularVector() {
        // Oja's rule converges to the leading eigenvector of E[x x^T] = X^T X / n,
        // which is the first right singular vector of the data matrix X.
        Random rng = new Random(21);
        Matrix mix = Matrix.random(5, 5, -1, 1, rng);
        Vector[] data = new Vector[400];
        double[][] rows = new double[data.length][];
        double[] scales = {3, 1.5, 1, 0.5, 0.3};
        for (int i = 0; i < data.length; i++) {
            double[] z = new double[5];
            for (int j = 0; j < 5; j++) {
                z[j] = scales[j] * rng.nextGaussian();
            }
            Vector x = new Matrix(new double[][] {z}).multiply(mix).getRow(0);
            data[i] = x;
            rows[i] = x.toArray();
        }
        Vector v1 = new SingularValueDecomposition(new Matrix(rows)).rightSingularVector(0);
        OjaLearning oja = new OjaLearning(5, 0.002, new Random(3));
        oja.train(data, 60, new Random(4));
        double cos = Math.abs(LearningRuleUtils.cosineSimilarity(oja.getWeights(), v1));
        assertTrue(cos > 0.99, "|cos(w, v1)| = " + cos);
        assertEquals(1.0, oja.weightNorm(), 0.05);
    }
}
