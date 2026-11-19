package com.landwatch.fundamentals.math;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Rigorous JUnit 5 tests for pure Java Matrix implementation.
 */
class MatrixTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("Matrix initialization and factories")
    void testInitialization() {
        Matrix zeros = Matrix.zeros(2, 3);
        assertEquals(2, zeros.getRows());
        assertEquals(3, zeros.getCols());
        assertEquals(0.0, zeros.get(1, 2), EPSILON);

        Matrix ones = Matrix.ones(3, 2);
        assertEquals(1.0, ones.get(0, 0), EPSILON);
        assertEquals(1.0, ones.get(2, 1), EPSILON);

        Matrix id = Matrix.identity(3);
        assertEquals(3, id.getRows());
        assertEquals(3, id.getCols());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(i == j ? 1.0 : 0.0, id.get(i, j), EPSILON);
            }
        }
    }

    @Test
    @DisplayName("Matrix from rows and columns")
    void testFromRowsAndCols() {
        Vector r0 = Vector.of(1.0, 2.0);
        Vector r1 = Vector.of(3.0, 4.0);
        Matrix fromR = Matrix.fromRows(r0, r1);

        assertEquals(2, fromR.getRows());
        assertEquals(2, fromR.getCols());
        assertEquals(3.0, fromR.get(1, 0), EPSILON);

        Vector c0 = Vector.of(1.0, 3.0);
        Vector c1 = Vector.of(2.0, 4.0);
        Matrix fromC = Matrix.fromCols(c0, c1);
        assertTrue(fromR.equals(fromC, EPSILON));
    }

    @Test
    @DisplayName("Matrix addition and subtraction")
    void testAddAndSubtract() {
        Matrix a = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Matrix b = new Matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

        Matrix sum = a.add(b);
        assertEquals(6.0, sum.get(0, 0), EPSILON);
        assertEquals(8.0, sum.get(0, 1), EPSILON);
        assertEquals(10.0, sum.get(1, 0), EPSILON);
        assertEquals(12.0, sum.get(1, 1), EPSILON);

        Matrix diff = b.subtract(a);
        assertEquals(4.0, diff.get(0, 0), EPSILON);
        assertEquals(4.0, diff.get(1, 1), EPSILON);
    }

    @Test
    @DisplayName("Matrix multiplication (A * B)")
    void testMatrixMultiplication() {
        // A: 2x3
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        // B: 3x2
        Matrix b = new Matrix(new double[][]{
                {7.0, 8.0},
                {9.0, 1.0},
                {2.0, 3.0}
        });

        // C = A * B -> 2x2
        // C[0,0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
        // C[0,1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
        // C[1,0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
        // C[1,1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
        Matrix c = a.multiply(b);

        assertEquals(2, c.getRows());
        assertEquals(2, c.getCols());
        assertEquals(31.0, c.get(0, 0), EPSILON);
        assertEquals(19.0, c.get(0, 1), EPSILON);
        assertEquals(85.0, c.get(1, 0), EPSILON);
        assertEquals(55.0, c.get(1, 1), EPSILON);
    }

    @Test
    @DisplayName("Matrix-vector multiplication (A * v)")
    void testMatrixVectorMultiplication() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        Vector v = Vector.of(2.0, -1.0, 3.0);

        // Result:
        // u[0] = 1*2 + 2*(-1) + 3*3 = 2 - 2 + 9 = 9
        // u[1] = 4*2 + 5*(-1) + 6*3 = 8 - 5 + 18 = 21
        Vector u = a.multiply(v);

        assertEquals(2, u.size());
        assertEquals(9.0, u.get(0), EPSILON);
        assertEquals(21.0, u.get(1), EPSILON);
    }

    @Test
    @DisplayName("Matrix transpose")
    void testTranspose() {
        Matrix a = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });
        Matrix at = a.transpose();

        assertEquals(3, at.getRows());
        assertEquals(2, at.getCols());
        assertEquals(1.0, at.get(0, 0), EPSILON);
        assertEquals(4.0, at.get(0, 1), EPSILON);
        assertEquals(2.0, at.get(1, 0), EPSILON);
        assertEquals(5.0, at.get(1, 1), EPSILON);
        assertEquals(3.0, at.get(2, 0), EPSILON);
        assertEquals(6.0, at.get(2, 1), EPSILON);

        // (A^T)^T == A
        assertTrue(at.transpose().equals(a, EPSILON));
    }

    @Test
    @DisplayName("Element-wise multiplication (Hadamard)")
    void testElementwiseMultiply() {
        Matrix a = new Matrix(new double[][]{{2.0, 3.0}, {4.0, 5.0}});
        Matrix b = new Matrix(new double[][]{{10.0, 20.0}, {30.0, 40.0}});

        Matrix result = a.elementwiseMultiply(b);
        assertEquals(20.0, result.get(0, 0), EPSILON);
        assertEquals(60.0, result.get(0, 1), EPSILON);
        assertEquals(120.0, result.get(1, 0), EPSILON);
        assertEquals(200.0, result.get(1, 1), EPSILON);
    }

    @Test
    @DisplayName("Vector broadcasting to rows (bias addition simulation)")
    void testAddVectorBroadcast() {
        Matrix m = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
        });
        Vector bias = Vector.of(10.0, 20.0);

        Matrix withBias = m.addVectorBroadcast(bias, true);
        assertEquals(11.0, withBias.get(0, 0), EPSILON);
        assertEquals(22.0, withBias.get(0, 1), EPSILON);
        assertEquals(13.0, withBias.get(1, 0), EPSILON);
        assertEquals(24.0, withBias.get(1, 1), EPSILON);
        assertEquals(15.0, withBias.get(2, 0), EPSILON);
        assertEquals(26.0, withBias.get(2, 1), EPSILON);
    }

    @Test
    @DisplayName("Row and column aggregations")
    void testAggregations() {
        Matrix m = new Matrix(new double[][]{
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        });

        Vector rowSums = m.sumRows(); // sum along each row: size 2 -> [6.0, 15.0]
        assertEquals(2, rowSums.size());
        assertEquals(6.0, rowSums.get(0), EPSILON);
        assertEquals(15.0, rowSums.get(1), EPSILON);

        Vector colSums = m.sumCols(); // sum along each col: size 3 -> [5.0, 7.0, 9.0]
        assertEquals(3, colSums.size());
        assertEquals(5.0, colSums.get(0), EPSILON);
        assertEquals(7.0, colSums.get(1), EPSILON);
        assertEquals(9.0, colSums.get(2), EPSILON);

        assertEquals(21.0, m.sumAll(), EPSILON);

        // Frobenius norm: sqrt(1+4+9+16+25+36) = sqrt(91) ~ 9.539392
        assertEquals(Math.sqrt(91), m.frobeniusNorm(), EPSILON);
    }

    @Test
    @DisplayName("Dimension mismatch exceptions")
    void testExceptions() {
        Matrix a = new Matrix(2, 3);
        Matrix b = new Matrix(2, 2);
        Matrix c = new Matrix(4, 2);

        assertThrows(IllegalArgumentException.class, () -> a.add(b));
        assertThrows(IllegalArgumentException.class, () -> a.subtract(b));
        assertThrows(IllegalArgumentException.class, () -> a.multiply(c)); // 2x3 * 4x2 mismatch
        assertThrows(IllegalArgumentException.class, () -> a.multiply(Vector.of(1.0, 2.0))); // needs 3

        assertThrows(IndexOutOfBoundsException.class, () -> a.get(-1, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> a.get(0, 3));
        assertThrows(IllegalArgumentException.class, () -> new Matrix(0, 5));
    }
}
