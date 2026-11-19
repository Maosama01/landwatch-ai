package com.landwatch.fundamentals.math;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Rigorous JUnit 5 tests for pure Java Vector implementation.
 */
class VectorTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("Vector initialization and factories")
    void testInitialization() {
        Vector zeros = Vector.zeros(4);
        assertEquals(4, zeros.size());
        for (int i = 0; i < 4; i++) {
            assertEquals(0.0, zeros.get(i), EPSILON);
        }

        Vector ones = Vector.ones(3);
        assertEquals(3, ones.size());
        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, ones.get(i), EPSILON);
        }

        Vector v = Vector.of(1.5, 2.5, -3.5);
        assertEquals(3, v.size());
        assertEquals(1.5, v.get(0), EPSILON);
        assertEquals(2.5, v.get(1), EPSILON);
        assertEquals(-3.5, v.get(2), EPSILON);
    }

    @Test
    @DisplayName("Vector addition and subtraction")
    void testAddAndSubtract() {
        Vector u = Vector.of(1.0, 2.0, 3.0);
        Vector v = Vector.of(4.0, 5.0, 6.0);

        Vector sum = u.add(v);
        assertArrayEquals(new double[]{5.0, 7.0, 9.0}, sum.toArray(), EPSILON);

        Vector diff = v.subtract(u);
        assertArrayEquals(new double[]{3.0, 3.0, 3.0}, diff.toArray(), EPSILON);
    }

    @Test
    @DisplayName("Scalar multiplication and division")
    void testScalarOperations() {
        Vector u = Vector.of(2.0, -4.0, 8.0);

        Vector scaled = u.multiply(2.5);
        assertArrayEquals(new double[]{5.0, -10.0, 20.0}, scaled.toArray(), EPSILON);

        Vector divided = u.divide(2.0);
        assertArrayEquals(new double[]{1.0, -2.0, 4.0}, divided.toArray(), EPSILON);

        assertThrows(ArithmeticException.class, () -> u.divide(0.0));
    }

    @Test
    @DisplayName("Dot product and element-wise multiplication")
    void testDotAndElementwise() {
        Vector u = Vector.of(1.0, 3.0, -5.0);
        Vector v = Vector.of(4.0, -2.0, -1.0);

        // Dot product: 1*4 + 3*(-2) + (-5)*(-1) = 4 - 6 + 5 = 3
        double dot = u.dot(v);
        assertEquals(3.0, dot, EPSILON);

        // Element-wise: [1*4, 3*(-2), (-5)*(-1)] = [4, -6, 5]
        Vector ew = u.elementwiseMultiply(v);
        assertArrayEquals(new double[]{4.0, -6.0, 5.0}, ew.toArray(), EPSILON);
    }

    @Test
    @DisplayName("Vector norms and sum")
    void testNormsAndSum() {
        Vector v = Vector.of(3.0, -4.0);

        assertEquals(-1.0, v.sum(), EPSILON);
        assertEquals(5.0, v.norm(), EPSILON);    // sqrt(3^2 + (-4)^2) = 5
        assertEquals(7.0, v.normL1(), EPSILON);  // |3| + |-4| = 7
    }

    @Test
    @DisplayName("Map function (activation simulation)")
    void testMap() {
        Vector v = Vector.of(0.0, 1.0, 2.0);
        // Apply f(x) = 2*x + 1
        Vector mapped = v.map(x -> 2 * x + 1);
        assertArrayEquals(new double[]{1.0, 3.0, 5.0}, mapped.toArray(), EPSILON);
    }

    @Test
    @DisplayName("Outer product generating matrix")
    void testOuterProduct() {
        Vector u = Vector.of(1.0, 2.0);
        Vector v = Vector.of(3.0, 4.0, 5.0);

        Matrix outer = u.outerProduct(v);
        assertEquals(2, outer.getRows());
        assertEquals(3, outer.getCols());

        // Row 0: [1*3, 1*4, 1*5] = [3, 4, 5]
        assertArrayEquals(new double[]{3.0, 4.0, 5.0}, outer.getRow(0).toArray(), EPSILON);
        // Row 1: [2*3, 2*4, 2*5] = [6, 8, 10]
        assertArrayEquals(new double[]{6.0, 8.0, 10.0}, outer.getRow(1).toArray(), EPSILON);
    }

    @Test
    @DisplayName("Row and Column matrix conversions")
    void testMatrixConversions() {
        Vector v = Vector.of(10.0, 20.0, 30.0);

        Matrix rowM = v.toRowMatrix();
        assertEquals(1, rowM.getRows());
        assertEquals(3, rowM.getCols());
        assertEquals(20.0, rowM.get(0, 1), EPSILON);

        Matrix colM = v.toColumnMatrix();
        assertEquals(3, colM.getRows());
        assertEquals(1, colM.getCols());
        assertEquals(30.0, colM.get(2, 0), EPSILON);
    }

    @Test
    @DisplayName("Dimension mismatch and bounds checking")
    void testExceptions() {
        Vector u = Vector.of(1.0, 2.0);
        Vector v = Vector.of(1.0, 2.0, 3.0);

        assertThrows(IllegalArgumentException.class, () -> u.add(v));
        assertThrows(IllegalArgumentException.class, () -> u.subtract(v));
        assertThrows(IllegalArgumentException.class, () -> u.dot(v));
        assertThrows(IllegalArgumentException.class, () -> u.elementwiseMultiply(v));

        assertThrows(IndexOutOfBoundsException.class, () -> u.get(-1));
        assertThrows(IndexOutOfBoundsException.class, () -> u.get(2));
        assertThrows(IllegalArgumentException.class, () -> new Vector(0));
    }
}
