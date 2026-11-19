package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ActivationFunction and ActivationFunctions utilities.
 */
class ActivationFunctionsTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("Sigmoid activation and analytical derivative")
    void testSigmoid() {
        assertEquals(0.5, ActivationFunctions.sigmoid(0.0), EPSILON);
        assertEquals(0.25, ActivationFunctions.sigmoidDerivative(0.0), EPSILON);

        double s2 = ActivationFunctions.sigmoid(2.0);
        double ds2 = ActivationFunctions.sigmoidDerivative(2.0);
        assertEquals(s2 * (1.0 - s2), ds2, EPSILON);

        Vector v = Vector.of(-1.0, 0.0, 1.0);
        Vector sVec = ActivationFunctions.sigmoid(v);
        assertEquals(3, sVec.size());
        assertEquals(0.5, sVec.get(1), EPSILON);
    }

    @Test
    @DisplayName("ReLU activation and step derivative")
    void testRelu() {
        // Positive inputs
        assertEquals(5.5, ActivationFunctions.relu(5.5), EPSILON);
        assertEquals(1.0, ActivationFunctions.reluDerivative(5.5), EPSILON);

        // Negative inputs
        assertEquals(0.0, ActivationFunctions.relu(-3.2), EPSILON);
        assertEquals(0.0, ActivationFunctions.reluDerivative(-3.2), EPSILON);

        // Zero
        assertEquals(0.0, ActivationFunctions.relu(0.0), EPSILON);
        assertEquals(0.0, ActivationFunctions.reluDerivative(0.0), EPSILON);

        Vector v = Vector.of(-2.0, 0.0, 3.0);
        Vector rVec = ActivationFunctions.relu(v);
        assertArrayEquals(new double[]{0.0, 0.0, 3.0}, rVec.toArray(), EPSILON);

        Vector dVec = ActivationFunctions.reluDerivative(v);
        assertArrayEquals(new double[]{0.0, 0.0, 1.0}, dVec.toArray(), EPSILON);
    }
}
