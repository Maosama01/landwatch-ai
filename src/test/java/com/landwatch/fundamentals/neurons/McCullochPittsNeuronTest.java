package com.landwatch.fundamentals.neurons;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Rigorous JUnit 5 tests for McCulloch-Pitts Neuron.
 */
class McCullochPittsNeuronTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("McCulloch-Pitts 2-input AND gate truth table")
    void testTwoInputAndGate() {
        McCullochPittsNeuron andGate = McCullochPittsNeuron.andGate(2);

        assertEquals(0, andGate.activate(0, 0));
        assertEquals(0, andGate.activate(0, 1));
        assertEquals(0, andGate.activate(1, 0));
        assertEquals(1, andGate.activate(1, 1));
    }

    @Test
    @DisplayName("McCulloch-Pitts 3-input AND gate truth table")
    void testThreeInputAndGate() {
        McCullochPittsNeuron andGate = McCullochPittsNeuron.andGate(3);

        assertEquals(0, andGate.activate(0, 0, 0));
        assertEquals(0, andGate.activate(1, 1, 0));
        assertEquals(0, andGate.activate(1, 0, 1));
        assertEquals(1, andGate.activate(1, 1, 1));
    }

    @Test
    @DisplayName("McCulloch-Pitts 2-input OR gate truth table")
    void testTwoInputOrGate() {
        McCullochPittsNeuron orGate = McCullochPittsNeuron.orGate(2);

        assertEquals(0, orGate.activate(0, 0));
        assertEquals(1, orGate.activate(0, 1));
        assertEquals(1, orGate.activate(1, 0));
        assertEquals(1, orGate.activate(1, 1));
    }

    @Test
    @DisplayName("McCulloch-Pitts 2-input NOR gate truth table")
    void testNorGate() {
        McCullochPittsNeuron norGate = McCullochPittsNeuron.norGate(2);

        assertEquals(1, norGate.activate(0, 0));
        assertEquals(0, norGate.activate(0, 1));
        assertEquals(0, norGate.activate(1, 0));
        assertEquals(0, norGate.activate(1, 1));
    }

    @Test
    @DisplayName("Threshold boundary behavior: exact threshold equality vs strictly below")
    void testThresholdBehavior() {
        // Neuron with weights [2.0, 3.0], threshold 5.0
        McCullochPittsNeuron neuron = new McCullochPittsNeuron(new double[]{2.0, 3.0}, 5.0);

        // Sum = 2*1 + 3*1 = 5.0 == threshold -> Must fire (output 1)
        assertEquals(5.0, neuron.computeWeightedSum(1, 1), EPSILON);
        assertEquals(1, neuron.activate(1, 1));

        // Sum = 2*0 + 3*1 = 3.0 < 5.0 -> Must not fire (output 0)
        assertEquals(3.0, neuron.computeWeightedSum(0, 1), EPSILON);
        assertEquals(0, neuron.activate(0, 1));

        // Sum = 2*1 + 3*0 = 2.0 < 5.0 -> Must not fire (output 0)
        assertEquals(2.0, neuron.computeWeightedSum(1, 0), EPSILON);
        assertEquals(0, neuron.activate(1, 0));
    }

    @Test
    @DisplayName("Weighted sum calculation using Vector input")
    void testVectorInput() {
        McCullochPittsNeuron neuron = new McCullochPittsNeuron(Vector.of(1.5, -2.5, 4.0), 3.0);
        Vector input = Vector.of(1.0, 0.0, 1.0);

        // 1.5*1 + (-2.5)*0 + 4.0*1 = 5.5 >= 3.0 -> Fires
        assertEquals(5.5, neuron.computeWeightedSum(input), EPSILON);
        assertEquals(1, neuron.activate(input));
    }

    @Test
    @DisplayName("Input validation: rejects non-binary inputs")
    void testNonBinaryInputsThrowException() {
        McCullochPittsNeuron neuron = McCullochPittsNeuron.andGate(2);

        // Non-binary integer inputs (e.g. 2, -1)
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(2, 0));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(0, -1));

        // Non-binary Vector inputs
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(Vector.of(0.5, 1.0)));
    }

    @Test
    @DisplayName("Input validation: dimension mismatch and null inputs")
    void testDimensionMismatch() {
        McCullochPittsNeuron neuron = McCullochPittsNeuron.andGate(2);

        // Wrong number of arguments
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(1));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(Vector.of(1.0)));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate((Vector) null));
    }
}
