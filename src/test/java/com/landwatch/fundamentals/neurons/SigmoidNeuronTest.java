package com.landwatch.fundamentals.neurons;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Rigorous JUnit 5 tests for Sigmoid Neuron.
 */
class SigmoidNeuronTest {

    private static final double EPSILON = 1e-6;

    @Test
    @DisplayName("Sigmoid value at zero must be exactly 0.5")
    void testSigmoidAtZero() {
        assertEquals(0.5, SigmoidNeuron.sigmoid(0.0), 1e-12);
        assertEquals(0.25, SigmoidNeuron.sigmoidDerivative(0.0), 1e-12);
    }

    @Test
    @DisplayName("Sigmoid output range strictly in (0, 1) and symmetry property")
    void testSigmoidOutputRangeAndSymmetry() {
        double[] testPoints = {-10.0, -5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0, 10.0};

        for (double z : testPoints) {
            double s = SigmoidNeuron.sigmoid(z);
            assertTrue(s > 0.0 && s < 1.0, "Sigmoid output must be strictly in (0, 1). z=" + z + ", s=" + s);

            // Symmetry property: sigma(-z) == 1 - sigma(z)
            double sNeg = SigmoidNeuron.sigmoid(-z);
            assertEquals(1.0 - s, sNeg, 1e-12, "Sigmoid must satisfy symmetry sigma(-z) = 1 - sigma(z)");
        }
    }

    @Test
    @DisplayName("Numerical stability on extreme positive and negative inputs")
    void testNumericalStability() {
        // Large positive inputs: must approach 1.0 without NaN or overflow
        double extremePos = SigmoidNeuron.sigmoid(500.0);
        assertEquals(1.0, extremePos, 1e-9);
        assertFalse(Double.isNaN(extremePos));

        // Large negative inputs: must approach 0.0 without NaN or division by zero
        double extremeNeg = SigmoidNeuron.sigmoid(-500.0);
        assertEquals(0.0, extremeNeg, 1e-9);
        assertFalse(Double.isNaN(extremeNeg));

        // Derivative at extreme saturation should approach 0.0
        assertEquals(0.0, SigmoidNeuron.sigmoidDerivative(100.0), 1e-9);
        assertEquals(0.0, SigmoidNeuron.sigmoidDerivative(-100.0), 1e-9);
    }

    @Test
    @DisplayName("Net input and activation calculation with known weights and bias")
    void testNetInputAndActivation() {
        // weights = [2.0, -1.0], bias = 0.5
        SigmoidNeuron neuron = new SigmoidNeuron(new double[]{2.0, -1.0}, 0.5);

        // input = [1.0, 3.0]
        // z = 2*1 + (-1)*3 + 0.5 = 2 - 3 + 0.5 = -0.5
        double z = neuron.computeNetInput(1.0, 3.0);
        assertEquals(-0.5, z, EPSILON);

        // a = 1 / (1 + exp(0.5)) ~ 0.377540668798
        double expectedActivation = 1.0 / (1.0 + Math.exp(0.5));
        double actualActivation = neuron.activate(1.0, 3.0);
        assertEquals(expectedActivation, actualActivation, EPSILON);

        // Vector overload test
        Vector inputVec = Vector.of(1.0, 3.0);
        assertEquals(z, neuron.computeNetInput(inputVec), EPSILON);
        assertEquals(expectedActivation, neuron.activate(inputVec), EPSILON);
    }

    @Test
    @DisplayName("Sigmoid derivative from activation vs from pre-activation")
    void testDerivativeEquivalence() {
        double[] zValues = {-3.0, -1.5, 0.0, 1.5, 3.0};
        for (double z : zValues) {
            double fromZ = SigmoidNeuron.sigmoidDerivative(z);
            double act = SigmoidNeuron.sigmoid(z);
            double fromAct = SigmoidNeuron.derivativeFromActivation(act);
            assertEquals(fromZ, fromAct, 1e-12);
        }
    }

    @Test
    @DisplayName("Weights and bias mutators")
    void testMutators() {
        SigmoidNeuron neuron = new SigmoidNeuron(2);
        assertEquals(0.0, neuron.getBias(), EPSILON);

        neuron.setBias(1.5);
        assertEquals(1.5, neuron.getBias(), EPSILON);

        Vector newW = Vector.of(0.8, -0.4);
        neuron.setWeights(newW);
        assertTrue(newW.equals(neuron.getWeights(), EPSILON));

        // Wrong dimension weights throws exception
        assertThrows(IllegalArgumentException.class, () -> neuron.setWeights(Vector.of(1.0, 2.0, 3.0)));
    }

    @Test
    @DisplayName("Input dimension mismatch and null checking")
    void testDimensionMismatchExceptions() {
        SigmoidNeuron neuron = new SigmoidNeuron(2);

        assertThrows(IllegalArgumentException.class, () -> neuron.activate(1.0));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(1.0, 2.0, 3.0));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate(Vector.of(1.0)));
        assertThrows(IllegalArgumentException.class, () -> neuron.activate((Vector) null));
    }
}
