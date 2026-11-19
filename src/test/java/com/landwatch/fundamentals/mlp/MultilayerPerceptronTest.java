package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for MultilayerPerceptron architecture and forward propagation.
 */
class MultilayerPerceptronTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("MLP initialization and layer configuration")
    void testInitialization() {
        int[] topology = {3, 5, 4, 2};
        MultilayerPerceptron mlp = new MultilayerPerceptron(
                topology, ActivationFunction.RELU, ActivationFunction.SIGMOID, 42L);

        assertEquals(3, mlp.getNumLayers());
        assertArrayEquals(topology, mlp.getLayerSizes());

        // Check layer dimensions
        assertEquals(3, mlp.getLayer(0).getInputSize());
        assertEquals(5, mlp.getLayer(0).getOutputSize());
        assertEquals(ActivationFunction.RELU, mlp.getLayer(0).getActivationFunction());

        assertEquals(5, mlp.getLayer(1).getInputSize());
        assertEquals(4, mlp.getLayer(1).getOutputSize());
        assertEquals(ActivationFunction.RELU, mlp.getLayer(1).getActivationFunction());

        assertEquals(4, mlp.getLayer(2).getInputSize());
        assertEquals(2, mlp.getLayer(2).getOutputSize());
        assertEquals(ActivationFunction.SIGMOID, mlp.getLayer(2).getActivationFunction());
    }

    @Test
    @DisplayName("Deterministic initialization with identical seeds")
    void testReproducibility() {
        int[] topology = {2, 4, 1};
        MultilayerPerceptron mlp1 = new MultilayerPerceptron(topology, 12345L);
        MultilayerPerceptron mlp2 = new MultilayerPerceptron(topology, 12345L);

        Vector testInput = Vector.of(0.5, -0.5);
        Vector pred1 = mlp1.forward(testInput);
        Vector pred2 = mlp2.forward(testInput);

        assertTrue(pred1.equals(pred2, EPSILON), "Identical seeds must yield identical forward predictions");
    }

    @Test
    @DisplayName("Forward propagation output dimensions")
    void testForwardDimensions() {
        MultilayerPerceptron mlp = new MultilayerPerceptron(new int[]{4, 8, 3}, 99L);
        Vector input = Vector.of(1.0, 2.0, 3.0, 4.0);

        Vector output = mlp.forward(input);
        assertNotNull(output);
        assertEquals(3, output.size());

        // Prediction should also work identically
        Vector pred = mlp.predict(input);
        assertTrue(output.equals(pred, EPSILON));
    }

    @Test
    @DisplayName("Invalid network topology validation")
    void testInvalidTopologies() {
        // Less than 2 layers
        assertThrows(IllegalArgumentException.class, () -> new MultilayerPerceptron(new int[]{5}, 42L));
        assertThrows(IllegalArgumentException.class, () -> new MultilayerPerceptron(new int[]{}, 42L));

        // Negative or zero neurons in a layer
        assertThrows(IllegalArgumentException.class, () -> new MultilayerPerceptron(new int[]{2, 0, 1}, 42L));
        assertThrows(IllegalArgumentException.class, () -> new MultilayerPerceptron(new int[]{-3, 2}, 42L));

        // Null argument
        assertThrows(NullPointerException.class, () -> new MultilayerPerceptron(null, 42L));

        // Wrong input size during forward pass
        MultilayerPerceptron mlp = new MultilayerPerceptron(new int[]{3, 2}, 42L);
        assertThrows(IllegalArgumentException.class, () -> mlp.forward(Vector.of(1.0, 2.0)));
    }
}
