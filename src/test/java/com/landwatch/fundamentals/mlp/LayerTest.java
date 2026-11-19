package com.landwatch.fundamentals.mlp;

import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for Dense Neural Network Layer forward and backward propagation.
 */
class LayerTest {

    private static final double EPSILON = 1e-9;

    @Test
    @DisplayName("Layer forward propagation with known linear weights and bias")
    void testForwardPass() {
        // 2 inputs -> 2 outputs
        Matrix weights = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });
        Vector biases = Vector.of(0.5, -0.5);
        Layer layer = new Layer(weights, biases, ActivationFunction.LINEAR);

        Vector input = Vector.of(2.0, 3.0);
        // z = W*x + b = [1*2 + 2*3 + 0.5, 3*2 + 4*3 - 0.5] = [8.5, 17.5]
        Vector output = layer.forward(input);

        assertNotNull(output);
        assertEquals(2, output.size());
        assertEquals(8.5, output.get(0), EPSILON);
        assertEquals(17.5, output.get(1), EPSILON);
    }

    @Test
    @DisplayName("Layer backward pass and analytical gradient calculation")
    void testBackwardPass() {
        Matrix weights = new Matrix(new double[][]{
                {1.0, 2.0},
                {3.0, 4.0}
        });
        Vector biases = Vector.of(0.5, -0.5);
        Layer layer = new Layer(weights, biases, ActivationFunction.LINEAR);

        Vector input = Vector.of(2.0, 3.0);
        layer.forward(input);

        // Upstream gradient dLoss/da = [1.0, 2.0]
        Vector upstreamGrad = Vector.of(1.0, 2.0);
        Vector downstreamGrad = layer.backward(upstreamGrad);

        // For LINEAR, f'(z) = 1, so delta = upstreamGrad = [1.0, 2.0]
        // gradW = delta * x^T = [[1*2, 1*3], [2*2, 2*3]] = [[2, 3], [4, 6]]
        Matrix gradW = layer.getGradW();
        assertEquals(2.0, gradW.get(0, 0), EPSILON);
        assertEquals(3.0, gradW.get(0, 1), EPSILON);
        assertEquals(4.0, gradW.get(1, 0), EPSILON);
        assertEquals(6.0, gradW.get(1, 1), EPSILON);

        // gradB = delta = [1.0, 2.0]
        Vector gradB = layer.getGradB();
        assertEquals(1.0, gradB.get(0), EPSILON);
        assertEquals(2.0, gradB.get(1), EPSILON);

        // Downstream grad = W^T * delta = [[1, 3], [2, 4]] * [1, 2] = [1*1 + 3*2, 2*1 + 4*2] = [7.0, 10.0]
        assertEquals(7.0, downstreamGrad.get(0), EPSILON);
        assertEquals(10.0, downstreamGrad.get(1), EPSILON);
    }

    @Test
    @DisplayName("Parameter update via SGD")
    void testParameterUpdate() {
        Matrix weights = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        Vector biases = Vector.of(0.5, -0.5);
        Layer layer = new Layer(weights, biases, ActivationFunction.LINEAR);

        layer.forward(Vector.of(2.0, 3.0));
        layer.backward(Vector.of(1.0, 2.0)); // gradW = [[2, 3], [4, 6]], gradB = [1, 2]

        double lr = 0.1;
        layer.updateParameters(lr);

        // new W = [[1 - 0.2, 2 - 0.3], [3 - 0.4, 4 - 0.6]] = [[0.8, 1.7], [2.6, 3.4]]
        Matrix updatedW = layer.getWeights();
        assertEquals(0.8, updatedW.get(0, 0), EPSILON);
        assertEquals(1.7, updatedW.get(0, 1), EPSILON);
        assertEquals(2.6, updatedW.get(1, 0), EPSILON);
        assertEquals(3.4, updatedW.get(1, 1), EPSILON);

        // new b = [0.5 - 0.1, -0.5 - 0.2] = [0.4, -0.7]
        Vector updatedB = layer.getBiases();
        assertEquals(0.4, updatedB.get(0), EPSILON);
        assertEquals(-0.7, updatedB.get(1), EPSILON);
    }

    @Test
    @DisplayName("Layer dimension validation and error handling")
    void testDimensionMismatch() {
        Layer layer = new Layer(3, 2, ActivationFunction.SIGMOID, 42L);

        // Input size mismatch (expected 3, provided 2)
        assertThrows(IllegalArgumentException.class, () -> layer.forward(Vector.of(1.0, 2.0)));
        assertThrows(IllegalArgumentException.class, () -> layer.forward(null));

        // Backward without forward
        Layer freshLayer = new Layer(3, 2, ActivationFunction.SIGMOID, 42L);
        assertThrows(IllegalStateException.class, () -> freshLayer.backward(Vector.of(1.0, 1.0)));
    }
}
