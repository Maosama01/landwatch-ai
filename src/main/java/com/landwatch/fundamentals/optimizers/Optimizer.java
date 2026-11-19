package com.landwatch.fundamentals.optimizers;

import com.landwatch.fundamentals.mlp.Layer;
import com.landwatch.fundamentals.mlp.MultilayerPerceptron;

/**
 * Base interface for neural network parameter optimizers.
 */
public interface Optimizer {

    /**
     * Returns the current learning rate eta.
     */
    double getLearningRate();

    /**
     * Configures the learning rate eta.
     */
    void setLearningRate(double learningRate);

    /**
     * Updates the weights and biases of a specific layer using its stored gradients.
     *
     * @param layer      the layer whose parameters to update
     * @param layerIndex index of the layer in the network (for state tracking)
     */
    void update(Layer layer, int layerIndex);

    /**
     * Updates all layers of a MultilayerPerceptron.
     *
     * @param mlp network model
     */
    default void update(MultilayerPerceptron mlp) {
        for (int i = 0; i < mlp.getNumLayers(); i++) {
            update(mlp.getLayer(i), i);
        }
    }

    /**
     * Resets any internal optimizer state (velocities, momentum buffers, etc.).
     */
    void reset();
}
