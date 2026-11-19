package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;

import java.nio.file.Path;

/**
 * Shared plumbing for building, training and running the change-detection model with DJL.
 *
 * <p>A DJL training step has four parts: run the model forward while a {@link GradientCollector}
 * records the computation, compute the loss, call {@code backward} to obtain gradients, and let the
 * optimiser ({@link Trainer#step()}) update the weights.</p>
 */
public final class TrainingSupport {

    /** Name used for the DJL model and its parameter file ({@code fc_siam_diff-0000.params}). */
    public static final String MODEL_NAME = "fc_siam_diff";

    private TrainingSupport() {
    }

    /**
     * A path as it should appear in console output and in the result files: relative to the working
     * directory when it lies inside it (so committed reports never contain a personal home-directory
     * path), otherwise absolute. Always uses forward slashes.
     */
    public static String displayPath(Path path) {
        Path abs = path.toAbsolutePath().normalize();
        Path cwd = Path.of("").toAbsolutePath().normalize();
        Path shown = abs.startsWith(cwd) ? cwd.relativize(abs) : abs;
        String s = shown.toString().replace('\\', '/');
        return s.isEmpty() ? "." : s;
    }

    /** Creates an (uninitialised) DJL model wrapping an {@link FcSiamDiff} block. */
    public static Model newModel(int[] widths, Device device) {
        Model model = Model.newInstance(MODEL_NAME, device, "PyTorch");
        model.setBlock(new FcSiamDiff(widths));
        return model;
    }

    /**
     * Creates a trainer with Adam and He (Kaiming) initialisation, and initialises the parameters
     * for inputs of the given patch size.
     */
    public static Trainer newTrainer(Model model, Loss loss, float learningRate, int batchSize, int patchSize) {
        return newTrainer(model, loss, Tracker.fixed(learningRate), batchSize, patchSize);
    }

    /** As {@link #newTrainer(Model, Loss, float, int, int)} with a learning-rate schedule. */
    public static Trainer newTrainer(Model model, Loss loss, Tracker learningRate, int batchSize, int patchSize) {
        Device device = model.getNDManager().getDevice();
        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(Optimizer.adam().optLearningRateTracker(learningRate).build())
                .optDevices(new Device[] {device})
                // He initialisation suits ReLU networks: Gaussian, variance 2 / fan_in
                .optInitializer(new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN,
                        XavierInitializer.FactorType.IN, 2f), Parameter.Type.WEIGHT);
        Trainer trainer = model.newTrainer(config);
        Shape inputShape = new Shape(batchSize, 3, patchSize, patchSize);
        trainer.initialize(inputShape, inputShape);
        return trainer;
    }

    /**
     * One optimisation step on one batch.
     *
     * <p>All intermediate arrays (activations, loss) are created in a temporary sub-manager that is
     * closed at the end of the step, so no native memory accumulates across steps. Parameters,
     * gradients and Adam state belong to the model's manager and are unaffected.</p>
     *
     * @param inputs NDList(before, after), each (N, 3, H, W)
     * @param labels NDList(mask), (N, 1, H, W) with 0/1 values
     * @return the batch loss before the update
     */
    public static float trainStep(Trainer trainer, NDList inputs, NDList labels) {
        try (NDManager scope = inputs.head().getManager().newSubManager()) {
            inputs.tempAttach(scope);
            labels.tempAttach(scope);
            float lossValue;
            try (GradientCollector collector = trainer.newGradientCollector()) {
                NDList predictions = trainer.forward(inputs, labels);
                NDArray loss = trainer.getLoss().evaluate(labels, predictions);
                collector.backward(loss);
                lossValue = toCpuFloats(loss)[0];
            }
            trainer.step();
            return lossValue;
        }
    }

    /**
     * Inference-mode forward pass (BatchNorm uses its running statistics) returning change
     * probabilities {@code sigmoid(logits)} as a flat CPU array in (N, 1, H, W) order. Intermediate
     * arrays are released before returning.
     */
    public static float[] predictProbabilities(Block block, ParameterStore parameterStore, NDList inputs) {
        try (NDManager scope = inputs.head().getManager().newSubManager()) {
            inputs.tempAttach(scope);
            NDArray logits = block.forward(parameterStore, inputs, false).singletonOrThrow();
            return toCpuFloats(Activation.sigmoid(logits));
        }
    }

    /** Same as {@link #predictProbabilities(Block, ParameterStore, NDList)} using a trainer's weights. */
    public static float[] predictProbabilities(Trainer trainer, NDList inputs) {
        try (NDManager scope = inputs.head().getManager().newSubManager()) {
            inputs.tempAttach(scope);
            NDArray logits = trainer.evaluate(inputs).singletonOrThrow();
            return toCpuFloats(Activation.sigmoid(logits));
        }
    }

    /** Number of trainable weights (BatchNorm running statistics are not counted). */
    public static long countTrainableParameters(Block block) {
        long count = 0;
        for (Pair<String, Parameter> p : block.getParameters()) {
            if (p.getValue().requiresGradient()) {
                count += p.getValue().getArray().getShape().size();
            }
        }
        return count;
    }

    /**
     * Moves the arrays of a list to the device of {@code target} and makes {@code target} own the
     * copies (they are freed when {@code target} is closed).
     */
    public static NDList copyTo(NDList list, NDManager target) {
        NDList out = new NDList(list.size());
        for (NDArray a : list) {
            NDArray copy = a.toDevice(target.getDevice(), true);
            copy.attach(target);
            out.add(copy);
        }
        return out;
    }

    /** Copies an NDArray to the CPU (if needed) and returns its float contents. */
    public static float[] toCpuFloats(NDArray array) {
        if (array.getDevice().equals(Device.cpu())) {
            return array.toFloatArray();
        }
        try (NDArray cpu = array.toDevice(Device.cpu(), true)) {
            return cpu.toFloatArray();
        }
    }
}
