package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.landwatch.deeplearning.TrainingSupport;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * DJL plumbing for the EuroSAT classifier: building the model and trainer, one training step,
 * inference, and checkpoint files.
 *
 * <p>Memory discipline: every step creates its arrays (inputs, activations, logits, loss) in a
 * temporary sub-{@link NDManager} that is closed at the end of the step, so native memory does not
 * pile up over thousands of steps. Weights, gradients and Adam state belong to the model's manager
 * and live for the whole run.</p>
 */
public final class EuroSatTraining {

    /** Name of the DJL model and of its weight file ({@code eurosat_resnet-<epoch>.params}). */
    public static final String MODEL_NAME = "eurosat_resnet";

    private static final Pattern PARAMS_FILE = Pattern.compile(Pattern.quote(MODEL_NAME) + "-\\d{4}\\.params");

    private EuroSatTraining() {
    }

    /** A DJL model (PyTorch engine) wrapping a new, uninitialised {@link EuroSatResNet}. */
    public static Model newModel(int[] widths, int blocksPerStage, int numClasses, Device device) {
        Model model = Model.newInstance(MODEL_NAME, device, "PyTorch");
        model.setBlock(new EuroSatResNet(widths, blocksPerStage, numClasses));
        return model;
    }

    /**
     * A trainer with softmax cross-entropy loss, Adam and He (Kaiming) initialisation; parameters are
     * initialised for (batchSize, 3, inputSize, inputSize) inputs.
     *
     * @param weightDecay L2 penalty added to the gradients by DJL's Adam (0 = none)
     */
    public static Trainer newTrainer(Model model, Tracker learningRate, float weightDecay, int batchSize, int inputSize) {
        Device device = model.getNDManager().getDevice();
        DefaultTrainingConfig config = new DefaultTrainingConfig(new SoftmaxCrossEntropyLoss("softmax_cross_entropy"))
                .optOptimizer(Optimizer.adam().optLearningRateTracker(learningRate).optWeightDecays(weightDecay).build())
                .optDevices(new Device[] {device})
                // He initialisation suits ReLU networks: Gaussian with variance 2 / fan_in
                .optInitializer(new XavierInitializer(XavierInitializer.RandomType.GAUSSIAN,
                        XavierInitializer.FactorType.IN, 2f), Parameter.Type.WEIGHT);
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(batchSize, 3, inputSize, inputSize));
        return trainer;
    }

    /** Mean loss of a batch and the number of correctly classified images in it. */
    public record StepResult(float loss, int correct) {
    }

    /**
     * One optimisation step: forward pass (recorded for autograd), softmax cross-entropy, backward
     * pass, Adam update.
     *
     * @param images normalised images, NCHW, length n*3*size*size
     * @param labels class indices, length at least n
     * @return the batch loss before the update and the number of correct (arg-max) predictions
     */
    public static StepResult trainStep(Trainer trainer, float[] images, long[] labels, int n, int size) {
        float lossValue;
        int correct;
        try (NDManager scope = trainer.getManager().newSubManager()) {
            NDArray x = scope.create(images, new Shape(n, 3, size, size));
            NDArray y = scope.create(java.util.Arrays.copyOf(labels, n));
            try (GradientCollector collector = trainer.newGradientCollector()) {
                NDArray logits = trainer.forward(new NDList(x)).singletonOrThrow();
                NDArray loss = trainer.getLoss().evaluate(new NDList(y), new NDList(logits));
                collector.backward(loss);
                lossValue = TrainingSupport.toCpuFloats(loss)[0];
                correct = countCorrect(TrainingSupport.toCpuFloats(logits), labels, n);
            }
            trainer.step();
        }
        return new StepResult(lossValue, correct);
    }

    /**
     * Inference-mode forward pass (BatchNorm uses its running statistics) returning the logits as a
     * flat CPU array of n x K values; also computes the mean loss when {@code labels} is given.
     *
     * @param lossOut if not null and labels given, {@code lossOut[0]} receives the batch mean loss
     */
    public static float[] evaluateBatch(Trainer trainer, float[] images, long[] labels, int n, int size, float[] lossOut) {
        try (NDManager scope = trainer.getManager().newSubManager()) {
            NDArray x = scope.create(images, new Shape(n, 3, size, size));
            NDArray logits = trainer.evaluate(new NDList(x)).singletonOrThrow();
            if (labels != null && lossOut != null) {
                NDArray y = scope.create(java.util.Arrays.copyOf(labels, n));
                lossOut[0] = TrainingSupport.toCpuFloats(trainer.getLoss().evaluate(new NDList(y), new NDList(logits)))[0];
            }
            return TrainingSupport.toCpuFloats(logits);
        }
    }

    /**
     * Inference with a loaded block: softmax probabilities as a flat CPU array of n x K values.
     * Intermediate arrays are released before returning.
     */
    public static float[] predictProbabilities(Block block, ParameterStore ps, NDManager parent, float[] images,
                                               int n, int size) {
        try (NDManager scope = parent.newSubManager()) {
            NDArray x = scope.create(images, new Shape(n, 3, size, size));
            NDArray logits = block.forward(ps, new NDList(x), false).singletonOrThrow();
            return TrainingSupport.toCpuFloats(logits.softmax(1));
        }
    }

    /** Index of the largest value in row {@code row} of a flat (n x k) array; ties go to the lower index. */
    public static int argMax(float[] values, int row, int k) {
        int best = 0;
        float bestValue = values[row * k];
        for (int c = 1; c < k; c++) {
            float v = values[row * k + c];
            if (v > bestValue) {
                bestValue = v;
                best = c;
            }
        }
        return best;
    }

    private static int countCorrect(float[] logits, long[] labels, int n) {
        int k = logits.length / n;
        int correct = 0;
        for (int i = 0; i < n; i++) {
            if (argMax(logits, i, k) == labels[i]) {
                correct++;
            }
        }
        return correct;
    }

    /**
     * Saves the weights as {@code eurosat_resnet-<epoch>.params}, first deleting older
     * {@code eurosat_resnet-*.params} files so the folder always holds exactly one set of weights.
     */
    public static void saveCheckpoint(Model model, Path dir, int epoch) throws IOException {
        Files.createDirectories(dir);
        try (Stream<Path> files = Files.list(dir)) {
            for (Path f : (Iterable<Path>) files::iterator) {
                if (PARAMS_FILE.matcher(f.getFileName().toString()).matches()) {
                    Files.delete(f);
                }
            }
        }
        model.setProperty("Epoch", String.valueOf(epoch));
        model.save(dir, MODEL_NAME);
    }
}
