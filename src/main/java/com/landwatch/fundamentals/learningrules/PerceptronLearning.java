package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Arrays;
import java.util.Random;

/**
 * Rosenblatt's perceptron learning rule (Frank Rosenblatt, 1958): error-correction learning
 * with a hard-threshold (step) activation.
 * <p>
 * <b>Forward pass</b>:
 * <pre>
 *   v = w . x + b                         (net input / induced local field)
 *   y = step(v) = 1 if v &gt;= 0, else 0
 * </pre>
 * <b>Error-correction update</b> (targets t in {0, 1}):
 * <pre>
 *   e       = t - y                       (always -1, 0 or +1)
 *   Delta w = eta * (t - y) * x
 *   Delta b = eta * (t - y)
 * </pre>
 * The weights change only when a sample is misclassified (e != 0). A false negative (t=1, y=0)
 * adds eta*x to w, and a false positive (t=0, y=1) subtracts it.
 * <p>
 * <b>Perceptron convergence theorem.</b> If the two classes are linearly separable, the rule
 * finds a separating hyperplane after a finite number of updates, for any eta &gt; 0.
 * AND and OR are linearly separable.
 * <p>
 * <b>The XOR limit.</b> XOR is not linearly separable. Suppose a line classified all four
 * patterns correctly:
 * <pre>
 *   (0,0) -&gt; 0 :  b &lt; 0
 *   (0,1) -&gt; 1 :  w2 + b &gt;= 0
 *   (1,0) -&gt; 1 :  w1 + b &gt;= 0
 *   (1,1) -&gt; 0 :  w1 + w2 + b &lt; 0
 * </pre>
 * Adding the middle two gives w1 + w2 + b &gt;= -b &gt; 0, which contradicts the last line. So a
 * single perceptron can get at most 3 of the 4 XOR patterns right, and the rule cycles forever.
 * The project's {@code MultilayerPerceptron}, which adds a hidden layer and is trained by
 * backpropagation, does solve XOR (see {@code BackpropagationEngineTest}).
 */
public class PerceptronLearning {

    /**
     * Summary of a training run.
     *
     * @param epochsRun         number of epochs actually performed
     * @param converged         true if an epoch finished with zero misclassifications
     * @param errorsPerEpoch    number of misclassified (i.e. weight-changing) samples in each epoch
     */
    public record TrainingResult(int epochsRun, boolean converged, int[] errorsPerEpoch) {

        /** Misclassifications in the last epoch performed (0 if converged). */
        public int errorsInLastEpoch() {
            return errorsPerEpoch.length == 0 ? 0 : errorsPerEpoch[errorsPerEpoch.length - 1];
        }

        @Override
        public String toString() {
            return String.format("TrainingResult(epochsRun=%d, converged=%b, errorsPerEpoch=%s)",
                    epochsRun, converged, Arrays.toString(errorsPerEpoch));
        }
    }

    private Vector weights;
    private double bias;
    private final double learningRate;

    /**
     * Creates a perceptron with explicit initial weights and bias.
     *
     * @param initialWeights starting weights (copied)
     * @param initialBias    starting bias b
     * @param learningRate   eta &gt; 0
     */
    public PerceptronLearning(Vector initialWeights, double initialBias, double learningRate) {
        if (initialWeights == null) {
            throw new IllegalArgumentException("Initial weights must not be null.");
        }
        LearningRuleUtils.requireValidLearningRate(learningRate);
        this.weights = initialWeights.copy();
        this.bias = initialBias;
        this.learningRate = learningRate;
    }

    /**
     * Creates a perceptron with small random weights and bias drawn uniformly from [-0.5, 0.5].
     *
     * @param numInputs    number of inputs
     * @param learningRate eta &gt; 0
     * @param rng          seeded random number generator
     */
    public PerceptronLearning(int numInputs, double learningRate, Random rng) {
        this(Vector.random(numInputs, -0.5, 0.5, rng), -0.5 + rng.nextDouble(), learningRate);
    }

    /** Net input v = w . x + b. */
    public double netInput(Vector x) {
        LearningRuleUtils.requireSize(x, weights.size(), "Input");
        return weights.dot(x) + bias;
    }

    /** Step activation y = 1 if v &gt;= 0, else 0. */
    public int predict(Vector x) {
        return netInput(x) >= 0.0 ? 1 : 0;
    }

    /**
     * Applies one perceptron step: Delta w = eta * (t - y) * x, Delta b = eta * (t - y).
     *
     * @param x      input pattern
     * @param target desired output t in {0, 1}
     * @return the error e = t - y (0 means the sample was already correct and nothing changed)
     */
    public int update(Vector x, int target) {
        requireBinaryTarget(target);
        int y = predict(x);
        int error = target - y;
        if (error != 0) {
            weights = weights.add(x.multiply(learningRate * error));
            bias += learningRate * error;
        }
        return error;
    }

    /**
     * Presents the samples in their given order, epoch after epoch, until an epoch has no
     * misclassifications (convergence) or {@code maxEpochs} is reached.
     *
     * @param inputs    input patterns
     * @param targets   desired outputs in {0, 1}
     * @param maxEpochs upper limit on epochs
     * @return the training summary
     */
    public TrainingResult train(Vector[] inputs, int[] targets, int maxEpochs) {
        validateDataset(inputs, targets);
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("maxEpochs must be positive, got: " + maxEpochs);
        }
        int[] errorsPerEpoch = new int[maxEpochs];
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            int errors = 0;
            for (int n = 0; n < inputs.length; n++) {
                if (update(inputs[n], targets[n]) != 0) {
                    errors++;
                }
            }
            errorsPerEpoch[epoch] = errors;
            if (errors == 0) {
                return new TrainingResult(epoch + 1, true, Arrays.copyOf(errorsPerEpoch, epoch + 1));
            }
        }
        return new TrainingResult(maxEpochs, false, errorsPerEpoch);
    }

    /**
     * Fraction of samples classified correctly by the current weights (in [0, 1]).
     */
    public double accuracy(Vector[] inputs, int[] targets) {
        validateDataset(inputs, targets);
        int correct = 0;
        for (int n = 0; n < inputs.length; n++) {
            if (predict(inputs[n]) == targets[n]) {
                correct++;
            }
        }
        return (double) correct / inputs.length;
    }

    /** Returns a copy of the weight vector w. */
    public Vector getWeights() {
        return weights.copy();
    }

    public double getBias() {
        return bias;
    }

    public double getLearningRate() {
        return learningRate;
    }

    private void validateDataset(Vector[] inputs, int[] targets) {
        LearningRuleUtils.requireNonEmpty(inputs, "Inputs");
        if (targets == null || targets.length != inputs.length) {
            throw new IllegalArgumentException("Targets must be non-null and match the number of inputs.");
        }
        for (int t : targets) {
            requireBinaryTarget(t);
        }
    }

    private static void requireBinaryTarget(int target) {
        if (target != 0 && target != 1) {
            throw new IllegalArgumentException("Perceptron targets must be 0 or 1, got: " + target);
        }
    }

    @Override
    public String toString() {
        return String.format("PerceptronLearning(eta=%.4f, w=%s, b=%.4f)", learningRate, weights, bias);
    }
}
