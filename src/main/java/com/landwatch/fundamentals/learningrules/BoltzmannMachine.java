package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Matrix;
import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * A small, fully visible, stochastic binary Boltzmann machine
 * (Ackley, Hinton and Sejnowski, 1985). It is used here to demonstrate energy-based Boltzmann learning.
 * <p>
 * <b>Units and energy.</b> There are n binary units with states s_i in {0, 1}, a symmetric weight
 * matrix W (w_ij = w_ji, w_ii = 0) and biases b. Every joint state s has an energy
 * <pre>
 *   E(s) = -1/2 * s^T W s - b^T s
 *        = - sum_{i &lt; j} w_ij s_i s_j - sum_i b_i s_i
 * </pre>
 * and the machine assigns it the Boltzmann (Gibbs) probability
 * <pre>
 *   P(s) = exp(-E(s) / T) / Z,      Z = sum over all 2^n states of exp(-E(s') / T)
 * </pre>
 * where T is the temperature (T = 1 by default). Low energy means high probability.
 * <p>
 * <b>Stochastic unit update (Gibbs sampling).</b> Changing only unit i, the energy gap is
 * <pre>
 *   E(s_i = 0) - E(s_i = 1) = h_i,      h_i = sum_j w_ij s_j + b_i     (local field)
 * </pre>
 * so the probability of switching the unit on, given all the others, is
 * <pre>
 *   P(s_i = 1 | all other units) = sigmoid(h_i / T) = 1 / (1 + exp(-h_i / T))
 * </pre>
 * Updating every unit in turn like this (one "sweep") and repeating many times produces
 * samples from P(s). This is Gibbs sampling.
 * <p>
 * <b>Boltzmann learning rule.</b> Write rho_ij for the average of s_i * s_j:
 * <pre>
 *   Delta w_ij = eta * ( rho_ij(data) - rho_ij(model) )      for i != j
 *   Delta b_i  = eta * ( mean_i(data) - mean_i(model) )
 * </pre>
 * rho_ij(data) is measured with the units clamped to the training patterns (the "positive"
 * or clamped phase). rho_ij(model) is measured while the machine runs freely, estimated here
 * by Gibbs sampling (the "negative" or free-running phase). The rule is gradient ascent on the
 * average log-likelihood of the training data (the gradient carries a factor 1/T, which is
 * absorbed into eta). Learning stops when the model reproduces the data's pairwise statistics. A bias is a weight to a unit that is always on, so the same rule
 * gives Delta b_i.
 * <p>
 * The effect is that the energy of the training patterns goes down and the energy of
 * patterns not in the data goes up.
 * <p>
 * Haykin's textbook uses bipolar states x in {-1, +1}. The two forms are equivalent: substituting
 * s = (x + 1) / 2 turns one energy function into the other, with rescaled weights and shifted biases.
 * <p>
 * Because the machine is tiny, the exact distribution can also be computed by enumerating all
 * 2^n states ({@link #exactModelStatistics()}, {@link #averageLogLikelihood(int[][])}). This lets
 * us check the Gibbs-sampling estimates and report the true log-likelihood during training.
 */
public class BoltzmannMachine {

    /** Largest machine for which exact enumeration of all 2^n states is allowed. */
    public static final int MAX_UNITS_FOR_EXACT = 16;

    /**
     * First- and second-order statistics of binary states.
     *
     * @param pairwise n x n matrix rho with rho_ij = average of s_i * s_j
     * @param means    vector m with m_i = average of s_i
     */
    public record Statistics(Matrix pairwise, Vector means) {
    }

    private final int numUnits;
    private final double[][] weights;
    private final double[] biases;
    private final double temperature;
    private final Random rng;

    /**
     * Creates a machine with all weights and biases equal to zero.
     *
     * @param numUnits    number of binary units (at least 2)
     * @param temperature T &gt; 0
     * @param rng         seeded random generator used for Gibbs sampling
     */
    public BoltzmannMachine(int numUnits, double temperature, Random rng) {
        if (numUnits < 2) {
            throw new IllegalArgumentException("A Boltzmann machine needs at least 2 units, got: " + numUnits);
        }
        if (!(temperature > 0.0) || Double.isInfinite(temperature)) {
            throw new IllegalArgumentException("Temperature must be positive and finite, got: " + temperature);
        }
        if (rng == null) {
            throw new IllegalArgumentException("Random number generator must not be null.");
        }
        this.numUnits = numUnits;
        this.weights = new double[numUnits][numUnits];
        this.biases = new double[numUnits];
        this.temperature = temperature;
        this.rng = rng;
    }

    /**
     * Creates a machine at temperature T = 1 with zero weights and biases.
     */
    public BoltzmannMachine(int numUnits, Random rng) {
        this(numUnits, 1.0, rng);
    }

    // --- Parameters ---

    public int getNumUnits() {
        return numUnits;
    }

    public double getTemperature() {
        return temperature;
    }

    /** Weight w_ij (equal to w_ji; 0 on the diagonal). */
    public double getWeight(int i, int j) {
        checkUnit(i);
        checkUnit(j);
        return weights[i][j];
    }

    /**
     * Sets w_ij AND w_ji to the same value, which keeps W symmetric.
     * Self-connections (i == j) are not allowed.
     */
    public void setWeight(int i, int j, double value) {
        checkUnit(i);
        checkUnit(j);
        if (i == j) {
            throw new IllegalArgumentException("Self-connections are not allowed (w_ii must stay 0).");
        }
        weights[i][j] = value;
        weights[j][i] = value;
    }

    public double getBias(int i) {
        checkUnit(i);
        return biases[i];
    }

    public void setBias(int i, double value) {
        checkUnit(i);
        biases[i] = value;
    }

    /** Copy of the weight matrix W. */
    public Matrix getWeights() {
        return new Matrix(weights);
    }

    /** Copy of the bias vector b. */
    public Vector getBiases() {
        return new Vector(biases);
    }

    // --- Energy and local field ---

    /**
     * Energy E(s) = -1/2 * s^T W s - b^T s = - sum_{i&lt;j} w_ij s_i s_j - sum_i b_i s_i.
     */
    public double energy(int[] state) {
        checkState(state);
        double energy = 0.0;
        for (int i = 0; i < numUnits; i++) {
            for (int j = i + 1; j < numUnits; j++) {
                energy -= weights[i][j] * state[i] * state[j];   // pairwise term
            }
        }
        for (int i = 0; i < numUnits; i++) {
            energy -= biases[i] * state[i];                      // bias term
        }
        return energy;
    }

    /**
     * Local field h_i = sum_j w_ij s_j + b_i, which equals E(s_i = 0) - E(s_i = 1).
     */
    public double localField(int[] state, int i) {
        checkState(state);
        checkUnit(i);
        double h = biases[i];
        for (int j = 0; j < numUnits; j++) {
            h += weights[i][j] * state[j];   // w_ii = 0, so s_i itself never contributes
        }
        return h;
    }

    /**
     * Probability that unit i is on given all other units: sigmoid(h_i / T).
     */
    public double probabilityOn(int[] state, int i) {
        return LearningRuleUtils.sigmoid(localField(state, i) / temperature);
    }

    // --- Gibbs sampling ---

    /** A uniformly random binary state, drawn from the machine's seeded generator. */
    public int[] randomState() {
        int[] s = new int[numUnits];
        for (int i = 0; i < numUnits; i++) {
            s[i] = rng.nextBoolean() ? 1 : 0;
        }
        return s;
    }

    /**
     * One Gibbs sweep, done in place: every unit i = 0..n-1 in turn is set to 1 with probability
     * sigmoid(h_i / T), and to 0 otherwise.
     */
    public void gibbsSweep(int[] state) {
        checkState(state);
        for (int i = 0; i < numUnits; i++) {
            state[i] = rng.nextDouble() < probabilityOn(state, i) ? 1 : 0;
        }
    }

    /**
     * Draws samples from the free-running machine by Gibbs sampling.
     * <p>
     * The chain starts from a random state, discards {@code burnInSweeps} sweeps so it can forget
     * the starting point, then records one state after each further sweep.
     *
     * @param numSamples   number of states to return
     * @param burnInSweeps number of initial sweeps to discard
     * @return numSamples binary states
     */
    public int[][] sample(int numSamples, int burnInSweeps) {
        if (numSamples <= 0) {
            throw new IllegalArgumentException("Number of samples must be positive, got: " + numSamples);
        }
        if (burnInSweeps < 0) {
            throw new IllegalArgumentException("Burn-in must be non-negative, got: " + burnInSweeps);
        }
        int[] state = randomState();
        for (int k = 0; k < burnInSweeps; k++) {
            gibbsSweep(state);
        }
        int[][] samples = new int[numSamples][];
        for (int k = 0; k < numSamples; k++) {
            gibbsSweep(state);
            samples[k] = state.clone();
        }
        return samples;
    }

    // --- Statistics ---

    /**
     * Clamped-phase statistics of a set of binary patterns:
     * rho_ij = (1/N) sum_n s_i s_j and m_i = (1/N) sum_n s_i.
     */
    public static Statistics dataStatistics(int[][] patterns) {
        if (patterns == null || patterns.length == 0 || patterns[0] == null) {
            throw new IllegalArgumentException("Patterns must not be null or empty.");
        }
        int n = patterns[0].length;
        if (n == 0) {
            throw new IllegalArgumentException("Patterns must have at least one unit.");
        }
        double[][] pair = new double[n][n];
        double[] mean = new double[n];
        for (int[] s : patterns) {
            if (s == null || s.length != n) {
                throw new IllegalArgumentException("All patterns must have length " + n);
            }
            for (int i = 0; i < n; i++) {
                if (s[i] != 0 && s[i] != 1) {
                    throw new IllegalArgumentException("Pattern values must be 0 or 1, got: " + s[i]);
                }
                mean[i] += s[i];
                for (int j = 0; j < n; j++) {
                    pair[i][j] += s[i] * s[j];
                }
            }
        }
        double inv = 1.0 / patterns.length;
        for (int i = 0; i < n; i++) {
            mean[i] *= inv;
            for (int j = 0; j < n; j++) {
                pair[i][j] *= inv;
            }
        }
        return new Statistics(new Matrix(pair), new Vector(mean));
    }

    /**
     * Free-running (model) statistics estimated by Gibbs sampling.
     *
     * @param numSamples   number of Gibbs samples to average over
     * @param burnInSweeps sweeps discarded before collecting samples
     * @return estimated rho(model) and m(model)
     */
    public Statistics estimateModelStatistics(int numSamples, int burnInSweeps) {
        return dataStatistics(sample(numSamples, burnInSweeps));
    }

    /**
     * Exact model statistics, computed by summing over all 2^n states weighted by P(s).
     * This is only practical for small machines (n &lt;= {@value #MAX_UNITS_FOR_EXACT}).
     */
    public Statistics exactModelStatistics() {
        checkSmallEnoughForExact();
        double logZ = logPartitionFunction();
        double[][] pair = new double[numUnits][numUnits];
        double[] mean = new double[numUnits];
        int numStates = 1 << numUnits;
        for (int index = 0; index < numStates; index++) {
            int[] s = stateFromIndex(index, numUnits);
            double p = Math.exp(-energy(s) / temperature - logZ);
            for (int i = 0; i < numUnits; i++) {
                mean[i] += p * s[i];
                for (int j = 0; j < numUnits; j++) {
                    pair[i][j] += p * s[i] * s[j];
                }
            }
        }
        return new Statistics(new Matrix(pair), new Vector(mean));
    }

    // --- Boltzmann learning rule ---

    /**
     * Weight change of the Boltzmann learning rule:
     * Delta w_ij = eta * (rho_ij(data) - rho_ij(model)) for i != j, and 0 on the diagonal.
     */
    public static Matrix weightUpdate(Statistics data, Statistics model, double learningRate) {
        LearningRuleUtils.requireValidLearningRate(learningRate);
        checkCompatible(data, model);
        Matrix delta = data.pairwise().subtract(model.pairwise()).multiply(learningRate);
        for (int i = 0; i < delta.getRows(); i++) {
            delta.set(i, i, 0.0);   // no self-connections
        }
        return delta;
    }

    /**
     * Bias change of the Boltzmann learning rule: Delta b_i = eta * (m_i(data) - m_i(model)).
     */
    public static Vector biasUpdate(Statistics data, Statistics model, double learningRate) {
        LearningRuleUtils.requireValidLearningRate(learningRate);
        checkCompatible(data, model);
        return data.means().subtract(model.means()).multiply(learningRate);
    }

    /**
     * Adds Delta W and Delta b to the parameters. Only the upper triangle (i &lt; j) of Delta W is read,
     * and it is written to both w_ij and w_ji, so W stays symmetric with a zero diagonal.
     */
    public void applyUpdate(Matrix deltaW, Vector deltaB) {
        if (deltaW == null || deltaB == null) {
            throw new IllegalArgumentException("Updates must not be null.");
        }
        if (deltaW.getRows() != numUnits || deltaW.getCols() != numUnits || deltaB.size() != numUnits) {
            throw new IllegalArgumentException("Update dimensions do not match the number of units " + numUnits);
        }
        for (int i = 0; i < numUnits; i++) {
            for (int j = i + 1; j < numUnits; j++) {
                double w = weights[i][j] + deltaW.get(i, j);
                weights[i][j] = w;
                weights[j][i] = w;
            }
            biases[i] += deltaB.get(i);
        }
    }

    /**
     * One learning step: clamped statistics from the patterns, free-running statistics from Gibbs
     * sampling, then Delta W and Delta b are applied.
     *
     * @param patterns     training patterns (clamped phase)
     * @param learningRate eta &gt; 0
     * @param numSamples   Gibbs samples for the free-running phase
     * @param burnInSweeps Gibbs sweeps discarded before sampling
     * @return the weight change Delta W that was applied
     */
    public Matrix trainStep(int[][] patterns, double learningRate, int numSamples, int burnInSweeps) {
        checkPatternLength(patterns);
        Statistics data = dataStatistics(patterns);
        Statistics model = estimateModelStatistics(numSamples, burnInSweeps);
        Matrix deltaW = weightUpdate(data, model, learningRate);
        Vector deltaB = biasUpdate(data, model, learningRate);
        applyUpdate(deltaW, deltaB);
        return deltaW;
    }

    /**
     * Runs {@link #trainStep} for a number of epochs.
     *
     * @return array of length epochs + 1 with the exact average log-likelihood of the patterns
     *         before training and after each epoch (higher is better). This needs a small machine.
     */
    public double[] train(int[][] patterns, int epochs, double learningRate, int numSamples, int burnInSweeps) {
        LearningRuleUtils.requireNonNegativeEpochs(epochs);
        double[] logLikelihood = new double[epochs + 1];
        logLikelihood[0] = averageLogLikelihood(patterns);
        for (int epoch = 1; epoch <= epochs; epoch++) {
            trainStep(patterns, learningRate, numSamples, burnInSweeps);
            logLikelihood[epoch] = averageLogLikelihood(patterns);
        }
        return logLikelihood;
    }

    // --- Exact quantities (small machines only) ---

    /**
     * log Z = log sum_s exp(-E(s) / T), computed with the log-sum-exp trick for numerical stability.
     */
    public double logPartitionFunction() {
        checkSmallEnoughForExact();
        int numStates = 1 << numUnits;
        double[] logTerms = new double[numStates];
        double max = Double.NEGATIVE_INFINITY;
        for (int index = 0; index < numStates; index++) {
            logTerms[index] = -energy(stateFromIndex(index, numUnits)) / temperature;
            max = Math.max(max, logTerms[index]);
        }
        double sum = 0.0;
        for (double t : logTerms) {
            sum += Math.exp(t - max);
        }
        return max + Math.log(sum);
    }

    /** Exact probability P(s) = exp(-E(s) / T) / Z. */
    public double probability(int[] state) {
        return Math.exp(-energy(state) / temperature - logPartitionFunction());
    }

    /**
     * Exact average log-likelihood (1/N) * sum_n log P(s_n) of the patterns under the model.
     * The Boltzmann learning rule performs gradient ascent on this quantity.
     */
    public double averageLogLikelihood(int[][] patterns) {
        checkPatternLength(patterns);
        double logZ = logPartitionFunction();
        double total = 0.0;
        for (int[] s : patterns) {
            total += -energy(s) / temperature - logZ;
        }
        return total / patterns.length;
    }

    /**
     * Decodes an integer in [0, 2^n) into a binary state, unit 0 being the most significant bit.
     * For example, with n = 3, index 6 (binary 110) gives the state (1, 1, 0).
     */
    public static int[] stateFromIndex(int index, int numUnits) {
        if (numUnits <= 0 || numUnits > 30 || index < 0 || index >= (1 << numUnits)) {
            throw new IllegalArgumentException("Invalid state index " + index + " for " + numUnits + " units.");
        }
        int[] s = new int[numUnits];
        for (int i = 0; i < numUnits; i++) {
            s[i] = (index >> (numUnits - 1 - i)) & 1;
        }
        return s;
    }

    // --- Validation helpers ---

    private void checkUnit(int i) {
        if (i < 0 || i >= numUnits) {
            throw new IndexOutOfBoundsException("Unit index " + i + " out of range for " + numUnits + " units.");
        }
    }

    private void checkState(int[] state) {
        if (state == null || state.length != numUnits) {
            throw new IllegalArgumentException("State must have length " + numUnits);
        }
        for (int v : state) {
            if (v != 0 && v != 1) {
                throw new IllegalArgumentException("State values must be 0 or 1, got: " + v);
            }
        }
    }

    private void checkPatternLength(int[][] patterns) {
        if (patterns == null || patterns.length == 0) {
            throw new IllegalArgumentException("Patterns must not be null or empty.");
        }
        for (int[] p : patterns) {
            checkState(p);
        }
    }

    private void checkSmallEnoughForExact() {
        if (numUnits > MAX_UNITS_FOR_EXACT) {
            throw new IllegalStateException("Exact enumeration is limited to " + MAX_UNITS_FOR_EXACT
                    + " units; this machine has " + numUnits);
        }
    }

    private static void checkCompatible(Statistics data, Statistics model) {
        if (data == null || model == null) {
            throw new IllegalArgumentException("Statistics must not be null.");
        }
        if (data.means().size() != model.means().size()) {
            throw new IllegalArgumentException("Data and model statistics have different numbers of units.");
        }
    }

    @Override
    public String toString() {
        return String.format("BoltzmannMachine(units=%d, T=%.2f)", numUnits, temperature);
    }
}
