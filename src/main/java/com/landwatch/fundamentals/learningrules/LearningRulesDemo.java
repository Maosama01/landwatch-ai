package com.landwatch.fundamentals.learningrules;

import com.landwatch.fundamentals.math.Vector;

import java.util.Random;

/**
 * Console demonstration of the classic neural-network learning rules implemented in this package:
 * Hebbian, Oja (modified Hebbian), competitive, error-correction (perceptron and LMS) and Boltzmann.
 * <p>
 * Every number printed is computed live by the code. All randomness uses fixed seeds, so the
 * output is identical on every run.
 * <p>
 * Run with:
 * <pre>
 *   mvn -B compile
 *   java -cp target/classes com.landwatch.fundamentals.learningrules.LearningRulesDemo
 * </pre>
 */
public final class LearningRulesDemo {

    private static final long SEED = 42L;

    private LearningRulesDemo() {
    }

    public static void main(String[] args) {
        System.out.println("==============================================================");
        System.out.println(" LANDWATCH AI - Learning Rules Demonstration (seed " + SEED + ")");
        System.out.println("==============================================================");
        demoHebbianAndOja();
        demoCompetitive();
        demoPerceptron();
        demoLms();
        demoBoltzmann();
        System.out.println();
        System.out.println("Demo complete.");
    }

    /**
     * Runs {@link #main} and returns everything it printed, for the dashboard's NN Fundamentals page.
     * The demo itself is unchanged: {@code System.out} is redirected into a buffer for the duration of the
     * run and restored afterwards (calls are serialised, so two runs never interleave).
     */
    public static synchronized String runToString() {
        java.io.PrintStream original = System.out;
        java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
        try (java.io.PrintStream capture = new java.io.PrintStream(buffer, true, java.nio.charset.StandardCharsets.UTF_8)) {
            System.setOut(capture);
            main(new String[0]);
        } finally {
            System.setOut(original);
        }
        return buffer.toString(java.nio.charset.StandardCharsets.UTF_8);
    }

    // ------------------------------------------------------------------
    // 1 + 2. Hebbian vs. Oja on the same correlated data
    // ------------------------------------------------------------------
    private static void demoHebbianAndOja() {
        double angle = Math.toRadians(30.0);
        Vector principal = SyntheticDatasets.unitDirection(angle);
        Vector[] data = SyntheticDatasets.correlatedGaussian2D(500, 2.0, 0.5, angle, new Random(SEED));

        System.out.println();
        System.out.println("[1] HEBBIAN LEARNING   Delta w = eta * y * x,  y = w . x");
        System.out.println("    Data: 500 zero-mean 2-D points, std 2.0 along 30 deg, 0.5 across;");
        System.out.println("    analytic principal eigenvector e1 = " + principal);

        HebbianLearning hebb = new HebbianLearning(2, 0.001, new Random(SEED + 1));
        Vector hebbStart = hebb.getWeights();
        double[] hebbNorms = hebb.train(data, 8, new Random(SEED + 2));
        System.out.println("    eta = 0.001, initial w = " + hebbStart);
        System.out.print("    |w| per epoch:");
        for (double n : hebbNorms) {
            System.out.printf(" %.3g", n);
        }
        System.out.println();
        System.out.printf("    |w| grew by a factor of %.3g in 8 epochs (no upper bound).%n",
                hebbNorms[hebbNorms.length - 1] / hebbNorms[0]);
        System.out.printf("    |cos(w, e1)| = %.4f (direction aligns with e1, but length explodes)%n",
                Math.abs(LearningRuleUtils.cosineSimilarity(hebb.getWeights(), principal)));

        System.out.println();
        System.out.println("[2] OJA'S RULE (modified Hebbian)   Delta w = eta * y * (x - y * w)");
        OjaLearning oja = new OjaLearning(2, 0.005, new Random(SEED + 1));
        Vector ojaStart = oja.getWeights();
        double[] ojaNorms = oja.train(data, 30, new Random(SEED + 2));
        Vector w = oja.getWeights();
        double cos = Math.abs(LearningRuleUtils.cosineSimilarity(w, principal));
        System.out.println("    eta = 0.005, 30 epochs, initial w = " + ojaStart);
        System.out.printf("    |w| after epochs 1, 5, 30: %.4f, %.4f, %.4f%n",
                ojaNorms[1], ojaNorms[5], ojaNorms[30]);
        System.out.println("    final w = " + w + "   (e1 = " + principal + ")");
        System.out.printf("    |w| = %.4f (self-normalised to ~1), |cos(w, e1)| = %.5f, angle error = %.2f deg%n",
                oja.weightNorm(), cos, Math.toDegrees(Math.acos(Math.min(1.0, cos))));
    }

    // ------------------------------------------------------------------
    // 3. Competitive learning
    // ------------------------------------------------------------------
    private static void demoCompetitive() {
        Vector[] centres = {Vector.of(0.0, 4.0), Vector.of(-4.0, -2.0), Vector.of(4.0, -2.0)};
        Vector[] data = SyntheticDatasets.gaussianClusters(centres, 100, 0.5, new Random(SEED));

        System.out.println();
        System.out.println("[3] COMPETITIVE LEARNING (winner-take-all)");
        System.out.println("    winner k* = argmin_k |x - w_k|,  Delta w_k* = eta * (x - w_k*),  losers unchanged");
        System.out.println("    Data: 3 Gaussian clusters x 100 points (std 0.5) at (0,4), (-4,-2), (4,-2)");

        CompetitiveLearning net = CompetitiveLearning.withRandomPrototypes(3, 2, -0.5, 0.5, 0.1,
                new Random(SEED + 1));
        double[] qe = net.train(data, 20, new Random(SEED + 2));
        System.out.printf("    eta = 0.1, 20 epochs; mean distance to winner: %.3f -> %.3f%n", qe[0], qe[20]);
        for (int k = 0; k < net.getNumPrototypes(); k++) {
            Vector p = net.getPrototype(k);
            int nearest = 0;
            for (int c = 1; c < centres.length; c++) {
                if (LearningRuleUtils.euclideanDistance(p, centres[c])
                        < LearningRuleUtils.euclideanDistance(p, centres[nearest])) {
                    nearest = c;
                }
            }
            System.out.printf("    prototype %d = %s  -> nearest centre %s, distance %.3f%n",
                    k, p, centres[nearest], LearningRuleUtils.euclideanDistance(p, centres[nearest]));
        }
    }

    // ------------------------------------------------------------------
    // 4a. Perceptron (error-correction with step activation)
    // ------------------------------------------------------------------
    private static void demoPerceptron() {
        Vector[] inputs = SyntheticDatasets.logicGateInputs();

        System.out.println();
        System.out.println("[4a] PERCEPTRON RULE (Rosenblatt)   Delta w = eta * (t - y) * x,  y = step(w . x + b)");
        reportGate("AND", inputs, SyntheticDatasets.andTargets(), 100);
        reportGate("OR ", inputs, SyntheticDatasets.orTargets(), 100);
        reportGate("XOR", inputs, SyntheticDatasets.xorTargets(), 1000);

        // Re-run XOR one epoch at a time to record the best accuracy ever reached at an epoch boundary.
        int[] xor = SyntheticDatasets.xorTargets();
        PerceptronLearning p = new PerceptronLearning(2, 0.1, new Random(SEED));
        double best = 0.0;
        boolean everConverged = false;
        for (int epoch = 0; epoch < 1000; epoch++) {
            everConverged |= p.train(inputs, xor, 1).converged();
            best = Math.max(best, p.accuracy(inputs, xor));
        }
        System.out.printf("    XOR: best accuracy at any of the 1000 epoch boundaries = %.0f%%, converged = %b%n",
                100.0 * best, everConverged);
        System.out.println("    XOR is not linearly separable: a single perceptron can get at most 3/4 right,");
        System.out.println("    so its weights keep cycling. The project's MultilayerPerceptron solves XOR.");
    }

    private static void reportGate(String name, Vector[] inputs, int[] targets, int maxEpochs) {
        PerceptronLearning p = new PerceptronLearning(2, 0.1, new Random(SEED));
        PerceptronLearning.TrainingResult r = p.train(inputs, targets, maxEpochs);
        String status = r.converged()
                ? String.format("converged in %d epochs", r.epochsRun())
                : String.format("NOT converged after %d epochs", r.epochsRun());
        System.out.printf("    %s: %-32s accuracy %3.0f%%  w = %s, b = %.4f%n",
                name, status, 100.0 * p.accuracy(inputs, targets), p.getWeights(), p.getBias());
    }

    // ------------------------------------------------------------------
    // 4b. LMS / delta rule
    // ------------------------------------------------------------------
    private static void demoLms() {
        Vector trueWeights = Vector.of(2.0, -3.0, 0.5);
        double trueBias = 1.0;
        Random rng = new Random(SEED);
        Vector[] inputs = new Vector[200];
        double[] targets = new double[200];
        for (int n = 0; n < inputs.length; n++) {
            inputs[n] = Vector.random(3, -1.0, 1.0, rng);
            targets[n] = trueWeights.dot(inputs[n]) + trueBias;   // noiseless linear target
        }

        System.out.println();
        System.out.println("[4b] LMS / WIDROW-HOFF DELTA RULE   Delta w = eta * (t - w . x - b) * x");
        System.out.println("    Target: t = 2*x1 - 3*x2 + 0.5*x3 + 1  (200 noiseless samples, x in [-1,1]^3)");
        LmsLearning lms = new LmsLearning(3, 0.05, new Random(SEED + 1));
        double[] mse = lms.train(inputs, targets, 100);
        System.out.printf("    eta = 0.05, 100 epochs; MSE: %.4g -> %.4g (epoch 10: %.4g)%n", mse[0], mse[100], mse[10]);
        System.out.println("    learned w = " + lms.getWeights() + ", b = " + String.format("%.4f", lms.getBias())
                + "   (true w = " + trueWeights + ", b = 1.0000)");
    }

    // ------------------------------------------------------------------
    // 5. Boltzmann learning
    // ------------------------------------------------------------------
    private static void demoBoltzmann() {
        // Units 0 and 1 are always equal; unit 2 is independent of them.
        int[][] patterns = {{0, 0, 0}, {0, 0, 1}, {1, 1, 0}, {1, 1, 1}};

        System.out.println();
        System.out.println("[5] BOLTZMANN LEARNING   E(s) = -1/2 s^T W s - b^T s");
        System.out.println("    Delta w_ij = eta * (<s_i s_j>_data - <s_i s_j>_model), model term by Gibbs sampling");
        System.out.println("    3 binary units; training patterns 000, 001, 110, 111 (s0 always equals s1)");

        BoltzmannMachine bm = new BoltzmannMachine(3, new Random(SEED));
        double[] ll = bm.train(patterns, 300, 0.5, 1000, 20);
        System.out.printf("    eta = 0.5, 300 epochs x 1000 Gibbs samples%n");
        System.out.printf("    avg log-likelihood: %.4f -> %.4f   (best possible = ln(1/4) = %.4f)%n",
                ll[0], ll[300], Math.log(0.25));
        System.out.printf("    weights: w01 = %.3f, w02 = %.3f, w12 = %.3f;  biases: %s%n",
                bm.getWeight(0, 1), bm.getWeight(0, 2), bm.getWeight(1, 2), bm.getBiases());

        int[][] samples = bm.sample(10000, 100);
        int equal = 0;
        for (int[] s : samples) {
            if (s[0] == s[1]) {
                equal++;
            }
        }
        System.out.printf("    10000 free-running Gibbs samples: s0 == s1 in %.1f%% (chance = 50%%)%n",
                100.0 * equal / samples.length);

        System.out.println("    state  energy   P(state)   in training set?");
        for (int index = 0; index < 8; index++) {
            int[] s = BoltzmannMachine.stateFromIndex(index, 3);
            boolean trained = s[0] == s[1];
            System.out.printf("     %d%d%d  %7.3f   %.4f     %s%n",
                    s[0], s[1], s[2], bm.energy(s), bm.probability(s), trained ? "yes" : "no (s0 != s1)");
        }
    }
}
