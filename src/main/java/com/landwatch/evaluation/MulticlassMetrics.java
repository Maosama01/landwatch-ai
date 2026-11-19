package com.landwatch.evaluation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Evaluation of single-label multi-class classification (for example the 10 EuroSAT land-use /
 * land-cover classes) from a K x K confusion matrix.
 *
 * <p>Convention: {@code matrix[actual][predicted]}, i.e. rows are the true classes and columns are
 * the predicted classes. The diagonal holds the correct predictions.</p>
 *
 * <p>For class {@code c}:</p>
 * <ul>
 *   <li>TP = {@code matrix[c][c]};</li>
 *   <li>FP = column sum of {@code c} minus TP (other classes predicted as {@code c});</li>
 *   <li>FN = row sum of {@code c} minus TP ({@code c} predicted as something else);</li>
 *   <li>precision = TP / (TP + FP), recall = TP / (TP + FN),
 *       F1 = 2TP / (2TP + FP + FN) (the harmonic mean of precision and recall).</li>
 * </ul>
 *
 * <p>Overall accuracy = trace / total. Macro averages are the unweighted means over all K classes,
 * so every class counts equally regardless of how many images it has (EuroSAT has 2,000 to 3,000
 * images per class).</p>
 *
 * <p>Convention, as in {@link BinaryChangeMetrics}: a ratio whose denominator is zero (for example
 * the precision of a class that was never predicted) is reported as 0.0.</p>
 *
 * <p>This class is independent of {@link BinaryChangeMetrics}, which handles the binary
 * change / no-change pixel problem.</p>
 */
public final class MulticlassMetrics {

    private final int numClasses;
    private final long[][] matrix;

    /** Creates an empty K x K confusion matrix. */
    public MulticlassMetrics(int numClasses) {
        if (numClasses < 2) {
            throw new IllegalArgumentException("Need at least 2 classes, got " + numClasses);
        }
        this.numClasses = numClasses;
        this.matrix = new long[numClasses][numClasses];
    }

    /** Creates metrics from known counts, {@code counts[actual][predicted]} (square, non-negative). */
    public static MulticlassMetrics fromConfusionMatrix(long[][] counts) {
        if (counts == null) {
            throw new IllegalArgumentException("Confusion matrix must not be null");
        }
        MulticlassMetrics m = new MulticlassMetrics(counts.length);
        for (int a = 0; a < counts.length; a++) {
            if (counts[a].length != counts.length) {
                throw new IllegalArgumentException("Confusion matrix must be square; row " + a + " has "
                        + counts[a].length + " entries, expected " + counts.length);
            }
            for (int p = 0; p < counts.length; p++) {
                if (counts[a][p] < 0) {
                    throw new IllegalArgumentException("Counts must be non-negative");
                }
                m.matrix[a][p] = counts[a][p];
            }
        }
        return m;
    }

    /** Records one prediction. */
    public void add(int actual, int predicted) {
        checkClass(actual);
        checkClass(predicted);
        matrix[actual][predicted]++;
    }

    /** Records many predictions; both arrays must have the same length. */
    public void addAll(int[] actual, int[] predicted) {
        if (actual.length != predicted.length) {
            throw new IllegalArgumentException("Lengths differ: " + actual.length + " actual vs "
                    + predicted.length + " predicted");
        }
        for (int i = 0; i < actual.length; i++) {
            add(actual[i], predicted[i]);
        }
    }

    /** Adds the counts of another matrix with the same number of classes. */
    public void merge(MulticlassMetrics other) {
        if (other.numClasses != numClasses) {
            throw new IllegalArgumentException("Class counts differ: " + numClasses + " vs " + other.numClasses);
        }
        for (int a = 0; a < numClasses; a++) {
            for (int p = 0; p < numClasses; p++) {
                matrix[a][p] += other.matrix[a][p];
            }
        }
    }

    public int getNumClasses() {
        return numClasses;
    }

    /** Number of images of class {@code actual} predicted as {@code predicted}. */
    public long count(int actual, int predicted) {
        checkClass(actual);
        checkClass(predicted);
        return matrix[actual][predicted];
    }

    /** A copy of the confusion matrix, {@code [actual][predicted]}. */
    public long[][] getConfusionMatrix() {
        long[][] copy = new long[numClasses][];
        for (int a = 0; a < numClasses; a++) {
            copy[a] = matrix[a].clone();
        }
        return copy;
    }

    /** Total number of recorded predictions. */
    public long getTotal() {
        long total = 0;
        for (long[] row : matrix) {
            for (long v : row) {
                total += v;
            }
        }
        return total;
    }

    /** Number of correct predictions (the trace of the matrix). */
    public long getCorrect() {
        long correct = 0;
        for (int c = 0; c < numClasses; c++) {
            correct += matrix[c][c];
        }
        return correct;
    }

    /** Number of images whose true class is {@code c} (row sum). */
    public long support(int c) {
        checkClass(c);
        long s = 0;
        for (long v : matrix[c]) {
            s += v;
        }
        return s;
    }

    /** Number of images predicted as {@code c} (column sum). */
    public long predictedCount(int c) {
        checkClass(c);
        long s = 0;
        for (long[] row : matrix) {
            s += row[c];
        }
        return s;
    }

    public long truePositives(int c) {
        checkClass(c);
        return matrix[c][c];
    }

    public long falsePositives(int c) {
        return predictedCount(c) - truePositives(c);
    }

    public long falseNegatives(int c) {
        return support(c) - truePositives(c);
    }

    /** Overall accuracy: correct / total. */
    public double accuracy() {
        return ratio(getCorrect(), getTotal());
    }

    /** Precision of class {@code c}: TP / (TP + FP). */
    public double precision(int c) {
        return ratio(truePositives(c), predictedCount(c));
    }

    /** Recall of class {@code c}: TP / (TP + FN). */
    public double recall(int c) {
        return ratio(truePositives(c), support(c));
    }

    /** F1 of class {@code c}: 2TP / (2TP + FP + FN). */
    public double f1(int c) {
        long tp = truePositives(c);
        return ratio(2.0 * tp, 2.0 * tp + falsePositives(c) + falseNegatives(c));
    }

    /** Unweighted mean of the per-class precisions. */
    public double macroPrecision() {
        double s = 0;
        for (int c = 0; c < numClasses; c++) {
            s += precision(c);
        }
        return s / numClasses;
    }

    /** Unweighted mean of the per-class recalls (also called balanced accuracy). */
    public double macroRecall() {
        double s = 0;
        for (int c = 0; c < numClasses; c++) {
            s += recall(c);
        }
        return s / numClasses;
    }

    /** Unweighted mean of the per-class F1 scores. */
    public double macroF1() {
        double s = 0;
        for (int c = 0; c < numClasses; c++) {
            s += f1(c);
        }
        return s / numClasses;
    }

    /**
     * The off-diagonal cells sorted by count (largest first), ignoring empty cells.
     *
     * @param limit maximum number of entries returned
     * @return entries {actual, predicted, count}
     */
    public List<long[]> topConfusions(int limit) {
        List<long[]> cells = new ArrayList<>();
        for (int a = 0; a < numClasses; a++) {
            for (int p = 0; p < numClasses; p++) {
                if (a != p && matrix[a][p] > 0) {
                    cells.add(new long[] {a, p, matrix[a][p]});
                }
            }
        }
        cells.sort((x, y) -> x[2] != y[2] ? Long.compare(y[2], x[2])
                : x[0] != y[0] ? Long.compare(x[0], y[0]) : Long.compare(x[1], y[1]));
        return cells.subList(0, Math.min(limit, cells.size()));
    }

    /**
     * Multi-line report: overall accuracy, macro scores, the per-class table and the confusion
     * matrix. The lines {@code Overall accuracy : 0.xxxx} and {@code Macro F1         : 0.xxxx}
     * start at the beginning of a line so that they can be parsed by other tools.
     *
     * @param classNames K names in class-index order
     */
    public String toReport(List<String> classNames) {
        checkNames(classNames);
        int nameWidth = "actual \\ predicted".length();
        for (String n : classNames) {
            nameWidth = Math.max(nameWidth, n.length());
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT, "Overall accuracy : %.4f%n", accuracy()));
        sb.append(String.format(Locale.ROOT, "Macro F1         : %.4f%n", macroF1()));
        sb.append(String.format(Locale.ROOT, "Macro precision  : %.4f%n", macroPrecision()));
        sb.append(String.format(Locale.ROOT, "Macro recall     : %.4f%n", macroRecall()));
        sb.append(String.format(Locale.ROOT, "Images           : %d (%d correct, %d wrong)%n",
                getTotal(), getCorrect(), getTotal() - getCorrect()));
        sb.append('\n').append("Per-class metrics (one-vs-rest from the confusion matrix):\n");
        String rowFormat = "%3s  %-" + nameWidth + "s  %9s  %9s  %9s  %7s%n";
        sb.append(String.format(Locale.ROOT, rowFormat, "idx", "class", "precision", "recall", "F1", "support"));
        for (int c = 0; c < numClasses; c++) {
            sb.append(String.format(Locale.ROOT, "%3d  %-" + nameWidth + "s  %9.4f  %9.4f  %9.4f  %7d%n",
                    c, classNames.get(c), precision(c), recall(c), f1(c), support(c)));
        }
        sb.append(String.format(Locale.ROOT, "%3s  %-" + nameWidth + "s  %9.4f  %9.4f  %9.4f  %7d%n",
                "", "macro average", macroPrecision(), macroRecall(), macroF1(), getTotal()));

        sb.append('\n').append("Confusion matrix (rows = actual class, columns = predicted class, in index order):\n");
        int cell = 5;
        for (long[] row : matrix) {
            for (long v : row) {
                cell = Math.max(cell, Long.toString(v).length() + 1);
            }
        }
        sb.append(String.format(Locale.ROOT, "%3s  %-" + nameWidth + "s", "", "actual \\ predicted"));
        for (int p = 0; p < numClasses; p++) {
            sb.append(String.format(Locale.ROOT, "%" + cell + "d", p));
        }
        sb.append('\n');
        for (int a = 0; a < numClasses; a++) {
            sb.append(String.format(Locale.ROOT, "%3d  %-" + nameWidth + "s", a, classNames.get(a)));
            for (int p = 0; p < numClasses; p++) {
                sb.append(String.format(Locale.ROOT, "%" + cell + "d", matrix[a][p]));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * The confusion matrix as CSV: a header row {@code actual\predicted,<names...>}, then one row
     * per actual class.
     */
    public String toConfusionMatrixCsv(List<String> classNames) {
        checkNames(classNames);
        StringBuilder sb = new StringBuilder("actual\\predicted");
        for (String n : classNames) {
            sb.append(',').append(n);
        }
        sb.append('\n');
        for (int a = 0; a < numClasses; a++) {
            sb.append(classNames.get(a));
            for (int p = 0; p < numClasses; p++) {
                sb.append(',').append(matrix[a][p]);
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "MulticlassMetrics[K=%d, n=%d, accuracy=%.4f, macroF1=%.4f, matrix=%s]",
                numClasses, getTotal(), accuracy(), macroF1(), Arrays.deepToString(matrix));
    }

    private void checkClass(int c) {
        if (c < 0 || c >= numClasses) {
            throw new IllegalArgumentException("Class index " + c + " outside 0.." + (numClasses - 1));
        }
    }

    private void checkNames(List<String> classNames) {
        if (classNames == null || classNames.size() != numClasses) {
            throw new IllegalArgumentException("Need " + numClasses + " class names, got "
                    + (classNames == null ? "null" : classNames.size()));
        }
    }

    private static double ratio(double numerator, double denominator) {
        return denominator == 0 ? 0.0 : numerator / denominator;
    }
}
