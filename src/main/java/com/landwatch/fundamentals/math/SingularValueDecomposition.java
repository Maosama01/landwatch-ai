package com.landwatch.fundamentals.math;

import java.util.Arrays;
import java.util.Comparator;

/**
 * Singular value decomposition {@code A = U * diag(s) * V^T} of an m x n matrix, written from scratch
 * with the one-sided Jacobi method (Hestenes, 1958).
 *
 * <h2>What the SVD says</h2>
 * <p>Every matrix maps the unit sphere to an ellipsoid. The right singular vectors (columns of
 * {@code V}) are the orthonormal input directions that land on the ellipsoid's axes, the left
 * singular vectors (columns of {@code U}) are those axes' directions, and the singular values
 * {@code s1 >= s2 >= ... >= 0} are their lengths. For a data matrix whose rows are samples,
 * {@code V}'s first column is the direction of largest spread (the first principal component), and
 * {@code s1^2 / n} is the mean squared projection onto it. Keeping the k largest singular values
 * gives the best rank-k approximation of {@code A} (Eckart-Young theorem).</p>
 *
 * <h2>Algorithm</h2>
 * <p>Start with {@code W = A} and {@code V = I}. Repeatedly pick two columns p, q of {@code W} and
 * rotate them in their plane by the angle that makes them orthogonal; apply the same rotation to
 * {@code V}. Each rotation keeps {@code A = W V^T} true. When all column pairs are orthogonal,
 * the column lengths of {@code W} are the singular values and the normalised columns are {@code U}.
 * The rotation for columns with {@code alpha = |w_p|^2}, {@code beta = |w_q|^2},
 * {@code gamma = w_p . w_q} uses {@code zeta = (beta - alpha) / (2 gamma)},
 * {@code t = sign(zeta) / (|zeta| + sqrt(1 + zeta^2))}, {@code c = 1 / sqrt(1 + t^2)}, {@code s = c t}.</p>
 *
 * <p>Results are sorted by decreasing singular value. Suited to the small matrices of the
 * fundamentals package (cost about {@code m * n^2} per sweep); it is not meant for images.</p>
 */
public final class SingularValueDecomposition {

    private static final int MAX_SWEEPS = 100;
    private static final double TOLERANCE = 1e-15;

    private final Matrix u;
    private final double[] singularValues;
    private final Matrix v;
    private final int sweeps;

    /** Decomposes {@code a} (it is not modified). */
    public SingularValueDecomposition(Matrix a) {
        int m = a.getRows();
        int n = a.getCols();
        double[][] w = a.toArray();
        double[][] vv = Matrix.identity(n).toArray();
        int sweep = 0;
        boolean rotated = true;
        while (rotated && sweep < MAX_SWEEPS) {
            rotated = false;
            sweep++;
            for (int p = 0; p < n - 1; p++) {
                for (int q = p + 1; q < n; q++) {
                    double alpha = 0;
                    double beta = 0;
                    double gamma = 0;
                    for (int i = 0; i < m; i++) {
                        alpha += w[i][p] * w[i][p];
                        beta += w[i][q] * w[i][q];
                        gamma += w[i][p] * w[i][q];
                    }
                    if (gamma == 0 || Math.abs(gamma) <= TOLERANCE * Math.sqrt(alpha * beta)) {
                        continue;
                    }
                    rotated = true;
                    double zeta = (beta - alpha) / (2 * gamma);
                    double t = Math.signum(zeta) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
                    if (zeta == 0) {
                        t = 1;
                    }
                    double c = 1 / Math.sqrt(1 + t * t);
                    double s = c * t;
                    rotate(w, m, p, q, c, s);
                    rotate(vv, n, p, q, c, s);
                }
            }
        }
        this.sweeps = sweep;

        int k = Math.min(m, n);
        double[] norms = new double[n];
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int i = 0; i < m; i++) {
                sum += w[i][j] * w[i][j];
            }
            norms[j] = Math.sqrt(sum);
        }
        Integer[] order = new Integer[n];
        for (int j = 0; j < n; j++) {
            order[j] = j;
        }
        Arrays.sort(order, Comparator.comparingDouble((Integer j) -> norms[j]).reversed());

        singularValues = new double[k];
        double[][] uu = new double[m][k];
        double[][] vSorted = new double[n][k];
        double scale = Arrays.stream(norms).max().orElse(0);
        for (int r = 0; r < k; r++) {
            int j = order[r];
            singularValues[r] = norms[j];
            for (int i = 0; i < n; i++) {
                vSorted[i][r] = vv[i][j];
            }
            if (norms[j] > 1e-12 * Math.max(scale, 1e-300)) {
                for (int i = 0; i < m; i++) {
                    uu[i][r] = w[i][j] / norms[j];
                }
            }
        }
        this.u = new Matrix(uu);
        this.v = new Matrix(vSorted);
    }

    /** Rotates columns p and q of x (rows 0..rows-1) by the Jacobi rotation (c, s). */
    private static void rotate(double[][] x, int rows, int p, int q, double c, double s) {
        for (int i = 0; i < rows; i++) {
            double xp = x[i][p];
            double xq = x[i][q];
            x[i][p] = c * xp - s * xq;
            x[i][q] = s * xp + c * xq;
        }
    }

    /** Left singular vectors as columns, m x min(m, n). Columns for zero singular values are zero. */
    public Matrix getU() {
        return u.copy();
    }

    /** Singular values, largest first (a copy). */
    public double[] getSingularValues() {
        return singularValues.clone();
    }

    /** Right singular vectors as columns, n x min(m, n). */
    public Matrix getV() {
        return v.copy();
    }

    /** The i-th right singular vector (i = 0 is the direction of largest spread). */
    public Vector rightSingularVector(int i) {
        return v.getCol(i);
    }

    /** {@code U * diag(s) * V^T}, which equals the original matrix up to rounding. */
    public Matrix reconstruct() {
        return reconstruct(singularValues.length);
    }

    /** Best rank-k approximation: keeps the k largest singular values. */
    public Matrix reconstruct(int k) {
        if (k < 0 || k > singularValues.length) {
            throw new IllegalArgumentException("k must be in 0.." + singularValues.length);
        }
        int m = u.getRows();
        int n = v.getRows();
        double[][] out = new double[m][n];
        for (int r = 0; r < k; r++) {
            for (int i = 0; i < m; i++) {
                double us = u.get(i, r) * singularValues[r];
                for (int j = 0; j < n; j++) {
                    out[i][j] += us * v.get(j, r);
                }
            }
        }
        return new Matrix(out);
    }

    /** Number of Jacobi sweeps used (for the curious; a few for small matrices). */
    public int getSweeps() {
        return sweeps;
    }
}
