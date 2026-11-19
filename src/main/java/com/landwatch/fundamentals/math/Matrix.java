package com.landwatch.fundamentals.math;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Fundamental 2D Matrix class for neural network mathematics.
 * Implemented from scratch in pure Java without third-party dependencies.
 */
public class Matrix {

    private final int rows;
    private final int cols;
    private final double[][] data;

    /**
     * Constructs a zero-initialized matrix of size (rows x cols).
     */
    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException(String.format(
                    "Matrix dimensions must be strictly positive, got: %d x %d", rows, cols));
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    /**
     * Constructs a matrix from a 2D double array with defensive deep copy.
     */
    public Matrix(double[][] values) {
        if (values == null || values.length == 0 || values[0].length == 0) {
            throw new IllegalArgumentException("Matrix values must not be null or empty.");
        }
        this.rows = values.length;
        this.cols = values[0].length;
        this.data = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            if (values[i] == null || values[i].length != cols) {
                throw new IllegalArgumentException(String.format(
                        "Inconsistent row length at row %d: expected %d", i, cols));
            }
            System.arraycopy(values[i], 0, this.data[i], 0, cols);
        }
    }

    // --- Static Factories ---

    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    public static Matrix ones(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            Arrays.fill(m.data[i], 1.0);
        }
        return m;
    }

    public static Matrix identity(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("Identity dimension must be positive, got: " + n);
        }
        Matrix m = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            m.data[i][i] = 1.0;
        }
        return m;
    }

    public static Matrix random(int rows, int cols, double min, double max, Random rng) {
        if (min >= max) {
            throw new IllegalArgumentException("min must be strictly less than max.");
        }
        Matrix m = new Matrix(rows, cols);
        double range = max - min;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = min + range * rng.nextDouble();
            }
        }
        return m;
    }

    public static Matrix random(int rows, int cols, double min, double max) {
        return random(rows, cols, min, max, new Random());
    }

    /**
     * Constructs a matrix by stacking row vectors.
     */
    public static Matrix fromRows(Vector... rowVectors) {
        if (rowVectors == null || rowVectors.length == 0) {
            throw new IllegalArgumentException("Row vectors must not be null or empty.");
        }
        int numRows = rowVectors.length;
        int numCols = rowVectors[0].size();
        Matrix m = new Matrix(numRows, numCols);

        for (int i = 0; i < numRows; i++) {
            if (rowVectors[i].size() != numCols) {
                throw new IllegalArgumentException(String.format(
                        "Row vector %d size mismatch: expected %d, got %d", i, numCols, rowVectors[i].size()));
            }
            for (int j = 0; j < numCols; j++) {
                m.data[i][j] = rowVectors[i].get(j);
            }
        }
        return m;
    }

    /**
     * Constructs a matrix by stacking column vectors.
     */
    public static Matrix fromCols(Vector... colVectors) {
        if (colVectors == null || colVectors.length == 0) {
            throw new IllegalArgumentException("Column vectors must not be null or empty.");
        }
        int numCols = colVectors.length;
        int numRows = colVectors[0].size();
        Matrix m = new Matrix(numRows, numCols);

        for (int j = 0; j < numCols; j++) {
            if (colVectors[j].size() != numRows) {
                throw new IllegalArgumentException(String.format(
                        "Column vector %d size mismatch: expected %d, got %d", j, numRows, colVectors[j].size()));
            }
            for (int i = 0; i < numRows; i++) {
                m.data[i][j] = colVectors[j].get(i);
            }
        }
        return m;
    }

    // --- Accessors ---

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double get(int r, int c) {
        checkRowIndex(r);
        checkColIndex(c);
        return data[r][c];
    }

    public void set(int r, int c, double val) {
        checkRowIndex(r);
        checkColIndex(c);
        data[r][c] = val;
    }

    public Vector getRow(int r) {
        checkRowIndex(r);
        return new Vector(data[r]);
    }

    public Vector getCol(int c) {
        checkColIndex(c);
        double[] colData = new double[rows];
        for (int i = 0; i < rows; i++) {
            colData[i] = data[i][c];
        }
        return new Vector(colData);
    }

    public void setRow(int r, Vector v) {
        checkRowIndex(r);
        if (v.size() != cols) {
            throw new IllegalArgumentException(String.format(
                    "Cannot set row: vector size %d does not match matrix cols %d", v.size(), cols));
        }
        for (int j = 0; j < cols; j++) {
            data[r][j] = v.get(j);
        }
    }

    public void setCol(int c, Vector v) {
        checkColIndex(c);
        if (v.size() != rows) {
            throw new IllegalArgumentException(String.format(
                    "Cannot set column: vector size %d does not match matrix rows %d", v.size(), rows));
        }
        for (int i = 0; i < rows; i++) {
            data[i][c] = v.get(i);
        }
    }

    public double[][] toArray() {
        double[][] copy = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, cols);
        }
        return copy;
    }

    public Matrix copy() {
        return new Matrix(this.data);
    }

    // --- Mathematical Operations ---

    /**
     * Matrix addition: A + B.
     */
    public Matrix add(Matrix other) {
        checkSameDimensions(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    /**
     * Matrix subtraction: A - B.
     */
    public Matrix subtract(Matrix other) {
        checkSameDimensions(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    /**
     * Scalar multiplication: alpha * A.
     */
    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }

    /**
     * Standard matrix multiplication: A (M x K) * B (K x N) -> C (M x N).
     */
    public Matrix multiply(Matrix other) {
        if (other == null) {
            throw new IllegalArgumentException("Multiplicand matrix must not be null.");
        }
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(String.format(
                    "Matrix multiplication dimension mismatch: this is %dx%d, other is %dx%d (inner dimensions %d != %d)",
                    this.rows, this.cols, other.rows, other.cols, this.cols, other.rows));
        }

        int m = this.rows;
        int k = this.cols;
        int n = other.cols;

        Matrix result = new Matrix(m, n);

        // Cache-friendly inner loop ordering (ikj)
        for (int i = 0; i < m; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int p = 0; p < k; p++) {
                double a = thisRow[p];
                double[] otherRow = other.data[p];
                for (int j = 0; j < n; j++) {
                    resultRow[j] += a * otherRow[j];
                }
            }
        }

        return result;
    }

    /**
     * Matrix-vector product: A (M x N) * v (N) -> u (M).
     */
    public Vector multiply(Vector v) {
        if (v == null) {
            throw new IllegalArgumentException("Vector must not be null.");
        }
        if (this.cols != v.size()) {
            throw new IllegalArgumentException(String.format(
                    "Matrix-vector multiplication mismatch: Matrix is %dx%d, Vector size is %d",
                    this.rows, this.cols, v.size()));
        }

        Vector result = new Vector(this.rows);
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            double[] row = data[i];
            for (int j = 0; j < cols; j++) {
                sum += row[j] * v.get(j);
            }
            result.set(i, sum);
        }
        return result;
    }

    /**
     * Element-wise (Hadamard) product: A_ij * B_ij.
     */
    public Matrix elementwiseMultiply(Matrix other) {
        checkSameDimensions(other);
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    /**
     * Matrix transpose: A^T of dimensions (cols x rows).
     */
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    /**
     * Applies a unary function to each element.
     */
    public Matrix map(DoubleUnaryOperator function) {
        Objects.requireNonNull(function, "Function must not be null.");
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = function.applyAsDouble(this.data[i][j]);
            }
        }
        return result;
    }

    /**
     * Broadcasts a 1D vector across the matrix.
     *
     * @param v      vector to add
     * @param toRows if true, vector length must equal cols (adds v to each row, e.g. mini-batch bias);
     *               if false, vector length must equal rows (adds v to each column).
     */
    public Matrix addVectorBroadcast(Vector v, boolean toRows) {
        if (v == null) {
            throw new IllegalArgumentException("Vector must not be null.");
        }
        Matrix result = new Matrix(rows, cols);

        if (toRows) {
            if (v.size() != cols) {
                throw new IllegalArgumentException(String.format(
                        "Broadcast dimension mismatch: Vector size %d != Matrix cols %d", v.size(), cols));
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = this.data[i][j] + v.get(j);
                }
            }
        } else {
            if (v.size() != rows) {
                throw new IllegalArgumentException(String.format(
                        "Broadcast dimension mismatch: Vector size %d != Matrix rows %d", v.size(), rows));
            }
            for (int i = 0; i < rows; i++) {
                double val = v.get(i);
                for (int j = 0; j < cols; j++) {
                    result.data[i][j] = this.data[i][j] + val;
                }
            }
        }
        return result;
    }

    /**
     * Sums across rows, producing a column vector of row sums (size = rows).
     */
    public Vector sumRows() {
        Vector result = new Vector(rows);
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
            result.set(i, sum);
        }
        return result;
    }

    /**
     * Sums across columns, producing a row vector of column sums (size = cols).
     */
    public Vector sumCols() {
        Vector result = new Vector(cols);
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            result.set(j, sum);
        }
        return result;
    }

    /**
     * Computes sum of all elements in the matrix.
     */
    public double sumAll() {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
        }
        return sum;
    }

    /**
     * Computes the Frobenius norm of the matrix: sqrt(sum(A_ij^2)).
     */
    public double frobeniusNorm() {
        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j] * data[i][j];
            }
        }
        return Math.sqrt(sum);
    }

    // --- Validation and Object Overrides ---

    private void checkRowIndex(int r) {
        if (r < 0 || r >= rows) {
            throw new IndexOutOfBoundsException(String.format("Row index %d out of bounds for %d rows", r, rows));
        }
    }

    private void checkColIndex(int c) {
        if (c < 0 || c >= cols) {
            throw new IndexOutOfBoundsException(String.format("Col index %d out of bounds for %d cols", c, cols));
        }
    }

    private void checkSameDimensions(Matrix other) {
        if (other == null) {
            throw new IllegalArgumentException("Operand matrix must not be null.");
        }
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException(String.format(
                    "Matrix dimension mismatch: this is %dx%d, other is %dx%d",
                    this.rows, this.cols, other.rows, other.cols));
        }
    }

    public boolean equals(Matrix other, double tolerance) {
        if (other == null || this.rows != other.rows || this.cols != other.cols) {
            return false;
        }
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(this.data[i][j] - other.data[i][j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Matrix other)) return false;
        if (rows != other.rows || cols != other.cols) return false;
        return Arrays.deepEquals(data, other.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(rows, cols);
        result = 31 * result + Arrays.deepHashCode(data);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Matrix %dx%d:\n", rows, cols));
        for (int i = 0; i < rows; i++) {
            sb.append("  [");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%8.4f", data[i][j]));
                if (j < cols - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}
