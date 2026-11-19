package com.landwatch.fundamentals.math;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Fundamental 1D Vector class for neural network mathematics.
 * Implemented from scratch in pure Java without third-party dependencies.
 */
public class Vector {

    private final double[] data;
    private final int size;

    /**
     * Constructs a zero-initialized vector of given size.
     *
     * @param size number of elements (must be positive)
     */
    public Vector(int size) {
        if (size <= 0) {
            throw new IllegalArgumentException("Vector size must be positive, got: " + size);
        }
        this.size = size;
        this.data = new double[size];
    }

    /**
     * Constructs a vector initialized with the given double values.
     *
     * @param values initial elements
     */
    public Vector(double... values) {
        if (values == null || values.length == 0) {
            throw new IllegalArgumentException("Vector elements must not be null or empty.");
        }
        this.size = values.length;
        this.data = Arrays.copyOf(values, values.length);
    }

    /**
     * Factory method creating a vector from varargs.
     */
    public static Vector of(double... values) {
        return new Vector(values);
    }

    /**
     * Factory method creating a vector of zeros.
     */
    public static Vector zeros(int size) {
        return new Vector(size);
    }

    /**
     * Factory method creating a vector of ones.
     */
    public static Vector ones(int size) {
        Vector v = new Vector(size);
        Arrays.fill(v.data, 1.0);
        return v;
    }

    /**
     * Factory method creating a randomly initialized vector using uniform distribution [min, max].
     */
    public static Vector random(int size, double min, double max, Random rng) {
        if (min >= max) {
            throw new IllegalArgumentException("min must be strictly less than max.");
        }
        Vector v = new Vector(size);
        for (int i = 0; i < size; i++) {
            v.data[i] = min + (max - min) * rng.nextDouble();
        }
        return v;
    }

    public static Vector random(int size, double min, double max) {
        return random(size, min, max, new Random());
    }

    // --- Accessors ---

    public int size() {
        return size;
    }

    public double get(int index) {
        checkIndex(index);
        return data[index];
    }

    public void set(int index, double value) {
        checkIndex(index);
        data[index] = value;
    }

    public double[] toArray() {
        return Arrays.copyOf(data, size);
    }

    public Vector copy() {
        return new Vector(this.data);
    }

    // --- Mathematical Operations ---

    /**
     * Vector addition: this + other.
     */
    public Vector add(Vector other) {
        checkDimension(other);
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] + other.data[i];
        }
        return result;
    }

    /**
     * Vector subtraction: this - other.
     */
    public Vector subtract(Vector other) {
        checkDimension(other);
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    /**
     * Scalar multiplication: alpha * this.
     */
    public Vector multiply(double scalar) {
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }

    /**
     * Scalar division: this / scalar.
     */
    public Vector divide(double scalar) {
        if (Math.abs(scalar) < 1e-15) {
            throw new ArithmeticException("Division by zero or near-zero scalar: " + scalar);
        }
        return multiply(1.0 / scalar);
    }

    /**
     * Dot product: sum(u_i * v_i).
     */
    public double dot(Vector other) {
        checkDimension(other);
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += this.data[i] * other.data[i];
        }
        return sum;
    }

    /**
     * Element-wise (Hadamard) product: u_i * v_i.
     */
    public Vector elementwiseMultiply(Vector other) {
        checkDimension(other);
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    /**
     * Applies a unary function to each element (e.g. activation, derivative).
     */
    public Vector map(DoubleUnaryOperator function) {
        Objects.requireNonNull(function, "Mapping function must not be null.");
        Vector result = new Vector(size);
        for (int i = 0; i < size; i++) {
            result.data[i] = function.applyAsDouble(this.data[i]);
        }
        return result;
    }

    /**
     * Computes sum of all elements.
     */
    public double sum() {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += data[i];
        }
        return sum;
    }

    /**
     * Computes the Euclidean (L2) norm: sqrt(sum(v_i^2)).
     */
    public double norm() {
        return Math.sqrt(dot(this));
    }

    /**
     * Computes the Manhattan (L1) norm: sum(|v_i|).
     */
    public double normL1() {
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += Math.abs(data[i]);
        }
        return sum;
    }

    /**
     * Outer product with another vector: u * v^T.
     * Produces a Matrix of dimensions (size x other.size).
     */
    public Matrix outerProduct(Vector other) {
        Objects.requireNonNull(other, "Other vector must not be null.");
        Matrix result = new Matrix(this.size, other.size);
        for (int i = 0; i < this.size; i++) {
            for (int j = 0; j < other.size; j++) {
                result.set(i, j, this.data[i] * other.data[j]);
            }
        }
        return result;
    }

    /**
     * Converts this vector to a 1 x N row Matrix.
     */
    public Matrix toRowMatrix() {
        Matrix m = new Matrix(1, size);
        for (int j = 0; j < size; j++) {
            m.set(0, j, data[j]);
        }
        return m;
    }

    /**
     * Converts this vector to an N x 1 column Matrix.
     */
    public Matrix toColumnMatrix() {
        Matrix m = new Matrix(size, 1);
        for (int i = 0; i < size; i++) {
            m.set(i, 0, data[i]);
        }
        return m;
    }

    // --- Validation and Object Overrides ---

    private void checkIndex(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(String.format("Index %d out of bounds for Vector size %d", index, size));
        }
    }

    private void checkDimension(Vector other) {
        if (other == null) {
            throw new IllegalArgumentException("Operand vector must not be null.");
        }
        if (this.size != other.size) {
            throw new IllegalArgumentException(String.format(
                    "Vector dimension mismatch: expected %d, got %d", this.size, other.size));
        }
    }

    public boolean equals(Vector other, double tolerance) {
        if (other == null || this.size != other.size) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            if (Math.abs(this.data[i] - other.data[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Vector other)) return false;
        if (size != other.size) return false;
        return Arrays.equals(data, other.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(size);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < size; i++) {
            sb.append(String.format("%.4f", data[i]));
            if (i < size - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}
