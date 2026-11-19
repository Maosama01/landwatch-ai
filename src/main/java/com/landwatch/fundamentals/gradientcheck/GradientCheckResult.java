package com.landwatch.fundamentals.gradientcheck;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Encapsulates the results of a finite-difference gradient check.
 */
public class GradientCheckResult {

    private final boolean passed;
    private final double maxRelativeError;
    private final double maxAbsoluteError;
    private final double tolerance;
    private final double epsilon;
    private final int totalParametersChecked;
    private final int failedParametersCount;
    private final List<ComparisonEntry> comparisons;

    public GradientCheckResult(boolean passed,
                               double maxRelativeError,
                               double maxAbsoluteError,
                               double tolerance,
                               double epsilon,
                               int totalParametersChecked,
                               int failedParametersCount,
                               List<ComparisonEntry> comparisons) {
        this.passed = passed;
        this.maxRelativeError = maxRelativeError;
        this.maxAbsoluteError = maxAbsoluteError;
        this.tolerance = tolerance;
        this.epsilon = epsilon;
        this.totalParametersChecked = totalParametersChecked;
        this.failedParametersCount = failedParametersCount;
        this.comparisons = (comparisons != null) ? new ArrayList<>(comparisons) : Collections.emptyList();
    }

    public boolean isPassed() { return passed; }
    public double getMaxRelativeError() { return maxRelativeError; }
    public double getMaxAbsoluteError() { return maxAbsoluteError; }
    public double getTolerance() { return tolerance; }
    public double getEpsilon() { return epsilon; }
    public int getTotalParametersChecked() { return totalParametersChecked; }
    public int getFailedParametersCount() { return failedParametersCount; }
    public List<ComparisonEntry> getComparisons() { return Collections.unmodifiableList(comparisons); }

    public String getSummary() {
        return String.format(
                "Gradient Check Summary:\n" +
                "  • Status: %s\n" +
                "  • Total Parameters Checked: %d\n" +
                "  • Failed Parameters: %d\n" +
                "  • Max Relative Error: %.2e (Tolerance: %.2e)\n" +
                "  • Max Absolute Error: %.2e\n" +
                "  • Epsilon: %.2e",
                passed ? "PASSED (Analytical gradients verified)" : "FAILED (Gradient mismatch)",
                totalParametersChecked,
                failedParametersCount,
                maxRelativeError, tolerance,
                maxAbsoluteError,
                epsilon);
    }

    @Override
    public String toString() {
        return getSummary();
    }

    /**
     * Entry comparing a single parameter's analytical and numerical gradient.
     */
    public static class ComparisonEntry {
        private final String parameterName;
        private final double analyticalGrad;
        private final double numericalGrad;
        private final double absoluteError;
        private final double relativeError;
        private final boolean passed;

        public ComparisonEntry(String parameterName, double analyticalGrad, double numericalGrad,
                               double absoluteError, double relativeError, boolean passed) {
            this.parameterName = parameterName;
            this.analyticalGrad = analyticalGrad;
            this.numericalGrad = numericalGrad;
            this.absoluteError = absoluteError;
            this.relativeError = relativeError;
            this.passed = passed;
        }

        public String getParameterName() { return parameterName; }
        public double getAnalyticalGrad() { return analyticalGrad; }
        public double getNumericalGrad() { return numericalGrad; }
        public double getAbsoluteError() { return absoluteError; }
        public double getRelativeError() { return relativeError; }
        public boolean isPassed() { return passed; }

        @Override
        public String toString() {
            return String.format("%s: Ana=%.6e, Num=%.6e, RelErr=%.2e (%s)",
                    parameterName, analyticalGrad, numericalGrad, relativeError, passed ? "OK" : "FAIL");
        }
    }
}
