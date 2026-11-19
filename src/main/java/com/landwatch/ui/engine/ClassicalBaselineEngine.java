package com.landwatch.ui.engine;

import com.landwatch.geospatial.BaselineChangeDetector;
import com.landwatch.geospatial.ChangeDetectionResult;

import java.awt.image.BufferedImage;

/**
 * Classical (non-learned) baseline: absolute luminance differencing, Otsu threshold,
 * 3x3 morphological opening and minimum-region filtering, via {@link BaselineChangeDetector}.
 */
public final class ClassicalBaselineEngine implements ChangeDetectionEngine {

    public static final String ID = "baseline-otsu";
    public static final String DISPLAY_NAME = "Classical baseline (Otsu differencing)";

    private final int minRegionSize;

    public ClassicalBaselineEngine() {
        this(BaselineChangeDetector.DEFAULT_MIN_REGION_SIZE);
    }

    public ClassicalBaselineEngine(int minRegionSize) {
        if (minRegionSize < 1) {
            throw new IllegalArgumentException("Minimum region size must be at least 1 pixel.");
        }
        this.minRegionSize = minRegionSize;
    }

    @Override public String getId() { return ID; }
    @Override public String getDisplayName() { return DISPLAY_NAME; }
    @Override public String getShortLabel() { return "Otsu baseline"; }
    @Override public boolean isAvailable() { return true; }

    @Override
    public String getStatusNote() {
        return "CPU: |ΔY| luminance differencing → Otsu threshold → 3×3 opening → regions ≥ "
                + minRegionSize + " px";
    }

    public int getMinRegionSize() { return minRegionSize; }

    @Override
    public EngineResult detect(BufferedImage before, BufferedImage after) {
        ChangeDetectionResult r = BaselineChangeDetector.analyze(before, after, minRegionSize);
        return new EngineResult(ID, DISPLAY_NAME,
                r.getBinaryMask(),
                r.getExecutionTimeMs(),
                r.getOtsuThreshold(),
                r.getRawChangedPixels(),
                r.getBoundingBoxes());
    }

    @Override
    public String toString() { return DISPLAY_NAME; }
}
