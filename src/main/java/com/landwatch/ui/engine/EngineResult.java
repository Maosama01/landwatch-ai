package com.landwatch.ui.engine;

import com.landwatch.geospatial.BaselineChangeDetector;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Engine-independent result of one change-detection run.
 *
 * <p>Holds the binary change mask ({@code mask[y][x]}, {@code true} = changed), the name of the
 * engine that produced it and the engine latency. All statistics shown in the dashboard are derived
 * from the mask, so they mean the same thing for every engine. Engine-specific extras (such as the
 * Otsu threshold of the classical baseline) are optional and {@code null} when not applicable.</p>
 *
 * <p>Areas are counted in pixels only: no ground-sample-distance calibration is applied.</p>
 */
public final class EngineResult {

    /**
     * Heuristic change level based only on the share of changed pixels. It is a rule of thumb for
     * the demo, not a verified encroachment assessment. Thresholds match
     * {@code ChangeDetectionResult#getRiskCategory()}.
     */
    public enum ChangeLevel {
        NONE("NO CHANGE", "level-none"),
        LOW("LOW (< 1 %)", "level-low"),
        MODERATE("MODERATE (1–5 %)", "level-moderate"),
        HIGH("HIGH (5–15 %)", "level-high"),
        SEVERE("SEVERE (≥ 15 %)", "level-severe");

        private final String label;
        private final String styleClass;

        ChangeLevel(String label, String styleClass) {
            this.label = label;
            this.styleClass = styleClass;
        }

        public String getLabel() { return label; }
        public String getStyleClass() { return styleClass; }

        public static ChangeLevel classify(long changedPixels, double changePercentage) {
            if (changedPixels == 0) return NONE;
            if (changePercentage < 1.0) return LOW;
            if (changePercentage < 5.0) return MODERATE;
            if (changePercentage < 15.0) return HIGH;
            return SEVERE;
        }
    }

    private final String engineId;
    private final String engineName;
    private final boolean[][] mask;
    private final int width;
    private final int height;
    private final long latencyMs;
    private final long changedPixels;
    private final List<Rectangle> regions;
    private final Integer otsuThreshold;
    private final Long rawChangedPixels;

    /**
     * @param engineId          stable engine id (used in file names)
     * @param engineName        human-readable engine name
     * @param mask              binary change mask {@code [y][x]}; copied defensively
     * @param latencyMs         time the engine needed to produce the mask
     * @param otsuThreshold     Otsu threshold (classical baseline only), otherwise {@code null}
     * @param rawChangedPixels  changed pixels before post-processing, or {@code null} if not applicable
     * @param regions           bounding boxes of connected change regions, or {@code null} to derive
     *                          them from the mask (8-connectivity)
     */
    public EngineResult(String engineId, String engineName, boolean[][] mask, long latencyMs,
                        Integer otsuThreshold, Long rawChangedPixels, List<Rectangle> regions) {
        if (engineId == null || engineId.isBlank()) {
            throw new IllegalArgumentException("Engine id must not be blank.");
        }
        if (engineName == null || engineName.isBlank()) {
            throw new IllegalArgumentException("Engine name must not be blank.");
        }
        if (mask == null || mask.length == 0 || mask[0] == null || mask[0].length == 0) {
            throw new IllegalArgumentException("Mask must be a non-empty boolean[height][width] array.");
        }
        if (latencyMs < 0) {
            throw new IllegalArgumentException("Latency must not be negative.");
        }
        this.engineId = engineId;
        this.engineName = engineName;
        this.height = mask.length;
        this.width = mask[0].length;

        this.mask = new boolean[height][];
        long count = 0;
        for (int y = 0; y < height; y++) {
            if (mask[y] == null || mask[y].length != width) {
                throw new IllegalArgumentException("Mask is not rectangular (row " + y + ").");
            }
            this.mask[y] = mask[y].clone();
            for (int x = 0; x < width; x++) {
                if (mask[y][x]) count++;
            }
        }
        this.changedPixels = count;
        this.latencyMs = latencyMs;
        this.otsuThreshold = otsuThreshold;
        this.rawChangedPixels = rawChangedPixels;
        this.regions = (regions != null) ? new ArrayList<>(regions) : computeRegions(this.mask, width, height);
    }

    /** Factory for engines that only produce a mask (e.g. a deep model). Regions are derived from the mask. */
    public static EngineResult fromMask(String engineId, String engineName, boolean[][] mask, long latencyMs) {
        return new EngineResult(engineId, engineName, mask, latencyMs, null, null, null);
    }

    private static List<Rectangle> computeRegions(boolean[][] mask, int width, int height) {
        List<Rectangle> boxes = new ArrayList<>();
        BaselineChangeDetector.filterConnectedComponents(mask, width, height, 1, boxes);
        return boxes;
    }

    public String getEngineId() { return engineId; }
    public String getEngineName() { return engineName; }
    public int getWidth() { return width; }
    public int getHeight() { return height; }
    public long getTotalPixels() { return (long) width * height; }
    public long getLatencyMs() { return latencyMs; }
    public long getChangedPixels() { return changedPixels; }
    public long getUnchangedPixels() { return getTotalPixels() - changedPixels; }
    public double getChangePercentage() { return changedPixels * 100.0 / getTotalPixels(); }
    public int getRegionCount() { return regions.size(); }
    public List<Rectangle> getRegions() { return Collections.unmodifiableList(regions); }

    /** Otsu threshold (0..255) for the classical baseline; {@code null} for engines that do not use one. */
    public Integer getOtsuThreshold() { return otsuThreshold; }

    /** Changed pixels before morphological filtering; {@code null} if the engine does not report it. */
    public Long getRawChangedPixels() { return rawChangedPixels; }

    public ChangeLevel getChangeLevel() { return ChangeLevel.classify(changedPixels, getChangePercentage()); }

    /** Returns a copy of the binary mask, indexed {@code [y][x]}. */
    public boolean[][] getMask() {
        boolean[][] copy = new boolean[height][];
        for (int y = 0; y < height; y++) {
            copy[y] = mask[y].clone();
        }
        return copy;
    }

    public boolean isChanged(int x, int y) {
        return mask[y][x];
    }
}
