package com.landwatch.ui.engine;

import java.awt.image.BufferedImage;

/**
 * A change-detection engine that can be selected in the dashboard.
 *
 * <p>Every engine produces the same thing: a binary change mask {@code boolean[y][x]} wrapped in an
 * {@link EngineResult} together with the engine name and its latency. The dashboard builds all
 * visualisations (binary mask, red overlay, thematic change map) and all statistics from that mask
 * only, so a new engine can be plugged in without touching the UI code.</p>
 */
public interface ChangeDetectionEngine {

    /** Stable machine-friendly id, used in exported file names (e.g. {@code baseline-otsu}). */
    String getId();

    /** Human-readable name shown in the engine selector, metrics panel and log. */
    String getDisplayName();

    /** Short upper-case label for badges (e.g. {@code BASELINE · OTSU}). */
    String getShortLabel();

    /** Whether the engine can run right now (e.g. a deep model needs trained weights). */
    boolean isAvailable();

    /** One-line description, or the reason the engine is unavailable. */
    String getStatusNote();

    /**
     * Runs change detection on a co-registered image pair of identical size.
     *
     * @return result whose mask has the same width and height as the inputs
     * @throws Exception if the engine is unavailable or the computation fails; an engine must
     *                   never return simulated or placeholder output
     */
    EngineResult detect(BufferedImage before, BufferedImage after) throws Exception;
}
