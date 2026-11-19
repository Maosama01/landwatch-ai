package com.landwatch.ui.engine;

import com.landwatch.geospatial.SatelliteImageLoader;

import java.awt.image.BufferedImage;

/**
 * Single, UI-independent entry point that runs a {@link ChangeDetectionEngine} and checks its output.
 */
public final class EngineRunner {

    private EngineRunner() {
    }

    /**
     * Runs {@code engine} on the image pair.
     *
     * @throws IllegalStateException    if the engine is unavailable or returns an invalid result
     * @throws IllegalArgumentException if the images are missing or differ in size
     */
    public static EngineResult run(ChangeDetectionEngine engine, BufferedImage before, BufferedImage after)
            throws Exception {
        if (engine == null) {
            throw new IllegalStateException("No detection engine selected.");
        }
        if (!engine.isAvailable()) {
            throw new IllegalStateException(engine.getDisplayName() + " is unavailable: " + engine.getStatusNote());
        }
        SatelliteImageLoader.validateDimensions(before, after);

        EngineResult result = engine.detect(before, after);
        if (result == null) {
            throw new IllegalStateException(engine.getDisplayName() + " returned no result.");
        }
        if (result.getWidth() != before.getWidth() || result.getHeight() != before.getHeight()) {
            throw new IllegalStateException(String.format(
                    "%s returned a %dx%d mask for %dx%d images.", engine.getDisplayName(),
                    result.getWidth(), result.getHeight(), before.getWidth(), before.getHeight()));
        }
        return result;
    }
}
