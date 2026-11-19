package com.landwatch.ui;

import com.landwatch.geospatial.ChangeVisualizer;
import com.landwatch.ui.engine.EngineResult;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * The three dashboard visualisations of one engine result, all built from the result's binary mask.
 */
public final class ChangeVisualizationSet {

    private final BufferedImage binaryMask;
    private final BufferedImage redOverlay;
    private final BufferedImage thematicMap;

    private ChangeVisualizationSet(BufferedImage binaryMask, BufferedImage redOverlay, BufferedImage thematicMap) {
        this.binaryMask = binaryMask;
        this.redOverlay = redOverlay;
        this.thematicMap = thematicMap;
    }

    /**
     * Builds the binary mask, red overlay and thematic change map from {@code result}'s mask.
     *
     * @param result     engine output (mask, regions)
     * @param afterImage post-event image used as context for the overlay and thematic map
     */
    public static ChangeVisualizationSet build(EngineResult result, BufferedImage afterImage) {
        Objects.requireNonNull(result, "result");
        Objects.requireNonNull(afterImage, "afterImage");
        if (afterImage.getWidth() != result.getWidth() || afterImage.getHeight() != result.getHeight()) {
            throw new IllegalArgumentException("Post-event image size does not match the engine mask.");
        }
        boolean[][] mask = result.getMask();
        return new ChangeVisualizationSet(
                ChangeVisualizer.createBlackWhiteMask(mask, result.getWidth(), result.getHeight()),
                ChangeVisualizer.createOverlayChangeMap(afterImage, mask, result.getRegions()),
                ChangeVisualizer.createThematicChangeMap(afterImage, mask));
    }

    public BufferedImage getBinaryMask() { return binaryMask; }
    public BufferedImage getRedOverlay() { return redOverlay; }
    public BufferedImage getThematicMap() { return thematicMap; }

    public BufferedImage get(VisualizationMode mode) {
        switch (mode) {
            case BINARY_MASK: return binaryMask;
            case RED_OVERLAY: return redOverlay;
            case CHANGE_MAP: return thematicMap;
            default: throw new IllegalArgumentException("Unknown mode: " + mode);
        }
    }
}
