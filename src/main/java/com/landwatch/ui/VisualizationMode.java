package com.landwatch.ui;

import com.landwatch.geospatial.ChangeVisualizer;

import java.util.List;

/**
 * The three change-visualisation modes offered by the dashboard, with the legend shown under the image.
 * Legend colours are taken from {@link ChangeVisualizer} so the legend always matches the pixels.
 */
public enum VisualizationMode implements ChangeView {

    BINARY_MASK("Binary Mask",
            "Raw engine output: every pixel is either unchanged (black) or changed (white).",
            List.of(
                    LegendEntry.fill(ChangeVisualizer.BW_UNCHANGED_RGB, "Unchanged (0)"),
                    LegendEntry.fill(ChangeVisualizer.BW_CHANGED_RGB, "Changed (1)"))),

    RED_OVERLAY("Red Overlay",
            "Post-event image with changed pixels tinted red; amber boxes bound each connected change region.",
            List.of(
                    LegendEntry.fill(ChangeVisualizer.OVERLAY_TINT_RGB, "Changed pixels (red tint)"),
                    LegendEntry.outline(ChangeVisualizer.OVERLAY_BOX_RGB, "Change region bounding box"),
                    LegendEntry.texture("Unchanged: post-event image"))),

    CHANGE_MAP("Change Map",
            "Thematic map: post-event image as dimmed grayscale context, change regions filled cyan with a yellow outline.",
            List.of(
                    LegendEntry.fill(0x3A3A3A, "Unchanged context (dimmed grayscale)"),
                    LegendEntry.fill(ChangeVisualizer.THEMATIC_CHANGE_RGB, "Changed region"),
                    LegendEntry.outline(ChangeVisualizer.THEMATIC_OUTLINE_RGB, "Region outline")));

    /** One legend item: a colour swatch (filled, outlined or image-texture) and its meaning. */
    public record LegendEntry(String cssColor, String label, SwatchStyle style) {

        public enum SwatchStyle { FILL, OUTLINE, TEXTURE }

        static LegendEntry fill(int rgb, String label) {
            return new LegendEntry(toCss(rgb), label, SwatchStyle.FILL);
        }

        static LegendEntry outline(int rgb, String label) {
            return new LegendEntry(toCss(rgb), label, SwatchStyle.OUTLINE);
        }

        static LegendEntry texture(String label) {
            return new LegendEntry(null, label, SwatchStyle.TEXTURE);
        }

        private static String toCss(int rgb) {
            return String.format("#%06x", rgb & 0xFFFFFF);
        }
    }

    private final String displayName;
    private final String description;
    private final List<LegendEntry> legend;

    VisualizationMode(String displayName, String description, List<LegendEntry> legend) {
        this.displayName = displayName;
        this.description = description;
        this.legend = legend;
    }

    @Override public String getDisplayName() { return displayName; }
    @Override public String getDescription() { return description; }
    @Override public List<LegendEntry> getLegend() { return legend; }

    @Override
    public String toString() { return displayName; }
}
