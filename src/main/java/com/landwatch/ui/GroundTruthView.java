package com.landwatch.ui;

import com.landwatch.geospatial.ChangeVisualizer;

import java.util.List;

/**
 * Views that compare the engine mask with a ground-truth label. They are only offered when the inputs
 * came with a label (a LEVIR-CD pair); the synthetic demo pair has none.
 * Legend colours are taken from {@link GroundTruthComparison} so the legend always matches the pixels.
 */
public enum GroundTruthView implements ChangeView {

    GROUND_TRUTH("Ground Truth",
            "LEVIR-CD reference label for this pair. It marks building changes only (construction or demolition).",
            List.of(
                    entry(ChangeVisualizer.BW_UNCHANGED_RGB, "No change (label 0)"),
                    entry(ChangeVisualizer.BW_CHANGED_RGB, "Building change (label > 0)"))),

    ERRORS("Errors vs GT",
            "Engine mask vs. label, pixel by pixel. TN pixels show the post-event image, dimmed.",
            List.of(
                    entry(GroundTruthComparison.TP_RGB, "TP: change found"),
                    entry(GroundTruthComparison.FP_RGB, "FP: false alarm"),
                    entry(GroundTruthComparison.FN_RGB, "FN: missed change"),
                    entry(GroundTruthComparison.TN_LEGEND_RGB, "TN: dimmed context")));

    private final String displayName;
    private final String description;
    private final List<VisualizationMode.LegendEntry> legend;

    GroundTruthView(String displayName, String description, List<VisualizationMode.LegendEntry> legend) {
        this.displayName = displayName;
        this.description = description;
        this.legend = legend;
    }

    private static VisualizationMode.LegendEntry entry(int rgb, String label) {
        return new VisualizationMode.LegendEntry(String.format("#%06x", rgb & 0xFFFFFF), label,
                VisualizationMode.LegendEntry.SwatchStyle.FILL);
    }

    @Override public String getDisplayName() { return displayName; }
    @Override public String getDescription() { return description; }
    @Override public List<VisualizationMode.LegendEntry> getLegend() { return legend; }

    @Override
    public String toString() { return displayName; }
}
