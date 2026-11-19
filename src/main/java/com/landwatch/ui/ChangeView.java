package com.landwatch.ui;

import java.util.List;

/**
 * Anything the change-map panel can show: the three engine views ({@link VisualizationMode}) and the
 * two views that need a ground-truth label ({@link GroundTruthView}).
 */
public interface ChangeView {

    /** Short name shown on the toggle and in the legend header. */
    String getDisplayName();

    /** One-sentence caption shown under the legend. */
    String getDescription();

    /** Legend entries; colours always match the rendered pixels. */
    List<VisualizationMode.LegendEntry> getLegend();

    /** Name used in toggle {@code userData} (the enum constant name). */
    String name();
}
