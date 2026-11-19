package com.landwatch.ui.engine;

import java.awt.image.BufferedImage;

/**
 * Placeholder slot for the deep-learning engine (Siamese U-Net).
 *
 * <p>No trained model is wired into the dashboard yet, so this engine reports itself as
 * unavailable and {@link #detect} refuses to run. It never produces simulated output.</p>
 *
 * <pre>
 * ====================================================================================
 * HOOK (deep model): wiring com.landwatch.deeplearning.DeepChangeDetector
 * ------------------------------------------------------------------------------------
 * That class is developed separately and does not exist in this module yet, so it is
 * deliberately NOT referenced here. Once it is merged:
 *
 *   1. Add a field  {@code private DeepChangeDetector detector;}  and a method
 *        public void loadModel(Path weights) throws Exception {
 *            detector = DeepChangeDetector.load(weights);
 *        }
 *   2. isAvailable()   -> return detector != null;
 *      getStatusNote() -> describe the loaded model (e.g. weights file name).
 *   3. detect(before, after):
 *        long t0 = System.nanoTime();
 *        boolean[][] mask = detector.predictMask(before, after);   // [y][x], true = changed
 *        long ms = (System.nanoTime() - t0) / 1_000_000;
 *        return EngineResult.fromMask(ID, DISPLAY_NAME, mask, ms);
 *   4. In DashboardController#createEngines(), load the weights (e.g. from a path chosen in
 *      the UI) before adding this engine. No other UI change is needed: the visualisations,
 *      metrics, log and export are all built from the returned mask.
 * ====================================================================================
 * </pre>
 */
public final class SiameseUNetEngine implements ChangeDetectionEngine {

    public static final String ID = "siamese-unet";
    public static final String DISPLAY_NAME = "Deep model – Siamese U-Net";
    public static final String UNAVAILABLE_NOTE = "no trained model loaded";

    @Override public String getId() { return ID; }
    @Override public String getDisplayName() { return DISPLAY_NAME; }
    @Override public String getShortLabel() { return "SIAMESE U-NET"; }

    /** Always false until a trained model is wired in (see the HOOK above). */
    @Override public boolean isAvailable() { return false; }

    @Override public String getStatusNote() { return UNAVAILABLE_NOTE; }

    @Override
    public EngineResult detect(BufferedImage before, BufferedImage after) {
        throw new IllegalStateException(DISPLAY_NAME + " is unavailable: " + UNAVAILABLE_NOTE
                + ". No deep-learning output is produced or simulated.");
    }

    @Override
    public String toString() { return DISPLAY_NAME; }
}
