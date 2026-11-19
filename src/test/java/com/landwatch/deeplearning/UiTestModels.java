package com.landwatch.deeplearning;

import java.nio.file.Path;

/** Public bridge to {@link TestModels} for tests outside this package (e.g. the dashboard's deep engine). */
public final class UiTestModels {

    private UiTestModels() {
    }

    /** Writes a tiny, briefly trained CPU model (weights + model.properties) with 32x32 tiles into {@code dir}. */
    public static void saveTinyModelDir(Path dir, float threshold) throws Exception {
        java.nio.file.Files.createDirectories(dir);
        TestModels.saveModelDir(dir, TestModels.TINY_WIDTHS, 32, threshold);
    }
}
