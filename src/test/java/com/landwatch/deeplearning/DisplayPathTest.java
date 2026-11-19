package com.landwatch.deeplearning;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/** Paths written to console output and result files must not leak the user's home directory. */
class DisplayPathTest {

    @Test
    @DisplayName("Paths inside the working directory are shown relative, with forward slashes")
    void insideWorkingDirectoryIsRelative() {
        Path cwd = Path.of("").toAbsolutePath();
        assertEquals("models/siamese_unet_levir", TrainingSupport.displayPath(Path.of("models/siamese_unet_levir")));
        assertEquals("results/deeplearning/test_metrics.txt",
                TrainingSupport.displayPath(cwd.resolve("results").resolve("deeplearning").resolve("test_metrics.txt")));
        assertEquals("models", TrainingSupport.displayPath(cwd.resolve("x").resolve("..").resolve("models")));
        assertEquals(".", TrainingSupport.displayPath(cwd));
        assertFalse(TrainingSupport.displayPath(cwd.resolve("models")).contains(cwd.toString()));
    }

    @Test
    @DisplayName("Paths outside the working directory stay absolute")
    void outsideWorkingDirectoryStaysAbsolute() {
        Path outside = Path.of("").toAbsolutePath().getRoot().resolve("landwatch-outside-check").resolve("model");
        assertEquals(outside.toString().replace('\\', '/'), TrainingSupport.displayPath(outside));
    }
}
