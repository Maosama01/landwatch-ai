package com.landwatch.deeplearning.eurosat;

import java.nio.file.Path;

/**
 * One row of a split CSV: an image path relative to the EuroSAT root folder, its class name and
 * its class index (classes are numbered in alphabetical order of their folder names).
 *
 * @param relativePath for example {@code EuroSAT_RGB/Forest/Forest_1.jpg} (always with '/')
 * @param className    for example {@code Forest}
 * @param classIndex   0-based index in alphabetical class order
 */
public record EuroSatSample(String relativePath, String className, int classIndex) {

    public EuroSatSample {
        if (relativePath == null || relativePath.isBlank() || relativePath.contains(",")) {
            throw new IllegalArgumentException("Invalid relative path: " + relativePath);
        }
        if (className == null || className.isBlank() || className.contains(",")) {
            throw new IllegalArgumentException("Invalid class name: " + className);
        }
        if (classIndex < 0) {
            throw new IllegalArgumentException("Class index must be >= 0: " + classIndex);
        }
    }

    /** The image file, resolved against the EuroSAT root folder. */
    public Path resolve(Path euroSatRoot) {
        return euroSatRoot.resolve(relativePath);
    }
}
