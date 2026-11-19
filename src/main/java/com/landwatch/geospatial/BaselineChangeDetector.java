package com.landwatch.geospatial;

import java.awt.Rectangle;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;

/**
 * Baseline Image-Processing Change Detection Engine.
 * <p>
 * Implements:
 * 1. Pixel-wise absolute luminance differencing: |I_post - I_pre|
 * 2. Automatic adaptive Otsu thresholding for foreground/background segmentation
 * 3. 3x3 Morphological opening (erosion + dilation) for sensor noise suppression
 * 4. 8-connectivity connected component analysis and minimum cluster area filtering
 * 5. Extraction of geospatial encroachment bounding boxes
 */
public class BaselineChangeDetector {

    public static final int DEFAULT_MIN_REGION_SIZE = 12; // Minimum pixels to qualify as an encroachment cluster

    /**
     * Executes the complete baseline change-detection pipeline on a bi-temporal satellite pair.
     *
     * @param preImage       Baseline / Pre-event satellite image
     * @param postImage      Current / Post-event satellite image
     * @param minRegionSize  Minimum pixel count for valid changed clusters
     * @return Full detection results including metrics, masks, and visual maps
     */
    public static ChangeDetectionResult analyze(BufferedImage preImage, BufferedImage postImage, int minRegionSize) {
        long startTime = System.currentTimeMillis();

        // 1. Validation
        SatelliteImageLoader.validateDimensions(preImage, postImage);
        int width = preImage.getWidth();
        int height = preImage.getHeight();

        // 2. Grayscale Luminance Conversion (Y = 0.299R + 0.587G + 0.114B)
        double[][] preGray = ImageNormalizer.toGrayscaleMatrix(preImage);
        double[][] postGray = ImageNormalizer.toGrayscaleMatrix(postImage);

        // 3. Absolute Pixel-Wise Difference Matrix
        double[][] diffMatrix = new double[height][width];
        int[] histogram = new int[256];
        double maxDiff = 0.0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double diff = Math.abs(postGray[y][x] - preGray[y][x]);
                diffMatrix[y][x] = diff;
                if (diff > maxDiff) {
                    maxDiff = diff;
                }
                int bin = Math.min(255, (int) Math.round(diff));
                histogram[bin]++;
            }
        }

        // 4. Handle Edge Case: Identical Images (Zero Difference)
        if (maxDiff < 1e-6) {
            long duration = System.currentTimeMillis() - startTime;
            boolean[][] emptyMask = new boolean[height][width];
            ChangeDetectionResult result = new ChangeDetectionResult(
                    width, height, 0, 0, 0, 0, duration,
                    emptyMask, diffMatrix, new ArrayList<>()
            );
            result.setBinaryChangeMap(ChangeVisualizer.createBinaryChangeMap(emptyMask, width, height));
            result.setOverlayChangeMap(ChangeVisualizer.createOverlayChangeMap(postImage, emptyMask, new ArrayList<>()));
            return result;
        }

        // 5. Calculate Otsu Adaptive Threshold
        int otsuThreshold = calculateOtsuThreshold(histogram, width * height);

        // 6. Generate Raw Binary Change Mask
        boolean[][] rawMask = new boolean[height][width];
        long rawChangedCount = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (diffMatrix[y][x] > otsuThreshold && diffMatrix[y][x] > 0) {
                    rawMask[y][x] = true;
                    rawChangedCount++;
                }
            }
        }

        // 7. Morphological Noise Removal: Opening (Erosion followed by Dilation)
        boolean[][] openedMask = morphologicalOpen(rawMask, width, height);

        // 8. Connected Component Analysis & Minimum Size Filtering
        List<Rectangle> boundingBoxes = new ArrayList<>();
        boolean[][] filteredMask = filterConnectedComponents(openedMask, width, height, minRegionSize, boundingBoxes);

        // Count filtered changed pixels
        long filteredChangedCount = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (filteredMask[y][x]) {
                    filteredChangedCount++;
                }
            }
        }

        long executionTime = System.currentTimeMillis() - startTime;

        // 9. Construct Result Object
        ChangeDetectionResult result = new ChangeDetectionResult(
                width, height,
                rawChangedCount,
                filteredChangedCount,
                boundingBoxes.size(),
                otsuThreshold,
                executionTime,
                filteredMask,
                diffMatrix,
                boundingBoxes
        );

        // 10. Generate Visual Maps
        result.setBinaryChangeMap(ChangeVisualizer.createBinaryChangeMap(filteredMask, width, height));
        result.setOverlayChangeMap(ChangeVisualizer.createOverlayChangeMap(postImage, filteredMask, boundingBoxes));

        return result;
    }

    /**
     * Overload using default minimum region size.
     */
    public static ChangeDetectionResult analyze(BufferedImage preImage, BufferedImage postImage) {
        return analyze(preImage, postImage, DEFAULT_MIN_REGION_SIZE);
    }

    /**
     * Calculates the optimal global threshold using Otsu's method.
     * Maximizes between-class variance: sigma_B^2 = w0 * w1 * (mu0 - mu1)^2
     *
     * @param histogram 256-bin histogram of pixel intensity differences
     * @param totalPixels total number of pixels in image
     * @return optimal threshold [0..255]
     */
    public static int calculateOtsuThreshold(int[] histogram, int totalPixels) {
        if (totalPixels <= 0) return 0;

        double sumTotal = 0;
        for (int i = 0; i < 256; i++) {
            sumTotal += i * histogram[i];
        }

        double sumBackground = 0;
        long weightBackground = 0;
        double maxVariance = -1.0;
        int optimalThreshold = 0;

        for (int t = 0; t < 256; t++) {
            weightBackground += histogram[t];
            if (weightBackground == 0) continue;

            long weightForeground = totalPixels - weightBackground;
            if (weightForeground == 0) break;

            sumBackground += (double) t * histogram[t];

            double meanBackground = sumBackground / weightBackground;
            double meanForeground = (sumTotal - sumBackground) / weightForeground;

            double meanDifference = meanBackground - meanForeground;
            double varianceBetween = (double) weightBackground * (double) weightForeground * meanDifference * meanDifference;

            if (varianceBetween > maxVariance) {
                maxVariance = varianceBetween;
                optimalThreshold = t;
            }
        }

        return optimalThreshold;
    }

    /**
     * Morphological Opening = Erosion followed by Dilation (3x3 window).
     * Eliminates isolated sensor speckles and noise while preserving coherent change regions.
     */
    public static boolean[][] morphologicalOpen(boolean[][] input, int width, int height) {
        boolean[][] eroded = erode3x3(input, width, height);
        return dilate3x3(eroded, width, height);
    }

    private static boolean[][] erode3x3(boolean[][] mask, int width, int height) {
        boolean[][] result = new boolean[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (!mask[y][x]) continue;

                boolean allNeighborsActive = true;
                for (int dy = -1; dy <= 1 && allNeighborsActive; dy++) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;
                        if (!mask[ny][nx]) {
                            allNeighborsActive = false;
                            break;
                        }
                    }
                }
                result[y][x] = allNeighborsActive;
            }
        }
        return result;
    }

    private static boolean[][] dilate3x3(boolean[][] mask, int width, int height) {
        boolean[][] result = new boolean[height][width];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x]) {
                    for (int dy = -1; dy <= 1; dy++) {
                        int ny = y + dy;
                        if (ny < 0 || ny >= height) continue;
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = x + dx;
                            if (nx >= 0 && nx < width) {
                                result[ny][nx] = true;
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    /**
     * Performs 8-connectivity Connected Component Labeling via Breadth-First Search (BFS).
     * Discards components whose total pixel count is less than minRegionSize.
     * Computes bounding boxes for all retained clusters.
     */
    public static boolean[][] filterConnectedComponents(boolean[][] mask, int width, int height,
                                                        int minRegionSize, List<Rectangle> outBoundingBoxes) {
        boolean[][] visited = new boolean[height][width];
        boolean[][] filtered = new boolean[height][width];

        // 8-directional offsets
        int[] dx = {-1,  0,  1, -1, 1, -1, 0, 1};
        int[] dy = {-1, -1, -1,  0, 0,  1, 1, 1};

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x] && !visited[y][x]) {
                    // Start BFS for new component
                    List<int[]> componentPixels = new ArrayList<>();
                    Queue<int[]> queue = new ArrayDeque<>();

                    queue.add(new int[]{x, y});
                    visited[y][x] = true;

                    int minX = x, maxX = x;
                    int minY = y, maxY = y;

                    while (!queue.isEmpty()) {
                        int[] p = queue.poll();
                        int px = p[0];
                        int py = p[1];
                        componentPixels.add(p);

                        if (px < minX) minX = px;
                        if (px > maxX) maxX = px;
                        if (py < minY) minY = py;
                        if (py > maxY) maxY = py;

                        for (int i = 0; i < 8; i++) {
                            int nx = px + dx[i];
                            int ny = py + dy[i];
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                if (mask[ny][nx] && !visited[ny][nx]) {
                                    visited[ny][nx] = true;
                                    queue.add(new int[]{nx, ny});
                                }
                            }
                        }
                    }

                    // Check area threshold
                    if (componentPixels.size() >= minRegionSize) {
                        for (int[] p : componentPixels) {
                            filtered[p[1]][p[0]] = true;
                        }
                        if (outBoundingBoxes != null) {
                            outBoundingBoxes.add(new Rectangle(minX, minY, (maxX - minX + 1), (maxY - minY + 1)));
                        }
                    }
                }
            }
        }

        return filtered;
    }
}
