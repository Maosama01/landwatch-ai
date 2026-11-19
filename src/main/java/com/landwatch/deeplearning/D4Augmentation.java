package com.landwatch.deeplearning;

/**
 * Geometric data augmentation with the 8 symmetries of a square (the dihedral group D4):
 * rotations by 0, 90, 180 and 270 degrees, each optionally followed by a horizontal flip. A
 * vertical flip equals "rotate 180 + horizontal flip", so all flip/rotation combinations are
 * covered.
 *
 * <p>The same transform must be applied to image A, image B and the label so they stay aligned;
 * {@link #apply(float[], int, int, int)} is deterministic for a given transform id, which makes
 * that easy. Only square patches are supported (rotating a non-square patch changes its shape).</p>
 */
public final class D4Augmentation {

    /** Number of distinct transforms (ids 0..7); id 0 is the identity. */
    public static final int COUNT = 8;

    private D4Augmentation() {
    }

    /**
     * Applies transform {@code id} to a channel-first array.
     *
     * @param chw      data of {@code channels} planes of size x size, layout c*size*size + y*size + x
     * @param channels number of channels (3 for images, 1 for labels)
     * @param size     side length of the square patch
     * @param id       0..7: rotation = id % 4 quarter turns counter-clockwise, flip = id &gt;= 4
     * @return a new transformed array (the input is not modified)
     */
    public static float[] apply(float[] chw, int channels, int size, int id) {
        if (id < 0 || id >= COUNT) {
            throw new IllegalArgumentException("Transform id must be 0..7, got " + id);
        }
        int plane = size * size;
        if (chw.length != channels * plane) {
            throw new IllegalArgumentException("Array length " + chw.length + " != " + channels + "x" + size + "x" + size);
        }
        if (id == 0) {
            return chw.clone();
        }
        int rotations = id % 4;
        boolean flip = id >= 4;
        int last = size - 1;
        float[] out = new float[chw.length];
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                // (y, x) is the output pixel; find the source pixel it comes from.
                int sx = flip ? last - x : x;
                int sy = y;
                for (int r = 0; r < rotations; r++) {
                    // inverse of one counter-clockwise quarter turn
                    int t = sy;
                    sy = sx;
                    sx = last - t;
                }
                int dst = y * size + x;
                int src = sy * size + sx;
                for (int c = 0; c < channels; c++) {
                    out[c * plane + dst] = chw[c * plane + src];
                }
            }
        }
        return out;
    }
}
