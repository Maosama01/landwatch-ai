package com.landwatch.deeplearning;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * DJL dataset over LEVIR-CD patch triplets.
 *
 * <p>Each record is {@code data = NDList(before (3,H,W), after (3,H,W))} and
 * {@code labels = NDList(mask (1,H,W))}; DJL's default batchifier stacks records into
 * (N,3,H,W) / (N,1,H,W) batches. Images are decoded from PNG on demand (holding the whole training
 * split in memory would take about 3.3 GB even as 8-bit pixels, or about 13 GB as float32 - too much
 * for an 8 GB laptop), normalised with the TRAIN
 * statistics, and - for the training split only - augmented with a random D4 transform (flips and
 * 90-degree rotations) that is applied identically to A, B and the mask.</p>
 */
public final class LevirCdDataset extends RandomAccessDataset {

    private final List<LevirCdPatch> patches;
    private final Normalization normalization;
    private final boolean augment;

    private LevirCdDataset(Builder builder) {
        super(builder);
        this.patches = builder.patches;
        this.normalization = builder.normalization;
        this.augment = builder.augment;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** The patches this dataset reads (in metadata order). */
    public List<LevirCdPatch> getPatches() {
        return patches;
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        LoadedSample s = load(patches.get(Math.toIntExact(index)), normalization,
                augment ? ThreadLocalRandom.current().nextInt(D4Augmentation.COUNT) : 0);
        Shape img = new Shape(3, s.height, s.width);
        NDList data = new NDList(manager.create(s.before, img), manager.create(s.after, img));
        NDList labels = new NDList(manager.create(s.mask, new Shape(1, s.height, s.width)));
        return new Record(data, labels);
    }

    @Override
    protected long availableSize() {
        return patches.size();
    }

    @Override
    public void prepare(Progress progress) {
        // nothing to download or pre-process: patches are read lazily from disk
    }

    /**
     * Decodes, normalises and (optionally) augments one patch triplet in plain Java.
     *
     * @param transformId D4 transform to apply to all three arrays (0 = none)
     */
    public static LoadedSample load(LevirCdPatch patch, Normalization normalization, int transformId)
            throws IOException {
        BufferedImage a = ImageTensors.read(patch.imageA());
        BufferedImage b = ImageTensors.read(patch.imageB());
        BufferedImage l = ImageTensors.read(patch.label());
        int w = a.getWidth();
        int h = a.getHeight();
        if (b.getWidth() != w || b.getHeight() != h || l.getWidth() != w || l.getHeight() != h) {
            throw new IOException("Size mismatch in patch " + patch.name());
        }
        float[] before = normalization.applyInPlace(ImageTensors.toChw01(a));
        float[] after = normalization.applyInPlace(ImageTensors.toChw01(b));
        float[] mask = ImageTensors.toBinaryMask(l);
        if (transformId != 0) {
            if (w != h) {
                throw new IOException("Augmentation needs square patches: " + patch.name());
            }
            before = D4Augmentation.apply(before, 3, w, transformId);
            after = D4Augmentation.apply(after, 3, w, transformId);
            mask = D4Augmentation.apply(mask, 1, w, transformId);
        }
        return new LoadedSample(before, after, mask, w, h);
    }

    /** A decoded patch: normalised CHW images and a 0/1 row-major mask. */
    public record LoadedSample(float[] before, float[] after, float[] mask, int width, int height) {
    }

    /** Builder; call {@link #setSampling(int, boolean)} to choose batch size and shuffling. */
    public static final class Builder extends BaseBuilder<Builder> {
        private List<LevirCdPatch> patches;
        private Normalization normalization = Normalization.none();
        private boolean augment;

        public Builder setPatches(List<LevirCdPatch> patches) {
            this.patches = patches;
            return this;
        }

        public Builder optNormalization(Normalization normalization) {
            this.normalization = normalization;
            return this;
        }

        /** Enables random D4 augmentation; use for the TRAIN split only. */
        public Builder optAugment(boolean augment) {
            this.augment = augment;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        public LevirCdDataset build() {
            if (patches == null || patches.isEmpty()) {
                throw new IllegalArgumentException("No patches given");
            }
            return new LevirCdDataset(this);
        }
    }
}
