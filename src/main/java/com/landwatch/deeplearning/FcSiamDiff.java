package com.landwatch.deeplearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.Arrays;

/**
 * FC-Siam-diff: a fully convolutional Siamese network for bi-temporal change detection.
 *
 * <p>Reference: R. Caye Daudt, B. Le Saux and A. Boulch, "Fully Convolutional Siamese Networks for
 * Change Detection", IEEE ICIP 2018. The paper proposes three U-Net style models (FC-EF,
 * FC-Siam-conc and FC-Siam-diff); this class implements FC-Siam-diff.</p>
 *
 * <h2>Idea</h2>
 * <p>The same encoder (identical weights - that is what "Siamese" means) looks at the "before" image
 * A and the "after" image B. At every resolution level the absolute difference of the two feature
 * maps, {@code |enc(A) - enc(B)|}, is passed to the decoder through a skip connection. Unchanged
 * regions produce similar features and therefore small differences; real changes (a new building)
 * produce large differences. The decoder upsamples back to the input resolution and outputs one
 * logit per pixel: a positive logit means "change". {@code sigmoid(logit)} is the change
 * probability.</p>
 *
 * <h2>Architecture (default widths w = 16, 32, 64, 128, as in the paper)</h2>
 * <pre>
 * Input: A and B, each (N, 3, H, W); H and W must be multiples of 16.
 *
 * Shared encoder (applied to A and B; "CBR" = 3x3 Conv + BatchNorm + ReLU)
 *   E1: CBR(w1) x2   -> s1  (H,    W)      then 2x2 max-pool
 *   E2: CBR(w2) x2   -> s2  (H/2,  W/2)    then 2x2 max-pool
 *   E3: CBR(w3) x3   -> s3  (H/4,  W/4)    then 2x2 max-pool
 *   E4: CBR(w4) x3   -> s4  (H/8,  W/8)    then 2x2 max-pool -> bottleneck (H/16, W/16)
 *
 * Decoder (single path; "up" = 2x nearest-neighbour upsampling followed by a 3x3 Conv)
 *   d = |bottleneck(A) - bottleneck(B)|
 *   d = up(w4)(d);  d = concat(d, |s4(A) - s4(B)|);  D4: CBR(w4), CBR(w4), CBR(w3)
 *   d = up(w3)(d);  d = concat(d, |s3(A) - s3(B)|);  D3: CBR(w3), CBR(w3), CBR(w2)
 *   d = up(w2)(d);  d = concat(d, |s2(A) - s2(B)|);  D2: CBR(w2), CBR(w1)
 *   d = up(w1)(d);  d = concat(d, |s1(A) - s1(B)|);  D1: CBR(w1)
 *   head: 3x3 Conv(1)  -> change logits (N, 1, H, W)
 * </pre>
 *
 * <h2>Implementation notes</h2>
 * <ul>
 *   <li>For speed, A and B are stacked along the batch axis and pushed through the encoder in one
 *       pass (shape (2N, ...)), then split again. The weights are therefore shared by construction.
 *       A side effect is that BatchNorm statistics are computed over A and B together.</li>
 *   <li>Differences from the authors' reference code: no dropout (training here is only a few
 *       epochs, so regularisation comes from data augmentation instead); the bottleneck fed to the
 *       decoder is also the absolute difference of A and B features (the reference code feeds only
 *       B's bottleneck); and upsampling is nearest-neighbour 2x upsampling followed by a 3x3
 *       convolution instead of a transposed convolution, because DJL 0.40's PyTorch engine does
 *       not implement the deconvolution operator ({@code UnsupportedOperationException}).</li>
 *   <li>With the default widths the model has about 1.4 million parameters (see
 *       {@code FcSiamDiffTest} for the exact count).</li>
 * </ul>
 */
public final class FcSiamDiff extends AbstractBlock {

    private static final byte VERSION = 1;

    /** Default channel widths of the four encoder stages, as in the paper. */
    public static final int[] DEFAULT_WIDTHS = {16, 32, 64, 128};

    /** Spatial size must be divisible by this (four 2x2 poolings). */
    public static final int SIZE_DIVISOR = 16;

    private final int[] widths;

    private final Block enc1;
    private final Block enc2;
    private final Block enc3;
    private final Block enc4;
    private final Block up4;
    private final Block up3;
    private final Block up2;
    private final Block up1;
    private final Block dec4;
    private final Block dec3;
    private final Block dec2;
    private final Block dec1;
    private final Block head;

    /** Creates the model with the paper's widths 16-32-64-128. */
    public FcSiamDiff() {
        this(DEFAULT_WIDTHS);
    }

    /**
     * Creates the model with custom widths.
     *
     * @param widths four positive channel counts for encoder stages 1-4
     */
    public FcSiamDiff(int[] widths) {
        super(VERSION);
        if (widths == null || widths.length != 4) {
            throw new IllegalArgumentException("Exactly four widths are required");
        }
        for (int w : widths) {
            if (w <= 0) {
                throw new IllegalArgumentException("Widths must be positive: " + Arrays.toString(widths));
            }
        }
        this.widths = widths.clone();
        int w1 = widths[0];
        int w2 = widths[1];
        int w3 = widths[2];
        int w4 = widths[3];

        enc1 = addChildBlock("enc1", cbrStack(w1, w1));
        enc2 = addChildBlock("enc2", cbrStack(w2, w2));
        enc3 = addChildBlock("enc3", cbrStack(w3, w3, w3));
        enc4 = addChildBlock("enc4", cbrStack(w4, w4, w4));

        up4 = addChildBlock("up4", upsample(w4));
        dec4 = addChildBlock("dec4", cbrStack(w4, w4, w3));
        up3 = addChildBlock("up3", upsample(w3));
        dec3 = addChildBlock("dec3", cbrStack(w3, w3, w2));
        up2 = addChildBlock("up2", upsample(w2));
        dec2 = addChildBlock("dec2", cbrStack(w2, w1));
        up1 = addChildBlock("up1", upsample(w1));
        dec1 = addChildBlock("dec1", cbrStack(w1));
        head = addChildBlock("head", Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(1)
                .build());
    }

    /** The four encoder widths this model was built with (a copy). */
    public int[] getWidths() {
        return widths.clone();
    }

    /** A stack of 3x3 Conv (no bias, BatchNorm follows) + BatchNorm + ReLU blocks. */
    private static SequentialBlock cbrStack(int... filters) {
        SequentialBlock block = new SequentialBlock();
        for (int f : filters) {
            block.add(Conv2d.builder()
                    .setKernelShape(new Shape(3, 3))
                    .optPadding(new Shape(1, 1))
                    .optBias(false)
                    .setFilters(f)
                    .build());
            block.add(BatchNorm.builder().build());
            block.add(Activation.reluBlock());
        }
        return block;
    }

    /**
     * The learnable part of an "up" step: a 3x3 convolution applied after nearest-neighbour 2x
     * upsampling (see {@link #upsampleNearest2x(NDArray)}). Together they play the role of the
     * transposed convolution in the paper.
     */
    private static Block upsample(int filters) {
        return Conv2d.builder()
                .setKernelShape(new Shape(3, 3))
                .optPadding(new Shape(1, 1))
                .setFilters(filters)
                .build();
    }

    /**
     * Nearest-neighbour 2x upsampling of an (N, C, H, W) array: every value is copied into a 2x2
     * block, giving (N, C, 2H, 2W). It has no parameters and gradients flow through it.
     */
    static NDArray upsampleNearest2x(NDArray x) {
        return x.repeat(2, 2).repeat(3, 2);
    }

    private static NDArray maxPool(NDArray x) {
        return Pool.maxPool2d(x, new Shape(2, 2), new Shape(2, 2), new Shape(0, 0), false);
    }

    /** |first half - second half| along the batch axis, i.e. |f(A) - f(B)|. */
    private static NDArray absDiffOfHalves(NDArray x) {
        NDList halves = x.split(2, 0);
        return halves.get(0).sub(halves.get(1)).abs();
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("FcSiamDiff expects two inputs (before, after), got " + inputs.size());
        }
        NDArray a = inputs.get(0);
        NDArray b = inputs.get(1);
        checkInputShapes(a.getShape(), b.getShape());

        // Shared encoder: A and B stacked on the batch axis -> (2N, 3, H, W)
        NDArray x = NDArrays.concat(new NDList(a, b), 0);
        NDArray s1 = enc1.forward(ps, new NDList(x), training, params).singletonOrThrow();
        NDArray s2 = enc2.forward(ps, new NDList(maxPool(s1)), training, params).singletonOrThrow();
        NDArray s3 = enc3.forward(ps, new NDList(maxPool(s2)), training, params).singletonOrThrow();
        NDArray s4 = enc4.forward(ps, new NDList(maxPool(s3)), training, params).singletonOrThrow();
        NDArray bottleneck = maxPool(s4);

        // Decoder fed with |enc(A) - enc(B)| at every level
        NDArray d = absDiffOfHalves(bottleneck);
        d = decodeLevel(ps, training, params, up4, dec4, d, absDiffOfHalves(s4));
        d = decodeLevel(ps, training, params, up3, dec3, d, absDiffOfHalves(s3));
        d = decodeLevel(ps, training, params, up2, dec2, d, absDiffOfHalves(s2));
        d = decodeLevel(ps, training, params, up1, dec1, d, absDiffOfHalves(s1));
        return head.forward(ps, new NDList(d), training, params);
    }

    private static NDArray decodeLevel(ParameterStore ps, boolean training, PairList<String, Object> params,
                                       Block up, Block dec, NDArray d, NDArray skipDiff) {
        NDArray upsampled = up.forward(ps, new NDList(upsampleNearest2x(d)), training, params).singletonOrThrow();
        NDArray merged = NDArrays.concat(new NDList(upsampled, skipDiff), 1);
        return dec.forward(ps, new NDList(merged), training, params).singletonOrThrow();
    }

    private static void checkInputShapes(Shape a, Shape b) {
        if (a.dimension() != 4 || !a.equals(b)) {
            throw new IllegalArgumentException("Inputs must both be (N, 3, H, W) with equal shapes, got " + a + " and " + b);
        }
        if (a.get(1) != 3) {
            throw new IllegalArgumentException("Inputs must have 3 channels (RGB), got " + a);
        }
        if (a.get(2) % SIZE_DIVISOR != 0 || a.get(3) % SIZE_DIVISOR != 0) {
            throw new IllegalArgumentException("Height and width must be multiples of " + SIZE_DIVISOR + ", got " + a);
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        if (inputShapes.length != 2) {
            throw new IllegalArgumentException("FcSiamDiff expects two input shapes, got " + inputShapes.length);
        }
        checkInputShapes(inputShapes[0], inputShapes[1]);
        Shape s = inputShapes[0];
        return new Shape[] {new Shape(s.get(0), 1, s.get(2), s.get(3))};
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        checkInputShapes(inputShapes[0], inputShapes[1]);
        long n = inputShapes[0].get(0);
        long h = inputShapes[0].get(2);
        long w = inputShapes[0].get(3);
        int w1 = widths[0];
        int w2 = widths[1];
        int w3 = widths[2];
        int w4 = widths[3];

        enc1.initialize(manager, dataType, new Shape(2 * n, 3, h, w));
        enc2.initialize(manager, dataType, new Shape(2 * n, w1, h / 2, w / 2));
        enc3.initialize(manager, dataType, new Shape(2 * n, w2, h / 4, w / 4));
        enc4.initialize(manager, dataType, new Shape(2 * n, w3, h / 8, w / 8));

        // each "up" convolution sees the already-upsampled map
        up4.initialize(manager, dataType, new Shape(n, w4, h / 8, w / 8));
        dec4.initialize(manager, dataType, new Shape(n, w4 + w4, h / 8, w / 8));
        up3.initialize(manager, dataType, new Shape(n, w3, h / 4, w / 4));
        dec3.initialize(manager, dataType, new Shape(n, w3 + w3, h / 4, w / 4));
        up2.initialize(manager, dataType, new Shape(n, w2, h / 2, w / 2));
        dec2.initialize(manager, dataType, new Shape(n, w2 + w2, h / 2, w / 2));
        up1.initialize(manager, dataType, new Shape(n, w1, h, w));
        dec1.initialize(manager, dataType, new Shape(n, w1 + w1, h, w));
        head.initialize(manager, dataType, new Shape(n, w1, h, w));
    }

    @Override
    public String toString() {
        return "FC-Siam-diff(widths=" + Arrays.toString(widths) + ")";
    }
}
