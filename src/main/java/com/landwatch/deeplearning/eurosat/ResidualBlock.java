package com.landwatch.deeplearning.eurosat;

import ai.djl.ndarray.NDArray;
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
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * The "basic block" of ResNet-18/34 (He, Zhang, Ren and Sun, "Deep Residual Learning for Image
 * Recognition", CVPR 2016), written as a DJL block.
 *
 * <h2>The residual idea: y = F(x) + x</h2>
 * <p>Instead of asking a few layers to learn a desired mapping H(x) directly, the block learns the
 * <em>residual</em> F(x) = H(x) - x and adds the input back through a shortcut:</p>
 * <pre>
 *   x --+--> Conv3x3(stride s) -> BN -> ReLU -> Conv3x3 -> BN --> F(x) --(+)--> ReLU --> y
 *       |                                                          ^
 *       +-------------------- shortcut(x) -------------------------+
 *
 *   y = ReLU( F(x) + shortcut(x) )
 * </pre>
 * <p>If the best thing a block can do is nothing, it only has to push F(x) towards zero, which is
 * easy; and during back-propagation the "+ x" path passes gradients straight to earlier layers, so
 * deep networks remain trainable. That is why ResNets can be much deeper than plain CNNs.</p>
 *
 * <h2>Two kinds of shortcut</h2>
 * <ul>
 *   <li><b>Identity</b> (stride 1 and same number of channels): {@code shortcut(x) = x}. No
 *       parameters, the shapes of x and F(x) already match.</li>
 *   <li><b>Projection</b> (stride 2 or a change in channel count): {@code shortcut(x) = BN(Conv1x1(x))}
 *       with the same stride, so the shortcut output has the same shape as F(x) and the two can be
 *       added. This is option (B) of the ResNet paper.</li>
 * </ul>
 *
 * <p>The convolutions have no bias because each is followed by BatchNorm, whose shift (beta) plays
 * the role of a bias.</p>
 */
public final class ResidualBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    private final int inChannels;
    private final int outChannels;
    private final int stride;

    private final Block conv1;
    private final Block bn1;
    private final Block conv2;
    private final Block bn2;
    /** {@code null} for an identity shortcut. */
    private final Block projection;

    /**
     * @param inChannels  channels of the input x
     * @param outChannels channels of the output y
     * @param stride      1 (same resolution) or 2 (halves height and width)
     */
    public ResidualBlock(int inChannels, int outChannels, int stride) {
        super(VERSION);
        if (inChannels <= 0 || outChannels <= 0) {
            throw new IllegalArgumentException("Channel counts must be positive");
        }
        if (stride != 1 && stride != 2) {
            throw new IllegalArgumentException("Stride must be 1 or 2, got " + stride);
        }
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.stride = stride;
        conv1 = addChildBlock("conv1", conv(3, outChannels, stride));
        bn1 = addChildBlock("bn1", BatchNorm.builder().build());
        conv2 = addChildBlock("conv2", conv(3, outChannels, 1));
        bn2 = addChildBlock("bn2", BatchNorm.builder().build());
        if (stride != 1 || inChannels != outChannels) {
            projection = addChildBlock("projection", new SequentialBlock()
                    .add(conv(1, outChannels, stride))
                    .add(BatchNorm.builder().build()));
        } else {
            projection = null;
        }
    }

    /** A k x k convolution without bias; padding k/2 keeps the size for stride 1. */
    static Conv2d conv(int kernel, int filters, int stride) {
        return Conv2d.builder()
                .setKernelShape(new Shape(kernel, kernel))
                .optStride(new Shape(stride, stride))
                .optPadding(new Shape(kernel / 2, kernel / 2))
                .optBias(false)
                .setFilters(filters)
                .build();
    }

    /** True when the shortcut is a 1x1 projection, false when it is the identity. */
    public boolean hasProjectionShortcut() {
        return projection != null;
    }

    public int getInChannels() {
        return inChannels;
    }

    public int getOutChannels() {
        return outChannels;
    }

    public int getStride() {
        return stride;
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        // residual branch F(x)
        NDArray f = conv1.forward(ps, new NDList(x), training, params).singletonOrThrow();
        f = bn1.forward(ps, new NDList(f), training, params).singletonOrThrow();
        f = Activation.relu(f);
        f = conv2.forward(ps, new NDList(f), training, params).singletonOrThrow();
        f = bn2.forward(ps, new NDList(f), training, params).singletonOrThrow();
        // shortcut: identity or 1x1 projection
        NDArray shortcut = projection == null
                ? x
                : projection.forward(ps, new NDList(x), training, params).singletonOrThrow();
        // y = ReLU(F(x) + shortcut(x))
        return new NDList(Activation.relu(f.add(shortcut)));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape in = inputShapes[0];
        checkInput(in);
        return new Shape[] {new Shape(in.get(0), outChannels, outSize(in.get(2)), outSize(in.get(3)))};
    }

    /** Output side length for an input side length: (n - 1) / stride + 1 (3x3 with padding 1, or 1x1). */
    private long outSize(long n) {
        return (n - 1) / stride + 1;
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape in = inputShapes[0];
        checkInput(in);
        Shape mid = new Shape(in.get(0), outChannels, outSize(in.get(2)), outSize(in.get(3)));
        conv1.initialize(manager, dataType, in);
        bn1.initialize(manager, dataType, mid);
        conv2.initialize(manager, dataType, mid);
        bn2.initialize(manager, dataType, mid);
        if (projection != null) {
            projection.initialize(manager, dataType, in);
        }
    }

    private void checkInput(Shape in) {
        if (in.dimension() != 4 || in.get(1) != inChannels) {
            throw new IllegalArgumentException("ResidualBlock(" + inChannels + "->" + outChannels
                    + ") expects (N, " + inChannels + ", H, W), got " + in);
        }
    }

    @Override
    public String toString() {
        return "ResidualBlock(" + inChannels + "->" + outChannels + ", stride " + stride
                + (projection == null ? ", identity shortcut)" : ", 1x1 projection shortcut)");
    }
}
