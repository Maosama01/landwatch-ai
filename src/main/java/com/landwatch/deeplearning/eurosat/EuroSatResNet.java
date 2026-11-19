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
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.Arrays;

/**
 * A compact ResNet-18-style classifier for 64 x 64 EuroSAT tiles, implemented as a DJL block.
 *
 * <h2>Architecture (default widths 32-64-128-256, two blocks per stage, 10 classes)</h2>
 * <pre>
 * input            (N,   3, 64, 64)  RGB, normalised with TRAIN mean/std
 * stem             Conv3x3(32, stride 1) + BN + ReLU          -> (N,  32, 64, 64)
 * stage1           ResidualBlock(32->32,  s1) x2  identity     -> (N,  32, 64, 64)
 * stage2           ResidualBlock(32->64,  s2, projection),
 *                  ResidualBlock(64->64,  s1)                  -> (N,  64, 32, 32)
 * stage3           ResidualBlock(64->128, s2, projection),
 *                  ResidualBlock(128->128, s1)                 -> (N, 128, 16, 16)
 * stage4           ResidualBlock(128->256, s2, projection),
 *                  ResidualBlock(256->256, s1)                 -> (N, 256,  8,  8)
 * global avg pool  mean over height and width                 -> (N, 256)
 * classifier       Linear(256 -> 10)                           -> (N, 10) logits
 * </pre>
 * <p>{@code softmax(logits)} gives the class probabilities; training uses softmax cross-entropy.
 * Each {@link ResidualBlock} computes {@code y = ReLU(F(x) + shortcut(x))}; see its Javadoc for the
 * residual idea.</p>
 *
 * <h2>Differences from the ImageNet ResNet-18</h2>
 * <ul>
 *   <li>The ImageNet stem (7x7 convolution with stride 2 followed by a 3x3 max-pool) shrinks the
 *       input 4x before the first residual block. That suits 224 x 224 photos but would leave only
 *       16 x 16 pixels of a 64 x 64 EuroSAT tile, so a 3x3 stride-1 stem without max-pool is used
 *       instead (as in the CIFAR variants of ResNet).</li>
 *   <li>Channel widths are halved (32-64-128-256 instead of 64-128-256-512), giving about 2.8 million
 *       parameters instead of 11 million: enough for 27,000 small images and fast to train on a
 *       laptop. The block layout [2, 2, 2, 2] is the same as ResNet-18 (1 stem convolution + 16
 *       convolutions in residual blocks + 1 linear layer = 18 weighted layers).</li>
 *   <li>Global average pooling makes the network accept any input size, but it was trained only on
 *       64 x 64 tiles at 10 m ground sample distance.</li>
 * </ul>
 */
public final class EuroSatResNet extends AbstractBlock {

    private static final byte VERSION = 1;

    /** Default stage widths (half of ResNet-18). */
    public static final int[] DEFAULT_WIDTHS = {32, 64, 128, 256};
    /** Residual blocks per stage (ResNet-18 uses 2). */
    public static final int DEFAULT_BLOCKS_PER_STAGE = 2;
    /** EuroSAT tiles are 64 x 64 pixels. */
    public static final int INPUT_SIZE = 64;

    private final int[] widths;
    private final int blocksPerStage;
    private final int numClasses;

    private final Block stem;
    private final Block[] stages;
    private final Block classifier;

    /** Default model: widths 32-64-128-256, 2 blocks per stage. */
    public EuroSatResNet(int numClasses) {
        this(DEFAULT_WIDTHS, DEFAULT_BLOCKS_PER_STAGE, numClasses);
    }

    /**
     * @param widths         channel width of each stage (stage 1 keeps the resolution, every further
     *                       stage halves it)
     * @param blocksPerStage number of residual blocks per stage
     * @param numClasses     number of output classes
     */
    public EuroSatResNet(int[] widths, int blocksPerStage, int numClasses) {
        super(VERSION);
        if (widths == null || widths.length == 0) {
            throw new IllegalArgumentException("At least one stage width is required");
        }
        for (int w : widths) {
            if (w <= 0) {
                throw new IllegalArgumentException("Widths must be positive: " + Arrays.toString(widths));
            }
        }
        if (blocksPerStage < 1) {
            throw new IllegalArgumentException("blocksPerStage must be >= 1");
        }
        if (numClasses < 2) {
            throw new IllegalArgumentException("numClasses must be >= 2");
        }
        this.widths = widths.clone();
        this.blocksPerStage = blocksPerStage;
        this.numClasses = numClasses;

        stem = addChildBlock("stem", new SequentialBlock()
                .add(ResidualBlock.conv(3, widths[0], 1))
                .add(BatchNorm.builder().build())
                .add(Activation.reluBlock()));
        stages = new Block[widths.length];
        int in = widths[0];
        for (int s = 0; s < widths.length; s++) {
            SequentialBlock stage = new SequentialBlock();
            for (int b = 0; b < blocksPerStage; b++) {
                int stride = (s > 0 && b == 0) ? 2 : 1;
                stage.add(new ResidualBlock(in, widths[s], stride));
                in = widths[s];
            }
            stages[s] = addChildBlock("stage" + (s + 1), stage);
        }
        classifier = addChildBlock("classifier", Linear.builder().setUnits(numClasses).build());
    }

    public int[] getWidths() {
        return widths.clone();
    }

    public int getBlocksPerStage() {
        return blocksPerStage;
    }

    public int getNumClasses() {
        return numClasses;
    }

    /** The stage blocks (each a {@link SequentialBlock} of {@link ResidualBlock}s), for inspection. */
    Block stage(int index) {
        return stages[index];
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training,
                                     PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        checkInput(x.getShape());
        x = stem.forward(ps, new NDList(x), training, params).singletonOrThrow();
        for (Block stage : stages) {
            x = stage.forward(ps, new NDList(x), training, params).singletonOrThrow();
        }
        // global average pooling (N, C, h, w) -> (N, C): mean over the width, then over the height
        // (DJL's PyTorch engine reduces one axis per call)
        NDArray pooled = x.mean(new int[] {3}).mean(new int[] {2});
        return classifier.forward(ps, new NDList(pooled), training, params);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        checkInput(inputShapes[0]);
        return new Shape[] {new Shape(inputShapes[0].get(0), numClasses)};
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape shape = inputShapes[0];
        checkInput(shape);
        stem.initialize(manager, dataType, shape);
        shape = stem.getOutputShapes(new Shape[] {shape})[0];
        for (Block stage : stages) {
            stage.initialize(manager, dataType, shape);
            shape = stage.getOutputShapes(new Shape[] {shape})[0];
        }
        classifier.initialize(manager, dataType, new Shape(shape.get(0), shape.get(1)));
    }

    private static void checkInput(Shape in) {
        if (in.dimension() != 4 || in.get(1) != 3) {
            throw new IllegalArgumentException("EuroSatResNet expects (N, 3, H, W) RGB input, got " + in);
        }
    }

    @Override
    public String toString() {
        return "EuroSatResNet(widths=" + Arrays.toString(widths) + ", blocks/stage=" + blocksPerStage
                + ", classes=" + numClasses + ")";
    }
}
