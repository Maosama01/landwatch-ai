package com.landwatch.deeplearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;

/**
 * A Transformer encoder that mixes the bottleneck features of the two dates of an image pair.
 *
 * <p>It is used inside {@link FcSiamDiff} to build the "FC-Siam-diff + Transformer" change detector.
 * The idea follows BIT (H. Chen, Z. Qi and Z. Shi, "Remote Sensing Image Change Detection With
 * Transformers", IEEE TGRS 2022): turn the CNN features of both dates into one sequence of tokens and
 * let self-attention relate every location of each date to every location of both dates. A
 * convolution only sees a small neighbourhood; attention sees the whole patch at once, so the network
 * can use context ("this roof looks like the roofs over there, so it is not new") when it compares
 * the two dates. This class is a simplified version written for this project, not a copy of BIT:
 * every bottleneck position is a token (BIT first compresses each image into a few "semantic tokens").</p>
 *
 * <h2>Data flow</h2>
 * <pre>
 * input  (2N, C, h, w)   first N = date A, last N = date B (as stacked by FcSiamDiff)
 *   -> tokens (N, 2T, C)  T = h*w tokens per date, the two dates side by side
 *   -> + 2-D sinusoidal position code (the same code for A and B at the same place)
 *   -> L x pre-norm Transformer layer:
 *        x = x + MultiHeadSelfAttention(LayerNorm(x))
 *        x = x + MLP(LayerNorm(x))            MLP = Linear(C->m), GELU, Linear(m->C)
 *   -> final LayerNorm
 *   -> back to (2N, C, h, w)
 * </pre>
 *
 * <p>Multi-head self-attention, written out with plain array operations so every step is visible:
 * {@code Q = XWq, K = XWk, V = XWv}, split into {@code H} heads of size {@code d = C/H}, then
 * {@code softmax(Q K^T / sqrt(d)) V} per head, heads concatenated and projected by {@code Wo}.</p>
 *
 * <p>No date embedding is added, so the block treats the two dates symmetrically: swapping A and B
 * swaps the two halves of the output. FC-Siam-diff then takes {@code |A - B|}, so the whole model
 * gives the same change map for (A, B) and (B, A), like the plain FC-Siam-diff.</p>
 */
public final class BitemporalTransformer extends AbstractBlock {

    private static final byte VERSION = 1;

    public static final int DEFAULT_LAYERS = 2;
    public static final int DEFAULT_HEADS = 4;
    public static final int DEFAULT_MLP_HIDDEN = 256;

    private final int channels;
    private final int layers;
    private final int heads;
    private final int mlpHidden;
    private final List<Block> transformerLayers = new ArrayList<>();
    private final Block finalNorm;

    /**
     * @param channels  feature channels C of the bottleneck (128 for FC-Siam-diff's default widths)
     * @param layers    number of Transformer layers L (at least 1)
     * @param heads     attention heads H; C must be divisible by H
     * @param mlpHidden hidden size m of each layer's MLP
     */
    public BitemporalTransformer(int channels, int layers, int heads, int mlpHidden) {
        super(VERSION);
        if (channels <= 0 || layers <= 0 || heads <= 0 || mlpHidden <= 0) {
            throw new IllegalArgumentException("All sizes must be positive");
        }
        if (channels % heads != 0) {
            throw new IllegalArgumentException("channels (" + channels + ") must be divisible by heads (" + heads + ")");
        }
        if (channels % 4 != 0) {
            throw new IllegalArgumentException("channels must be divisible by 4 for the 2-D position code, got " + channels);
        }
        this.channels = channels;
        this.layers = layers;
        this.heads = heads;
        this.mlpHidden = mlpHidden;
        for (int i = 0; i < layers; i++) {
            transformerLayers.add(addChildBlock("layer" + (i + 1), new Layer(channels, heads, mlpHidden)));
        }
        finalNorm = addChildBlock("final_norm", LayerNorm.builder().axis(2).build());
    }

    public int getChannels() {
        return channels;
    }

    public int getLayers() {
        return layers;
    }

    public int getHeads() {
        return heads;
    }

    public int getMlpHidden() {
        return mlpHidden;
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        Shape s = checkShape(x.getShape());
        long n = s.get(0) / 2;
        long h = s.get(2);
        long w = s.get(3);
        long t = h * w;

        // (2N, C, h, w) -> (2, N, C, T) -> (N, 2, T, C) -> (N, 2T, C)
        NDArray tokens = x.reshape(2, n, channels, t).transpose(1, 0, 3, 2).reshape(n, 2 * t, channels);
        float[] code = positionCode((int) h, (int) w, channels);
        float[] both = new float[code.length * 2];
        System.arraycopy(code, 0, both, 0, code.length);
        System.arraycopy(code, 0, both, code.length, code.length);
        NDArray position = x.getManager().create(both, new Shape(1, 2 * t, channels));
        tokens = tokens.add(position);

        for (Block layer : transformerLayers) {
            tokens = layer.forward(ps, new NDList(tokens), training, params).singletonOrThrow();
        }
        tokens = finalNorm.forward(ps, new NDList(tokens), training, params).singletonOrThrow();

        // (N, 2T, C) -> (N, 2, T, C) -> (2, N, C, T) -> (2N, C, h, w)
        NDArray out = tokens.reshape(n, 2, t, channels).transpose(1, 0, 3, 2).reshape(2 * n, channels, h, w);
        return new NDList(out);
    }

    private Shape checkShape(Shape s) {
        if (s.dimension() != 4 || s.get(1) != channels || s.get(0) % 2 != 0) {
            throw new IllegalArgumentException("Expected (2N, " + channels + ", h, w) with the two dates stacked, got " + s);
        }
        return s;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {checkShape(inputShapes[0])};
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape s = checkShape(inputShapes[0]);
        Shape tokenShape = new Shape(s.get(0) / 2, 2 * s.get(2) * s.get(3), channels);
        for (Block layer : transformerLayers) {
            layer.initialize(manager, dataType, tokenShape);
        }
        finalNorm.initialize(manager, dataType, tokenShape);
    }

    /**
     * Fixed 2-D sinusoidal position code, as a flat (h*w, C) array in row-major token order.
     * The first C/2 values encode the row y and the last C/2 the column x; each half alternates
     * {@code sin(p * f_i)} and {@code cos(p * f_i)} with frequencies {@code f_i = 10000^(-i / (C/4))}.
     * Nothing here is learned, so the code works for any h and w.
     */
    static float[] positionCode(int h, int w, int channels) {
        int quarter = channels / 4;
        float[] code = new float[h * w * channels];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int base = (y * w + x) * channels;
                for (int i = 0; i < quarter; i++) {
                    double f = Math.pow(10000.0, -(double) i / quarter);
                    code[base + 2 * i] = (float) Math.sin(y * f);
                    code[base + 2 * i + 1] = (float) Math.cos(y * f);
                    code[base + 2 * quarter + 2 * i] = (float) Math.sin(x * f);
                    code[base + 2 * quarter + 2 * i + 1] = (float) Math.cos(x * f);
                }
            }
        }
        return code;
    }

    @Override
    public String toString() {
        return "BitemporalTransformer(C=" + channels + ", layers=" + layers + ", heads=" + heads + ", mlp=" + mlpHidden + ")";
    }

    /** One pre-norm Transformer encoder layer: self-attention and an MLP, each with a residual connection. */
    static final class Layer extends AbstractBlock {

        private final int channels;
        private final int heads;
        private final Block norm1;
        private final Block qkv;
        private final Block projection;
        private final Block norm2;
        private final Block mlpIn;
        private final Block mlpOut;

        Layer(int channels, int heads, int mlpHidden) {
            super(VERSION);
            this.channels = channels;
            this.heads = heads;
            norm1 = addChildBlock("norm1", LayerNorm.builder().axis(2).build());
            qkv = addChildBlock("qkv", Linear.builder().setUnits(3L * channels).build());
            projection = addChildBlock("projection", Linear.builder().setUnits(channels).build());
            norm2 = addChildBlock("norm2", LayerNorm.builder().axis(2).build());
            mlpIn = addChildBlock("mlp_in", Linear.builder().setUnits(mlpHidden).build());
            mlpOut = addChildBlock("mlp_out", Linear.builder().setUnits(channels).build());
        }

        @Override
        protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
            NDArray x = inputs.singletonOrThrow();
            NDArray attended = selfAttention(ps, training, params,
                    norm1.forward(ps, new NDList(x), training, params).singletonOrThrow());
            x = x.add(attended);
            NDArray hidden = mlpIn.forward(ps, norm2.forward(ps, new NDList(x), training, params), training, params)
                    .singletonOrThrow();
            NDArray mlp = mlpOut.forward(ps, new NDList(Activation.gelu(hidden)), training, params).singletonOrThrow();
            return new NDList(x.add(mlp));
        }

        /** softmax(Q K^T / sqrt(d)) V for every head, then the output projection. Input and output (N, S, C). */
        private NDArray selfAttention(ParameterStore ps, boolean training, PairList<String, Object> params, NDArray x) {
            Shape s = x.getShape();
            long n = s.get(0);
            long len = s.get(1);
            long d = channels / heads;
            NDList qkvParts = qkv.forward(ps, new NDList(x), training, params).singletonOrThrow().split(3, 2);
            NDArray q = toHeads(qkvParts.get(0), n, len, d);
            NDArray k = toHeads(qkvParts.get(1), n, len, d);
            NDArray v = toHeads(qkvParts.get(2), n, len, d);
            // a float scale: a double would create a float64 scalar, which the Apple GPU (MPS) cannot hold
            NDArray scores = q.batchMatMul(k.transpose(0, 2, 1)).mul((float) (1.0 / Math.sqrt(d))); // (N*H, S, S)
            NDArray weights = scores.softmax(2);
            NDArray mixed = weights.batchMatMul(v)                                          // (N*H, S, d)
                    .reshape(n, heads, len, d).transpose(0, 2, 1, 3).reshape(n, len, channels);
            return projection.forward(ps, new NDList(mixed), training, params).singletonOrThrow();
        }

        /** (N, S, C) -> (N*H, S, d): one row of the batch per sample and head. */
        private NDArray toHeads(NDArray a, long n, long len, long d) {
            return a.reshape(n, len, heads, d).transpose(0, 2, 1, 3).reshape(n * heads, len, d);
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            return new Shape[] {inputShapes[0]};
        }

        @Override
        protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
            Shape s = inputShapes[0];
            norm1.initialize(manager, dataType, s);
            qkv.initialize(manager, dataType, s);
            projection.initialize(manager, dataType, s);
            norm2.initialize(manager, dataType, s);
            mlpIn.initialize(manager, dataType, s);
            mlpOut.initialize(manager, dataType, new Shape(s.get(0), s.get(1), mlpIn.getOutputShapes(new Shape[] {s})[0].get(2)));
        }
    }
}
