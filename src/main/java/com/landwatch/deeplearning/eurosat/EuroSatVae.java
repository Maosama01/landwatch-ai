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
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * A convolutional variational autoencoder (VAE) for 64x64 RGB EuroSAT patches.
 *
 * <p>Reference: D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes", ICLR 2014.</p>
 *
 * <h2>Idea</h2>
 * <p>An ordinary autoencoder squeezes an image into a short code and rebuilds it. A VAE makes the
 * code probabilistic: the encoder outputs a mean {@code mu} and a log-variance {@code logvar} for each
 * of the {@code L} latent numbers, a code {@code z} is sampled from that Gaussian, and the decoder
 * rebuilds the image from {@code z}. Training maximises the evidence lower bound (ELBO), i.e. minimises
 * <pre>
 *   loss = reconstruction error  +  KL( N(mu, sigma^2) || N(0, I) )
 * </pre>
 * The KL term pulls every code distribution towards the standard normal, so the latent space has no
 * holes: decoding any {@code z ~ N(0, I)} gives a plausible new image. That makes the VAE a
 * generative model, unlike the classifier and the change detector.</p>
 *
 * <p>Sampling is made differentiable with the reparameterisation trick:
 * {@code z = mu + exp(logvar / 2) * eps} with {@code eps ~ N(0, I)}, so gradients flow through
 * {@code mu} and {@code logvar} while the randomness sits in {@code eps}.</p>
 *
 * <h2>Architecture (L = 64)</h2>
 * <pre>
 * encoder: 4 x [4x4 Conv stride 2 + ReLU], widths 32, 64, 128, 256:  64 -> 32 -> 16 -> 8 -> 4 px
 *          flatten (256*4*4 = 4096) -> Linear(2L) -> mu, logvar
 * decoder: Linear(4096) + ReLU -> (256, 4, 4)
 *          4 x [2x nearest upsampling, 3x3 Conv + ReLU], widths 128, 64, 32, 32: 4 -> 64 px
 *          3x3 Conv(3) -> logits; sigmoid(logits) is the reconstructed image in [0, 1]
 * </pre>
 * <p>Upsampling plus convolution replaces transposed convolution, which DJL 0.40's PyTorch engine
 * does not implement (the same choice as in FC-Siam-diff).</p>
 *
 * <p>Output of {@link #forward}: {@code NDList(logits (N,3,64,64), mu (N,L), logvar (N,L))}. In
 * training mode {@code z} is sampled; in inference mode {@code z = mu}, so reconstructions are
 * deterministic.</p>
 */
public final class EuroSatVae extends AbstractBlock {

    private static final byte VERSION = 1;

    public static final int INPUT_SIZE = 64;
    public static final int DEFAULT_LATENT = 64;
    private static final int[] WIDTHS = {32, 64, 128, 256};
    private static final int BOTTOM = INPUT_SIZE / 16; // 4 px after four stride-2 convolutions

    private final int latent;
    private final Block encoder;
    private final Block toStats;
    private final Block fromLatent;
    private final Block[] up = new Block[4];
    private final Block output;

    public EuroSatVae() {
        this(DEFAULT_LATENT);
    }

    /** @param latent number of latent dimensions L */
    public EuroSatVae(int latent) {
        super(VERSION);
        if (latent <= 0) {
            throw new IllegalArgumentException("latent must be positive");
        }
        this.latent = latent;
        SequentialBlock enc = new SequentialBlock();
        for (int w : WIDTHS) {
            enc.add(Conv2d.builder().setKernelShape(new Shape(4, 4)).optStride(new Shape(2, 2))
                    .optPadding(new Shape(1, 1)).setFilters(w).build());
            enc.add(Activation.reluBlock());
        }
        encoder = addChildBlock("encoder", enc);
        toStats = addChildBlock("to_stats", Linear.builder().setUnits(2L * latent).build());
        fromLatent = addChildBlock("from_latent", Linear.builder().setUnits((long) WIDTHS[3] * BOTTOM * BOTTOM).build());
        int[] decoderWidths = {128, 64, 32, 32};
        for (int i = 0; i < 4; i++) {
            up[i] = addChildBlock("up" + (i + 1), conv3x3(decoderWidths[i]));
        }
        output = addChildBlock("output", conv3x3(3));
    }

    private static Block conv3x3(int filters) {
        return Conv2d.builder().setKernelShape(new Shape(3, 3)).optPadding(new Shape(1, 1)).setFilters(filters).build();
    }

    public int getLatent() {
        return latent;
    }

    @Override
    protected NDList forwardInternal(ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        checkShape(x.getShape());
        long n = x.getShape().get(0);
        NDArray h = encoder.forward(ps, new NDList(x), training, params).singletonOrThrow().reshape(n, -1);
        NDList stats = toStats.forward(ps, new NDList(h), training, params).singletonOrThrow().split(2, 1);
        NDArray mu = stats.get(0);
        NDArray logvar = stats.get(1).clip(-10f, 10f); // keeps exp(logvar) finite early in training
        NDArray z = mu;
        if (training) {
            NDArray eps = x.getManager().randomNormal(mu.getShape());
            z = mu.add(logvar.mul(0.5f).exp().mul(eps));
        }
        return new NDList(decode(ps, z, training, params), mu, logvar);
    }

    /** Decodes latent codes (N, L) into logits (N, 3, 64, 64); used for sampling from the prior. */
    public NDArray decode(ParameterStore ps, NDArray z, boolean training, PairList<String, Object> params) {
        long n = z.getShape().get(0);
        NDArray d = Activation.relu(fromLatent.forward(ps, new NDList(z), training, params).singletonOrThrow())
                .reshape(n, WIDTHS[3], BOTTOM, BOTTOM);
        for (Block u : up) {
            d = Activation.relu(u.forward(ps, new NDList(d.repeat(2, 2).repeat(3, 2)), training, params).singletonOrThrow());
        }
        return output.forward(ps, new NDList(d), training, params).singletonOrThrow();
    }

    private static void checkShape(Shape s) {
        if (s.dimension() != 4 || s.get(1) != 3 || s.get(2) != INPUT_SIZE || s.get(3) != INPUT_SIZE) {
            throw new IllegalArgumentException("Expected (N, 3, 64, 64), got " + s);
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape s = inputShapes[0];
        checkShape(s);
        long n = s.get(0);
        return new Shape[] {new Shape(n, 3, INPUT_SIZE, INPUT_SIZE), new Shape(n, latent), new Shape(n, latent)};
    }

    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        Shape s = inputShapes[0];
        checkShape(s);
        long n = s.get(0);
        encoder.initialize(manager, dataType, s);
        toStats.initialize(manager, dataType, new Shape(n, (long) WIDTHS[3] * BOTTOM * BOTTOM));
        fromLatent.initialize(manager, dataType, new Shape(n, latent));
        long[] inChannels = {WIDTHS[3], 128, 64, 32};
        long size = BOTTOM;
        for (int i = 0; i < 4; i++) {
            size *= 2;
            up[i].initialize(manager, dataType, new Shape(n, inChannels[i], size, size));
        }
        output.initialize(manager, dataType, new Shape(n, 32, INPUT_SIZE, INPUT_SIZE));
    }

    @Override
    public String toString() {
        return "EuroSatVae(latent=" + latent + ")";
    }
}
