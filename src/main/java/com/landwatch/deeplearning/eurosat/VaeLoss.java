package com.landwatch.deeplearning.eurosat;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

/**
 * Negative ELBO of {@link EuroSatVae}, averaged over the batch.
 *
 * <p>Per image, in nats:</p>
 * <ul>
 *   <li><b>Reconstruction</b>: binary cross-entropy between the image {@code x} (pixel values in
 *       [0, 1]) and {@code sigmoid(logits)}, summed over the 3 x 64 x 64 values. This is the negative
 *       log-likelihood of a Bernoulli decoder, the choice of Kingma and Welling's paper. Because real
 *       pixels are not 0 or 1, its minimum is above 0 even for a perfect reconstruction.</li>
 *   <li><b>KL</b>: {@code -1/2 * sum(1 + logvar - mu^2 - exp(logvar))}, the closed-form KL divergence
 *       from {@code N(mu, exp(logvar))} to {@code N(0, I)}, summed over the latent dimensions.</li>
 * </ul>
 * <p>{@code loss = reconstruction + beta * KL}; {@code beta = 1} is the standard VAE.</p>
 *
 * <p>Labels: {@code NDList(x)}. Predictions: {@code NDList(logits, mu, logvar)} from {@link EuroSatVae}.</p>
 */
public final class VaeLoss extends Loss {

    private final float beta;

    public VaeLoss(float beta) {
        super("vae_negative_elbo");
        if (!(beta >= 0)) {
            throw new IllegalArgumentException("beta must be >= 0");
        }
        this.beta = beta;
    }

    public float getBeta() {
        return beta;
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray x = labels.head();
        return reconstructionPerImage(predictions.get(0), x)
                .add(klPerImage(predictions.get(1), predictions.get(2)).mul(beta))
                .mean();
    }

    /**
     * Binary cross-entropy with logits summed per image, written in the numerically stable form
     * {@code max(l, 0) - l * x + log(1 + exp(-|l|))}. Returns shape (N).
     */
    public static NDArray reconstructionPerImage(NDArray logits, NDArray x) {
        NDArray bce = logits.maximum(0f).sub(logits.mul(x)).add(logits.abs().neg().exp().add(1f).log());
        return bce.sum(new int[] {1, 2, 3});
    }

    /** KL( N(mu, exp(logvar)) || N(0, I) ) per image, shape (N). */
    public static NDArray klPerImage(NDArray mu, NDArray logvar) {
        return logvar.add(1f).sub(mu.square()).sub(logvar.exp()).sum(new int[] {1}).mul(-0.5f);
    }
}
