package com.landwatch.deeplearning;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.training.loss.Loss;

/**
 * Loss for heavily imbalanced binary segmentation: weighted binary cross-entropy plus soft Dice.
 *
 * <p>In LEVIR-CD only about 4.65% of pixels are "change". Plain cross-entropy is then minimised
 * almost as well by predicting "no change" everywhere. Two standard remedies are combined:</p>
 * <ul>
 *   <li><b>Weighted BCE</b> (computed from logits, numerically stable): every change pixel's error
 *       is multiplied by {@code posWeight}, so missing a change costs more than a false alarm:
 *       <pre>loss = (1 - y) * x + (1 + (posWeight - 1) * y) * softplus(-x)</pre>
 *       which is the same formula as PyTorch's {@code BCEWithLogitsLoss(pos_weight=...)}.</li>
 *   <li><b>Soft Dice</b> over the whole batch: {@code 1 - (2 * sum(p*y) + eps) / (sum(p) + sum(y) + eps)}
 *       with {@code p = sigmoid(x)}. Dice measures overlap of the predicted and true change regions
 *       directly, so it is insensitive to the huge number of easy "no change" pixels.</li>
 * </ul>
 * <p>Total loss = {@code BCE + diceWeight * Dice}. Predictions are raw logits of shape (N, 1, H, W);
 * labels are 0/1 floats of the same shape.</p>
 */
public final class BceDiceLoss extends Loss {

    private static final float DICE_EPSILON = 1f;

    private final float posWeight;
    private final float diceWeight;

    /**
     * @param posWeight  weight of the positive (change) class in the BCE term; 1 means unweighted
     * @param diceWeight weight of the Dice term; 0 disables it
     */
    public BceDiceLoss(float posWeight, float diceWeight) {
        super("BceDiceLoss");
        if (posWeight <= 0 || diceWeight < 0) {
            throw new IllegalArgumentException("posWeight must be > 0 and diceWeight >= 0");
        }
        this.posWeight = posWeight;
        this.diceWeight = diceWeight;
    }

    public float getPosWeight() {
        return posWeight;
    }

    public float getDiceWeight() {
        return diceWeight;
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray logits = predictions.singletonOrThrow();
        NDArray y = labels.singletonOrThrow();
        if (!logits.getShape().equals(y.getShape())) {
            throw new IllegalArgumentException("Prediction shape " + logits.getShape()
                    + " does not match label shape " + y.getShape());
        }
        return weightedBce(logits, y).add(dice(logits, y).mul(diceWeight));
    }

    /** Mean weighted binary cross-entropy computed from logits. */
    public NDArray weightedBce(NDArray logits, NDArray y) {
        // softplus(-x) = log(1 + exp(-|x|)) + max(-x, 0), stable for large |x|
        NDArray softplusNeg = logits.abs().neg().exp().add(1f).log().add(logits.neg().maximum(0f));
        NDArray logWeight = y.mul(posWeight - 1f).add(1f);
        return y.neg().add(1f).mul(logits).add(logWeight.mul(softplusNeg)).mean();
    }

    /** Soft Dice loss over the whole batch (0 = perfect overlap, 1 = no overlap). */
    public NDArray dice(NDArray logits, NDArray y) {
        NDArray p = Activation.sigmoid(logits);
        NDArray intersection = p.mul(y).sum();
        NDArray denominator = p.sum().add(y.sum());
        return intersection.mul(2f).add(DICE_EPSILON).div(denominator.add(DICE_EPSILON)).neg().add(1f);
    }
}
