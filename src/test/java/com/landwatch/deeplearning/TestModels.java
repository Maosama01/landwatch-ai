package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.Trainer;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.Random;

/** Helpers shared by the deep-learning tests (all on CPU, small sizes, fast). */
final class TestModels {

    static final int[] TINY_WIDTHS = {4, 8, 8, 16};

    private TestModels() {
    }

    /**
     * Builds a tiny model, runs a few training steps on synthetic data (so BatchNorm running
     * statistics are non-trivial), saves weights + model.properties into {@code dir}.
     */
    static void saveModelDir(Path dir, int[] widths, int patchSize, float threshold) throws Exception {
        DeviceBenchmark.SyntheticBatch data = DeviceBenchmark.SyntheticBatch.create(2, patchSize, 7L);
        try (Model model = TrainingSupport.newModel(widths, Device.cpu());
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(2f, 1f), 1e-3f, 2, patchSize)) {
            NDManager m = trainer.getManager();
            NDList x = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            NDList y = new NDList(m.create(data.y, data.labelShape()));
            for (int i = 0; i < 2; i++) {
                TrainingSupport.trainStep(trainer, x, y);
            }
            TrainChangeDetector.saveCheckpoint(model, dir, 1);
        }
        new ModelConfig()
                .setWidths(widths)
                .set(ModelConfig.PATCH_SIZE, patchSize)
                .setNormalization(new Normalization(new float[] {0.4f, 0.39f, 0.33f}, new float[] {0.2f, 0.19f, 0.17f}))
                .set(ModelConfig.THRESHOLD, threshold)
                .set(ModelConfig.PARAMS_NAME, TrainingSupport.MODEL_NAME)
                .save(dir);
    }

    /** A random RGB image. */
    static BufferedImage randomImage(int w, int h, long seed) {
        Random rnd = new Random(seed);
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.setRGB(x, y, rnd.nextInt(0x1000000));
            }
        }
        return img;
    }

    /** Copy of a sub-rectangle as an independent image. */
    static BufferedImage crop(BufferedImage src, int x, int y, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        out.getGraphics().drawImage(src.getSubimage(x, y, w, h), 0, 0, null);
        return out;
    }
}
