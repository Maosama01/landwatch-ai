package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.Trainer;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Sanity check (b): the full-width FC-Siam-diff can drive the loss down on a handful of examples.
 * If a network cannot overfit a tiny batch, something in the model, loss or optimiser is broken.
 */
class TinyOverfitTest {

    private static final int SYNTHETIC_STEPS = 40;
    private static final int REAL_STEPS = 30;

    @Test
    void lossDropsOnSyntheticBatch() {
        Engine.getInstance().setRandomSeed(1);
        DeviceBenchmark.SyntheticBatch data = DeviceBenchmark.SyntheticBatch.create(4, 64, 3L);
        float[] losses = train(data.a, data.b, data.y, 4, 64, SYNTHETIC_STEPS);
        float last = losses[losses.length - 1];
        System.out.printf("Synthetic overfit: loss %.4f -> %.4f%n", losses[0], last);
        assertTrue(last < 0.5f * losses[0], "loss should at least halve: " + losses[0] + " -> " + last);
    }

    /**
     * Real LEVIR-CD data: 8 training patches that contain change, each reduced to the 96x96
     * window with the most change pixels (keeps the CPU test well under a minute). Skipped when the
     * dataset is not available (set LEVIR_CD_ROOT or run from the project root).
     */
    @Test
    void lossDropsOnEightRealTrainingPatches() throws Exception {
        Path root = LevirCdPatch.locateDatasetRoot();
        assumeTrue(root != null, "LEVIR-CD processed_256 not found; skipping real-data overfit test");
        Engine.getInstance().setRandomSeed(1);
        List<LevirCdPatch> candidates = new ArrayList<>();
        for (LevirCdPatch p : LevirCdPatch.readSplit(root, "train")) {
            double ratio = p.changePixelCount() / (double) p.pixelCount();
            if (ratio > 0.05 && ratio < 0.5) {
                candidates.add(p);
            }
        }
        candidates.sort(Comparator.comparing(LevirCdPatch::name));
        int n = 8;
        int crop = 96;
        int plane = crop * crop;
        float[] a = new float[n * 3 * plane];
        float[] b = new float[n * 3 * plane];
        float[] y = new float[n * plane];
        Normalization norm = new Normalization(new float[] {0.398f, 0.392f, 0.335f}, new float[] {0.197f, 0.188f, 0.173f});
        for (int i = 0; i < n; i++) {
            LevirCdDataset.LoadedSample s = LevirCdDataset.load(candidates.get(i * (candidates.size() / n)), norm, 0);
            int[] best = bestQuadrant(s.mask(), s.width(), crop);
            copyCrop(s.before(), 3, s.width(), best[0], best[1], crop, a, i * 3 * plane);
            copyCrop(s.after(), 3, s.width(), best[0], best[1], crop, b, i * 3 * plane);
            copyCrop(s.mask(), 1, s.width(), best[0], best[1], crop, y, i * plane);
        }
        float[] losses = train(a, b, y, n, crop, REAL_STEPS);
        float last = losses[losses.length - 1];
        System.out.printf("Real-data overfit (8 patches, 96x96): loss %.4f -> %.4f%n", losses[0], last);
        assertTrue(last < 0.5f * losses[0], "loss should at least halve: " + losses[0] + " -> " + last);
    }

    private static float[] train(float[] a, float[] b, float[] y, int n, int size, int steps) {
        float[] losses = new float[steps];
        try (Model model = TrainingSupport.newModel(FcSiamDiff.DEFAULT_WIDTHS, Device.cpu());
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(4.5f, 1f), 1e-3f, n, size)) {
            NDManager m = trainer.getManager();
            Shape img = new Shape(n, 3, size, size);
            NDList x = new NDList(m.create(a, img), m.create(b, img));
            NDList labels = new NDList(m.create(y, new Shape(n, 1, size, size)));
            for (int i = 0; i < steps; i++) {
                losses[i] = TrainingSupport.trainStep(trainer, x, labels);
                assertTrue(Float.isFinite(losses[i]), "loss must stay finite");
            }
        }
        return losses;
    }

    private static int[] bestQuadrant(float[] mask, int size, int crop) {
        int[] best = {0, 0};
        double bestCount = -1;
        for (int oy = 0; oy + crop <= size; oy += 32) {
            for (int ox = 0; ox + crop <= size; ox += 32) {
                double c = 0;
                for (int yy = oy; yy < oy + crop; yy++) {
                    for (int xx = ox; xx < ox + crop; xx++) {
                        c += mask[yy * size + xx];
                    }
                }
                if (c > bestCount) {
                    bestCount = c;
                    best = new int[] {ox, oy};
                }
            }
        }
        return best;
    }

    private static void copyCrop(float[] src, int channels, int size, int ox, int oy, int crop, float[] dst, int offset) {
        for (int c = 0; c < channels; c++) {
            for (int yy = 0; yy < crop; yy++) {
                System.arraycopy(src, c * size * size + (oy + yy) * size + ox, dst, offset + c * crop * crop + yy * crop, crop);
            }
        }
    }
}
