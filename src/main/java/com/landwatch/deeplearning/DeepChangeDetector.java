package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Inference API for the trained FC-Siam-diff change detector, used by the JavaFX dashboard.
 *
 * <pre>
 * try (DeepChangeDetector detector = DeepChangeDetector.load(Path.of("models/siamese_unet_levir"))) {
 *     float[][] probability = detector.predictProbabilities(before, after); // [y][x], 0..1
 *     boolean[][] changed   = detector.predictMask(before, after);          // [y][x]
 * }
 * </pre>
 *
 * <p>Images of any size are accepted (both must have the same size). They are cut into
 * non-overlapping {@code patch_size} x {@code patch_size} tiles (256 for the shipped model); tiles that
 * stick out past the right or bottom edge are filled by mirror (reflect) padding. Each tile is
 * normalised with the TRAIN statistics stored in {@code model.properties}, passed through the
 * network, and the tile outputs are stitched together and cropped back to the original size.</p>
 *
 * <p>The device is chosen with the system property {@code landwatch.deeplearning.device}
 * ({@code auto} by default: Apple MPS GPU when usable, otherwise CPU). Methods are synchronized, so
 * one instance can be shared between threads.</p>
 */
public final class DeepChangeDetector implements AutoCloseable {

    /** System property selecting the inference device: auto, cpu or mps. */
    public static final String DEVICE_PROPERTY = "landwatch.deeplearning.device";

    /** Number of tiles pushed through the network at once. */
    private static final int TILE_BATCH = 4;

    private final Path modelDir;
    private final ModelConfig config;
    private final Model model;
    private final ParameterStore parameterStore;
    private final Normalization normalization;
    private final float threshold;
    private final int tile;
    private final long parameterCount;
    private boolean closed;

    private DeepChangeDetector(Path modelDir, ModelConfig config, Model model) {
        this.modelDir = modelDir;
        this.config = config;
        this.model = model;
        this.parameterStore = new ParameterStore(model.getNDManager(), false);
        this.normalization = config.getNormalization();
        this.threshold = config.getThreshold();
        this.tile = config.getPatchSize();
        this.parameterCount = TrainingSupport.countTrainableParameters(model.getBlock());
    }

    /**
     * Loads a model directory written by {@link TrainChangeDetector} (weights + {@code model.properties}).
     * The device comes from the {@value #DEVICE_PROPERTY} system property (default {@code auto}).
     */
    public static DeepChangeDetector load(Path modelDir) throws IOException {
        return load(modelDir, System.getProperty(DEVICE_PROPERTY, "auto"));
    }

    /**
     * Loads a model directory on an explicit device.
     *
     * @param device {@code auto}, {@code cpu} or {@code mps}
     */
    public static DeepChangeDetector load(Path modelDir, String device) throws IOException {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            throw new IOException("Model directory not found: " + modelDir);
        }
        ModelConfig config = ModelConfig.load(modelDir);
        Device dev = DjlRuntime.resolveDevice(device);
        Model model = TrainingSupport.newModel(config, dev);
        try {
            model.load(modelDir, config.get(ModelConfig.PARAMS_NAME, TrainingSupport.MODEL_NAME));
        } catch (MalformedModelException | RuntimeException e) {
            model.close();
            throw new IOException("Cannot load weights from " + modelDir + ": " + e.getMessage(), e);
        }
        return new DeepChangeDetector(modelDir, config, model);
    }

    /**
     * Per-pixel change probabilities.
     *
     * @param before earlier image (any type; converted to RGB)
     * @param after  later image, same width and height as {@code before}
     * @return probabilities in [0, 1], indexed {@code [y][x]}, same size as the inputs
     */
    public synchronized float[][] predictProbabilities(BufferedImage before, BufferedImage after) {
        ensureOpen();
        if (before == null || after == null) {
            throw new IllegalArgumentException("Both images are required");
        }
        int w = before.getWidth();
        int h = before.getHeight();
        if (after.getWidth() != w || after.getHeight() != h) {
            throw new IllegalArgumentException(String.format(
                    "Image sizes differ: before %dx%d, after %dx%d", w, h, after.getWidth(), after.getHeight()));
        }
        float[] a = normalization.applyInPlace(ImageTensors.toChw01(before));
        float[] b = normalization.applyInPlace(ImageTensors.toChw01(after));

        int tilesX = (w + tile - 1) / tile;
        int tilesY = (h + tile - 1) / tile;
        int total = tilesX * tilesY;
        float[][] out = new float[h][w];
        int plane = tile * tile;
        for (int first = 0; first < total; first += TILE_BATCH) {
            int n = Math.min(TILE_BATCH, total - first);
            float[] batchA = new float[n * 3 * plane];
            float[] batchB = new float[n * 3 * plane];
            for (int i = 0; i < n; i++) {
                int t = first + i;
                int ox = (t % tilesX) * tile;
                int oy = (t / tilesX) * tile;
                copyTile(a, w, h, ox, oy, batchA, i * 3 * plane);
                copyTile(b, w, h, ox, oy, batchB, i * 3 * plane);
            }
            float[] probs;
            try (NDManager scope = model.getNDManager().newSubManager()) {
                Shape shape = new Shape(n, 3, tile, tile);
                NDList inputs = new NDList(scope.create(batchA, shape), scope.create(batchB, shape));
                probs = TrainingSupport.predictProbabilities(model.getBlock(), parameterStore, inputs);
            }
            for (int i = 0; i < n; i++) {
                int t = first + i;
                int ox = (t % tilesX) * tile;
                int oy = (t / tilesX) * tile;
                int maxY = Math.min(tile, h - oy);
                int maxX = Math.min(tile, w - ox);
                for (int y = 0; y < maxY; y++) {
                    System.arraycopy(probs, i * plane + y * tile, out[oy + y], ox, maxX);
                }
            }
        }
        return out;
    }

    /**
     * Binary change mask using the decision threshold tuned on the validation split
     * (stored in {@code model.properties}).
     *
     * @return {@code true} = change, indexed {@code [y][x]}
     */
    public synchronized boolean[][] predictMask(BufferedImage before, BufferedImage after) {
        return applyThreshold(predictProbabilities(before, after), threshold);
    }

    /** Thresholds a probability map: {@code probability >= threshold} means change. */
    public static boolean[][] applyThreshold(float[][] probabilities, float threshold) {
        boolean[][] mask = new boolean[probabilities.length][];
        for (int y = 0; y < probabilities.length; y++) {
            mask[y] = new boolean[probabilities[y].length];
            for (int x = 0; x < probabilities[y].length; x++) {
                mask[y][x] = probabilities[y][x] >= threshold;
            }
        }
        return mask;
    }

    /** The validation-tuned decision threshold used by {@link #predictMask}. */
    public float getThreshold() {
        return threshold;
    }

    /** The tile size the network runs on (256 for the shipped model). */
    public int getTileSize() {
        return tile;
    }

    /** The device inference runs on. */
    public Device getDevice() {
        return model.getNDManager().getDevice();
    }

    /** The configuration stored with the model. */
    public ModelConfig getConfig() {
        return config;
    }

    /** Human-readable description: architecture, training data, and where the metrics come from. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        if (ModelConfig.TYPE_TRANSFORMER.equals(config.getModelType())) {
            sb.append("FC-Siam-diff + Transformer change detector (Siamese U-Net with ").append(config.get(ModelConfig.TF_LAYERS, "?"))
                    .append(" self-attention layers at the bottleneck, after BIT, Chen et al., TGRS 2022)\n");
        } else {
            sb.append("FC-Siam-diff change detector (fully convolutional Siamese U-Net, Daudt, Le Saux & Boulch, ICIP 2018)\n");
        }
        sb.append("  Encoder widths ").append(Arrays.toString(config.getWidths()))
                .append(", ").append(String.format("%,d", parameterCount)).append(" trainable parameters + BatchNorm statistics\n");
        sb.append("  Trained with DJL ").append(config.get("djl_version", "?"))
                .append(" / ").append(config.get("engine", "PyTorch")).append(' ').append(config.get("engine_version", "?"))
                .append(" on ").append(config.get("training.device", "?")).append('\n');
        sb.append("  Training data: LEVIR-CD train split, ").append(config.get("data.train_patches", "?"))
                .append(" patches of ").append(tile).append('x').append(tile)
                .append(" (").append(config.get("training.epochs_completed", "?")).append(" epochs, best epoch ")
                .append(config.get("training.best_epoch", "?")).append(")\n");
        sb.append("  Checkpoint and decision threshold (").append(threshold).append(") chosen on the LEVIR-CD val split (")
                .append(config.get("data.val_patches", "?")).append(" patches)\n");
        String valF1 = config.get("metrics.val.f1", null);
        if (valF1 != null) {
            sb.append("  Val  : F1 ").append(valF1).append(", IoU ").append(config.get("metrics.val.iou", "?"))
                    .append(", precision ").append(config.get("metrics.val.precision", "?"))
                    .append(", recall ").append(config.get("metrics.val.recall", "?")).append('\n');
        }
        String testF1 = config.get("metrics.test.f1", null);
        if (testF1 != null) {
            sb.append("  Test : F1 ").append(testF1).append(", IoU ").append(config.get("metrics.test.iou", "?"))
                    .append(", precision ").append(config.get("metrics.test.precision", "?"))
                    .append(", recall ").append(config.get("metrics.test.recall", "?"))
                    .append(" (held-out test split, ").append(config.get("data.test_patches", "?"))
                    .append(" patches, evaluated once)\n");
        } else {
            sb.append("  Test : not evaluated yet (run EvaluateChangeDetector test)\n");
        }
        sb.append("  Pixel-level metrics from com.landwatch.evaluation.BinaryChangeMetrics; full reports in results/deeplearning/\n");
        sb.append("  Model directory: ").append(TrainingSupport.displayPath(modelDir)).append(", inference device: ").append(getDevice());
        return sb.toString();
    }

    /** Copies one tile (with mirror padding beyond the image border) from a CHW image into a batch buffer. */
    private void copyTile(float[] chw, int w, int h, int ox, int oy, float[] dst, int dstOffset) {
        int plane = w * h;
        int tilePlane = tile * tile;
        int[] srcX = new int[tile];
        for (int x = 0; x < tile; x++) {
            srcX[x] = reflect(ox + x, w);
        }
        for (int y = 0; y < tile; y++) {
            int sy = reflect(oy + y, h);
            for (int c = 0; c < 3; c++) {
                int srcRow = c * plane + sy * w;
                int dstRow = dstOffset + c * tilePlane + y * tile;
                for (int x = 0; x < tile; x++) {
                    dst[dstRow + x] = chw[srcRow + srcX[x]];
                }
            }
        }
    }

    /**
     * Mirror index without repeating the edge pixel ("reflect" padding), valid for any offset:
     * for n = 5 the indices 0..9 map to 0 1 2 3 4 3 2 1 0 1.
     */
    static int reflect(int i, int n) {
        if (n == 1) {
            return 0;
        }
        int period = 2 * (n - 1);
        int m = i % period;
        if (m < 0) {
            m += period;
        }
        return m < n ? m : period - m;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("DeepChangeDetector is closed");
        }
    }

    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            model.close();
        }
    }
}
