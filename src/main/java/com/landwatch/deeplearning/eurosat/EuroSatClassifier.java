package com.landwatch.deeplearning.eurosat;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.training.ParameterStore;
import com.landwatch.deeplearning.DjlRuntime;
import com.landwatch.deeplearning.ImageTensors;
import com.landwatch.deeplearning.Normalization;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * Inference API for the trained EuroSAT land-use / land-cover classifier.
 *
 * <pre>
 * try (EuroSatClassifier classifier = EuroSatClassifier.load(Path.of("models/eurosat_resnet"))) {
 *     EuroSatClassifier.Prediction p = classifier.predict(tile);   // a BufferedImage
 *     System.out.println(p.className() + " " + p.probability());
 * }
 * </pre>
 *
 * <p>What the model expects: a 64 x 64 RGB tile of Sentinel-2 imagery at 10 m ground sample
 * distance (640 m x 640 m on the ground), like the EuroSAT tiles it was trained on. It assigns one of
 * the 10 EuroSAT classes to the whole tile; it is a scene classifier, not a change detector, and it
 * says nothing about which pixels belong to which class.</p>
 *
 * <p>Images of another size are resized to 64 x 64 (the whole image is stretched, no cropping;
 * bilinear interpolation, with repeated halving first for large images) and the returned
 * {@link Prediction#note()} says so. Such predictions are less reliable: the ground footprint and
 * resolution then differ from the training data (for example, a 256 x 256 aerial photo at 0.5 m covers
 * 128 m x 128 m, a far smaller area seen in far more detail).</p>
 *
 * <p>The device is chosen with the system property {@value #DEVICE_PROPERTY} ({@code auto} by
 * default: Apple MPS GPU when usable, otherwise CPU), or explicitly with {@link #load(Path, String)}.
 * Methods are synchronized, so one instance can be shared between threads.</p>
 */
public final class EuroSatClassifier implements AutoCloseable {

    /** System property selecting the inference device: auto, cpu or mps. */
    public static final String DEVICE_PROPERTY = "landwatch.eurosat.device";

    /** Images pushed through the network at once by {@link #predict(List)}. */
    private static final int BATCH = 64;

    private final Path modelDir;
    private final EuroSatModelConfig config;
    private final Model model;
    private final ParameterStore parameterStore;
    private final Normalization normalization;
    private final List<String> classNames;
    private final int inputSize;
    private boolean closed;

    /**
     * The result for one image.
     *
     * @param className     predicted class name
     * @param classIndex    predicted class index (arg-max of the probabilities)
     * @param probabilities softmax probability of every class, in class-index order (sums to 1)
     * @param note          empty when the image was already 64 x 64; otherwise explains the resizing
     */
    public record Prediction(String className, int classIndex, float[] probabilities, String note) {

        public Prediction {
            probabilities = probabilities.clone();
        }

        @Override
        public float[] probabilities() {
            return probabilities.clone();
        }

        /** Probability of the predicted class (the model's confidence). */
        public float probability() {
            return probabilities[classIndex];
        }

        /** True when the input was not 64 x 64 and had to be resized. */
        public boolean wasResized() {
            return !note.isEmpty();
        }

        /** Class indices sorted by decreasing probability, at most {@code k}. */
        public int[] topIndices(int k) {
            Integer[] idx = new Integer[probabilities.length];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = i;
            }
            Arrays.sort(idx, (a, b) -> Float.compare(probabilities[b], probabilities[a]));
            int[] out = new int[Math.min(k, idx.length)];
            for (int i = 0; i < out.length; i++) {
                out[i] = idx[i];
            }
            return out;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "%s (index %d, p=%.4f)%s", className, classIndex, probability(),
                    note.isEmpty() ? "" : " [" + note + "]");
        }
    }

    private EuroSatClassifier(Path modelDir, EuroSatModelConfig config, Model model) {
        this.modelDir = modelDir;
        this.config = config;
        this.model = model;
        this.parameterStore = new ParameterStore(model.getNDManager(), false);
        this.normalization = config.getNormalization();
        this.classNames = config.getClassNames();
        this.inputSize = config.getInputSize();
    }

    /**
     * Loads a model folder written by {@link TrainEuroSatClassifier} ({@code eurosat_resnet-*.params}
     * plus {@code model.properties}). The device comes from {@value #DEVICE_PROPERTY} (default auto).
     */
    public static EuroSatClassifier load(Path modelDir) throws IOException {
        return load(modelDir, System.getProperty(DEVICE_PROPERTY, "auto"));
    }

    /**
     * Loads a model folder on an explicit device.
     *
     * @param device {@code auto}, {@code cpu} or {@code mps}
     */
    public static EuroSatClassifier load(Path modelDir, String device) throws IOException {
        if (modelDir == null || !Files.isDirectory(modelDir)) {
            throw new IOException("Model directory not found: " + modelDir);
        }
        EuroSatModelConfig config = EuroSatModelConfig.load(modelDir);
        Device dev = DjlRuntime.resolveDevice(device);
        Model model = EuroSatTraining.newModel(config.getWidths(), config.getBlocksPerStage(),
                config.getClassNames().size(), dev);
        try {
            model.load(modelDir, config.get(EuroSatModelConfig.PARAMS_NAME, EuroSatTraining.MODEL_NAME));
        } catch (MalformedModelException | RuntimeException e) {
            model.close();
            throw new IOException("Cannot load weights from " + modelDir + ": " + e.getMessage(), e);
        }
        return new EuroSatClassifier(modelDir, config, model);
    }

    /** Classifies one image (any size and type; converted to RGB, resized to 64 x 64 if needed). */
    public synchronized Prediction predict(BufferedImage image) {
        return predict(List.of(image)).get(0);
    }

    /** Classifies several images, running them through the network in batches. */
    public synchronized List<Prediction> predict(List<BufferedImage> images) {
        ensureOpen();
        if (images == null) {
            throw new IllegalArgumentException("images must not be null");
        }
        List<Prediction> out = new ArrayList<>(images.size());
        int per = 3 * inputSize * inputSize;
        int k = classNames.size();
        for (int first = 0; first < images.size(); first += BATCH) {
            int n = Math.min(BATCH, images.size() - first);
            float[] batch = new float[n * per];
            String[] notes = new String[n];
            for (int i = 0; i < n; i++) {
                BufferedImage img = images.get(first + i);
                if (img == null) {
                    throw new IllegalArgumentException("Image " + (first + i) + " is null");
                }
                notes[i] = "";
                if (img.getWidth() != inputSize || img.getHeight() != inputSize) {
                    notes[i] = resizeNote(img.getWidth(), img.getHeight(), inputSize);
                    img = resize(img, inputSize);
                }
                float[] chw = normalization.applyInPlace(ImageTensors.toChw01(img));
                System.arraycopy(chw, 0, batch, i * per, per);
            }
            float[] probs = EuroSatTraining.predictProbabilities(model.getBlock(), parameterStore,
                    model.getNDManager(), batch, n, inputSize);
            for (int i = 0; i < n; i++) {
                float[] p = Arrays.copyOfRange(probs, i * k, (i + 1) * k);
                int best = EuroSatTraining.argMax(p, 0, k);
                out.add(new Prediction(classNames.get(best), best, p, notes[i]));
            }
        }
        return out;
    }

    /** The note attached to predictions of images that are not {@code size x size}. */
    static String resizeNote(int w, int h, int size) {
        return String.format(Locale.ROOT,
                "input was %dx%d px and was resized to %dx%d (stretched, aspect ratio not kept); the model was "
                        + "trained on %dx%d Sentinel-2 tiles at 10 m/pixel, so predictions for other sizes or "
                        + "resolutions are less reliable", w, h, size, size, size, size);
    }

    /**
     * Resizes to {@code size x size} RGB with bilinear interpolation. Large images are first halved
     * repeatedly (each halving averages 2 x 2 pixels) until they are at most twice the target size,
     * so shrinking a big image does not simply skip most of its pixels (aliasing).
     */
    static BufferedImage resize(BufferedImage src, int size) {
        BufferedImage current = src;
        int w = src.getWidth();
        int h = src.getHeight();
        while (w > 2 * size || h > 2 * size) {
            w = w > 2 * size ? w / 2 : w;
            h = h > 2 * size ? h / 2 : h;
            current = drawBilinear(current, w, h);
        }
        return drawBilinear(current, size, size);
    }

    private static BufferedImage drawBilinear(BufferedImage src, int w, int h) {
        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        try {
            g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g.drawImage(src, 0, 0, w, h, null);
        } finally {
            g.dispose();
        }
        return out;
    }

    /** Class names in class-index order. */
    public List<String> getClassNames() {
        return classNames;
    }

    /** Side length of the tiles the network was trained on (64). */
    public int getInputSize() {
        return inputSize;
    }

    /** The device inference runs on. */
    public Device getDevice() {
        return model.getNDManager().getDevice();
    }

    /** The configuration stored with the model. */
    public EuroSatModelConfig getConfig() {
        return config;
    }

    /** Human-readable description: architecture, training data, and where the metrics come from. */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        sb.append("EuroSAT land-use / land-cover classifier: ResNet-18-style network (")
                .append(config.get(EuroSatModelConfig.ARCHITECTURE, "?")).append(")\n");
        sb.append("  Widths ").append(Arrays.toString(config.getWidths())).append(", ")
                .append(config.get("model.trainable_parameters", "?")).append(" trainable parameters, ")
                .append(classNames.size()).append(" classes: ").append(String.join(", ", classNames)).append('\n');
        sb.append("  Trained with DJL ").append(config.get("djl_version", "?")).append(" / ")
                .append(config.get("engine", "PyTorch")).append(' ').append(config.get("engine_version", "?"))
                .append(" on ").append(config.get("training.device", "?")).append(", ")
                .append(config.get("training.epochs_completed", "?")).append(" epochs, selected epoch ")
                .append(config.get("training.best_epoch", "?")).append(" (highest val accuracy)\n");
        sb.append("  Data: EuroSAT RGB (Sentinel-2, 10 m), stratified split train ")
                .append(config.get("data.train_images", "?")).append(" / val ")
                .append(config.get("data.val_images", "?")).append(" / test ")
                .append(config.get("data.test_images", "?")).append(" images\n");
        appendMetrics(sb, "val", "Val ");
        appendMetrics(sb, "test", "Test");
        sb.append("  Scope: classifies a whole 64x64 tile into one land-use class; not a change detector.\n");
        sb.append("  Model directory: ").append(modelDir.toAbsolutePath()).append(", inference device: ").append(getDevice());
        return sb.toString();
    }

    private void appendMetrics(StringBuilder sb, String split, String label) {
        String acc = config.get("metrics." + split + ".accuracy", null);
        if (acc == null) {
            sb.append("  ").append(label).append(": not evaluated yet (run EvaluateEuroSatClassifier ").append(split).append(")\n");
        } else {
            sb.append("  ").append(label).append(": accuracy ").append(acc).append(", macro F1 ")
                    .append(config.get("metrics." + split + ".macro_f1", "?")).append(" (")
                    .append(config.get("metrics." + split + ".images", "?")).append(" images, ")
                    .append(config.get("metrics." + split + ".source", "")).append(")\n");
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("EuroSatClassifier is closed");
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
