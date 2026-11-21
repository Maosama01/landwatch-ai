package com.landwatch.ui.engine;

import com.landwatch.deeplearning.DeepChangeDetector;
import com.landwatch.deeplearning.eurosat.EuroSatClassifier;
import com.landwatch.deeplearning.eurosat.EuroSatModelConfig;
import com.landwatch.ui.data.AppPaths;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * The trained EuroSAT land-use classifier ({@link EuroSatClassifier}) as the dashboard's "Land use
 * (EuroSAT)" page uses it. It is a scene classifier for 64 x 64 Sentinel-2 tiles at 10 m/pixel, not a
 * change detector, so it deliberately does NOT implement {@link ChangeDetectionEngine} and is never
 * offered for LEVIR-CD pairs.
 *
 * <p>Life cycle as in {@link SiameseUNetEngine}: {@link State#NOT_LOADED} until {@link #load()} runs
 * on a background thread (native PyTorch + weights), then {@link State#READY} or {@link State#FAILED}
 * with a precise note such as {@code model not found at models/eurosat_resnet}. Nothing is ever
 * simulated: without a loaded model {@link #classify} throws.</p>
 *
 * <p>Model folder: {@link AppPaths#euroSatModelDir()} ({@value AppPaths#DEFAULT_EUROSAT_MODEL_DIR} or
 * {@code -D}{@value AppPaths#EUROSAT_MODEL_DIR_PROPERTY}). Device: {@code -D}{@value
 * EuroSatClassifier#DEVICE_PROPERTY} if set, otherwise the dashboard's {@code -D}{@value
 * DeepChangeDetector#DEVICE_PROPERTY}, otherwise {@code auto}.</p>
 */
public final class EuroSatEngine implements AutoCloseable {

    public static final String UNAVAILABLE_NOTE = "classifier not loaded";
    public static final String LOADING_NOTE = "loading…";

    /** Loading state. */
    public enum State { NOT_LOADED, LOADING, READY, FAILED, CLOSED }

    /** One classified image and how long the network took. */
    public record Classified(EuroSatClassifier.Prediction prediction, long millis) {
    }

    private final Path modelDir;
    private final String device;

    private volatile State state = State.NOT_LOADED;
    private volatile String statusNote = UNAVAILABLE_NOTE;
    private volatile String deviceName;
    private volatile String summary;
    private volatile List<String> classNames = List.of();

    /** Guarded by {@code this}. */
    private EuroSatClassifier classifier;

    /** Engine for the configured model folder and device; nothing is loaded yet. */
    public EuroSatEngine() {
        this(AppPaths.euroSatModelDir(), resolveDevice(System.getProperty(EuroSatClassifier.DEVICE_PROPERTY),
                System.getProperty(DeepChangeDetector.DEVICE_PROPERTY)));
    }

    /** Engine for an explicit model folder and device ({@code auto}, {@code cpu} or {@code mps}). */
    public EuroSatEngine(Path modelDir, String device) {
        if (modelDir == null) {
            throw new IllegalArgumentException("Model directory must not be null.");
        }
        this.modelDir = modelDir;
        this.device = device == null || device.isBlank() ? "auto" : device.trim();
    }

    /** The EuroSAT device property if set, else the dashboard's deep-model device property, else auto. */
    public static String resolveDevice(String euroSatDevice, String dashboardDevice) {
        if (euroSatDevice != null && !euroSatDevice.isBlank()) {
            return euroSatDevice.trim();
        }
        if (dashboardDevice != null && !dashboardDevice.isBlank()) {
            return dashboardDevice.trim();
        }
        return "auto";
    }

    // ------------------------------------------------------------------ loading

    /** Marks the engine as loading (call on the UI thread right before {@link #load()} is scheduled). */
    public synchronized void markLoading() {
        if (state == State.NOT_LOADED || state == State.FAILED) {
            state = State.LOADING;
            statusNote = LOADING_NOTE;
        }
    }

    /**
     * Loads the classifier. Blocking (native library + weights): call it from a background thread.
     *
     * @return {@code true} if ready; otherwise {@link #getStatusNote()} says precisely why not
     */
    public boolean load() {
        synchronized (this) {
            if (state == State.READY) {
                return true;
            }
            if (state == State.CLOSED) {
                return false;
            }
            state = State.LOADING;
            statusNote = LOADING_NOTE;
        }
        String problem = checkModelDir(modelDir);
        if (problem != null) {
            fail(problem);
            return false;
        }
        EuroSatClassifier loaded;
        try {
            loaded = EuroSatClassifier.load(modelDir, device);
        } catch (IOException | RuntimeException | LinkageError e) {
            fail("classifier failed to load from " + modelDir + ": " + messageOf(e));
            return false;
        }
        synchronized (this) {
            if (state == State.CLOSED) {
                loaded.close();
                return false;
            }
            classifier = loaded;
            classNames = List.copyOf(loaded.getClassNames());
            deviceName = String.valueOf(loaded.getDevice());
            summary = summarise(loaded);
            statusNote = summary;
            state = State.READY;
        }
        return true;
    }

    /**
     * Checks that {@code dir} looks like a folder written by {@code TrainEuroSatClassifier}.
     *
     * @return {@code null} if it does, otherwise a precise, user-facing reason
     */
    public static String checkModelDir(Path dir) {
        if (!Files.exists(dir)) {
            return "model not found at " + dir;
        }
        if (!Files.isDirectory(dir)) {
            return "model path is not a directory: " + dir;
        }
        if (!Files.isRegularFile(dir.resolve(EuroSatModelConfig.FILE_NAME))) {
            return "model not found at " + dir + " (no " + EuroSatModelConfig.FILE_NAME + ")";
        }
        try (DirectoryStream<Path> params = Files.newDirectoryStream(dir, "*.params")) {
            if (!params.iterator().hasNext()) {
                return "model not found at " + dir + " (no *.params weights file)";
            }
        } catch (IOException e) {
            return "cannot read model directory " + dir + ": " + messageOf(e);
        }
        return null;
    }

    private synchronized void fail(String reason) {
        if (state != State.CLOSED) {
            state = State.FAILED;
            statusNote = reason;
        }
    }

    /** One line about the loaded model; every number comes from its model.properties. */
    private static String summarise(EuroSatClassifier c) {
        EuroSatModelConfig config = c.getConfig();
        StringBuilder sb = new StringBuilder("ResNet-18-style EuroSAT classifier, ")
                .append(c.getClassNames().size()).append(" classes");
        String params = config.get("model.trainable_parameters", null);
        if (params != null) {
            try {
                sb.append(String.format(Locale.US, ", %,d params", Long.parseLong(params.trim())));
            } catch (NumberFormatException e) {
                sb.append(", ").append(params).append(" params");
            }
        }
        String epoch = config.get("training.best_epoch", null);
        if (epoch != null) {
            sb.append(", checkpoint of epoch ").append(epoch).append(" (best val accuracy)");
        }
        sb.append(", device ").append(c.getDevice());
        return sb.toString();
    }

    private static String messageOf(Throwable e) {
        String m = e.getMessage();
        return (m == null || m.isBlank()) ? e.getClass().getSimpleName() : m;
    }

    // ------------------------------------------------------------------ inference

    private synchronized EuroSatClassifier readyClassifier() {
        if (state != State.READY || classifier == null) {
            throw new IllegalStateException("EuroSAT classifier is unavailable: " + statusNote
                    + ". No prediction is produced or simulated.");
        }
        return classifier;
    }

    /** Classifies one image (any size; other sizes than 64 x 64 are resized and the prediction's note says so). */
    public Classified classify(BufferedImage image) {
        EuroSatClassifier c = readyClassifier();
        long t0 = System.nanoTime();
        EuroSatClassifier.Prediction p = c.predict(image);
        return new Classified(p, (System.nanoTime() - t0) / 1_000_000);
    }

    /** Classifies several images in batches. */
    public List<EuroSatClassifier.Prediction> classifyAll(List<BufferedImage> images) {
        return readyClassifier().predict(images);
    }

    // ------------------------------------------------------------------ accessors

    public State getState() { return state; }
    public boolean isAvailable() { return state == State.READY; }
    public boolean isLoading() { return state == State.LOADING; }
    /** True until a load has been attempted (or while it runs). */
    public boolean needsLoad() { return state == State.NOT_LOADED; }
    public String getStatusNote() { return statusNote; }
    /** The configured model folder (as given, usually relative to the working directory). */
    public Path getModelDir() { return modelDir; }
    /** The requested device ({@code auto}, {@code cpu}, {@code mps}). */
    public String getRequestedDevice() { return device; }
    /** Device in use, e.g. {@code cpu()} or {@code mps(0)}, or {@code null} before loading. */
    public String getDeviceName() { return deviceName; }
    /** One-line model summary, or {@code null} before loading. */
    public String getSummary() { return summary; }
    /** Class names in index order (empty before loading). */
    public List<String> getClassNames() { return classNames; }

    /** Releases the native model. Safe to call more than once and while a load is in progress. */
    @Override
    public synchronized void close() {
        state = State.CLOSED;
        statusNote = "classifier closed";
        if (classifier != null) {
            classifier.close();
            classifier = null;
        }
    }
}
