package com.landwatch.ui.engine;

import com.landwatch.deeplearning.DeepChangeDetector;
import com.landwatch.deeplearning.ModelConfig;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Deep-learning engine: the trained FC-Siam-diff change detector (a fully convolutional Siamese U-Net)
 * from {@link DeepChangeDetector}.
 *
 * <p>Life cycle: a new engine is {@link State#NOT_LOADED} and unavailable ("no trained model loaded").
 * The dashboard calls {@link #markLoading()} on the FX thread and then {@link #load()} on a background
 * thread, so start-up never blocks on the native PyTorch library or the weights. Loading either ends in
 * {@link State#READY} (the engine becomes selectable) or {@link State#FAILED} with a precise note, for
 * example {@code model not found at models/siamese_unet_levir}. {@link #close()} releases the native
 * model when the application exits.</p>
 *
 * <p>The engine never produces simulated or placeholder output: {@link #detect} only returns the mask
 * predicted by the loaded network, thresholded with the decision threshold stored in the model's
 * {@code model.properties} (tuned on the LEVIR-CD validation split).</p>
 *
 * <p>Model directory: {@value #DEFAULT_MODEL_DIR} relative to the working directory, overridable with
 * the system property {@value #MODEL_DIR_PROPERTY}. The inference device comes from
 * {@value DeepChangeDetector#DEVICE_PROPERTY} ({@code auto}, {@code cpu} or {@code mps}).</p>
 */
public final class SiameseUNetEngine implements ChangeDetectionEngine, AutoCloseable {

    public static final String ID = "siamese-unet";
    public static final String DISPLAY_NAME = "Deep model – Siamese U-Net";
    public static final String UNAVAILABLE_NOTE = "no trained model loaded";
    public static final String LOADING_NOTE = "loading…";

    /** System property that overrides the model directory. */
    public static final String MODEL_DIR_PROPERTY = "landwatch.model.dir";
    /** Default model directory, relative to the working directory (the project root under mvn javafx:run). */
    public static final String DEFAULT_MODEL_DIR = "models/siamese_unet_levir";

    /** Loading state of the engine. */
    public enum State { NOT_LOADED, LOADING, READY, FAILED, CLOSED }

    private final Path modelDir;

    private volatile State state = State.NOT_LOADED;
    private volatile String statusNote = UNAVAILABLE_NOTE;
    private volatile String description;
    private volatile String deviceName;
    private volatile Float threshold;

    /** Guarded by {@code this}. */
    private DeepChangeDetector detector;

    /** Engine for the configured model directory ({@link #resolveModelDir()}); nothing is loaded yet. */
    public SiameseUNetEngine() {
        this(resolveModelDir());
    }

    /** Engine for an explicit model directory; nothing is loaded until {@link #load()} is called. */
    public SiameseUNetEngine(Path modelDir) {
        if (modelDir == null) {
            throw new IllegalArgumentException("Model directory must not be null.");
        }
        this.modelDir = modelDir;
    }

    /** The model directory from {@value #MODEL_DIR_PROPERTY}, or {@value #DEFAULT_MODEL_DIR} when unset. */
    public static Path resolveModelDir() {
        String configured = System.getProperty(MODEL_DIR_PROPERTY);
        return Path.of(configured == null || configured.isBlank() ? DEFAULT_MODEL_DIR : configured.trim());
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
     * Loads the trained model. Blocking (native library + weights): call it from a background thread.
     *
     * @return {@code true} if the model is ready; otherwise {@link #getStatusNote()} says precisely why not
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

        DeepChangeDetector loaded;
        try {
            loaded = DeepChangeDetector.load(modelDir);
        } catch (IOException | RuntimeException | LinkageError e) {
            fail("model failed to load from " + modelDir + ": " + messageOf(e));
            return false;
        }

        synchronized (this) {
            if (state == State.CLOSED) {
                // The application shut down while the model was loading.
                loaded.close();
                return false;
            }
            detector = loaded;
            description = loaded.describe();
            deviceName = String.valueOf(loaded.getDevice());
            threshold = loaded.getThreshold();
            statusNote = summarise(loaded);
            state = State.READY;
        }
        return true;
    }

    /**
     * Checks that {@code dir} looks like a model directory written by the trainer.
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
        if (!Files.isRegularFile(dir.resolve(ModelConfig.FILE_NAME))) {
            return "model not found at " + dir + " (no " + ModelConfig.FILE_NAME + ")";
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

    /** One-line summary for the engine selector. Every number comes from the model's own files. */
    private static String summarise(DeepChangeDetector d) {
        ModelConfig c = d.getConfig();
        StringBuilder sb = new StringBuilder("FC-Siam-diff (Siamese U-Net)");
        String params = c.get("model.trainable_parameters", null);
        if (params != null) {
            try {
                sb.append(String.format(Locale.US, ", %,d params", Long.parseLong(params.trim())));
            } catch (NumberFormatException ignored) {
                sb.append(", ").append(params).append(" params");
            }
        }
        sb.append(String.format(Locale.US, ", threshold %.2f (tuned on val)", d.getThreshold()));
        sb.append(", device ").append(d.getDevice());
        String valF1 = c.get("metrics.val.f1", null);
        if (valF1 != null) {
            sb.append("; val F1 ").append(valF1);
            String valIou = c.get("metrics.val.iou", null);
            if (valIou != null) {
                sb.append(", IoU ").append(valIou);
            }
            sb.append(" (from model.properties)");
        }
        return sb.toString();
    }

    private static String messageOf(Throwable e) {
        String m = e.getMessage();
        return (m == null || m.isBlank()) ? e.getClass().getSimpleName() : m;
    }

    // ------------------------------------------------------------------ ChangeDetectionEngine

    @Override public String getId() { return ID; }
    @Override public String getDisplayName() { return DISPLAY_NAME; }
    @Override public String getShortLabel() { return "FC-Siam-diff"; }

    /** True only once a trained model has been loaded successfully. */
    @Override public boolean isAvailable() { return state == State.READY; }

    @Override public boolean isLoading() { return state == State.LOADING; }

    @Override public String getStatusNote() { return statusNote; }

    @Override
    public EngineResult detect(BufferedImage before, BufferedImage after) {
        DeepChangeDetector d;
        synchronized (this) {
            d = (state == State.READY) ? detector : null;
        }
        if (d == null) {
            throw new IllegalStateException(DISPLAY_NAME + " is unavailable: " + statusNote
                    + ". No deep-learning output is produced or simulated.");
        }
        long t0 = System.nanoTime();
        boolean[][] mask = d.predictMask(before, after);   // [y][x], true = changed
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return EngineResult.fromMask(ID, DISPLAY_NAME, mask, ms);
    }

    // ------------------------------------------------------------------ accessors

    public State getState() { return state; }

    /** The configured model directory (as given, usually relative to the working directory). */
    public Path getModelDir() { return modelDir; }

    /** Full multi-line {@link DeepChangeDetector#describe()} text, or {@code null} if no model is loaded. */
    public String getDescription() { return description; }

    /** Device the loaded model runs on (e.g. {@code cpu()} or {@code mps(0)}), or {@code null}. */
    public String getDeviceName() { return deviceName; }

    /** Validation-tuned decision threshold of the loaded model, or {@code null}. */
    public Float getThreshold() { return threshold; }

    /** A value from the loaded model's {@code model.properties}, or {@code null} if absent / not loaded. */
    public synchronized String getModelProperty(String key) {
        return detector == null ? null : detector.getConfig().get(key, null);
    }

    /** Releases the native model. Safe to call more than once and while a load is in progress. */
    @Override
    public synchronized void close() {
        state = State.CLOSED;
        statusNote = "model closed";
        if (detector != null) {
            detector.close();
            detector = null;
        }
    }

    @Override
    public String toString() { return DISPLAY_NAME; }
}
