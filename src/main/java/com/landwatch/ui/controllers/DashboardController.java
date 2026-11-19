package com.landwatch.ui.controllers;

import com.landwatch.geospatial.SatelliteImageLoader;
import com.landwatch.ui.ChangeVisualizationSet;
import com.landwatch.ui.ResultExporter;
import com.landwatch.ui.VisualizationMode;
import com.landwatch.ui.engine.ChangeDetectionEngine;
import com.landwatch.ui.engine.ClassicalBaselineEngine;
import com.landwatch.ui.engine.EngineResult;
import com.landwatch.ui.engine.EngineRunner;
import com.landwatch.ui.engine.SiameseUNetEngine;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ListCell;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.Toggle;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.ResourceBundle;
import java.util.stream.Collectors;

/**
 * Controller for the LandWatch AI Dashboard.
 *
 * <p>Flow: the user picks a detection engine, loads a before/after pair and clicks Analyze.
 * {@link #runSelectedEngine} runs the engine and returns an {@link EngineResult} (binary mask
 * {@code [y][x]} + engine name + latency). Every visualisation, statistic, log line and exported file
 * is derived from that mask, so a new engine only has to implement {@link ChangeDetectionEngine}.</p>
 */
public class DashboardController implements Initializable {

    private static final File RESULTS_DIR = new File("results");
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final String DEMO_PRE = "siteA_pre.bmp";
    private static final String DEMO_POST = "siteA_post.bmp";

    // Header & Status
    @FXML private Label systemStatusLabel;
    @FXML private Label engineModeLabel;
    @FXML private ScrollPane contentScrollPane;

    // Before (Date 1) Panel
    @FXML private StackPane beforeImageContainer;
    @FXML private ImageView beforeImageView;
    @FXML private VBox beforePlaceholder;
    @FXML private Label beforeFilenameLabel;
    @FXML private Label beforeDimensionLabel;
    @FXML private Button uploadBeforeButton;

    // After (Date 2) Panel
    @FXML private StackPane afterImageContainer;
    @FXML private ImageView afterImageView;
    @FXML private VBox afterPlaceholder;
    @FXML private Label afterFilenameLabel;
    @FXML private Label afterDimensionLabel;
    @FXML private Button uploadAfterButton;

    // Controls
    @FXML private VBox actionBar;
    @FXML private ComboBox<ChangeDetectionEngine> engineComboBox;
    @FXML private Label engineNoteLabel;
    @FXML private Button analyzeButton;
    @FXML private Button loadDemoButton;
    @FXML private Button resetButton;
    @FXML private ProgressBar analysisProgressBar;

    // Results & Change Map Panel
    @FXML private StackPane changeMapContainer;
    @FXML private ImageView changeMapImageView;
    @FXML private VBox changeMapPlaceholder;
    @FXML private Label changeMapStatusLabel;
    @FXML private Label resultEngineBadge;
    @FXML private ToggleGroup viewModeGroup;
    @FXML private ToggleButton modeMaskToggle;
    @FXML private ToggleButton modeOverlayToggle;
    @FXML private ToggleButton modeChangeMapToggle;
    @FXML private Label legendModeLabel;
    @FXML private FlowPane legendPane;
    @FXML private Label legendCaption;

    // Statistics
    @FXML private Label statEngine;
    @FXML private Label statChangePercent;
    @FXML private Label statChangedPixels;
    @FXML private Label statTotalPixels;
    @FXML private Label statRegions;
    @FXML private Label statOtsu;
    @FXML private Label statOtsuSub;
    @FXML private Label statLatency;
    @FXML private Label statImageSize;
    @FXML private Label statChangeLevel;

    // Activity / Audit Log
    @FXML private TextArea activityLogArea;

    // State
    private File beforeFile = null;
    private File afterFile = null;
    private BufferedImage beforeBufferedImage = null;
    private BufferedImage afterBufferedImage = null;
    private boolean beforeIsSyntheticDemo = false;
    private boolean afterIsSyntheticDemo = false;

    private EngineResult lastResult = null;
    private final Map<VisualizationMode, Image> fxImages = new EnumMap<>(VisualizationMode.class);
    private VisualizationMode currentMode = VisualizationMode.RED_OVERLAY;
    private volatile boolean analysisRunning = false;

    /** Output of one background analysis run. */
    private record AnalysisBundle(ChangeDetectionEngine engine, EngineResult result,
                                  ChangeVisualizationSet visuals, String exportBaseName,
                                  List<File> savedFiles, String saveError) {
    }

    // ------------------------------------------------------------------
    // Initialisation
    // ------------------------------------------------------------------

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        logMessage("LandWatch AI system initialized (Java 17 LTS).");

        setupEngineSelector();
        setupViewModes();
        bindImageWidth(beforeImageContainer, beforeImageView);
        bindImageWidth(afterImageContainer, afterImageView);
        bindImageWidth(changeMapContainer, changeMapImageView);

        systemStatusLabel.setText("System: Standby");
        resetDashboardState();
    }

    /**
     * Engines offered in the selector.
     *
     * <p>HOOK (deep model): {@link SiameseUNetEngine} is a clearly marked placeholder that reports
     * "no trained model loaded" and refuses to run. When {@code com.landwatch.deeplearning.DeepChangeDetector}
     * is merged, load the trained weights there (see the HOOK in SiameseUNetEngine) and it becomes
     * selectable here. No other UI code needs to change.</p>
     */
    static List<ChangeDetectionEngine> createEngines() {
        return List.of(new ClassicalBaselineEngine(), new SiameseUNetEngine());
    }

    private void setupEngineSelector() {
        List<ChangeDetectionEngine> engines = createEngines();
        engineComboBox.getItems().setAll(engines);
        engineComboBox.setCellFactory(list -> new EngineListCell(true));
        engineComboBox.setButtonCell(new EngineListCell(false));
        engineComboBox.setValue(firstAvailableEngine());
        onEngineSelected(engineComboBox.getValue(), false);

        engineComboBox.valueProperty().addListener((obs, oldEngine, newEngine) -> {
            if (newEngine != null && !newEngine.isAvailable()) {
                // Disabled entries cannot be clicked, but keyboard navigation could still reach them.
                logMessage(newEngine.getDisplayName() + " cannot be selected: " + newEngine.getStatusNote() + ".");
                ChangeDetectionEngine fallback = (oldEngine != null && oldEngine.isAvailable())
                        ? oldEngine : firstAvailableEngine();
                Platform.runLater(() -> engineComboBox.setValue(fallback));
                return;
            }
            onEngineSelected(newEngine, true);
        });

        String summary = engines.stream()
                .map(e -> e.getDisplayName() + (e.isAvailable() ? " [ready]" : " [unavailable: " + e.getStatusNote() + "]"))
                .collect(Collectors.joining("; "));
        logMessage("Detection engines: " + summary + ".");
    }

    private ChangeDetectionEngine firstAvailableEngine() {
        return engineComboBox.getItems().stream()
                .filter(ChangeDetectionEngine::isAvailable)
                .findFirst()
                .orElse(null);
    }

    private void onEngineSelected(ChangeDetectionEngine engine, boolean log) {
        if (engine == null) {
            engineModeLabel.setText("Engine: none available");
            engineNoteLabel.setText("");
            return;
        }
        engineModeLabel.setText("Engine: " + engine.getDisplayName());

        StringBuilder note = new StringBuilder(engine.getStatusNote());
        for (ChangeDetectionEngine other : engineComboBox.getItems()) {
            if (!other.isAvailable()) {
                note.append('\n').append(other.getDisplayName()).append(": unavailable, ").append(other.getStatusNote()).append('.');
            }
        }
        engineNoteLabel.setText(note.toString());

        if (log) {
            logMessage("Detection engine selected: " + engine.getDisplayName() + ".");
            if (lastResult != null && !lastResult.getEngineName().equals(engine.getDisplayName())) {
                logMessage("Results on screen were produced by " + lastResult.getEngineName()
                        + ". Click 'Analyze Changes' to run " + engine.getDisplayName() + ".");
            }
        }
    }

    /** Combo-box cell; in the popup, unavailable engines are disabled and show their status note. */
    private static final class EngineListCell extends ListCell<ChangeDetectionEngine> {
        private final boolean inPopup;

        EngineListCell(boolean inPopup) {
            this.inPopup = inPopup;
        }

        @Override
        protected void updateItem(ChangeDetectionEngine engine, boolean empty) {
            super.updateItem(engine, empty);
            getStyleClass().remove("engine-cell-unavailable");
            if (empty || engine == null) {
                setText(null);
                setDisable(false);
                return;
            }
            if (engine.isAvailable()) {
                setText(engine.getDisplayName());
                setDisable(false);
            } else {
                setText(engine.getDisplayName() + "  (" + engine.getStatusNote() + ")");
                setDisable(inPopup);
                getStyleClass().add("engine-cell-unavailable");
            }
        }
    }

    private void setupViewModes() {
        viewModeGroup.selectedToggleProperty().addListener((obs, oldToggle, newToggle) -> {
            if (newToggle == null) {
                // Keep exactly one mode selected (clicking the active toggle would otherwise clear it).
                if (oldToggle != null) {
                    oldToggle.setSelected(true);
                }
                return;
            }
            VisualizationMode mode = modeOf(newToggle);
            if (mode != currentMode) {
                currentMode = mode;
                applyMode(mode, true);
            }
        });
        Toggle selected = viewModeGroup.getSelectedToggle();
        currentMode = (selected != null) ? modeOf(selected) : VisualizationMode.RED_OVERLAY;
        applyMode(currentMode, false);
    }

    private static VisualizationMode modeOf(Toggle toggle) {
        return VisualizationMode.valueOf(String.valueOf(toggle.getUserData()));
    }

    private ToggleButton toggleFor(VisualizationMode mode) {
        switch (mode) {
            case BINARY_MASK: return modeMaskToggle;
            case CHANGE_MAP: return modeChangeMapToggle;
            case RED_OVERLAY:
            default: return modeOverlayToggle;
        }
    }

    /** Keeps an ImageView inside its container when the window is narrow (height stays fixed). */
    private static void bindImageWidth(StackPane container, ImageView view) {
        if (container == null || view == null) {
            return;
        }
        double initialFitWidth = view.getFitWidth();
        container.widthProperty().addListener((obs, oldW, newW) -> {
            double available = newW.doubleValue() - 16;
            if (available > 40) {
                view.setFitWidth(Math.min(initialFitWidth, available));
            }
        });
    }

    // ------------------------------------------------------------------
    // Reset / input loading
    // ------------------------------------------------------------------

    /**
     * Resets the dashboard back to clean unanalyzed state.
     */
    @FXML
    public void handleReset() {
        beforeFile = null;
        afterFile = null;
        beforeBufferedImage = null;
        afterBufferedImage = null;
        beforeIsSyntheticDemo = false;
        afterIsSyntheticDemo = false;
        lastResult = null;
        fxImages.clear();

        beforeImageView.setImage(null);
        beforeImageView.setVisible(false);
        beforePlaceholder.setVisible(true);
        beforeFilenameLabel.setText("No file selected");
        beforeDimensionLabel.setText("Resolution: -- x -- px");

        afterImageView.setImage(null);
        afterImageView.setVisible(false);
        afterPlaceholder.setVisible(true);
        afterFilenameLabel.setText("No file selected");
        afterDimensionLabel.setText("Resolution: -- x -- px");

        changeMapImageView.setImage(null);
        changeMapImageView.setVisible(false);
        changeMapPlaceholder.setVisible(true);
        changeMapStatusLabel.setText("Awaiting analysis: load a before/after pair and click 'Analyze Changes'");
        resultEngineBadge.setText("NO RESULT YET");
        setModeTogglesDisabled(true);

        statEngine.setText("-- (no analysis run yet)");
        statChangePercent.setText("--");
        statChangedPixels.setText("--");
        statTotalPixels.setText("of -- px");
        statRegions.setText("--");
        statOtsu.setText("--");
        statOtsuSub.setText("on |ΔY|, baseline only");
        statLatency.setText("--");
        statImageSize.setText("--");
        setChangeLevel(null);

        analysisProgressBar.setVisible(false);
        analyzeButton.setDisable(true);
        systemStatusLabel.setText("System: Standby");

        logMessage("Dashboard reset to initial standby state.");
    }

    private void resetDashboardState() {
        handleReset();
    }

    /**
     * Opens file chooser to upload Date 1 (Before) image.
     */
    @FXML
    public void handleUploadBefore() {
        File file = chooseImageFile("Select Pre-Event (Before) Image");
        if (file != null) {
            try {
                BufferedImage buf = SatelliteImageLoader.load(file);
                beforeFile = file;
                beforeBufferedImage = buf;
                beforeIsSyntheticDemo = false;
                showInput(beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                        buf, file.getName(), false);
                logMessage("Loaded Before image: " + file.getName() + " [" + buf.getWidth() + "x" + buf.getHeight() + "]");
                checkReadiness();
            } catch (Exception e) {
                showError("File Error", "Could not load image: " + e.getMessage());
            }
        }
    }

    /**
     * Opens file chooser to upload Date 2 (After) image.
     */
    @FXML
    public void handleUploadAfter() {
        File file = chooseImageFile("Select Post-Event (After) Image");
        if (file != null) {
            try {
                BufferedImage buf = SatelliteImageLoader.load(file);
                afterFile = file;
                afterBufferedImage = buf;
                afterIsSyntheticDemo = false;
                showInput(afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                        buf, file.getName(), false);
                logMessage("Loaded After image: " + file.getName() + " [" + buf.getWidth() + "x" + buf.getHeight() + "]");
                checkReadiness();
            } catch (Exception e) {
                showError("File Error", "Could not load image: " + e.getMessage());
            }
        }
    }

    /**
     * One-click loader for the bundled demo pair. The pair is a SYNTHETIC, computer-generated scene
     * (not real satellite imagery); it only exercises the pipeline end to end.
     */
    @FXML
    public void handleLoadDemoPair() {
        try {
            URL preUrl = getClass().getResource("/com/landwatch/sample_data/" + DEMO_PRE);
            URL postUrl = getClass().getResource("/com/landwatch/sample_data/" + DEMO_POST);

            if (preUrl == null || postUrl == null) {
                showError("Demo Data Missing", "Could not locate the bundled synthetic demo pair.");
                return;
            }
            beforeBufferedImage = SatelliteImageLoader.load(preUrl);
            afterBufferedImage = SatelliteImageLoader.load(postUrl);
            beforeFile = null;
            afterFile = null;
            beforeIsSyntheticDemo = true;
            afterIsSyntheticDemo = true;

            showInput(beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                    beforeBufferedImage, DEMO_PRE, true);
            showInput(afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                    afterBufferedImage, DEMO_POST, true);

            logMessage("Loaded the bundled SYNTHETIC demonstration pair (Site A: computer-generated test scene, "
                    + "not real satellite imagery). It only demonstrates the pipeline.");
            checkReadiness();
        } catch (Exception e) {
            showError("Demo Load Error", "Failed to load demo pair: " + e.getMessage());
        }
    }

    private static void showInput(ImageView view, VBox placeholder, Label nameLabel, Label dimLabel,
                                  BufferedImage image, String name, boolean syntheticDemo) {
        view.setImage(SwingFXUtils.toFXImage(image, null));
        view.setVisible(true);
        placeholder.setVisible(false);
        nameLabel.setText(syntheticDemo ? name + " (synthetic demo)" : name);
        String dims = String.format("Resolution: %d x %d px", image.getWidth(), image.getHeight());
        dimLabel.setText(syntheticDemo ? dims + " · synthetic, not satellite data" : dims);
    }

    /** Honest one-line description of the current inputs, written to the log and run summary. */
    private String describeInputs() {
        if (beforeIsSyntheticDemo && afterIsSyntheticDemo) {
            return "bundled SYNTHETIC demo pair (" + DEMO_PRE + " / " + DEMO_POST
                    + "): computer-generated test scene, not real satellite imagery";
        }
        String pre = beforeIsSyntheticDemo ? DEMO_PRE + " (synthetic demo)" : nameOf(beforeFile);
        String post = afterIsSyntheticDemo ? DEMO_POST + " (synthetic demo)" : nameOf(afterFile);
        return "before: " + pre + ", after: " + post;
    }

    private static String nameOf(File f) {
        return f != null ? f.getName() : "unknown";
    }

    /**
     * Validates whether both images are uploaded and dimensions align.
     */
    private void checkReadiness() {
        if (beforeBufferedImage != null && afterBufferedImage != null) {
            if (beforeBufferedImage.getWidth() != afterBufferedImage.getWidth() ||
                beforeBufferedImage.getHeight() != afterBufferedImage.getHeight()) {
                logMessage("WARNING: Spatial dimensions differ (" +
                        beforeBufferedImage.getWidth() + "x" + beforeBufferedImage.getHeight() +
                        " vs " + afterBufferedImage.getWidth() + "x" + afterBufferedImage.getHeight() +
                        "). Images must be aligned to run change detection.");
                analyzeButton.setDisable(true);
                systemStatusLabel.setText("System: Dimension Mismatch");
            } else {
                analyzeButton.setDisable(analysisRunning);
                systemStatusLabel.setText("System: Ready for Analysis");
                logMessage("Image dimensions verified matching: " + beforeBufferedImage.getWidth() + "x" + beforeBufferedImage.getHeight() + " px.");
            }
        } else {
            analyzeButton.setDisable(true);
        }
    }

    // ------------------------------------------------------------------
    // Analysis
    // ------------------------------------------------------------------

    /**
     * Runs the selected engine in a background Task, builds the three visualisations from the
     * returned mask and saves everything to results/.
     */
    @FXML
    public void handleAnalyzeChanges() {
        if (beforeBufferedImage == null || afterBufferedImage == null) {
            showError("Missing Inputs", "Please load both pre-event and post-event images before starting analysis.");
            return;
        }
        ChangeDetectionEngine engine = engineComboBox.getValue();
        if (engine == null || !engine.isAvailable()) {
            showError("Engine Unavailable", "The selected detection engine cannot run"
                    + (engine != null ? ": " + engine.getStatusNote() : "") + ".");
            return;
        }

        final BufferedImage before = beforeBufferedImage;
        final BufferedImage after = afterBufferedImage;
        final String inputDescription = describeInputs();

        logMessage("Running detection engine: " + engine.getDisplayName() + " on " + inputDescription + " ...");
        setAnalysisRunning(true);
        systemStatusLabel.setText("System: Processing...");

        Task<AnalysisBundle> detectionTask = new Task<>() {
            @Override
            protected AnalysisBundle call() throws Exception {
                EngineResult result = runSelectedEngine(engine, before, after);
                ChangeVisualizationSet visuals = ChangeVisualizationSet.build(result, after);

                LocalDateTime now = LocalDateTime.now();
                String baseName = ResultExporter.uniqueBaseName(RESULTS_DIR, result, now);
                List<File> saved = List.of();
                String saveError = null;
                try {
                    saved = ResultExporter.export(RESULTS_DIR, baseName, result, visuals, inputDescription, now);
                } catch (Exception e) {
                    saveError = e.getMessage();
                }
                return new AnalysisBundle(engine, result, visuals, baseName, saved, saveError);
            }
        };

        detectionTask.setOnSucceeded(event -> displayResults(detectionTask.getValue()));

        detectionTask.setOnFailed(event -> {
            Throwable error = detectionTask.getException();
            logMessage("ERROR during change detection (" + engine.getDisplayName() + "): " + error.getMessage());
            systemStatusLabel.setText("System: Analysis Failed");
            setAnalysisRunning(false);
            showError("Analysis Failure", "Error during change detection: " + error.getMessage());
        });

        Thread thread = new Thread(detectionTask, "landwatch-analysis");
        thread.setDaemon(true);
        thread.start();
    }

    /**
     * The single entry point that runs a detection engine. Returns the engine's binary mask together
     * with the engine name and latency; it never substitutes or simulates output.
     */
    public static EngineResult runSelectedEngine(ChangeDetectionEngine engine, BufferedImage before,
                                                 BufferedImage after) throws Exception {
        return EngineRunner.run(engine, before, after);
    }

    private void displayResults(AnalysisBundle bundle) {
        EngineResult r = bundle.result();
        lastResult = r;

        fxImages.clear();
        for (VisualizationMode mode : VisualizationMode.values()) {
            fxImages.put(mode, SwingFXUtils.toFXImage(bundle.visuals().get(mode), null));
        }
        changeMapImageView.setImage(fxImages.get(currentMode));
        changeMapImageView.setVisible(true);
        changeMapPlaceholder.setVisible(false);
        setModeTogglesDisabled(false);
        resultEngineBadge.setText(bundle.engine().getShortLabel());

        updateMetrics(r);

        logMessage(String.format("[Engine] %s produced a %d x %d change mask in %d ms.",
                r.getEngineName(), r.getWidth(), r.getHeight(), r.getLatencyMs()));
        if (r.getOtsuThreshold() != null) {
            String raw = (r.getRawChangedPixels() != null)
                    ? String.format("; %,d raw changed px -> %,d after 3x3 opening and region filter",
                    r.getRawChangedPixels(), r.getChangedPixels())
                    : "";
            logMessage(String.format("[Baseline] Otsu threshold on |ΔY| luminance difference: %d / 255%s.",
                    r.getOtsuThreshold(), raw));
        }
        logMessage(String.format("[Result] Changed: %,d of %,d px (%.2f %%) in %d connected region(s). "
                        + "Change level (heuristic): %s.",
                r.getChangedPixels(), r.getTotalPixels(), r.getChangePercentage(), r.getRegionCount(),
                r.getChangeLevel().getLabel()));
        if (bundle.saveError() == null) {
            String parts = bundle.savedFiles().stream()
                    .map(f -> f.getName().startsWith(bundle.exportBaseName() + "_")
                            ? f.getName().substring(bundle.exportBaseName().length() + 1) : f.getName())
                    .collect(Collectors.joining(", "));
            logMessage("[Export] Saved to " + RESULTS_DIR.getPath() + "/" + bundle.exportBaseName() + "_*: " + parts);
        } else {
            logMessage("WARNING: Results could not be saved to " + RESULTS_DIR.getPath() + "/: " + bundle.saveError());
        }

        systemStatusLabel.setText("System: Analysis Complete");
        setAnalysisRunning(false);
        scrollToResults();
    }

    private void updateMetrics(EngineResult r) {
        statEngine.setText(r.getEngineName());
        statChangePercent.setText(String.format("%.2f %%", r.getChangePercentage()));
        statChangedPixels.setText(String.format("%,d px", r.getChangedPixels()));
        statTotalPixels.setText(String.format("of %,d px", r.getTotalPixels()));
        statRegions.setText(String.valueOf(r.getRegionCount()));
        if (r.getOtsuThreshold() != null) {
            statOtsu.setText(r.getOtsuThreshold() + " / 255");
            statOtsuSub.setText("on |ΔY| luminance diff.");
        } else {
            statOtsu.setText("n/a");
            statOtsuSub.setText("not used by this engine");
        }
        statLatency.setText(r.getLatencyMs() + " ms");
        statImageSize.setText(r.getWidth() + " × " + r.getHeight() + " px");
        setChangeLevel(r.getChangeLevel());
    }

    private void setChangeLevel(EngineResult.ChangeLevel level) {
        for (EngineResult.ChangeLevel l : EngineResult.ChangeLevel.values()) {
            statChangeLevel.getStyleClass().remove(l.getStyleClass());
        }
        if (level == null) {
            statChangeLevel.setText("Awaiting Data");
        } else {
            statChangeLevel.setText(level.getLabel());
            statChangeLevel.getStyleClass().add(level.getStyleClass());
        }
    }

    private void setAnalysisRunning(boolean running) {
        analysisRunning = running;
        analysisProgressBar.setVisible(running);
        analyzeButton.setDisable(running || beforeBufferedImage == null || afterBufferedImage == null);
        engineComboBox.setDisable(running);
    }

    private void setModeTogglesDisabled(boolean disabled) {
        modeMaskToggle.setDisable(disabled);
        modeOverlayToggle.setDisable(disabled);
        modeChangeMapToggle.setDisable(disabled);
    }

    /** Scrolls the page so the action bar and results are visible after an analysis. */
    public void scrollToResults() {
        if (contentScrollPane == null || actionBar == null || contentScrollPane.getScene() == null) {
            return;
        }
        Parent root = contentScrollPane.getScene().getRoot();
        root.applyCss();
        root.layout();
        Node content = contentScrollPane.getContent();
        double scrollable = content.getLayoutBounds().getHeight() - contentScrollPane.getViewportBounds().getHeight();
        if (scrollable <= 0) {
            return;
        }
        double target = actionBar.getBoundsInParent().getMinY() - 10;
        double fraction = Math.max(0, Math.min(1, target / scrollable));
        contentScrollPane.setVvalue(contentScrollPane.getVmin()
                + fraction * (contentScrollPane.getVmax() - contentScrollPane.getVmin()));
    }

    // ------------------------------------------------------------------
    // View modes & legend
    // ------------------------------------------------------------------

    private void applyMode(VisualizationMode mode, boolean log) {
        updateLegend(mode);
        Image image = fxImages.get(mode);
        if (image != null) {
            changeMapImageView.setImage(image);
            if (log) {
                logMessage("Change view switched to: " + mode.getDisplayName() + ".");
            }
        }
    }

    private void updateLegend(VisualizationMode mode) {
        legendModeLabel.setText(mode.getDisplayName());
        legendCaption.setText(mode.getDescription());
        legendPane.getChildren().clear();
        for (VisualizationMode.LegendEntry entry : mode.getLegend()) {
            Region swatch = new Region();
            swatch.getStyleClass().add("legend-swatch");
            swatch.setMinSize(14, 14);
            swatch.setPrefSize(14, 14);
            swatch.setMaxSize(14, 14);
            switch (entry.style()) {
                case OUTLINE:
                    swatch.setStyle("-fx-background-color: transparent; -fx-border-color: "
                            + entry.cssColor() + "; -fx-border-width: 2;");
                    break;
                case TEXTURE:
                    swatch.getStyleClass().add("legend-swatch-texture");
                    break;
                case FILL:
                default:
                    swatch.setStyle("-fx-background-color: " + entry.cssColor() + ";");
                    break;
            }
            Label text = new Label(entry.label());
            text.getStyleClass().add("legend-text");
            text.setMinWidth(Region.USE_PREF_SIZE);
            HBox item = new HBox(6, swatch, text);
            item.setAlignment(Pos.CENTER_LEFT);
            legendPane.getChildren().add(item);
        }
    }

    // ------------------------------------------------------------------
    // Accessors used by the snapshot tool / future UI
    // ------------------------------------------------------------------

    public boolean isAnalysisRunning() {
        return analysisRunning;
    }

    public EngineResult getLastEngineResult() {
        return lastResult;
    }

    public VisualizationMode getCurrentMode() {
        return currentMode;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private File chooseImageFile(String title) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle(title);
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Images (*.png, *.jpg, *.tif, *.bmp)", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"),
                new FileChooser.ExtensionFilter("All Files", "*.*")
        );

        Stage stage = (Stage) analyzeButton.getScene().getWindow();
        return fileChooser.showOpenDialog(stage);
    }

    private void logMessage(String message) {
        String timestamp = LocalDateTime.now().format(TIME_FORMATTER);
        activityLogArea.appendText("[" + timestamp + "] " + message + "\n");
    }

    private void showError(String title, String content) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }
}
