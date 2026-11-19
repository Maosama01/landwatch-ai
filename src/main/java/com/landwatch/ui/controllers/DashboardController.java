package com.landwatch.ui.controllers;

import com.landwatch.deeplearning.DeepChangeDetector;
import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.geospatial.SatelliteImageLoader;
import com.landwatch.ui.ChangeView;
import com.landwatch.ui.ChangeVisualizationSet;
import com.landwatch.ui.GroundTruthComparison;
import com.landwatch.ui.GroundTruthView;
import com.landwatch.ui.ResultExporter;
import com.landwatch.ui.VisualizationMode;
import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.data.LevirPair;
import com.landwatch.ui.data.MetricsReport;
import com.landwatch.ui.engine.ChangeDetectionEngine;
import com.landwatch.ui.engine.ClassicalBaselineEngine;
import com.landwatch.ui.engine.EngineResult;
import com.landwatch.ui.engine.EngineRunner;
import com.landwatch.ui.engine.SiameseUNetEngine;
import com.landwatch.ui.pages.AboutPage;
import com.landwatch.ui.pages.BenchmarksPage;
import com.landwatch.ui.pages.FailureAnalysisPage;
import com.landwatch.ui.pages.FundamentalsPage;
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
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.ResourceBundle;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Controller for the LandWatch AI Dashboard.
 *
 * <p>Flow: the user picks a detection engine, loads a before/after pair and clicks Analyze.
 * {@link #runSelectedEngine} runs the engine and returns an {@link EngineResult} (binary mask
 * {@code [y][x]} + engine name + latency). Every visualisation, statistic, log line and exported file
 * is derived from that mask, so a new engine only has to implement {@link ChangeDetectionEngine}.</p>
 *
 * <p>When the inputs are a LEVIR-CD pair loaded with "Load LEVIR-CD Pair…", the pair's ground-truth
 * label is kept as well: the analysis then also renders the label and a TP/FP/FN error map and computes
 * per-image metrics with {@link BinaryChangeMetrics}. The deep model is loaded asynchronously at start-up
 * and becomes selectable once ready. The sidebar switches between the analysis view and the
 * Benchmarks, NN Fundamentals, Failure Analysis and About pages.</p>
 */
public class DashboardController implements Initializable {

    private static final File RESULTS_DIR = new File("results");
    private static final DateTimeFormatter TIME_FORMATTER = DateTimeFormatter.ofPattern("HH:mm:ss");
    private static final String DEMO_PRE = "siteA_pre.bmp";
    private static final String DEMO_POST = "siteA_post.bmp";

    // Navigation
    @FXML private StackPane pageStack;
    @FXML private Button navAnalysisButton;
    @FXML private Button navBenchmarksButton;
    @FXML private Button navFundamentalsButton;
    @FXML private Button navFailureButton;
    @FXML private Button navAboutButton;

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
    @FXML private Button loadLevirButton;
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
    @FXML private ToggleButton modeGroundTruthToggle;
    @FXML private ToggleButton modeErrorsToggle;
    @FXML private Label gtViewNote;
    @FXML private Label legendModeLabel;
    @FXML private FlowPane legendPane;
    @FXML private Label legendCaption;

    // Statistics
    @FXML private Label statEngine;
    @FXML private Label statInput;
    @FXML private Label statChangePercent;
    @FXML private Label statChangedPixels;
    @FXML private Label statTotalPixels;
    @FXML private Label statRegions;
    @FXML private Label statOtsu;
    @FXML private Label statOtsuSub;
    @FXML private Label statLatency;
    @FXML private Label statImageSize;
    @FXML private Label statChangeLevel;
    @FXML private Label statAreaM2;

    // Per-image ground-truth comparison
    @FXML private VBox gtBlock;
    @FXML private GridPane gtTiles;
    @FXML private Label gtF1;
    @FXML private Label gtIoU;
    @FXML private Label gtPrecision;
    @FXML private Label gtRecall;
    @FXML private Label gtCounts;
    @FXML private Label gtBenchmark;
    @FXML private Label gtNote;

    // Activity / Audit Log
    @FXML private TextArea activityLogArea;

    // State
    private File beforeFile = null;
    private File afterFile = null;
    private BufferedImage beforeBufferedImage = null;
    private BufferedImage afterBufferedImage = null;
    private boolean beforeIsSyntheticDemo = false;
    private boolean afterIsSyntheticDemo = false;
    /** The LEVIR-CD pair the current inputs came from, or null. */
    private LevirPair levirPair = null;
    /** Ground-truth label of the current inputs ({@code [y][x]}), or null if there is none. */
    private boolean[][] groundTruthMask = null;

    private EngineResult lastResult = null;
    private AnalysisBundle lastBundle = null;
    private final Map<ChangeView, Image> fxImages = new HashMap<>();
    private ChangeView currentView = VisualizationMode.RED_OVERLAY;
    private volatile boolean analysisRunning = false;
    private final AtomicInteger completedAnalyses = new AtomicInteger();

    private SiameseUNetEngine deepEngine;

    // Pages
    private BenchmarksPage benchmarksPage;
    private FundamentalsPage fundamentalsPage;
    private FailureAnalysisPage failureAnalysisPage;
    private AboutPage aboutPage;
    private Button activeNavButton;

    /** Output of one background analysis run. */
    private record AnalysisBundle(ChangeDetectionEngine engine, EngineResult result,
                                  ChangeVisualizationSet visuals, String exportBaseName,
                                  List<File> savedFiles, String saveError,
                                  String inputShortName, boolean syntheticInput, LevirPair levirPair,
                                  GroundTruthComparison comparison,
                                  BufferedImage groundTruthImage, BufferedImage errorsImage) {
        boolean hasGroundTruth() {
            return comparison != null;
        }
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
        bindImageHeight(changeMapContainer, changeMapImageView);
        setupGroundTruthTiles();
        setupPages();

        systemStatusLabel.setText("System: Standby");
        resetDashboardState();
        startDeepModelLoad();
    }

    /**
     * Engines offered in the selector: the classical baseline (always available) and the deep model,
     * which starts unloaded and is loaded in the background by {@link #startDeepModelLoad()}.
     */
    static List<ChangeDetectionEngine> createEngines() {
        return List.of(new ClassicalBaselineEngine(), new SiameseUNetEngine());
    }

    private void setupEngineSelector() {
        List<ChangeDetectionEngine> engines = createEngines();
        for (ChangeDetectionEngine e : engines) {
            if (e instanceof SiameseUNetEngine) {
                deepEngine = (SiameseUNetEngine) e;
            }
        }
        engineComboBox.getItems().setAll(engines);
        installEngineCells();
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
    }

    private void installEngineCells() {
        engineComboBox.setCellFactory(list -> new EngineListCell(true));
        engineComboBox.setButtonCell(new EngineListCell(false));
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
        updateEngineNote();

        if (log) {
            logMessage("Detection engine selected: " + engine.getDisplayName() + ".");
            if (lastResult != null && !lastResult.getEngineName().equals(engine.getDisplayName())) {
                logMessage("Results on screen were produced by " + lastResult.getEngineName()
                        + ". Click 'Analyze Changes' to run " + engine.getDisplayName() + ".");
            }
        }
    }

    /** Note next to the selector: the selected engine's description plus the state of every other engine. */
    private void updateEngineNote() {
        ChangeDetectionEngine engine = engineComboBox.getValue();
        StringBuilder note = new StringBuilder(engine != null ? engine.getStatusNote() : "");
        for (ChangeDetectionEngine other : engineComboBox.getItems()) {
            if (other == engine) {
                continue;
            }
            note.append(note.length() > 0 ? "\n" : "").append(other.getDisplayName());
            if (other.isLoading()) {
                note.append(": loading model in the background…");
            } else if (!other.isAvailable()) {
                note.append(": unavailable, ").append(other.getStatusNote()).append('.');
            } else {
                note.append(": ready (select it above).");
            }
        }
        engineNoteLabel.setText(note.toString());
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
                setText(engine.getDisplayName() + "  (" + (engine.isLoading() ? SiameseUNetEngine.LOADING_NOTE
                        : engine.getStatusNote()) + ")");
                setDisable(inPopup);
                getStyleClass().add("engine-cell-unavailable");
            }
        }
    }

    // ------------------------------------------------------------------
    // Deep model: asynchronous loading and shutdown
    // ------------------------------------------------------------------

    /** Loads the deep model on a background thread so start-up never blocks. */
    private void startDeepModelLoad() {
        if (deepEngine == null) {
            return;
        }
        deepEngine.markLoading();
        installEngineCells();
        updateEngineNote();
        String device = System.getProperty(DeepChangeDetector.DEVICE_PROPERTY, "auto");
        logMessage("Loading the deep model from " + deepEngine.getModelDir() + " in the background (device: "
                + device + "). The classical baseline is ready meanwhile.");

        Task<Boolean> loadTask = new Task<>() {
            @Override
            protected Boolean call() {
                return deepEngine.load();
            }
        };
        loadTask.setOnSucceeded(e -> onDeepModelLoadFinished());
        loadTask.setOnFailed(e -> {
            Throwable t = loadTask.getException();
            logMessage("ERROR while loading the deep model: " + t);
            onDeepModelLoadFinished();
        });
        Thread thread = new Thread(loadTask, "landwatch-model-load");
        thread.setDaemon(true);
        thread.start();
    }

    private void onDeepModelLoadFinished() {
        installEngineCells();
        if (engineComboBox.getValue() == null) {
            engineComboBox.setValue(firstAvailableEngine());
        }
        updateEngineNote();
        if (deepEngine.isAvailable()) {
            logMessage("Deep model ready: " + deepEngine.getStatusNote() + ".");
            String description = deepEngine.getDescription();
            if (description != null) {
                for (String line : description.split("\n")) {
                    logMessage("[Model] " + line.strip());
                }
            }
            logMessage("Select '" + deepEngine.getDisplayName() + "' in the engine selector to use it.");
        } else {
            logMessage("Deep model unavailable: " + deepEngine.getStatusNote()
                    + ". Only the classical baseline can run (set -D" + SiameseUNetEngine.MODEL_DIR_PROPERTY
                    + "=<dir> to use another model directory).");
        }
        refreshVisiblePage();
    }

    /** The deep engine (for the snapshot tool and pages). */
    public SiameseUNetEngine getDeepEngine() {
        return deepEngine;
    }

    /** Releases native resources (the deep model). Called from MainApp#stop. */
    public void shutdown() {
        if (deepEngine != null) {
            deepEngine.close();
        }
    }

    // ------------------------------------------------------------------
    // Pages & navigation
    // ------------------------------------------------------------------

    private void setupPages() {
        benchmarksPage = new BenchmarksPage(deepEngine);
        fundamentalsPage = new FundamentalsPage(this::logMessage);
        failureAnalysisPage = new FailureAnalysisPage(this::openLevirPairFromPage);
        aboutPage = new AboutPage(deepEngine);
        for (Node page : List.of(benchmarksPage.getRoot(), fundamentalsPage.getRoot(),
                failureAnalysisPage.getRoot(), aboutPage.getRoot())) {
            page.setVisible(false);
            pageStack.getChildren().add(page);
        }
        activeNavButton = navAnalysisButton;
    }

    @FXML public void handleNavAnalysis() { showPage(contentScrollPane, navAnalysisButton); }
    @FXML public void handleNavBenchmarks() { benchmarksPage.refresh(); showPage(benchmarksPage.getRoot(), navBenchmarksButton); }
    @FXML public void handleNavFundamentals() { showPage(fundamentalsPage.getRoot(), navFundamentalsButton); }
    @FXML public void handleNavFailure() { failureAnalysisPage.refresh(); showPage(failureAnalysisPage.getRoot(), navFailureButton); }
    @FXML public void handleNavAbout() { aboutPage.refresh(); showPage(aboutPage.getRoot(), navAboutButton); }

    private void showPage(Node page, Button navButton) {
        for (Node child : pageStack.getChildren()) {
            child.setVisible(child == page);
        }
        if (activeNavButton != null) {
            activeNavButton.getStyleClass().remove("nav-button-active");
        }
        if (!navButton.getStyleClass().contains("nav-button-active")) {
            navButton.getStyleClass().add("nav-button-active");
        }
        activeNavButton = navButton;
    }

    private void refreshVisiblePage() {
        if (benchmarksPage.getRoot().isVisible()) {
            benchmarksPage.refresh();
        } else if (aboutPage.getRoot().isVisible()) {
            aboutPage.refresh();
        }
    }

    /** Failure Analysis "Open pair": load the pair and switch to the analysis view. */
    private void openLevirPairFromPage(Path picked) {
        handleNavAnalysis();
        loadLevirPair(picked);
    }

    // ------------------------------------------------------------------
    // View modes
    // ------------------------------------------------------------------

    private void setupViewModes() {
        viewModeGroup.selectedToggleProperty().addListener((obs, oldToggle, newToggle) -> {
            if (newToggle == null) {
                // Keep exactly one mode selected (clicking the active toggle would otherwise clear it).
                if (oldToggle != null) {
                    oldToggle.setSelected(true);
                }
                return;
            }
            ChangeView view = viewOf(newToggle);
            if (view != currentView) {
                currentView = view;
                applyView(view, true);
            }
        });
        Toggle selected = viewModeGroup.getSelectedToggle();
        currentView = (selected != null) ? viewOf(selected) : VisualizationMode.RED_OVERLAY;
        applyView(currentView, false);
    }

    private static ChangeView viewOf(Toggle toggle) {
        String key = String.valueOf(toggle.getUserData());
        for (GroundTruthView v : GroundTruthView.values()) {
            if (v.name().equals(key)) {
                return v;
            }
        }
        return VisualizationMode.valueOf(key);
    }

    private ToggleButton toggleFor(ChangeView view) {
        if (view == GroundTruthView.GROUND_TRUTH) return modeGroundTruthToggle;
        if (view == GroundTruthView.ERRORS) return modeErrorsToggle;
        if (view == VisualizationMode.BINARY_MASK) return modeMaskToggle;
        if (view == VisualizationMode.CHANGE_MAP) return modeChangeMapToggle;
        return modeOverlayToggle;
    }

    /** Selects a view programmatically (used by the snapshot tool); ignored if its toggle is disabled. */
    public void selectView(ChangeView view) {
        ToggleButton toggle = toggleFor(view);
        if (!toggle.isDisabled()) {
            viewModeGroup.selectToggle(toggle);
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

    /** Lets the change map use the extra height when its card is stretched (the container's pref height is fixed). */
    private static void bindImageHeight(StackPane container, ImageView view) {
        double initialFitHeight = view.getFitHeight();
        container.heightProperty().addListener((obs, oldH, newH) -> {
            double available = newH.doubleValue() - 16;
            view.setFitHeight(Math.max(initialFitHeight, available));
        });
    }

    /** Minimum width of one per-image metric tile before the 4-in-a-row layout switches to 2 x 2. */
    private static final double GT_TILE_MIN_WIDTH = 76;

    /** The four per-image metric tiles sit in one row when there is room, otherwise in a 2 x 2 grid. */
    private void setupGroundTruthTiles() {
        layoutGroundTruthTiles(true);
        gtTiles.widthProperty().addListener((obs, oldW, newW) -> {
            boolean oneRow = newW.doubleValue() >= 4 * GT_TILE_MIN_WIDTH + 3 * gtTiles.getHgap();
            if (oneRow != (gtTiles.getColumnConstraints().size() == 4)) {
                layoutGroundTruthTiles(oneRow);
            }
        });
    }

    private void layoutGroundTruthTiles(boolean oneRow) {
        int columns = oneRow ? 4 : 2;
        gtTiles.getColumnConstraints().clear();
        for (int c = 0; c < columns; c++) {
            ColumnConstraints cc = new ColumnConstraints();
            cc.setPercentWidth(100.0 / columns);
            gtTiles.getColumnConstraints().add(cc);
        }
        List<Node> tiles = List.copyOf(gtTiles.getChildren());
        for (int i = 0; i < tiles.size(); i++) {
            GridPane.setColumnIndex(tiles.get(i), i % columns);
            GridPane.setRowIndex(tiles.get(i), i / columns);
        }
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
        levirPair = null;
        groundTruthMask = null;
        lastResult = null;
        lastBundle = null;
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
        updateGroundTruthControls(null);

        statEngine.setText("-- (no analysis run yet)");
        statInput.setText("");
        statChangePercent.setText("--");
        statChangedPixels.setText("--");
        statTotalPixels.setText("of -- px");
        statRegions.setText("--");
        statOtsu.setText("--");
        statOtsuSub.setText("on |ΔY|, baseline only");
        statLatency.setText("--");
        statImageSize.setText("--");
        setChangeLevel(null);
        showAreaM2(null);

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
        File file = chooseImageFile("Select Pre-Event (Before) Image", null);
        if (file != null) {
            try {
                BufferedImage buf = SatelliteImageLoader.load(file);
                beforeFile = file;
                beforeBufferedImage = buf;
                beforeIsSyntheticDemo = false;
                dropGroundTruth();
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
        File file = chooseImageFile("Select Post-Event (After) Image", null);
        if (file != null) {
            try {
                BufferedImage buf = SatelliteImageLoader.load(file);
                afterFile = file;
                afterBufferedImage = buf;
                afterIsSyntheticDemo = false;
                dropGroundTruth();
                showInput(afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                        buf, file.getName(), false);
                logMessage("Loaded After image: " + file.getName() + " [" + buf.getWidth() + "x" + buf.getHeight() + "]");
                checkReadiness();
            } catch (Exception e) {
                showError("File Error", "Could not load image: " + e.getMessage());
            }
        }
    }

    /** Inputs changed individually, so the LEVIR-CD label no longer belongs to them. */
    private void dropGroundTruth() {
        if (levirPair != null) {
            logMessage("Ground-truth label of " + levirPair.shortName()
                    + " dropped: the inputs no longer form that LEVIR-CD pair.");
        }
        levirPair = null;
        groundTruthMask = null;
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
            dropGroundTruth();

            showInput(beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                    beforeBufferedImage, DEMO_PRE, true);
            showInput(afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                    afterBufferedImage, DEMO_POST, true);

            logMessage("Loaded the bundled SYNTHETIC demonstration pair (Site A: computer-generated test scene, "
                    + "not real satellite imagery). It only demonstrates the pipeline and has no ground-truth label.");
            checkReadiness();
        } catch (Exception e) {
            showError("Demo Load Error", "Failed to load demo pair: " + e.getMessage());
        }
    }

    /**
     * "Load LEVIR-CD Pair…": the user picks an image in a split's A/ or B/ folder; A, B and label/ with
     * the same file name are loaded. The chooser starts in datasets/LEVIR-CD/test/A when it exists.
     */
    @FXML
    public void handleLoadLevirPair() {
        Path start = AppPaths.levirChooserStartDir();
        File file = chooseImageFile("Select an image in a LEVIR-CD split's A/ or B/ folder",
                start != null ? start.toFile() : null);
        if (file != null) {
            loadLevirPair(file.toPath());
        }
    }

    /**
     * Loads the LEVIR-CD pair that {@code picked} (a file in {@code <split>/A}, {@code /B} or {@code /label})
     * belongs to, including its ground-truth label when present.
     *
     * @return true if the pair was loaded
     */
    public boolean loadLevirPair(Path picked) {
        LevirPair.Loaded loaded;
        try {
            loaded = LevirPair.locate(picked).load();
        } catch (Exception e) {
            logMessage("Could not load a LEVIR-CD pair from " + picked + ": " + e.getMessage());
            showError("LEVIR-CD Pair", e.getMessage());
            return false;
        }
        LevirPair pair = loaded.pair();
        beforeBufferedImage = loaded.pre();
        afterBufferedImage = loaded.post();
        beforeFile = pair.pre().toFile();
        afterFile = pair.post().toFile();
        beforeIsSyntheticDemo = false;
        afterIsSyntheticDemo = false;
        levirPair = pair;
        groundTruthMask = loaded.groundTruth();

        showInput(beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                beforeBufferedImage, pair.name(), false);
        showInput(afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                afterBufferedImage, pair.name(), false);
        beforeFilenameLabel.setText(pair.name() + "  (LEVIR-CD " + pair.split() + "/A)");
        afterFilenameLabel.setText(pair.name() + "  (LEVIR-CD " + pair.split() + "/B)");
        String dims = String.format(Locale.ROOT, "Resolution: %d x %d px · real imagery, 0.5 m/px (LEVIR-CD)",
                beforeBufferedImage.getWidth(), beforeBufferedImage.getHeight());
        beforeDimensionLabel.setText(dims);
        afterDimensionLabel.setText(dims);

        if (groundTruthMask != null) {
            long changed = countTrue(groundTruthMask);
            long total = (long) beforeBufferedImage.getWidth() * beforeBufferedImage.getHeight();
            logMessage(String.format(Locale.ROOT,
                    "Loaded %s: A, B and ground-truth label (%,d labelled change px = %.2f %%). Real imagery at "
                            + "0.5 m/px; the label marks building changes only.",
                    pair.shortName(), changed, changed * 100.0 / total));
        } else {
            logMessage("Loaded " + pair.shortName() + " (A and B). No label/" + pair.name()
                    + " found, so no ground-truth comparison is possible for this pair.");
        }
        checkReadiness();
        return true;
    }

    private static long countTrue(boolean[][] mask) {
        long n = 0;
        for (boolean[] row : mask) {
            for (boolean b : row) {
                if (b) n++;
            }
        }
        return n;
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
        if (levirPair != null) {
            return levirPair.shortName() + " (A/B, real imagery at 0.5 m/px"
                    + (groundTruthMask != null ? "; label = building changes only)" : "; no label)");
        }
        String pre = beforeIsSyntheticDemo ? DEMO_PRE + " (synthetic demo)" : nameOf(beforeFile);
        String post = afterIsSyntheticDemo ? DEMO_POST + " (synthetic demo)" : nameOf(afterFile);
        return "before: " + pre + ", after: " + post;
    }

    /** Short label of the inputs for the metrics panel. */
    private String shortInputName() {
        if (beforeIsSyntheticDemo && afterIsSyntheticDemo) {
            return "synthetic demo pair (Site A)";
        }
        if (levirPair != null) {
            return levirPair.shortName();
        }
        return nameOf(beforeFile) + " → " + nameOf(afterFile);
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
     * Runs the selected engine in a background Task, builds the visualisations (and, for a LEVIR-CD pair,
     * the ground-truth comparison) from the returned mask and saves everything to results/.
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
        final String inputShortName = shortInputName();
        final boolean synthetic = beforeIsSyntheticDemo && afterIsSyntheticDemo;
        final LevirPair pair = levirPair;
        final boolean[][] groundTruth = groundTruthMask;

        logMessage("Running detection engine: " + engine.getDisplayName() + " on " + inputDescription + " ...");
        setAnalysisRunning(true);
        systemStatusLabel.setText("System: Processing...");

        Task<AnalysisBundle> detectionTask = new Task<>() {
            @Override
            protected AnalysisBundle call() throws Exception {
                EngineResult result = runSelectedEngine(engine, before, after);
                ChangeVisualizationSet visuals = ChangeVisualizationSet.build(result, after);

                GroundTruthComparison comparison = null;
                BufferedImage gtImage = null;
                BufferedImage errorsImage = null;
                if (groundTruth != null) {
                    boolean[][] predicted = result.getMask();
                    comparison = GroundTruthComparison.compare(predicted, groundTruth);
                    gtImage = GroundTruthComparison.renderGroundTruth(groundTruth);
                    errorsImage = GroundTruthComparison.renderErrors(predicted, groundTruth, after);
                }

                LocalDateTime now = LocalDateTime.now();
                String baseName = ResultExporter.uniqueBaseName(RESULTS_DIR, result, now);
                List<File> saved = List.of();
                String saveError = null;
                try {
                    saved = ResultExporter.export(RESULTS_DIR, baseName, result, visuals, inputDescription, now,
                            exportExtra(result, pair, comparison));
                } catch (Exception e) {
                    saveError = e.getMessage();
                }
                return new AnalysisBundle(engine, result, visuals, baseName, saved, saveError,
                        inputShortName, synthetic, pair, comparison, gtImage, errorsImage);
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

    /** Extra lines for the exported summary: LEVIR-CD area and the per-image ground-truth comparison. */
    private static String exportExtra(EngineResult result, LevirPair pair, GroundTruthComparison comparison) {
        if (pair == null) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.US,
                "LEVIR-CD pair %s. Changed area ~ %,.0f m2, ASSUMING the LEVIR-CD ground sample distance of 0.5 m/px "
                        + "(0.25 m2 per pixel).%n", pair.shortName(), LevirPair.areaSquareMetres(result.getChangedPixels())));
        if (comparison != null) {
            sb.append("\nCompared with ground truth (THIS IMAGE ONLY; label value > 0 = change, building changes only):\n");
            sb.append(comparison.getMetrics().toReport());
        }
        return sb.toString();
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
        lastBundle = bundle;

        fxImages.clear();
        for (VisualizationMode mode : VisualizationMode.values()) {
            fxImages.put(mode, SwingFXUtils.toFXImage(bundle.visuals().get(mode), null));
        }
        if (bundle.hasGroundTruth()) {
            fxImages.put(GroundTruthView.GROUND_TRUTH, SwingFXUtils.toFXImage(bundle.groundTruthImage(), null));
            fxImages.put(GroundTruthView.ERRORS, SwingFXUtils.toFXImage(bundle.errorsImage(), null));
        }
        setModeTogglesDisabled(false);
        updateGroundTruthControls(bundle);
        if (currentView instanceof GroundTruthView && !bundle.hasGroundTruth()) {
            viewModeGroup.selectToggle(modeOverlayToggle);
        }
        changeMapImageView.setImage(fxImages.get(currentView));
        changeMapImageView.setVisible(true);
        changeMapPlaceholder.setVisible(false);
        resultEngineBadge.setText(bundle.engine().getShortLabel());

        updateMetrics(r);
        statInput.setText("on " + bundle.inputShortName());
        showAreaM2(bundle.levirPair() != null ? r : null);

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
        if (bundle.hasGroundTruth()) {
            BinaryChangeMetrics m = bundle.comparison().getMetrics();
            logMessage(String.format(Locale.ROOT,
                    "[Ground truth, this image only] %s vs label: F1 %.4f, IoU %.4f, precision %.4f, recall %.4f "
                            + "(TP %,d, FP %,d, FN %,d px).", bundle.inputShortName(), m.f1(), m.iou(), m.precision(),
                    m.recall(), m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives()));
        }
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
        completedAnalyses.incrementAndGet();
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

    /** m² line, only for LEVIR-CD inputs (known 0.5 m/px); hidden otherwise. */
    private void showAreaM2(EngineResult r) {
        boolean show = r != null;
        statAreaM2.setVisible(show);
        statAreaM2.setManaged(show);
        statAreaM2.setText(show ? String.format(Locale.US,
                "Changed area ≈ %,.0f m², assuming the LEVIR-CD ground sample distance of 0.5 m/px (0.25 m² per pixel).",
                LevirPair.areaSquareMetres(r.getChangedPixels())) : "");
    }

    /** Ground-truth toggles and the per-image metrics block for the displayed result (null = no result). */
    private void updateGroundTruthControls(AnalysisBundle bundle) {
        boolean hasGt = bundle != null && bundle.hasGroundTruth();
        modeGroundTruthToggle.setDisable(!hasGt);
        modeErrorsToggle.setDisable(!hasGt);
        if (bundle == null) {
            gtViewNote.setText("needs a LEVIR-CD pair with a label");
        } else if (!hasGt) {
            gtViewNote.setText("no ground-truth label for these inputs");
        } else {
            gtViewNote.setText("label of " + bundle.levirPair().name());
        }
        gtBlock.getStyleClass().remove("gt-block-inactive");
        for (Node n : List.of(gtTiles, gtCounts, gtBenchmark)) {
            n.setVisible(hasGt);
            n.setManaged(hasGt);
        }

        if (!hasGt) {
            gtBlock.getStyleClass().add("gt-block-inactive");
            for (Label l : List.of(gtF1, gtIoU, gtPrecision, gtRecall)) {
                l.setText("--");
            }
            gtCounts.setText("");
            gtBenchmark.setText("");
            boolean synthetic = bundle != null && bundle.syntheticInput();
            gtNote.setText((synthetic ? "The synthetic demo pair has no ground-truth label, so there is nothing to compare with. "
                    : "No ground-truth label for the analysed inputs. ")
                    + "Use 'Load LEVIR-CD Pair…' to load A, B and the label, then analyse to see per-image F1, IoU, "
                    + "precision and recall.");
            return;
        }

        BinaryChangeMetrics m = bundle.comparison().getMetrics();
        gtF1.setText(String.format(Locale.ROOT, "%.4f", m.f1()));
        gtIoU.setText(String.format(Locale.ROOT, "%.4f", m.iou()));
        gtPrecision.setText(String.format(Locale.ROOT, "%.4f", m.precision()));
        gtRecall.setText(String.format(Locale.ROOT, "%.4f", m.recall()));
        gtCounts.setText(String.format(Locale.ROOT, "Pixels of this image: TP %,d · FP %,d · FN %,d · TN %,d",
                m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives(), m.getTrueNegatives()));
        gtBenchmark.setText(datasetBenchmarkLine(bundle));
        gtNote.setText("Per-image numbers for " + bundle.levirPair().name() + " only (BinaryChangeMetrics, label > 0 = "
                + "change). They vary a lot between images and are not the dataset benchmark. LEVIR-CD labels building "
                + "changes only.");
    }

    /** "Dataset benchmark" line for the engine of the displayed result, parsed from the results files. */
    private static String datasetBenchmarkLine(AnalysisBundle bundle) {
        String split = bundle.levirPair().split();
        if (!split.equals("val") && !split.equals("test")) {
            return "Dataset benchmark: none for the '" + split + "' split (benchmarks exist for val and test).";
        }
        boolean deep = SiameseUNetEngine.ID.equals(bundle.result().getEngineId());
        Path file = deep ? AppPaths.deepMetrics(split) : AppPaths.baselineMetrics(split);
        MetricsReport report = MetricsReport.readChangeDetection(file);
        String prefix = "Dataset benchmark for this engine, LEVIR-CD " + split + " (all images): ";
        if (!report.isOk()) {
            return prefix + "not available (" + (report.getStatus() == MetricsReport.Status.MISSING
                    ? file + " not found" : report.getProblem()) + ").";
        }
        OptionalDouble f1 = report.get(MetricsReport.Metric.F1);
        OptionalDouble iou = report.get(MetricsReport.Metric.IOU);
        return prefix + String.format(Locale.ROOT, "F1 %s · IoU %s  (%s)",
                f1.isPresent() ? String.format(Locale.ROOT, "%.4f", f1.getAsDouble()) : "n/a",
                iou.isPresent() ? String.format(Locale.ROOT, "%.4f", iou.getAsDouble()) : "n/a", file);
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

    private void applyView(ChangeView view, boolean log) {
        updateLegend(view);
        Image image = fxImages.get(view);
        if (image != null) {
            changeMapImageView.setImage(image);
            if (log) {
                logMessage("Change view switched to: " + view.getDisplayName() + ".");
            }
        }
    }

    private void updateLegend(ChangeView view) {
        legendModeLabel.setText(view.getDisplayName());
        legendCaption.setText(view.getDescription());
        legendPane.getChildren().clear();
        for (VisualizationMode.LegendEntry entry : view.getLegend()) {
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

    /** Number of analyses that finished successfully (lets callers wait for a specific run). */
    public int getCompletedAnalysisCount() {
        return completedAnalyses.get();
    }

    public EngineResult getLastEngineResult() {
        return lastResult;
    }

    /** Per-image metrics of the displayed result against its ground truth, or null if it had none. */
    public BinaryChangeMetrics getLastGroundTruthMetrics() {
        return (lastBundle != null && lastBundle.hasGroundTruth()) ? lastBundle.comparison().getMetrics() : null;
    }

    /** The current engine view, or null if a ground-truth view is shown. */
    public VisualizationMode getCurrentMode() {
        return currentView instanceof VisualizationMode ? (VisualizationMode) currentView : null;
    }

    public ChangeView getCurrentView() {
        return currentView;
    }

    public FundamentalsPage getFundamentalsPage() {
        return fundamentalsPage;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private File chooseImageFile(String title, File initialDirectory) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle(title);
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Images (*.png, *.jpg, *.tif, *.bmp)", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"),
                new FileChooser.ExtensionFilter("All Files", "*.*")
        );
        if (initialDirectory != null && initialDirectory.isDirectory()) {
            fileChooser.setInitialDirectory(initialDirectory);
        }

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
