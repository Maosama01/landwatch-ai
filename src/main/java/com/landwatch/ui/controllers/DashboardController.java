package com.landwatch.ui.controllers;

import com.landwatch.deeplearning.DeepChangeDetector;
import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.geospatial.SatelliteImageLoader;
import com.landwatch.ui.ChangeView;
import com.landwatch.ui.ChangeVisualizationSet;
import com.landwatch.ui.GroundTruthComparison;
import com.landwatch.ui.GroundTruthView;
import com.landwatch.ui.ResultExporter;
import com.landwatch.ui.SwipeCompareView;
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
import com.landwatch.ui.pages.EuroSatPage;
import com.landwatch.ui.pages.FailureAnalysisPage;
import com.landwatch.ui.pages.FundamentalsPage;
import javafx.application.Platform;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.concurrent.Task;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.geometry.Pos;
import javafx.geometry.Rectangle2D;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
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
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.input.MouseButton;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Region;
import javafx.scene.layout.RowConstraints;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Screen;
import javafx.stage.Stage;

import java.awt.image.BufferedImage;
import java.io.File;
import java.net.URL;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
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
 * <p>When the inputs are a LEVIR-CD pair loaded with "Open LEVIR-CD pair…" (or dropped onto a frame), the pair's ground-truth
 * label is kept as well: the analysis then also renders the label and a TP/FP/FN error map and computes
 * per-image metrics with {@link BinaryChangeMetrics}. The deep model is loaded asynchronously at start-up
 * and becomes selectable once ready. The sidebar switches between the analysis view and the
 * Land use (EuroSAT), Benchmarks, NN fundamentals, Failure analysis and About pages. The EuroSAT
 * land-use classifier is a separate scene classifier with its own page and background thread; it is
 * not a {@link ChangeDetectionEngine} and is never run on the change-detection inputs.</p>
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
    @FXML private Button navLandUseButton;
    @FXML private Button navAboutButton;

    // Header & Status
    @FXML private Label systemStatusLabel;
    @FXML private Region modelStatusDot;
    @FXML private Label modelStatusLabel;
    @FXML private Button themeToggleButton;
    @FXML private Label resultSummaryLabel;
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
    @FXML private Label changeMapHintLabel;
    @FXML private Label resultEngineBadge;
    @FXML private Button enlargeButton;
    @FXML private ToggleButton compareToggle;
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
    @FXML private StackPane shareTrack;
    @FXML private Region shareFill;
    @FXML private Label savedLabel;

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

    // Workspace layout
    @FXML private GridPane resultsGrid;
    @FXML private RowConstraints workspaceRow;
    @FXML private Label gtScopeLabel;
    @FXML private VBox measurementsCard;
    private Runnable fitWorkspace = () -> { };

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
    /** True while the before/after swipe comparison is shown instead of an engine view. */
    private boolean compareActive = false;
    private final SwipeCompareView swipeView = new SwipeCompareView();
    /** Share of changed pixels (0..1), drawn at its true ratio under the headline figure. */
    private final DoubleProperty shareFraction = new SimpleDoubleProperty(0);
    /** The inputs changed after the result on screen was computed. */
    private boolean resultsStale = false;
    /** The user picked an engine themselves, so the deep model is not auto-selected once it is ready. */
    private boolean userPickedEngine = false;
    private boolean programmaticEngineChange = false;

    private SiameseUNetEngine deepEngine;

    // Pages
    private BenchmarksPage benchmarksPage;
    private FundamentalsPage fundamentalsPage;
    private FailureAnalysisPage failureAnalysisPage;
    private EuroSatPage euroSatPage;
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
        swipeView.setVisible(false);
        changeMapContainer.getChildren().add(swipeView);
        setupViewModes();
        fitImageToContainer(beforeImageContainer, beforeImageView);
        fitImageToContainer(afterImageContainer, afterImageView);
        fitImageToContainer(changeMapContainer, changeMapImageView);
        setupDropTarget(beforeImageContainer, true);
        setupDropTarget(afterImageContainer, false);
        setupDropTarget(changeMapContainer, null);
        fitWorkspaceToWindow();
        shareFill.maxWidthProperty().bind(shareTrack.widthProperty().multiply(shareFraction));
        shareFill.prefWidthProperty().bind(shareFill.maxWidthProperty());
        setupGroundTruthTiles();
        setupPages();
        setupShortcuts();

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
        setEngineProgrammatically(firstAvailableEngine());
        onEngineSelected(engineComboBox.getValue(), false);

        engineComboBox.valueProperty().addListener((obs, oldEngine, newEngine) -> {
            if (newEngine != null && !newEngine.isAvailable()) {
                // Disabled entries cannot be clicked, but keyboard navigation could still reach them.
                logMessage(newEngine.getDisplayName() + " cannot be selected: " + newEngine.getStatusNote() + ".");
                ChangeDetectionEngine fallback = (oldEngine != null && oldEngine.isAvailable())
                        ? oldEngine : firstAvailableEngine();
                Platform.runLater(() -> setEngineProgrammatically(fallback));
                return;
            }
            if (!programmaticEngineChange) {
                userPickedEngine = true;
            }
            onEngineSelected(newEngine, true);
        });
    }

    /** Changes the selected engine without counting it as the user's own choice. */
    private void setEngineProgrammatically(ChangeDetectionEngine engine) {
        programmaticEngineChange = true;
        try {
            engineComboBox.setValue(engine);
        } finally {
            programmaticEngineChange = false;
        }
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
            engineNoteLabel.setText("");
            return;
        }
        updateEngineNote();
        updateSummary();

        if (log) {
            logMessage("Detection engine selected: " + engine.getDisplayName() + ".");
            if (lastResult != null && !lastResult.getEngineName().equals(engine.getDisplayName())) {
                logMessage("Results on screen were produced by " + lastResult.getEngineName()
                        + ". Press 'Analyze changes' to run " + engine.getDisplayName() + ".");
            }
        }
    }

    /** Note next to the selector: the selected engine's description plus the state of every other engine. */
    private void updateEngineNote() {
        ChangeDetectionEngine engine = engineComboBox.getValue();
        engineNoteLabel.setText(engine != null ? engine.getStatusNote() : "");
        StringBuilder note = new StringBuilder(engine != null ? engine.getStatusNote() : "");
        for (ChangeDetectionEngine other : engineComboBox.getItems()) {
            if (other == engine) {
                continue;
            }
            note.append(note.length() > 0 ? "\n" : "").append(other.getDisplayName());
            if (other.isLoading()) {
                note.append(": loading in the background…");
            } else if (!other.isAvailable()) {
                note.append(": unavailable, ").append(other.getStatusNote()).append('.');
            } else {
                note.append(": also ready, pick it in the menu.");
            }
        }
        engineNoteLabel.setTooltip(tooltip(note.toString()));
        engineComboBox.setTooltip(tooltip(note.toString()));
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
        updateModelStatus();
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
            setEngineProgrammatically(firstAvailableEngine());
        }
        updateEngineNote();
        updateModelStatus();
        if (deepEngine.isAvailable()) {
            logMessage("Deep model ready: " + deepEngine.getStatusNote() + ".");
            String description = deepEngine.getDescription();
            if (description != null) {
                for (String line : description.split("\n")) {
                    logMessage("[Model] " + line.strip());
                }
            }
            if (!userPickedEngine && !analysisRunning) {
                setEngineProgrammatically(deepEngine);
                logMessage("Selected '" + deepEngine.getDisplayName() + "' as the engine. The classical baseline "
                        + "stays available in the Engine menu.");
            } else {
                logMessage("Select '" + deepEngine.getDisplayName() + "' in the Engine menu to use it.");
            }
        } else {
            logMessage("Deep model unavailable: " + deepEngine.getStatusNote()
                    + ". Only the classical baseline can run (set -D" + SiameseUNetEngine.MODEL_DIR_PROPERTY
                    + "=<dir> to use another model directory).");
        }
        refreshVisiblePage();
    }

    /** Sidebar status of the deep model: a dot (ready, loading, unavailable) and one line of text. */
    private void updateModelStatus() {
        modelStatusDot.getStyleClass().removeAll("dot-ready", "dot-loading", "dot-off");
        if (deepEngine == null) {
            modelStatusDot.getStyleClass().add("dot-off");
            modelStatusLabel.setText("No deep model configured");
        } else if (deepEngine.isLoading()) {
            modelStatusDot.getStyleClass().add("dot-loading");
            modelStatusLabel.setText("Deep model loading…");
        } else if (deepEngine.isAvailable()) {
            modelStatusDot.getStyleClass().add("dot-ready");
            String device = deepEngine.getDeviceName();
            modelStatusLabel.setText("Deep model ready" + (device != null ? " · " + prettyDevice(device) : ""));
        } else {
            modelStatusDot.getStyleClass().add("dot-off");
            modelStatusLabel.setText("Deep model unavailable");
        }
        Tooltip.install(modelStatusLabel, new Tooltip(deepEngine != null ? deepEngine.getStatusNote() : ""));
    }

    /** "mps(0)" → "Apple GPU", "cpu()" → "CPU", anything else unchanged. */
    static String prettyDevice(String djlDevice) {
        if (djlDevice == null) {
            return "";
        }
        if (djlDevice.startsWith("mps")) {
            return "Apple GPU";
        }
        if (djlDevice.startsWith("cpu")) {
            return "CPU";
        }
        if (djlDevice.startsWith("gpu")) {
            return "GPU";
        }
        return djlDevice;
    }

    /**
     * Sizes the workspace row (inputs, result, measurements) to what is left of the window under the title
     * and engine rows, so the whole result is on the first screen; never below 560 px.
     */
    private void fitWorkspaceToWindow() {
        fitWorkspace = () -> {
            double viewport = contentScrollPane.getViewportBounds().getHeight();
            if (viewport > 0) {
                // The measurements cannot shrink (wrapped text), so the row is never shorter than they are.
                double width = measurementsCard.getWidth();
                double needed = width > 0 ? measurementsCard.prefHeight(width) : 0;
                double target = Math.max(Math.max(560, needed), viewport - resultsGrid.getLayoutY() - 24);
                if (Math.abs(workspaceRow.getPrefHeight() - target) > 0.5) {
                    workspaceRow.setPrefHeight(target);
                }
            }
        };
        contentScrollPane.viewportBoundsProperty().addListener((obs, o, n) -> fitWorkspace.run());
        resultsGrid.layoutYProperty().addListener((obs, o, n) -> fitWorkspace.run());
        measurementsCard.widthProperty().addListener((obs, o, n) -> fitWorkspace.run());
    }

    /** The deep engine (for the snapshot tool and pages). */
    public SiameseUNetEngine getDeepEngine() {
        return deepEngine;
    }

    /** Releases native resources (the deep model and the EuroSAT classifier). Called from MainApp#stop. */
    public void shutdown() {
        if (deepEngine != null) {
            deepEngine.close();
        }
        if (euroSatPage != null) {
            euroSatPage.shutdown();
        }
    }

    // ------------------------------------------------------------------
    // Pages & navigation
    // ------------------------------------------------------------------

    private void setupPages() {
        benchmarksPage = new BenchmarksPage(deepEngine);
        fundamentalsPage = new FundamentalsPage(this::logMessage);
        failureAnalysisPage = new FailureAnalysisPage(this::openLevirPairFromPage);
        euroSatPage = new EuroSatPage(this::logMessage);
        aboutPage = new AboutPage(deepEngine);
        for (Node page : List.of(benchmarksPage.getRoot(), fundamentalsPage.getRoot(),
                failureAnalysisPage.getRoot(), euroSatPage.getRoot(), aboutPage.getRoot())) {
            page.setVisible(false);
            pageStack.getChildren().add(page);
        }
        activeNavButton = navAnalysisButton;
    }

    @FXML public void handleNavAnalysis() { showPage(contentScrollPane, navAnalysisButton); }
    @FXML public void handleNavBenchmarks() { benchmarksPage.refresh(); showPage(benchmarksPage.getRoot(), navBenchmarksButton); }
    @FXML public void handleNavFundamentals() { showPage(fundamentalsPage.getRoot(), navFundamentalsButton); }
    @FXML public void handleNavFailure() { failureAnalysisPage.refresh(); showPage(failureAnalysisPage.getRoot(), navFailureButton); }
    @FXML public void handleNavLandUse() { euroSatPage.onShow(); showPage(euroSatPage.getRoot(), navLandUseButton); }
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
            if (newToggle == compareToggle) {
                compareActive = true;
                refreshViewer();
                logMessage("Compare view: drag the divider (or use the arrow keys) to compare the two dates.");
                return;
            }
            compareActive = false;
            ChangeView view = viewOf(newToggle);
            if (view != currentView) {
                currentView = view;
                applyView(view, true);
            } else {
                refreshViewer();
            }
        });
        Toggle selected = viewModeGroup.getSelectedToggle();
        currentView = (selected != null && selected != compareToggle) ? viewOf(selected) : VisualizationMode.RED_OVERLAY;
        applyView(currentView, false);

        String[] keys = {"1", "2", "3", "4", "5", "6"};
        ToggleButton[] toggles = {modeOverlayToggle, modeMaskToggle, modeChangeMapToggle,
                modeGroundTruthToggle, modeErrorsToggle, compareToggle};
        String[] tips = {
                "The after image with the detected change in red and an amber box around each region",
                "The engine's raw output: white = change, black = no change",
                "Change regions in cyan with a yellow outline over the dimmed after image",
                "The LEVIR-CD ground-truth label of this pair (building changes only)",
                "Engine against the label: yellow = found, orange = false alarm, blue = missed",
                "Drag a divider across the two dates of the same place"};
        for (int i = 0; i < toggles.length; i++) {
            toggles[i].setTooltip(tooltip(tips[i] + " (" + shortcutText(keys[i]) + ")"));
        }
    }

    /** Selects the before/after swipe comparison (used by the snapshot tool); ignored if unavailable. */
    public void selectCompare() {
        if (!compareToggle.isDisabled()) {
            viewModeGroup.selectToggle(compareToggle);
        }
    }

    /**
     * Shows exactly one thing in the result frame: the swipe comparison, the image of the current view,
     * or the placeholder sentence; and the legend that belongs to it.
     */
    private void refreshViewer() {
        boolean showing = (compareActive && swipeView.hasImages()) || fxImages.get(currentView) != null;
        changeMapContainer.getStyleClass().remove("has-image");
        if (showing) {
            changeMapContainer.getStyleClass().add("has-image");
        }
        if (compareActive && swipeView.hasImages()) {
            swipeView.setVisible(true);
            changeMapImageView.setVisible(false);
            changeMapPlaceholder.setVisible(false);
            enlargeButton.setDisable(true);
            legendPane.getChildren().clear();
            legendCaption.setText("Before on the left of the divider, after on the right, at the same place. "
                    + "Drag the divider, or click the image and use the arrow keys.");
            return;
        }
        swipeView.setVisible(false);
        Image image = fxImages.get(currentView);
        changeMapImageView.setVisible(image != null);
        changeMapPlaceholder.setVisible(image == null);
        if (image != null) {
            changeMapImageView.setImage(image);
            updateLegend(currentView);
        } else {
            legendPane.getChildren().clear();
            legendCaption.setText("");
        }
        enlargeButton.setDisable(image == null);
        Platform.runLater(fitWorkspace);
    }

    /** The swipe comparison needs two loaded images of the same size. */
    private void updateCompareAvailability() {
        boolean ok = beforeBufferedImage != null && afterBufferedImage != null
                && beforeBufferedImage.getWidth() == afterBufferedImage.getWidth()
                && beforeBufferedImage.getHeight() == afterBufferedImage.getHeight();
        swipeView.setImages(ok ? beforeImageView.getImage() : null, ok ? afterImageView.getImage() : null);
        compareToggle.setDisable(!ok || analysisRunning);
        if (!ok && compareActive) {
            viewModeGroup.selectToggle(toggleFor(currentView).isDisabled() ? modeOverlayToggle : toggleFor(currentView));
            compareActive = false;
        }
        refreshViewer();
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

    /**
     * Fits an image into its frame (8px inside it), centred, at any window size. The view is unmanaged,
     * so its size never feeds back into the frame's own size: the frame can shrink as well as grow.
     */
    private static void fitImageToContainer(StackPane container, ImageView view) {
        view.setManaged(false);
        Runnable layout = () -> {
            Image image = view.getImage();
            double boxW = container.getWidth() - 16;
            double boxH = container.getHeight() - 16;
            if (image == null || boxW <= 0 || boxH <= 0) {
                return;
            }
            double[] size = SwipeCompareView.fit(image.getWidth(), image.getHeight(), boxW, boxH);
            view.setFitWidth(size[0]);
            view.setFitHeight(size[1]);
            view.relocate((container.getWidth() - size[0]) / 2, (container.getHeight() - size[1]) / 2);
        };
        container.widthProperty().addListener((obs, o, n) -> layout.run());
        container.heightProperty().addListener((obs, o, n) -> layout.run());
        view.imageProperty().addListener((obs, o, n) -> layout.run());
    }

    /**
     * Makes an image frame accept dropped files and, while empty, open the file chooser on a click.
     * {@code before} is true for the before frame, false for the after frame and null for the result
     * frame (one image there fills whichever input is still empty).
     */
    private void setupDropTarget(StackPane container, Boolean before) {
        container.setOnDragOver(e -> {
            if (e.getDragboard().hasFiles() && !analysisRunning) {
                e.acceptTransferModes(TransferMode.COPY);
            }
            e.consume();
        });
        container.setOnDragEntered(e -> {
            if (e.getDragboard().hasFiles() && !analysisRunning && !container.getStyleClass().contains("drag-over")) {
                container.getStyleClass().add("drag-over");
            }
        });
        container.setOnDragExited(e -> container.getStyleClass().remove("drag-over"));
        container.setOnDragDropped(e -> {
            boolean done = e.getDragboard().hasFiles() && loadDroppedFiles(e.getDragboard().getFiles(), before);
            e.setDropCompleted(done);
            e.consume();
        });
        if (before != null) {
            container.setOnMouseClicked(e -> {
                boolean empty = before ? beforeBufferedImage == null : afterBufferedImage == null;
                if (e.getButton() == MouseButton.PRIMARY && empty && !analysisRunning) {
                    if (before) {
                        handleUploadBefore();
                    } else {
                        handleUploadAfter();
                    }
                }
            });
        }
    }

    /**
     * Two or more dropped images become the before and after inputs (in file-name order). One image from
     * a LEVIR-CD A/, B/ or label/ folder loads its whole pair with the label; any other single image
     * fills the frame it was dropped on.
     */
    public boolean loadDroppedFiles(List<File> files, Boolean before) {
        List<File> images = new ArrayList<>();
        for (File f : files) {
            String n = f.getName().toLowerCase(Locale.ROOT);
            if (f.isFile() && (n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".bmp")
                    || n.endsWith(".tif") || n.endsWith(".tiff"))) {
                images.add(f);
            }
        }
        if (images.isEmpty()) {
            logMessage("Ignored the dropped file(s): only PNG, JPG, BMP and TIFF images can be analysed.");
            return false;
        }
        if (images.size() >= 2) {
            images.sort(Comparator.comparing(File::getName));
            if (images.size() > 2) {
                logMessage("Dropped " + images.size() + " images; using the first two by name as before and after.");
            }
            boolean ok = loadBeforeFile(images.get(0));
            return loadAfterFile(images.get(1)) && ok;
        }
        File file = images.get(0);
        try {
            LevirPair.locate(file.toPath());
            return loadLevirPair(file.toPath());
        } catch (Exception notALevirPair) {
            boolean intoBefore = before != null ? before : beforeBufferedImage == null;
            return intoBefore ? loadBeforeFile(file) : loadAfterFile(file);
        }
    }

    /** Keyboard shortcuts on the main window (Shortcut is ⌘ on macOS, Ctrl elsewhere). */
    private void setupShortcuts() {
        analyzeButton.sceneProperty().addListener((obs, oldScene, scene) -> {
            if (scene != null) {
                installShortcuts(scene);
            }
        });
        analyzeButton.setTooltip(tooltip("Run the selected engine on the two images (" + shortcutText("ENTER") + ")"));
        loadDemoButton.setTooltip(tooltip("Load the bundled synthetic test pair: a computer-generated scene, not "
                + "satellite imagery (" + shortcutText("D") + ")"));
        loadLevirButton.setTooltip(tooltip("Pick an image in a LEVIR-CD split's A/ or B/ folder: the matching A, B "
                + "and label files are loaded, so the result can be checked against the ground truth ("
                + shortcutText("O") + "). You can also drop the file onto an image frame."));
        resetButton.setTooltip(tooltip("Clear the inputs and the result (" + shortcutText("BACK_SPACE") + ")"));
    }

    private void installShortcuts(Scene scene) {
        Map<KeyCombination, Runnable> keys = scene.getAccelerators();
        keys.put(combo("ENTER"), () -> {
            if (!analyzeButton.isDisabled()) {
                handleNavAnalysis();
                handleAnalyzeChanges();
            }
        });
        keys.put(combo("O"), () -> {
            if (!analysisRunning) {
                handleNavAnalysis();
                handleLoadLevirPair();
            }
        });
        keys.put(combo("D"), () -> {
            if (!analysisRunning) {
                handleNavAnalysis();
                handleLoadDemoPair();
            }
        });
        keys.put(combo("BACK_SPACE"), () -> {
            if (!analysisRunning) {
                handleReset();
            }
        });
        ToggleButton[] toggles = {modeOverlayToggle, modeMaskToggle, modeChangeMapToggle,
                modeGroundTruthToggle, modeErrorsToggle, compareToggle};
        for (int i = 0; i < toggles.length; i++) {
            ToggleButton t = toggles[i];
            keys.put(combo("DIGIT" + (i + 1)), () -> {
                if (!t.isDisabled()) {
                    handleNavAnalysis();
                    viewModeGroup.selectToggle(t);
                }
            });
        }
    }

    private static KeyCodeCombination combo(String keyCode) {
        return new KeyCodeCombination(KeyCode.valueOf(keyCode), KeyCombination.SHORTCUT_DOWN);
    }

    private static String shortcutText(String keyOrDigit) {
        String code = keyOrDigit.length() == 1 && Character.isDigit(keyOrDigit.charAt(0)) ? "DIGIT" + keyOrDigit : keyOrDigit;
        return combo(code).getDisplayText();
    }

    private static Tooltip tooltip(String text) {
        Tooltip t = new Tooltip(text);
        t.setWrapText(true);
        t.setMaxWidth(320);
        return t;
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
        resultsStale = false;
        fxImages.clear();

        beforeImageView.setImage(null);
        beforeImageView.setVisible(false);
        beforePlaceholder.setVisible(true);
        beforeImageContainer.getStyleClass().remove("has-image");
        beforeFilenameLabel.setText("No image yet");
        beforeDimensionLabel.setText("PNG, JPG, BMP or TIFF");

        afterImageView.setImage(null);
        afterImageView.setVisible(false);
        afterPlaceholder.setVisible(true);
        afterImageContainer.getStyleClass().remove("has-image");
        afterFilenameLabel.setText("No image yet");
        afterDimensionLabel.setText("PNG, JPG, BMP or TIFF");

        changeMapImageView.setImage(null);
        changeMapStatusLabel.setText("No result yet");
        changeMapHintLabel.setText("Load a before and an after image of the same site, then press Analyze changes. "
                + "You can drop image files on any frame.");
        resultEngineBadge.setText("No result yet");
        setModeTogglesDisabled(true);
        updateGroundTruthControls(null);
        if (compareActive || toggleFor(currentView).isDisabled()) {
            viewModeGroup.selectToggle(modeOverlayToggle);
        }
        updateCompareAvailability();

        statEngine.setText("No analysis yet");
        statInput.setText("");
        setValue(statChangePercent, null);
        statChangedPixels.setText("");
        statTotalPixels.setText("");
        shareFraction.set(0);
        setValue(statRegions, null);
        setValue(statOtsu, null);
        statOtsuSub.setText("");
        setValue(statLatency, null);
        setValue(statImageSize, null);
        setChangeLevel(null);
        showAreaM2(null);
        showSaved(null);

        analysisProgressBar.setVisible(false);
        analyzeButton.setDisable(true);
        systemStatusLabel.setText("Load an image pair to begin");
        updateSummary();

        logMessage("Dashboard reset: no inputs, no result.");
    }

    /** Sets a figure, or an em dash in the faint "missing" style when {@code text} is null. */
    private static void setValue(Label label, String text) {
        label.getStyleClass().remove("is-empty");
        if (text == null) {
            label.setText("—");
            label.getStyleClass().add("is-empty");
        } else {
            label.setText(text);
        }
    }

    /** The "saved to results/…" line under the measurements; hidden when there is nothing to report. */
    private void showSaved(String text) {
        savedLabel.setText(text == null ? "" : text);
        savedLabel.setVisible(text != null);
        savedLabel.setManaged(text != null);
    }

    /**
     * The sentence under the page title: what is loaded, what to do next, or the finding of the result
     * on screen.
     */
    private void updateSummary() {
        String text;
        if (analysisRunning) {
            ChangeDetectionEngine engine = engineComboBox.getValue();
            text = "Analysing " + shortInputName() + " with " + (engine != null ? engine.getDisplayName() : "the engine") + "…";
        } else if (lastBundle != null && !resultsStale) {
            EngineResult r = lastBundle.result();
            StringBuilder sb = new StringBuilder(String.format(Locale.ROOT, "%.2f %% of %s changed, in %,d region%s",
                    r.getChangePercentage(), lastBundle.inputShortName(), r.getRegionCount(),
                    r.getRegionCount() == 1 ? "" : "s"));
            if (lastBundle.hasGroundTruth()) {
                sb.append(String.format(Locale.ROOT, " · F1 %.4f against the label of this image",
                        lastBundle.comparison().getMetrics().f1()));
            } else if (lastBundle.syntheticInput()) {
                sb.append(" · computer-generated, not satellite imagery");
            }
            text = sb.toString();
        } else if (beforeBufferedImage != null && afterBufferedImage != null) {
            if (beforeBufferedImage.getWidth() != afterBufferedImage.getWidth()
                    || beforeBufferedImage.getHeight() != afterBufferedImage.getHeight()) {
                text = String.format(Locale.ROOT, "The two images differ in size (%d × %d and %d × %d px). "
                                + "Change detection needs two aligned images of the same size.",
                        beforeBufferedImage.getWidth(), beforeBufferedImage.getHeight(),
                        afterBufferedImage.getWidth(), afterBufferedImage.getHeight());
            } else {
                text = String.format(Locale.ROOT, "%s · %d × %d px · ready to analyse (%s)%s", shortInputName(),
                        beforeBufferedImage.getWidth(), beforeBufferedImage.getHeight(), shortcutText("ENTER"),
                        resultsStale ? ". The result shown is for the previous inputs" : "");
            }
        } else if (beforeBufferedImage != null) {
            text = "Before image loaded. Add the after image of the same site.";
        } else if (afterBufferedImage != null) {
            text = "After image loaded. Add the before image of the same site.";
        } else {
            text = "Compare two images of the same site taken on different dates. Start with Demo pair, "
                    + "open a LEVIR-CD pair, or drop two images on the frames below.";
        }
        resultSummaryLabel.setText(text);
    }

    /** Called whenever an input changes: an older result no longer matches the inputs. */
    private void inputsChanged() {
        if (lastBundle != null) {
            resultsStale = true;
        }
        updateCompareAvailability();
        checkReadiness();
        updateSummary();
    }

    private void resetDashboardState() {
        handleReset();
    }

    /**
     * Opens file chooser to upload Date 1 (Before) image.
     */
    @FXML
    public void handleUploadBefore() {
        File file = chooseImageFile("Select the before (earlier) image", null);
        if (file != null) {
            loadBeforeFile(file);
        }
    }

    /**
     * Opens file chooser to upload Date 2 (After) image.
     */
    @FXML
    public void handleUploadAfter() {
        File file = chooseImageFile("Select the after (later) image", null);
        if (file != null) {
            loadAfterFile(file);
        }
    }

    /** Loads {@code file} as the before image; returns false (after telling the user) if it cannot be read. */
    private boolean loadBeforeFile(File file) {
        try {
            BufferedImage buf = SatelliteImageLoader.load(file);
            beforeFile = file;
            beforeBufferedImage = buf;
            beforeIsSyntheticDemo = false;
            dropGroundTruth();
            showInput(beforeImageContainer, beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                    buf, file.getName(), false);
            logMessage("Loaded Before image: " + file.getName() + " [" + buf.getWidth() + "x" + buf.getHeight() + "]");
            inputsChanged();
            return true;
        } catch (Exception e) {
            showError("Could not open the image", file.getName() + ": " + e.getMessage());
            return false;
        }
    }

    /** Loads {@code file} as the after image; returns false (after telling the user) if it cannot be read. */
    private boolean loadAfterFile(File file) {
        try {
            BufferedImage buf = SatelliteImageLoader.load(file);
            afterFile = file;
            afterBufferedImage = buf;
            afterIsSyntheticDemo = false;
            dropGroundTruth();
            showInput(afterImageContainer, afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                    buf, file.getName(), false);
            logMessage("Loaded After image: " + file.getName() + " [" + buf.getWidth() + "x" + buf.getHeight() + "]");
            inputsChanged();
            return true;
        } catch (Exception e) {
            showError("Could not open the image", file.getName() + ": " + e.getMessage());
            return false;
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

            showInput(beforeImageContainer, beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                    beforeBufferedImage, DEMO_PRE, true);
            showInput(afterImageContainer, afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                    afterBufferedImage, DEMO_POST, true);

            logMessage("Loaded the bundled SYNTHETIC demonstration pair (Site A: computer-generated test scene, "
                    + "not real satellite imagery). It only demonstrates the pipeline and has no ground-truth label.");
            inputsChanged();
        } catch (Exception e) {
            showError("Demo Load Error", "Failed to load demo pair: " + e.getMessage());
        }
    }

    /**
     * "Open LEVIR-CD pair…": the user picks an image in a split's A/ or B/ folder; A, B and label/ with
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

        showInput(beforeImageContainer, beforeImageView, beforePlaceholder, beforeFilenameLabel, beforeDimensionLabel,
                beforeBufferedImage, pair.name(), false);
        showInput(afterImageContainer, afterImageView, afterPlaceholder, afterFilenameLabel, afterDimensionLabel,
                afterBufferedImage, pair.name(), false);
        beforeFilenameLabel.setText(pair.name() + " · LEVIR-CD " + pair.split() + "/A");
        afterFilenameLabel.setText(pair.name() + " · LEVIR-CD " + pair.split() + "/B");
        String dims = String.format(Locale.ROOT, "%d × %d px · 0.5 m/px",
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
        inputsChanged();
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

    private static void showInput(StackPane container, ImageView view, VBox placeholder, Label nameLabel, Label dimLabel,
                                  BufferedImage image, String name, boolean syntheticDemo) {
        view.setImage(SwingFXUtils.toFXImage(image, null));
        view.setVisible(true);
        placeholder.setVisible(false);
        if (!container.getStyleClass().contains("has-image")) {
            container.getStyleClass().add("has-image");
        }
        nameLabel.setText(syntheticDemo ? name + " · synthetic demo" : name);
        String dims = String.format(Locale.ROOT, "%d × %d px", image.getWidth(), image.getHeight());
        dimLabel.setText(syntheticDemo ? dims + " · computer-generated, not satellite imagery" : dims);
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
                systemStatusLabel.setText("The two images differ in size");
            } else {
                analyzeButton.setDisable(analysisRunning);
                systemStatusLabel.setText("Ready to analyse");
                logMessage("Image dimensions verified matching: " + beforeBufferedImage.getWidth() + "x" + beforeBufferedImage.getHeight() + " px.");
            }
        } else {
            analyzeButton.setDisable(true);
            systemStatusLabel.setText(beforeBufferedImage == null && afterBufferedImage == null
                    ? "Load an image pair to begin" : "Waiting for the second image");
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
        systemStatusLabel.setText("Analysing…");
        changeMapStatusLabel.setText("Running " + engine.getDisplayName() + "…");
        changeMapHintLabel.setText("on " + inputShortName);
        updateSummary();

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
            systemStatusLabel.setText("Analysis failed");
            setAnalysisRunning(false);
            changeMapStatusLabel.setText(lastBundle == null ? "No result yet" : changeMapStatusLabel.getText());
            changeMapHintLabel.setText("The last analysis failed: " + error.getMessage());
            updateSummary();
            showError("The analysis failed", engine.getDisplayName() + " could not finish: " + error.getMessage());
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
        resultsStale = false;
        setModeTogglesDisabled(false);
        updateGroundTruthControls(bundle);
        if (currentView instanceof GroundTruthView && !bundle.hasGroundTruth()) {
            currentView = VisualizationMode.RED_OVERLAY;
        }
        // A new result is what the user asked to see: leave the comparison for the current view.
        compareActive = false;
        viewModeGroup.selectToggle(toggleFor(currentView));
        refreshViewer();
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
            showSaved("Saved " + bundle.savedFiles().size() + " files to " + RESULTS_DIR.getPath() + "/");
            savedLabel.setTooltip(tooltip(RESULTS_DIR.getPath() + "/" + bundle.exportBaseName() + "_*: " + parts));
        } else {
            logMessage("WARNING: Results could not be saved to " + RESULTS_DIR.getPath() + "/: " + bundle.saveError());
            showSaved("Not saved: " + bundle.saveError());
        }

        systemStatusLabel.setText("Analysis complete");
        setAnalysisRunning(false);
        completedAnalyses.incrementAndGet();
        updateSummary();
        scrollToResults();
    }

    private void updateMetrics(EngineResult r) {
        statEngine.setText(r.getEngineName());
        setValue(statChangePercent, String.format(Locale.ROOT, "%.2f %%", r.getChangePercentage()));
        statChangedPixels.setText(String.format(Locale.ROOT, "%,d of %,d px", r.getChangedPixels(), r.getTotalPixels()));
        statTotalPixels.setText("");
        shareFraction.set(r.getTotalPixels() > 0 ? (double) r.getChangedPixels() / r.getTotalPixels() : 0);
        setValue(statRegions, String.format(Locale.ROOT, "%,d", r.getRegionCount()));
        if (r.getOtsuThreshold() != null) {
            setValue(statOtsu, r.getOtsuThreshold() + " / 255");
            statOtsuSub.setText("on the brightness difference");
        } else {
            setValue(statOtsu, null);
            statOtsuSub.setText("not used by this engine");
        }
        statOtsu.setTooltip(tooltip(r.getOtsuThreshold() != null
                ? "Otsu's cut-off on the brightness difference |ΔY| for this pair"
                : "Not used: only the classical baseline thresholds the brightness difference"));
        setValue(statLatency, String.format(Locale.ROOT, "%,d ms", r.getLatencyMs()));
        setValue(statImageSize, String.format(Locale.ROOT, "%,d × %,d px", r.getWidth(), r.getHeight()));
        setChangeLevel(r.getChangeLevel());
    }

    /** m² line, only for LEVIR-CD inputs (known 0.5 m/px); hidden otherwise. */
    private void showAreaM2(EngineResult r) {
        boolean show = r != null;
        statAreaM2.setVisible(show);
        statAreaM2.setManaged(show);
        statAreaM2.setText(show ? String.format(Locale.US,
                "≈ %,.0f m² at LEVIR-CD's 0.5 m per pixel",
                LevirPair.areaSquareMetres(r.getChangedPixels())) : "");
    }

    /** Ground-truth toggles and the per-image metrics block for the displayed result (null = no result). */
    private void updateGroundTruthControls(AnalysisBundle bundle) {
        boolean hasGt = bundle != null && bundle.hasGroundTruth();
        modeGroundTruthToggle.setDisable(!hasGt);
        modeErrorsToggle.setDisable(!hasGt);
        if (bundle == null) {
            gtViewNote.setText("Label and Errors need a LEVIR-CD pair, which comes with its ground-truth label.");
        } else if (!hasGt) {
            gtViewNote.setText("Label and Errors are off: these inputs have no ground-truth label.");
        } else {
            gtViewNote.setText("Label and Errors use the ground truth of " + bundle.levirPair().name() + ".");
        }
        gtBlock.getStyleClass().remove("gt-block-inactive");
        for (Node n : List.of(gtTiles, gtCounts, gtBenchmark)) {
            n.setVisible(hasGt);
            n.setManaged(hasGt);
        }

        if (!hasGt) {
            gtBlock.getStyleClass().add("gt-block-inactive");
            for (Label l : List.of(gtF1, gtIoU, gtPrecision, gtRecall)) {
                setValue(l, null);
            }
            gtCounts.setText("");
            gtBenchmark.setText("");
            boolean synthetic = bundle != null && bundle.syntheticInput();
            gtNote.setText((synthetic ? "The synthetic demo pair has no label, so there is nothing to check against. "
                    : "No label for these inputs. ")
                    + "Open a LEVIR-CD pair to see F1, IoU, precision and recall for the image.");
            return;
        }

        BinaryChangeMetrics m = bundle.comparison().getMetrics();
        setValue(gtF1, String.format(Locale.ROOT, "%.4f", m.f1()));
        setValue(gtIoU, String.format(Locale.ROOT, "%.4f", m.iou()));
        setValue(gtPrecision, String.format(Locale.ROOT, "%.4f", m.precision()));
        setValue(gtRecall, String.format(Locale.ROOT, "%.4f", m.recall()));
        gtCounts.setText(String.format(Locale.ROOT, "%,d px found · %,d false alarms · %,d missed",
                m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives()));
        gtBenchmark.setText(datasetBenchmarkLine(bundle));
        gtNote.setText("");
        gtScopeLabel.setTooltip(tooltip("Scores for " + bundle.levirPair().name() + " only. They vary a lot "
                + "between images and are not the benchmark (see Benchmarks). LEVIR-CD labels building changes only."));
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
        String prefix = "Whole " + split + " split: ";
        if (!report.isOk()) {
            return prefix + "not available (" + (report.getStatus() == MetricsReport.Status.MISSING
                    ? file + " not found" : report.getProblem()) + ").";
        }
        OptionalDouble f1 = report.get(MetricsReport.Metric.F1);
        OptionalDouble iou = report.get(MetricsReport.Metric.IOU);
        return prefix + String.format(Locale.ROOT, "F1 %s · IoU %s",
                f1.isPresent() ? String.format(Locale.ROOT, "%.4f", f1.getAsDouble()) : "n/a",
                iou.isPresent() ? String.format(Locale.ROOT, "%.4f", iou.getAsDouble()) : "n/a");
    }

    private void setChangeLevel(EngineResult.ChangeLevel level) {
        for (EngineResult.ChangeLevel l : EngineResult.ChangeLevel.values()) {
            statChangeLevel.getStyleClass().remove(l.getStyleClass());
        }
        statChangeLevel.getStyleClass().remove("is-empty");
        if (level == null) {
            statChangeLevel.setText("—");
            statChangeLevel.getStyleClass().add("is-empty");
        } else {
            statChangeLevel.setText(sentenceCase(level.getLabel()));
            statChangeLevel.getStyleClass().add(level.getStyleClass());
        }
    }

    /** "HIGH (5–15 %)" → "High (5–15 %)". */
    static String sentenceCase(String label) {
        if (label == null || label.isEmpty()) {
            return label;
        }
        return label.substring(0, 1).toUpperCase(Locale.ROOT) + label.substring(1).toLowerCase(Locale.ROOT);
    }

    private void setAnalysisRunning(boolean running) {
        analysisRunning = running;
        analysisProgressBar.setVisible(running);
        analyzeButton.setDisable(running || beforeBufferedImage == null || afterBufferedImage == null);
        engineComboBox.setDisable(running);
        for (Button b : List.of(loadDemoButton, loadLevirButton, resetButton, uploadBeforeButton, uploadAfterButton)) {
            b.setDisable(running);
        }
        analyzeButton.setText(running ? "Analysing…" : "Analyze changes");
        if (running) {
            compareToggle.setDisable(true);
        } else {
            updateCompareAvailability();
        }
    }

    private void setModeTogglesDisabled(boolean disabled) {
        modeMaskToggle.setDisable(disabled);
        modeOverlayToggle.setDisable(disabled);
        modeChangeMapToggle.setDisable(disabled);
    }

    /**
     * Scrolls back to the top of the page after an analysis: the title's finding, the inputs, the result and
     * its measurements all share the first screen.
     */
    public void scrollToResults() {
        if (contentScrollPane != null) {
            contentScrollPane.setVvalue(contentScrollPane.getVmin());
        }
    }

    // ------------------------------------------------------------------
    // View modes & legend
    // ------------------------------------------------------------------

    private void applyView(ChangeView view, boolean log) {
        Image image = fxImages.get(view);
        refreshViewer();
        if (image != null && log) {
            logMessage("Change view switched to: " + view.getDisplayName() + ".");
        }
    }

    // ------------------------------------------------------------------
    // Full-size view and theme
    // ------------------------------------------------------------------

    /** Opens the current result view at up to full resolution in its own window; Esc closes it. */
    @FXML
    public void handleEnlarge() {
        Image image = changeMapImageView.getImage();
        if (image == null || !changeMapImageView.isVisible() || changeMapImageView.getScene() == null) {
            return;
        }
        Rectangle2D screen = Screen.getPrimary().getVisualBounds();
        double[] size = SwipeCompareView.fit(image.getWidth(), image.getHeight(),
                Math.min(image.getWidth(), screen.getWidth() * 0.9), Math.min(image.getHeight(), screen.getHeight() * 0.85));
        ImageView view = new ImageView(image);
        view.setPreserveRatio(true);
        view.setSmooth(size[0] < image.getWidth());
        view.setFitWidth(size[0]);
        view.setFitHeight(size[1]);
        StackPane root = new StackPane(view);
        root.getStyleClass().add("root-pane");
        Scene mainScene = changeMapImageView.getScene();
        if (mainScene.getRoot().getStyleClass().contains("light")) {
            root.getStyleClass().add("light");
        }
        Scene scene = new Scene(root, size[0] + 24, size[1] + 24);
        scene.getStylesheets().addAll(mainScene.getStylesheets());
        Stage stage = new Stage();
        stage.initOwner(mainScene.getWindow());
        String input = lastBundle != null ? lastBundle.inputShortName() : "result";
        stage.setTitle(currentView.getDisplayName() + " · " + input + String.format(Locale.ROOT, " · %d × %d px",
                (int) image.getWidth(), (int) image.getHeight()));
        scene.setOnKeyPressed(e -> {
            if (e.getCode() == KeyCode.ESCAPE) {
                stage.close();
            }
        });
        stage.setScene(scene);
        stage.show();
    }

    /** Switches between the dark theme and the light one (for a projector in a bright room). */
    @FXML
    public void handleToggleTheme() {
        Parent root = themeToggleButton.getScene().getRoot();
        boolean light = !root.getStyleClass().contains("light");
        if (light) {
            root.getStyleClass().add("light");
        } else {
            root.getStyleClass().remove("light");
        }
        themeToggleButton.setText(light ? "Dark mode" : "Light mode");
        logMessage("Switched to the " + (light ? "light" : "dark") + " theme.");
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

    /** True while the before/after swipe comparison is shown. */
    public boolean isCompareActive() {
        return compareActive;
    }

    /** Divider position of the swipe comparison, 0 (all after) to 1 (all before). */
    public double getSwipePosition() {
        return swipeView.positionProperty().get();
    }

    /** The swipe comparison node (for the snapshot tool's interaction check). */
    public SwipeCompareView getSwipeView() {
        return swipeView;
    }

    public FundamentalsPage getFundamentalsPage() {
        return fundamentalsPage;
    }

    public EuroSatPage getEuroSatPage() {
        return euroSatPage;
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
        alert.setGraphic(null);
        alert.setContentText(content);
        Scene scene = analyzeButton.getScene();
        if (scene != null) {
            alert.initOwner(scene.getWindow());
            alert.getDialogPane().getStylesheets().addAll(scene.getStylesheets());
            if (scene.getRoot().getStyleClass().contains("light")) {
                alert.getDialogPane().getStyleClass().add("light");
            }
        }
        alert.showAndWait();
    }
}
