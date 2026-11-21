package com.landwatch.ui.pages;

import com.landwatch.deeplearning.eurosat.EuroSatClassifier;
import com.landwatch.deeplearning.eurosat.EuroSatSample;
import com.landwatch.geospatial.SatelliteImageLoader;
import com.landwatch.ui.EuroSatResultFormat;
import com.landwatch.ui.EuroSatResultFormat.ProbabilityRow;
import com.landwatch.ui.ImageFit;
import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.data.EuroSatTestReport;
import com.landwatch.ui.data.EuroSatTestSet;
import com.landwatch.ui.data.EuroSatTestSet.DatasetMatch;
import com.landwatch.ui.data.MetricsReport;
import com.landwatch.ui.data.MetricsReport.Metric;
import com.landwatch.ui.engine.EuroSatEngine;
import javafx.concurrent.Task;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.HPos;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.geometry.VPos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressBar;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.ToggleButton;
import javafx.scene.control.ToggleGroup;
import javafx.scene.control.Tooltip;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseButton;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.RowConstraints;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.TilePane;
import javafx.scene.layout.VBox;
import javafx.scene.shape.SVGPath;
import javafx.stage.FileChooser;
import javafx.stage.Window;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * "Land use (EuroSAT)" page: the trained EuroSAT ResNet classifies one image at a time (a file the user
 * opens or drops, or a random image of the EuroSAT TEST split) and a small gallery of test images,
 * showing predicted against true labels. The recorded test metrics are read from
 * {@code results/eurosat/test_metrics.txt}; everything else is computed live.
 *
 * <p>Model loading and inference run on one background thread ({@code landwatch-eurosat}), never on
 * the JavaFX thread, so the load is queued before any classification. The classifier is a scene
 * classifier for 64 x 64 Sentinel-2 tiles at 10 m/pixel; it is never applied to LEVIR-CD pairs (a
 * LEVIR-CD pair image opened here is shown but not classified, with the reason), and any other image
 * of another size is resized with the classifier's note shown next to the result.</p>
 *
 * <p>The EuroSAT images are not in git. Without {@code datasets/EuroSAT/EuroSAT_RGB} the page still
 * opens, says where the dataset goes, and Browse / drag-and-drop still work with the model alone.</p>
 */
public final class EuroSatPage {

    /** Images in the random and recorded-errors gallery samples. */
    static final int GALLERY_SIZE = 12;
    /** Gallery image side: exactly twice a 64 px tile, so each source pixel is a 2 x 2 block. */
    private static final double TILE_IMAGE = 128;
    private static final double TILE_WIDTH = 150;

    /** Which test images the gallery shows. */
    enum GalleryMode { RANDOM, PER_CLASS, RECORDED_ERRORS }

    private record SingleResult(BufferedImage image, String name, DatasetMatch match,
                                EuroSatEngine.Classified classified, String problem) {
    }

    private record GalleryItem(EuroSatSample sample, BufferedImage image, EuroSatClassifier.Prediction prediction) {
        boolean correct() {
            return prediction.className().equals(sample.className());
        }
    }

    private record GalleryResult(List<GalleryItem> items, int skipped, String description, String problem) {
    }

    private final Consumer<String> log;
    private final EuroSatEngine engine;
    private final ScrollPane root;
    private final Random random = new Random();
    private final AtomicInteger singleRequests = new AtomicInteger();
    private final AtomicInteger galleryRequests = new AtomicInteger();
    private final AtomicInteger pending = new AtomicInteger();
    private ExecutorService worker;
    private EuroSatTestSet testSet;
    private GalleryMode galleryMode = GalleryMode.RANDOM;
    private boolean galleryRequested;
    private boolean singleShown;
    private VBox selectedTile;

    // Header actions
    private final Button randomButton = new Button("Random test image");
    private final Button openButton = new Button("Open image…");

    // Image card
    private final StackPane imageWell = new StackPane();
    private final ImageView imageView = new ImageView();
    private final VBox placeholder;
    private final Label fileLabel = Ui.label("No image yet", "file-label");
    private final Label fileSubLabel = Ui.label("PNG, JPG, BMP or TIFF; EuroSAT tiles are 64×64 px", "file-sublabel");
    private final Label sourceLabel = Ui.label("", "file-sublabel");
    private final Label datasetLabel = Ui.label("", "lu-warning");

    // Prediction card
    private final Label statusPill = Ui.fixed("Classifier not loaded", "pill");
    private final Label inferenceLabel = Ui.fixed("", "card-meta");
    private final ProgressBar progress = new ProgressBar(ProgressBar.INDETERMINATE_PROGRESS);
    private final Label predictedLabel = Ui.label("No prediction yet", "metric-value-hero", "is-empty");
    private final Label confidenceLabel = Ui.label("Open an image, drop one on the frame, or pick a random test image.",
            "metric-sub");
    private final Label verdictLabel = Ui.fixed("", "pill");
    private final Label resizeLabel = Ui.label("", "lu-warning");
    private final Label problemLabel = Ui.label("", "lu-warning");
    private final GridPane probabilityGrid = new GridPane();
    private final Label probabilityCaption = Ui.label("All classes, most likely first", "metric-caption");
    private final Label modelLabel = Ui.label("", "bench-source");
    private final Label footnote = Ui.label("Softmax outputs of the network for this one image; how well these "
            + "probabilities are calibrated was not measured.", "bench-source");

    // Gallery card
    private final ToggleGroup galleryGroup = new ToggleGroup();
    private final ToggleButton randomToggle = new ToggleButton("Random");
    private final ToggleButton perClassToggle = new ToggleButton("One per class");
    private final ToggleButton errorsToggle = new ToggleButton("Recorded errors");
    private final Button newSampleButton = new Button("New sample");
    private final Label gallerySummary = Ui.label("", "page-note");
    private final TilePane galleryTiles = new TilePane(10, 10);
    private final ProgressBar galleryProgress = new ProgressBar(ProgressBar.INDETERMINATE_PROGRESS);

    // Metrics card
    private final VBox metricsBox = new VBox(12);

    public EuroSatPage(Consumer<String> log) {
        this(log, new EuroSatEngine());
    }

    EuroSatPage(Consumer<String> log, EuroSatEngine engine) {
        this.log = log;
        this.engine = engine;

        VBox content = Ui.pageContent("Land use (EuroSAT)",
                "Scene classification: the trained ResNet labels a whole 64×64 Sentinel-2 tile (10 m per pixel) with "
                        + "one EuroSAT land-use class. A separate task from change detection; it is not applied to "
                        + "LEVIR-CD pairs.",
                null);
        HBox header = (HBox) content.getChildren().get(0);
        header.setAlignment(Pos.CENTER_LEFT);
        randomButton.setId("euroSatRandomButton");
        randomButton.getStyleClass().add("btn-secondary");
        randomButton.setMinWidth(Region.USE_PREF_SIZE);
        randomButton.setOnAction(e -> showRandomTestImage());
        openButton.setId("euroSatOpenButton");
        openButton.getStyleClass().add("btn-secondary");
        openButton.setMinWidth(Region.USE_PREF_SIZE);
        openButton.setOnAction(e -> browse());
        openButton.setTooltip(tip("Classify any image file. EuroSAT tiles are 64×64 px; other sizes are resized "
                + "and the result says so."));
        header.getChildren().addAll(randomButton, openButton);

        placeholder = buildPlaceholder();
        GridPane top = new GridPane();
        top.setHgap(14);
        ColumnConstraints left = new ColumnConstraints();
        left.setPercentWidth(40);
        ColumnConstraints right = new ColumnConstraints();
        right.setPercentWidth(60);
        top.getColumnConstraints().addAll(left, right);
        RowConstraints row = new RowConstraints();
        row.setFillHeight(true);
        top.getRowConstraints().add(row);
        VBox imageCard = buildImageCard();
        VBox predictionCard = buildPredictionCard();
        imageCard.setMaxHeight(Double.MAX_VALUE);
        predictionCard.setMaxHeight(Double.MAX_VALUE);
        top.add(imageCard, 0, 0);
        top.add(predictionCard, 1, 0);

        content.getChildren().addAll(top, buildGalleryCard(), buildMetricsCard(), buildScopeCard());
        root = Ui.page(content);
        root.setId("euroSatPage");
        refreshData();
        updateEngineStatus();
    }

    public Node getRoot() {
        return root;
    }

    /** The classifier engine (for the snapshot tool and tests). */
    public EuroSatEngine getEngine() {
        return engine;
    }

    /** True while a model load, classification or gallery task is queued or running. */
    public boolean isBusy() {
        return pending.get() > 0;
    }

    /**
     * Called each time the page is shown: re-reads the split, dataset folder and metrics file, starts the
     * model load the first time, and fills the gallery once if the images are installed.
     */
    public void onShow() {
        refreshData();
        ensureModelLoad();
        if (!galleryRequested && testSet.isOk() && testSet.imagesAvailable()) {
            newGallerySample();
        }
    }

    /** Releases the native model and stops the background thread. */
    public void shutdown() {
        engine.close();
        if (worker != null) {
            worker.shutdownNow();
        }
    }

    // ------------------------------------------------------------------ building

    private VBox buildPlaceholder() {
        SVGPath icon = new SVGPath();
        icon.setContent("M10 3v9 M6 8l4 4 4-4 M3 13v4h14v-4");
        icon.getStyleClass().add("upload-icon");
        VBox box = new VBox(6, icon, Ui.fixed("Drop an image here", "placeholder-text"),
                Ui.fixed("or click to browse", "placeholder-subtext"));
        box.setAlignment(Pos.CENTER);
        return box;
    }

    private VBox buildImageCard() {
        VBox card = Ui.card("Image", null, null);
        imageWell.getStyleClass().addAll("image-container", "drop-target");
        imageWell.setMinHeight(220);
        imageWell.setPrefHeight(300);
        imageWell.setMinWidth(0);
        VBox.setVgrow(imageWell, Priority.ALWAYS);
        imageView.setPreserveRatio(true);
        imageView.setVisible(false);
        imageWell.getChildren().addAll(placeholder, imageView);
        fitImage(imageWell, imageView, 8);
        setupDropTarget();
        sourceLabel.setVisible(false);
        sourceLabel.setManaged(false);
        datasetLabel.setVisible(false);
        datasetLabel.setManaged(false);
        card.getChildren().addAll(imageWell, new VBox(1, fileLabel, fileSubLabel), sourceLabel, datasetLabel);
        return card;
    }

    private VBox buildPredictionCard() {
        VBox card = Ui.card("Prediction", null, null);
        HBox head = (HBox) card.getChildren().get(0);
        head.setAlignment(Pos.CENTER_LEFT);
        head.getChildren().addAll(statusPill, Ui.spacer(), inferenceLabel);

        progress.setMaxWidth(Double.MAX_VALUE);
        progress.setVisible(false);
        progress.setManaged(false);

        VBox hero = new VBox(2, predictedLabel, confidenceLabel);
        hide(verdictLabel);
        hide(resizeLabel);
        hide(problemLabel);

        probabilityGrid.setHgap(10);
        probabilityGrid.setVgap(7);
        ColumnConstraints name = new ColumnConstraints();
        name.setMinWidth(Region.USE_PREF_SIZE);
        ColumnConstraints bar = new ColumnConstraints();
        bar.setHgrow(Priority.ALWAYS);
        bar.setMinWidth(60);
        bar.setFillWidth(true);
        ColumnConstraints pct = new ColumnConstraints();
        pct.setMinWidth(76);
        pct.setHalignment(HPos.RIGHT);
        ColumnConstraints marker = new ColumnConstraints();
        marker.setMinWidth(Region.USE_PREF_SIZE);
        probabilityGrid.getColumnConstraints().addAll(name, bar, pct, marker);
        hide(probabilityCaption);
        hide(probabilityGrid);
        hide(footnote);

        Region push = new Region();
        VBox.setVgrow(push, Priority.ALWAYS);
        card.getChildren().addAll(progress, hero, verdictLabel, resizeLabel, problemLabel, probabilityCaption,
                probabilityGrid, push, footnote, modelLabel);
        return card;
    }

    private VBox buildGalleryCard() {
        VBox card = Ui.card("Test-split gallery", "classified live, click a tile to inspect it", null);
        HBox head = (HBox) card.getChildren().get(0);
        head.setAlignment(Pos.CENTER_LEFT);

        randomToggle.setUserData(GalleryMode.RANDOM);
        perClassToggle.setUserData(GalleryMode.PER_CLASS);
        errorsToggle.setUserData(GalleryMode.RECORDED_ERRORS);
        randomToggle.setId("euroSatGalleryRandomToggle");
        perClassToggle.setId("euroSatGalleryPerClassToggle");
        errorsToggle.setId("euroSatGalleryErrorsToggle");
        HBox segmented = new HBox();
        segmented.getStyleClass().add("segmented");
        for (ToggleButton t : List.of(randomToggle, perClassToggle, errorsToggle)) {
            t.getStyleClass().add("mode-toggle");
            t.setMinWidth(Region.USE_PREF_SIZE);
            t.setToggleGroup(galleryGroup);
            segmented.getChildren().add(t);
        }
        randomToggle.setTooltip(tip(GALLERY_SIZE + " test images picked uniformly at random"));
        perClassToggle.setTooltip(tip("One random test image of every class"));
        errorsToggle.setTooltip(tip("Test images that the recorded evaluation (results/eurosat/test_predictions.csv) "
                + "classified wrongly, classified again live now"));
        randomToggle.setSelected(true);
        galleryGroup.selectedToggleProperty().addListener((obs, oldToggle, newToggle) -> {
            if (newToggle == null) {
                if (oldToggle != null) {
                    oldToggle.setSelected(true);
                }
                return;
            }
            galleryMode = (GalleryMode) newToggle.getUserData();
            newGallerySample();
        });
        newSampleButton.setId("euroSatNewSampleButton");
        newSampleButton.getStyleClass().add("btn-secondary");
        newSampleButton.setMinWidth(Region.USE_PREF_SIZE);
        newSampleButton.setOnAction(e -> newGallerySample());
        head.getChildren().addAll(Ui.spacer(), segmented, newSampleButton);

        galleryProgress.setMaxWidth(Double.MAX_VALUE);
        hide(galleryProgress);
        galleryTiles.setPrefTileWidth(TILE_WIDTH);
        galleryTiles.setTileAlignment(Pos.TOP_LEFT);
        galleryTiles.setAlignment(Pos.TOP_LEFT);
        card.getChildren().addAll(gallerySummary, galleryProgress, galleryTiles);
        return card;
    }

    private VBox buildMetricsCard() {
        VBox card = Ui.card("Recorded test metrics", "read from results/eurosat/test_metrics.txt", null);
        card.getChildren().add(metricsBox);
        return card;
    }

    private VBox buildScopeCard() {
        VBox card = Ui.card("What this classifier can and cannot tell you", null, null);
        card.getChildren().add(Ui.bullets(
                "What it does: it labels a whole 64×64 Sentinel-2 tile (10 m per pixel, 640 m × 640 m on the ground) "
                        + "with one EuroSAT land-use class. It does not outline areas inside the tile and it does not "
                        + "detect change.",
                "Not for LEVIR-CD: those pairs are 0.5 m per pixel Google Earth images, 20 times finer than "
                        + "Sentinel-2, showing individual buildings rather than land-use scenes. The dashboard never runs "
                        + "this classifier on change-detection pairs.",
                "Other images: anything that is not 64×64 px is stretched to 64×64 (aspect ratio not kept) and the "
                        + "prediction carries a note saying so. Such predictions are less reliable, and their accuracy "
                        + "has not been measured.",
                "Numbers: the test metrics on this page are read from the results file of one evaluation of the whole "
                        + "test split. Everything else is computed live; gallery counts describe only the images on "
                        + "screen and are not an accuracy estimate."));
        return card;
    }

    // ------------------------------------------------------------------ data, metrics, model state

    /** Re-reads the split CSVs (when their location changed or they were unreadable) and the metrics file. */
    private void refreshData() {
        Path root = AppPaths.euroSatDatasetDir();
        Path splits = AppPaths.euroSatSplitDir();
        if (testSet == null || !testSet.isOk() || !testSet.getDatasetRoot().equals(root)
                || !testSet.getSplitDir().equals(splits)) {
            testSet = EuroSatTestSet.load(root, splits);
        }
        updateDatasetState();
        buildMetrics(EuroSatTestReport.read(AppPaths.euroSatMetrics()));
    }

    private void updateDatasetState() {
        boolean available = testSet.isOk() && testSet.imagesAvailable();
        randomButton.setDisable(!available);
        newSampleButton.setDisable(!available);
        for (ToggleButton t : List.of(randomToggle, perClassToggle, errorsToggle)) {
            t.setDisable(!available);
        }
        if (!testSet.isOk()) {
            randomButton.setTooltip(tip("The EuroSAT test split could not be read: " + testSet.getProblem()));
            galleryTiles.getChildren().clear();
            gallerySummary.setText("Not available: " + testSet.getProblem() + ".");
        } else if (!available) {
            randomButton.setTooltip(tip("The EuroSAT images are not installed; see the gallery below."));
            galleryTiles.getChildren().clear();
            gallerySummary.setText(testSet.missingImagesMessage());
            galleryRequested = false;
        } else {
            randomButton.setTooltip(tip("Classify a random image of the EuroSAT test split ("
                    + String.format(Locale.ROOT, "%,d", testSet.size()) + " images in "
                    + AppPaths.shortForDisplay(testSet.getSplitDir().resolve("test.csv")) + ")"));
        }
        gallerySummary.getStyleClass().remove("lu-warning");
        if (!available) {
            gallerySummary.getStyleClass().add("lu-warning");
            datasetLabel.setText(testSet.isOk()
                    ? "EuroSAT test images not installed, so Random test image and the gallery are off (where to put "
                    + "them: see the gallery card). Open image… and drag-and-drop still work."
                    : "EuroSAT test split unavailable: " + testSet.getProblem() + ".");
            show(datasetLabel);
        } else {
            hide(datasetLabel);
        }
    }

    private void buildMetrics(EuroSatTestReport report) {
        metricsBox.getChildren().clear();
        String source = AppPaths.shortForDisplay(report.getFile());
        if (!report.isOk()) {
            MetricsReport h = report.getHeadline();
            metricsBox.getChildren().add(Ui.label(h.getStatus() == MetricsReport.Status.MISSING
                    ? "Not available: " + source + " not found. Run EvaluateEuroSatClassifier test to create it."
                    : "Not available: " + h.getProblem(), "bench-na"));
            return;
        }
        MetricsReport h = report.getHeadline();
        HBox tiles = new HBox(10,
                tile("Overall accuracy", h.format(Metric.OVERALL_ACCURACY), null),
                tile("Macro F1", h.format(Metric.MACRO_F1), null),
                tile("Test images", report.getImages().isPresent()
                                ? String.format(Locale.ROOT, "%,d", report.getImages().getAsInt()) : "n/a",
                        report.getCorrect().isPresent()
                                ? String.format(Locale.ROOT, "%,d correct, %,d wrong", report.getCorrect().getAsInt(),
                                report.getWrong().getAsInt()) : null));

        VBox leftCol = new VBox(12, tiles);
        if (!report.getConfusions().isEmpty()) {
            VBox conf = new VBox(4, Ui.label("Most frequent confusions (actual → predicted: images)", "metric-caption"));
            int shown = 0;
            for (EuroSatTestReport.Confusion c : report.getConfusions()) {
                if (shown++ == 6) {
                    break;
                }
                conf.getChildren().add(Ui.label(c.actual() + " → " + c.predicted() + ": " + c.count(), "page-note"));
            }
            leftCol.getChildren().add(conf);
        }
        StringBuilder src = new StringBuilder("Source: ").append(source);
        report.getGenerated().ifPresent(g -> src.append(", generated ").append(g.length() > 16 ? g.substring(0, 16) : g));
        src.append('.');
        report.getProtocol().ifPresent(p -> src.append(" Protocol: ").append(p));
        leftCol.getChildren().add(Ui.label(src.toString(), "bench-source"));

        Node right = report.getPerClass().isEmpty()
                ? Ui.label("Per-class table not available: " + report.getPerClassProblem() + ".", "bench-na")
                : perClassTable(report.getPerClass());

        GridPane grid = new GridPane();
        grid.setHgap(24);
        ColumnConstraints c1 = new ColumnConstraints();
        c1.setPercentWidth(42);
        ColumnConstraints c2 = new ColumnConstraints();
        c2.setPercentWidth(58);
        grid.getColumnConstraints().addAll(c1, c2);
        grid.add(leftCol, 0, 0);
        grid.add(right, 1, 0);
        GridPane.setValignment(leftCol, VPos.TOP);
        GridPane.setValignment(right, VPos.TOP);
        metricsBox.getChildren().add(grid);
    }

    private static GridPane perClassTable(List<EuroSatTestReport.ClassRow> rows) {
        GridPane table = new GridPane();
        table.getStyleClass().add("bench-table");
        String[] headers = {"Class", "Precision", "Recall", "F1", "Images"};
        for (int i = 0; i < headers.length; i++) {
            ColumnConstraints c = new ColumnConstraints();
            if (i == 0) {
                c.setHgrow(Priority.ALWAYS);
                c.setMinWidth(Region.USE_PREF_SIZE);
            } else {
                c.setMinWidth(66);
                c.setHalignment(HPos.RIGHT);
            }
            table.getColumnConstraints().add(c);
            Label h = Ui.fixed(headers[i], "bench-header");
            h.setMaxWidth(Double.MAX_VALUE);
            h.setAlignment(i == 0 ? Pos.CENTER_LEFT : Pos.CENTER_RIGHT);
            table.add(h, i, 0);
        }
        int r = 1;
        for (EuroSatTestReport.ClassRow row : rows) {
            String[] values = {row.className(),
                    String.format(Locale.ROOT, "%.4f", row.precision()),
                    String.format(Locale.ROOT, "%.4f", row.recall()),
                    String.format(Locale.ROOT, "%.4f", row.f1()),
                    String.format(Locale.ROOT, "%,d", row.support())};
            for (int i = 0; i < values.length; i++) {
                Label v = Ui.fixed(values[i], i == 0 ? "bench-model" : (i == 3 ? "bench-value-primary" : "bench-value"));
                VBox cell = new VBox(v);
                cell.setAlignment(i == 0 ? Pos.CENTER_LEFT : Pos.CENTER_RIGHT);
                cell.getStyleClass().addAll("bench-cell", "lu-table-cell");
                cell.setMaxWidth(Double.MAX_VALUE);
                table.add(cell, i, r);
            }
            r++;
        }
        return table;
    }

    private static VBox tile(String caption, String value, String sub) {
        VBox t = new VBox(2, Ui.fixed(caption, "metric-caption"), Ui.fixed(value, "metric-value"));
        if (sub != null) {
            t.getChildren().add(Ui.fixed(sub, "metric-sub"));
        }
        t.getStyleClass().add("metric-tile");
        return t;
    }

    private void updateEngineStatus() {
        statusPill.getStyleClass().removeAll("pill-warning", "pill-negative", "pill-positive");
        switch (engine.getState()) {
            case READY:
                statusPill.setText("Classifier ready · " + EuroSatResultFormat.deviceText(engine.getDeviceName()));
                modelLabel.setText("Model: " + AppPaths.shortForDisplay(engine.getModelDir()) + " · " + engine.getSummary()
                        + ". Inference runs on a background thread.");
                break;
            case LOADING:
                statusPill.setText("Loading classifier…");
                statusPill.getStyleClass().add("pill-warning");
                modelLabel.setText("Loading " + AppPaths.shortForDisplay(engine.getModelDir()) + " (device "
                        + engine.getRequestedDevice() + ") in the background.");
                break;
            case FAILED:
            case CLOSED:
                statusPill.setText("Classifier unavailable");
                statusPill.getStyleClass().add("pill-negative");
                modelLabel.setText("EuroSAT classifier unavailable: " + engine.getStatusNote()
                        + ". Set -D" + AppPaths.EUROSAT_MODEL_DIR_PROPERTY + "=<folder> to use another model folder.");
                break;
            case NOT_LOADED:
            default:
                statusPill.setText("Classifier not loaded");
                modelLabel.setText("Model: " + AppPaths.shortForDisplay(engine.getModelDir())
                        + ", loaded in the background when this page is first opened.");
                break;
        }
    }

    // ------------------------------------------------------------------ background work

    private ExecutorService worker() {
        if (worker == null) {
            worker = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r, "landwatch-eurosat");
                t.setDaemon(true);
                return t;
            });
        }
        return worker;
    }

    /** Runs {@code task} on the page's single worker thread; the handlers run on the JavaFX thread. */
    private <T> void submit(Task<T> task, Consumer<T> onSuccess, Consumer<Throwable> onFailure) {
        pending.incrementAndGet();
        task.setOnSucceeded(e -> {
            try {
                onSuccess.accept(task.getValue());
            } finally {
                pending.decrementAndGet();
            }
        });
        task.setOnFailed(e -> {
            try {
                onFailure.accept(task.getException());
            } finally {
                pending.decrementAndGet();
            }
        });
        task.setOnCancelled(e -> pending.decrementAndGet());
        worker().execute(task);
    }

    /** Queues the model load the first time it is needed (it then runs before any queued classification). */
    private void ensureModelLoad() {
        if (!engine.needsLoad()) {
            return;
        }
        engine.markLoading();
        updateEngineStatus();
        log.accept("Loading the EuroSAT land-use classifier from " + AppPaths.shortForDisplay(engine.getModelDir())
                + " in the background (device: " + engine.getRequestedDevice() + ").");
        Task<Boolean> load = new Task<>() {
            @Override
            protected Boolean call() {
                return engine.load();
            }
        };
        submit(load, ok -> onModelLoadFinished(), error -> {
            log.accept("ERROR while loading the EuroSAT classifier: " + error);
            onModelLoadFinished();
        });
    }

    private void onModelLoadFinished() {
        updateEngineStatus();
        if (engine.isAvailable()) {
            log.accept("EuroSAT classifier ready: " + engine.getSummary() + ".");
        } else {
            log.accept("EuroSAT classifier unavailable: " + engine.getStatusNote() + ".");
        }
    }

    // ------------------------------------------------------------------ single image

    /** Picks a random test image and classifies it. */
    public void showRandomTestImage() {
        if (!testSet.isOk() || !testSet.imagesAvailable()) {
            return;
        }
        EuroSatSample sample = testSet.pickOne(random);
        classifyFile(testSet.resolve(sample));
    }

    /** Loads {@code file} and classifies it on the background thread. */
    public void classifyFile(Path file) {
        int id = singleRequests.incrementAndGet();
        EuroSatTestSet set = testSet;
        String name = file.getFileName().toString();
        setRunning(true, "Classifying " + name + "…");
        ensureModelLoad();
        Task<SingleResult> task = new Task<>() {
            @Override
            protected SingleResult call() throws Exception {
                BufferedImage image = SatelliteImageLoader.load(file.toFile());
                DatasetMatch match = set.lookup(file).orElse(null);
                Optional<String> levir = EuroSatResultFormat.levirCdRefusal(file);
                if (levir.isPresent()) {
                    return new SingleResult(image, name, null, null, levir.get());
                }
                if (!engine.isAvailable()) {
                    return new SingleResult(image, name, match, null, "Not classified: the EuroSAT classifier is "
                            + "unavailable (" + engine.getStatusNote() + ").");
                }
                return new SingleResult(image, name, match, engine.classify(image), null);
            }
        };
        submit(task, result -> {
            if (id == singleRequests.get()) {
                setRunning(false, null);
                showSingle(result);
            }
        }, error -> {
            if (id == singleRequests.get()) {
                setRunning(false, null);
                showProblem("Could not classify " + name + ": " + messageOf(error));
                log.accept("Land use: could not classify " + name + ": " + messageOf(error));
            }
        });
    }

    private void showSingle(SingleResult r) {
        selectTile(null);
        showImage(r.image(), r.name(), r.match());
        if (r.classified() == null) {
            showProblem(r.problem());
            if (r.match() == null) {
                hide(sourceLabel);   // nothing was predicted, so there is nothing to check
            }
            log.accept("Land use: " + r.name() + ". " + r.problem());
            return;
        }
        EuroSatClassifier.Prediction p = r.classified().prediction();
        showPrediction(p, r.match());
        inferenceLabel.setText(EuroSatResultFormat.latencyText(r.classified().millis(), engine.getDeviceName()));
        StringBuilder line = new StringBuilder("Land use: ").append(r.name()).append(" → ").append(p.className())
                .append(" (").append(EuroSatResultFormat.percent(p.probability())).append(')');
        if (r.match() != null) {
            line.append(", true label ").append(r.match().trueClass())
                    .append(r.match().inKnownSplit() ? " (" + r.match().split() + " split)" : "");
        }
        if (p.wasResized()) {
            line.append("; resized from ").append(r.image().getWidth()).append('x').append(r.image().getHeight());
        }
        log.accept(line.append('.').toString());
    }

    private void showImage(BufferedImage image, String name, DatasetMatch match) {
        singleShown = true;
        boolean small = Math.max(image.getWidth(), image.getHeight()) <= 128;
        BufferedImage shown = small
                ? ImageFit.enlargeNearest(image, ImageFit.nearestScale(image.getWidth(), image.getHeight(), 512))
                : image;
        imageView.setSmooth(true);
        imageView.setImage(SwingFXUtils.toFXImage(shown, null));
        imageView.setVisible(true);
        placeholder.setVisible(false);
        if (!imageWell.getStyleClass().contains("has-image")) {
            imageWell.getStyleClass().add("has-image");
        }
        fileLabel.setText(name);
        fileSubLabel.setText(EuroSatResultFormat.sizeText(image.getWidth(), image.getHeight())
                + (small ? " · enlarged with nearest-neighbour scaling, so the individual pixels stay visible" : ""));
        sourceLabel.setText(EuroSatResultFormat.sourceSentence(match));
        sourceLabel.getStyleClass().remove("lu-warning");
        if (match != null && !match.isTestImage()) {
            sourceLabel.getStyleClass().add("lu-warning");
        }
        show(sourceLabel);
    }

    private void showPrediction(EuroSatClassifier.Prediction p, DatasetMatch match) {
        hide(problemLabel);
        predictedLabel.getStyleClass().remove("is-empty");
        predictedLabel.setText(p.className());
        List<String> names = engine.getClassNames();
        confidenceLabel.setText("Probability " + EuroSatResultFormat.percent(p.probability()) + ", the highest of "
                + names.size() + " classes");

        verdictLabel.getStyleClass().removeAll("pill-positive", "pill-negative");
        if (match != null) {
            boolean correct = p.className().equals(match.trueClass());
            verdictLabel.setText(EuroSatResultFormat.verdict(p.className(), match.trueClass()));
            verdictLabel.getStyleClass().add(correct ? "pill-positive" : "pill-negative");
            show(verdictLabel);
        } else {
            hide(verdictLabel);
        }
        if (p.wasResized()) {
            resizeLabel.setText("Resized input: " + p.note() + ".");
            show(resizeLabel);
        } else {
            hide(resizeLabel);
        }

        int trueIndex = match == null ? -1 : names.indexOf(match.trueClass());
        probabilityGrid.getChildren().clear();
        int row = 0;
        for (ProbabilityRow pr : EuroSatResultFormat.sortedRows(p.probabilities(), names, trueIndex)) {
            Label name = Ui.fixed(pr.className(), pr.predicted() ? "lu-prob-name-top" : "lu-prob-name");
            StackPane track = new StackPane();
            track.getStyleClass().add("lu-prob-track");
            track.setAlignment(Pos.CENTER_LEFT);
            track.setMaxWidth(Double.MAX_VALUE);
            Region fill = new Region();
            fill.getStyleClass().add(pr.predicted() ? "lu-prob-fill-top" : "lu-prob-fill");
            fill.maxWidthProperty().bind(track.widthProperty().multiply(EuroSatResultFormat.barFraction(pr.probability())));
            fill.prefWidthProperty().bind(fill.maxWidthProperty());
            fill.setMinWidth(0);
            track.getChildren().add(fill);
            Label pct = Ui.fixed(EuroSatResultFormat.percent(pr.probability()),
                    pr.predicted() ? "lu-prob-value-top" : "lu-prob-value");
            Label marker = Ui.fixed(pr.trueClass() ? "true class" : "", pr.trueClass() ? "pill" : "lu-prob-value");
            probabilityGrid.addRow(row++, name, track, pct, marker);
            for (Node n : List.of(name, track, pct, marker)) {
                GridPane.setValignment(n, VPos.CENTER);
            }
        }
        show(probabilityCaption);
        show(probabilityGrid);
        show(footnote);
    }

    private void showProblem(String text) {
        predictedLabel.setText("Not classified");
        if (!predictedLabel.getStyleClass().contains("is-empty")) {
            predictedLabel.getStyleClass().add("is-empty");
        }
        confidenceLabel.setText("");
        problemLabel.setText(text);
        show(problemLabel);
        hide(verdictLabel);
        hide(resizeLabel);
        hide(probabilityCaption);
        hide(probabilityGrid);
        hide(footnote);
        inferenceLabel.setText("");
    }

    private void setRunning(boolean running, String text) {
        progress.setVisible(running);
        progress.setManaged(running);
        if (running) {
            inferenceLabel.setText(text);
        }
    }

    // ------------------------------------------------------------------ gallery

    private void newGallerySample() {
        if (!testSet.isOk() || !testSet.imagesAvailable()) {
            updateDatasetState();
            return;
        }
        galleryRequested = true;
        int id = galleryRequests.incrementAndGet();
        List<EuroSatSample> picks;
        String description;
        switch (galleryMode) {
            case PER_CLASS:
                picks = testSet.pickPerClass(1, random);
                description = "One random test image of each class; not an accuracy estimate.";
                break;
            case RECORDED_ERRORS: {
                Path csv = AppPaths.euroSatTestPredictions();
                List<String> errors;
                try {
                    errors = EuroSatTestSet.readRecordedErrors(csv);
                } catch (IOException | RuntimeException e) {
                    hide(galleryProgress);
                    galleryTiles.getChildren().clear();
                    gallerySummary.setText("Recorded errors not available: " + (Files.isRegularFile(csv)
                            ? messageOf(e) : AppPaths.shortForDisplay(csv) + " not found") + ".");
                    return;
                }
                picks = testSet.pickAmong(errors, GALLERY_SIZE, random);
                description = String.format(Locale.ROOT, "Picked from the %d test images that the recorded evaluation "
                                + "classified wrongly (%s) and classified again live now. Chosen to show failures, so "
                                + "deliberately not representative.", errors.size(),
                        AppPaths.shortForDisplay(csv));
                break;
            }
            case RANDOM:
            default:
                picks = testSet.pickRandom(GALLERY_SIZE, random);
                description = "A small uniformly random sample of the test split, far too few images to estimate "
                        + "accuracy; the recorded metrics below cover the whole split.";
                break;
        }
        gallerySummary.setText("Classifying " + picks.size() + " test images…");
        show(galleryProgress);
        ensureModelLoad();
        EuroSatTestSet set = testSet;
        List<EuroSatSample> chosen = picks;
        String desc = description;
        Task<GalleryResult> task = new Task<>() {
            @Override
            protected GalleryResult call() {
                List<EuroSatSample> read = new ArrayList<>();
                List<BufferedImage> images = new ArrayList<>();
                int skipped = 0;
                for (EuroSatSample s : chosen) {
                    try {
                        images.add(SatelliteImageLoader.load(set.resolve(s).toFile()));
                        read.add(s);
                    } catch (IOException | RuntimeException e) {
                        skipped++;
                    }
                }
                if (!engine.isAvailable()) {
                    return new GalleryResult(List.of(), skipped, desc, engine.getStatusNote());
                }
                List<EuroSatClassifier.Prediction> predictions = images.isEmpty()
                        ? List.of() : engine.classifyAll(images);
                List<GalleryItem> items = new ArrayList<>();
                for (int i = 0; i < read.size(); i++) {
                    items.add(new GalleryItem(read.get(i), images.get(i), predictions.get(i)));
                }
                return new GalleryResult(items, skipped, desc, null);
            }
        };
        submit(task, result -> {
            if (id == galleryRequests.get()) {
                hide(galleryProgress);
                showGallery(result);
            }
        }, error -> {
            if (id == galleryRequests.get()) {
                hide(galleryProgress);
                galleryTiles.getChildren().clear();
                gallerySummary.setText("The gallery could not be classified: " + messageOf(error));
            }
        });
    }

    private void showGallery(GalleryResult result) {
        galleryTiles.getChildren().clear();
        selectedTile = null;
        if (result.problem() != null) {
            gallerySummary.setText("Not classified: the EuroSAT classifier is unavailable (" + result.problem() + ").");
            return;
        }
        int correct = 0;
        for (GalleryItem item : result.items()) {
            if (item.correct()) {
                correct++;
            }
            galleryTiles.getChildren().add(galleryTile(item));
        }
        gallerySummary.setText(EuroSatResultFormat.sampleSummary(correct, result.items().size(), result.skipped(),
                result.description()));
        log.accept(String.format(Locale.ROOT, "Land use gallery (%s): %d of %d test images classified correctly.",
                galleryMode.name().toLowerCase(Locale.ROOT).replace('_', ' '), correct, result.items().size()));
        if (!singleShown && singleRequests.get() == 0 && !result.items().isEmpty()) {
            showGalleryItem(result.items().get(0), (VBox) galleryTiles.getChildren().get(0));
        }
    }

    private VBox galleryTile(GalleryItem item) {
        BufferedImage img = item.image();
        ImageView view = new ImageView(SwingFXUtils.toFXImage(ImageFit.enlargeNearest(img,
                ImageFit.nearestScale(img.getWidth(), img.getHeight(), (int) TILE_IMAGE)), null));
        view.setFitWidth(TILE_IMAGE);
        view.setFitHeight(TILE_IMAGE);
        view.setPreserveRatio(true);
        view.setSmooth(true);
        StackPane frame = new StackPane(view);
        frame.setMinSize(TILE_IMAGE, TILE_IMAGE);
        frame.setPrefSize(TILE_IMAGE, TILE_IMAGE);
        frame.setMaxSize(TILE_IMAGE, TILE_IMAGE);

        EuroSatClassifier.Prediction p = item.prediction();
        boolean correct = item.correct();
        Label pred = Ui.label(p.className(), "gallery-pred");
        Label mark = Ui.label((correct ? "✓ correct · " : "✗ wrong · ") + EuroSatResultFormat.percent(p.probability()),
                correct ? "gallery-ok" : "gallery-wrong");
        Label truth = Ui.label("true: " + item.sample().className(), "gallery-sub");
        for (Label l : List.of(pred, mark, truth)) {
            l.setMaxWidth(TILE_IMAGE);
        }
        VBox tile = new VBox(3, frame, pred, mark, truth);
        tile.getStyleClass().add("gallery-tile");
        if (!correct) {
            tile.getStyleClass().add("gallery-tile-wrong");
        }
        tile.setPrefWidth(TILE_WIDTH);
        tile.setMaxWidth(TILE_WIDTH);
        Tooltip.install(tile, tip(item.sample().relativePath() + "\nPredicted " + p.className() + " ("
                + EuroSatResultFormat.percent(p.probability()) + "), true " + item.sample().className()
                + "\nClick to show all class probabilities above."));
        tile.setOnMouseClicked(e -> {
            if (e.getButton() == MouseButton.PRIMARY) {
                showGalleryItem(item, tile);
                root.setVvalue(0);
            }
        });
        return tile;
    }

    /** Shows a gallery item in the image and prediction cards (its prediction is already computed). */
    private void showGalleryItem(GalleryItem item, VBox tile) {
        singleRequests.incrementAndGet();   // a pending single classification must not overwrite this
        setRunning(false, null);
        selectTile(tile);
        Path file = testSet.resolve(item.sample());
        DatasetMatch match = testSet.lookup(file).orElse(
                new DatasetMatch(item.sample().className(), item.sample().classIndex(), "test"));
        showImage(item.image(), file.getFileName().toString(), match);
        showPrediction(item.prediction(), match);
        inferenceLabel.setText("from the gallery batch · " + EuroSatResultFormat.deviceText(engine.getDeviceName()));
    }

    private void selectTile(VBox tile) {
        if (selectedTile != null) {
            selectedTile.getStyleClass().remove("gallery-tile-selected");
        }
        selectedTile = tile;
        if (tile != null) {
            tile.getStyleClass().add("gallery-tile-selected");
        }
    }

    // ------------------------------------------------------------------ input: browse and drop

    private void browse() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Open an image to classify");
        chooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("Images (*.png, *.jpg, *.tif, *.bmp)", "*.png", "*.jpg", "*.jpeg",
                        "*.tif", "*.tiff", "*.bmp"),
                new FileChooser.ExtensionFilter("All Files", "*.*"));
        Path start = testSet != null ? testSet.getImagesDir() : null;
        if (start != null && Files.isDirectory(start)) {
            chooser.setInitialDirectory(start.toFile());
        }
        Window window = root.getScene() != null ? root.getScene().getWindow() : null;
        File file = chooser.showOpenDialog(window);
        if (file != null) {
            classifyFile(file.toPath());
        }
    }

    private void setupDropTarget() {
        imageWell.setOnDragOver(e -> {
            if (e.getDragboard().hasFiles()) {
                e.acceptTransferModes(TransferMode.COPY);
            }
            e.consume();
        });
        imageWell.setOnDragEntered(e -> {
            if (e.getDragboard().hasFiles() && !imageWell.getStyleClass().contains("drag-over")) {
                imageWell.getStyleClass().add("drag-over");
            }
        });
        imageWell.setOnDragExited(e -> imageWell.getStyleClass().remove("drag-over"));
        imageWell.setOnDragDropped(e -> {
            boolean done = false;
            if (e.getDragboard().hasFiles()) {
                done = loadDroppedFiles(e.getDragboard().getFiles());
            }
            e.setDropCompleted(done);
            e.consume();
        });
        imageWell.setOnMouseClicked(e -> {
            if (e.getButton() == MouseButton.PRIMARY && imageView.getImage() == null) {
                browse();
            }
        });
    }

    /** Classifies the first image among dropped files (PNG, JPG, BMP or TIFF); false if there is none. */
    public boolean loadDroppedFiles(List<File> files) {
        for (File f : files) {
            String n = f.getName().toLowerCase(Locale.ROOT);
            if (f.isFile() && (n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".jpeg") || n.endsWith(".bmp")
                    || n.endsWith(".tif") || n.endsWith(".tiff"))) {
                if (files.size() > 1) {
                    log.accept("Land use: " + files.size() + " files dropped; classifying the first image, " + f.getName() + ".");
                }
                classifyFile(f.toPath());
                return true;
            }
        }
        log.accept("Land use: ignored the dropped file(s); only PNG, JPG, BMP and TIFF images can be classified.");
        return false;
    }

    // ------------------------------------------------------------------ helpers

    /** Fits an image inside its frame ({@code inset} px inside it), centred, as the change page does. */
    private static void fitImage(StackPane container, ImageView view, double inset) {
        view.setManaged(false);
        Runnable layout = () -> {
            Image image = view.getImage();
            double boxW = container.getWidth() - 2 * inset;
            double boxH = container.getHeight() - 2 * inset;
            if (image == null || boxW <= 0 || boxH <= 0) {
                return;
            }
            double[] size = ImageFit.fit(image.getWidth(), image.getHeight(), boxW, boxH);
            view.setFitWidth(size[0]);
            view.setFitHeight(size[1]);
            view.relocate((container.getWidth() - size[0]) / 2, (container.getHeight() - size[1]) / 2);
        };
        container.widthProperty().addListener((obs, o, n) -> layout.run());
        container.heightProperty().addListener((obs, o, n) -> layout.run());
        view.imageProperty().addListener((obs, o, n) -> layout.run());
    }

    private static void show(Node n) {
        n.setVisible(true);
        n.setManaged(true);
    }

    private static void hide(Node n) {
        n.setVisible(false);
        n.setManaged(false);
    }

    private static Tooltip tip(String text) {
        Tooltip t = new Tooltip(text);
        t.setWrapText(true);
        t.setMaxWidth(320);
        return t;
    }

    private static String messageOf(Throwable e) {
        if (e == null) {
            return "unknown error";
        }
        String m = e.getMessage();
        return (m == null || m.isBlank()) ? e.getClass().getSimpleName() : m;
    }
}
