package com.landwatch.ui.pages;

import com.landwatch.deeplearning.DeepChangeDetector;
import com.landwatch.deeplearning.DjlRuntime;
import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.engine.SiameseUNetEngine;
import javafx.scene.Node;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.VBox;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * "About / Runtime" page: what is actually running (Java, JavaFX, DJL engine and device), where the model
 * and data are expected and whether they were found. Everything is read live when the page is shown.
 */
public final class AboutPage {

    private final SiameseUNetEngine deepEngine;
    private final ScrollPane root;
    private final GridPane runtimeGrid = Ui.kvGrid();
    private final GridPane modelGrid = Ui.kvGrid();
    private final GridPane dataGrid = Ui.kvGrid();
    private final TextArea modelDescription = Ui.monoArea(9);

    public AboutPage(SiameseUNetEngine deepEngine) {
        this.deepEngine = deepEngine;
        VBox content = Ui.pageContent("About / Runtime",
                "LANDWATCH AI · Problem Statement 50: satellite land-use change and encroachment detection. "
                        + "This page reports the live runtime configuration; nothing here is configured from the UI.",
                "Academic DL PBL");

        VBox runtimeCard = Ui.card("RUNTIME", null, null);
        runtimeCard.getChildren().add(runtimeGrid);
        VBox modelCard = Ui.card("DEEP MODEL", null, null);
        modelCard.getChildren().addAll(modelGrid, modelDescription);
        VBox dataCard = Ui.card("DATASET & RESULTS FILES", null, null);
        dataCard.getChildren().add(dataGrid);

        content.getChildren().addAll(modelCard, runtimeCard, dataCard);
        root = Ui.page(content);
        root.setId("aboutPage");
        refresh();
    }

    public Node getRoot() {
        return root;
    }

    /** Re-reads the runtime state (called each time the page is shown). */
    public void refresh() {
        runtimeGrid.getChildren().clear();
        Runtime rt = Runtime.getRuntime();
        Ui.kv(runtimeGrid, "Java", System.getProperty("java.version") + " (" + System.getProperty("java.vendor")
                + ", " + System.getProperty("java.vm.name") + ")");
        Ui.kv(runtimeGrid, "JavaFX", String.valueOf(System.getProperty("javafx.runtime.version", "?")));
        Ui.kv(runtimeGrid, "OS", System.getProperty("os.name") + " " + System.getProperty("os.version")
                + ", " + System.getProperty("os.arch") + ", " + rt.availableProcessors() + " CPU cores visible");
        Ui.kv(runtimeGrid, "JVM heap", String.format(Locale.ROOT, "max %d MB", rt.maxMemory() / (1024 * 1024)));
        Ui.kv(runtimeGrid, "Working dir", AppPaths.forDisplay(Path.of("")));

        modelGrid.getChildren().clear();
        Path dir = deepEngine.getModelDir();
        String source = dir.equals(Path.of(SiameseUNetEngine.DEFAULT_MODEL_DIR))
                ? "default" : "set by -D" + SiameseUNetEngine.MODEL_DIR_PROPERTY;
        Ui.kv(modelGrid, "Model dir", dir + "  (" + source + ")");
        Ui.kv(modelGrid, "Absolute", AppPaths.forDisplay(dir) + (Files.isDirectory(dir) ? "  [found]" : "  [NOT FOUND]"));
        Ui.kv(modelGrid, "Status", stateText(), deepEngine.isAvailable() ? "kv-ok" : "kv-warn");
        Ui.kv(modelGrid, "Device", "requested " + System.getProperty(DeepChangeDetector.DEVICE_PROPERTY, "auto")
                + " (-D" + DeepChangeDetector.DEVICE_PROPERTY + "), in use: "
                + (deepEngine.getDeviceName() != null ? deepEngine.getDeviceName() : "none (no model loaded)"));
        Ui.kv(modelGrid, "DJL", djlText());
        String desc = deepEngine.getDescription();
        modelDescription.setText(desc != null ? desc
                : "No model loaded, so no description. Expected a directory with model.properties and a *.params "
                + "weights file written by com.landwatch.deeplearning.TrainChangeDetector.");

        dataGrid.getChildren().clear();
        Path levir = AppPaths.levirDatasetDir();
        Ui.kv(dataGrid, "LEVIR-CD", AppPaths.forDisplay(levir) + (Files.isDirectory(levir) ? "  [found]" : "  [NOT FOUND]"),
                Files.isDirectory(levir) ? "kv-ok" : "kv-warn");
        for (String split : List.of("train", "val", "test")) {
            Path a = levir.resolve(split).resolve("A");
            Ui.kv(dataGrid, "  " + split, Files.isDirectory(a) ? countImages(a) + " image pairs (A/)" : "not found");
        }
        for (Path p : List.of(AppPaths.baselineMetrics("val"), AppPaths.baselineMetrics("test"),
                AppPaths.baselinePerImage("test"), AppPaths.deepMetrics("val"), AppPaths.deepMetrics("test"),
                AppPaths.euroSatMetrics())) {
            boolean found = Files.isRegularFile(p);
            Ui.kv(dataGrid, found ? "found" : "missing", BenchmarksPage.relative(p), found ? "kv-ok" : "kv-warn");
        }
    }

    private String stateText() {
        switch (deepEngine.getState()) {
            case READY: return "ready: " + deepEngine.getStatusNote();
            case LOADING: return "loading…";
            case FAILED: return "unavailable: " + deepEngine.getStatusNote();
            case CLOSED: return "closed";
            case NOT_LOADED:
            default: return "not loaded";
        }
    }

    private String djlText() {
        String version;
        try {
            version = "DJL " + DjlRuntime.djlVersion();
        } catch (RuntimeException | LinkageError e) {
            return "DJL not available: " + e.getMessage();
        }
        if (deepEngine.isAvailable()) {
            try {
                return version + " / " + DjlRuntime.engineName() + " " + DjlRuntime.engineVersion();
            } catch (RuntimeException | LinkageError e) {
                return version + " (engine: " + e.getMessage() + ")";
            }
        }
        return version + " (native engine not started: no model loaded)";
    }

    private static long countImages(Path dir) {
        try (Stream<Path> s = Files.list(dir)) {
            return s.filter(p -> {
                String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                return !n.startsWith(".") && (n.endsWith(".png") || n.endsWith(".jpg") || n.endsWith(".tif"));
            }).count();
        } catch (IOException e) {
            return -1;
        }
    }
}
