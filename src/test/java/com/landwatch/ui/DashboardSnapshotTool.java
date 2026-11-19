package com.landwatch.ui;

import com.landwatch.ui.controllers.DashboardController;
import com.landwatch.ui.engine.ChangeDetectionEngine;
import com.landwatch.ui.engine.EngineResult;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXMLLoader;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Labeled;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.ToggleButton;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;

import javax.imageio.ImageIO;
import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

/**
 * On-demand visual-evidence tool (NOT a unit test; it is not picked up by Surefire).
 *
 * <p>Loads dashboard.fxml with the real controller and stylesheet in an off-screen Stage, clicks
 * "Load Demo Pair" and "Analyze Changes", then saves {@code scene.snapshot()} PNGs of every view mode
 * at 1280x840 and 1100x760 into {@code results/ui_snapshots/}. No window is shown; the tool exits with
 * {@link Platform#exit()}.</p>
 *
 * <p>Run (after {@code mvn test-compile}) with target/classes, target/test-classes and the JavaFX jars
 * on the classpath: {@code java -cp ... com.landwatch.ui.DashboardSnapshotTool [outputDir]}.</p>
 */
public final class DashboardSnapshotTool {

    private static final int[][] SIZES = {{1280, 840}, {1100, 760}};

    private DashboardSnapshotTool() {
    }

    public static void main(String[] args) throws Exception {
        File outDir = new File(args.length > 0 ? args[0] : "results/ui_snapshots");
        if (!outDir.exists() && !outDir.mkdirs()) {
            throw new IllegalStateException("Cannot create " + outDir);
        }

        CountDownLatch started = new CountDownLatch(1);
        Platform.setImplicitExit(false);
        Platform.startup(started::countDown);
        started.await(10, TimeUnit.SECONDS);

        List<String> written = new ArrayList<>();
        int exitCode = 0;
        try {
            for (int[] size : SIZES) {
                written.addAll(captureAtSize(size[0], size[1], outDir));
            }
        } catch (Throwable t) {
            t.printStackTrace();
            exitCode = 1;
        } finally {
            Platform.exit();
        }
        written.forEach(p -> System.out.println("[snapshot] " + p));
        System.exit(exitCode);
    }

    private static List<String> captureAtSize(int width, int height, File outDir) throws Exception {
        List<String> files = new ArrayList<>();
        String tag = width + "x" + height;

        // Build stage + scene exactly like MainApp (same FXML, controller and stylesheet).
        Object[] holder = onFx(() -> {
            URL fxml = DashboardSnapshotTool.class.getResource("/com/landwatch/ui/views/dashboard.fxml");
            FXMLLoader loader = new FXMLLoader(fxml);
            Parent root = loader.load();
            Scene scene = new Scene(root, width, height);
            URL css = DashboardSnapshotTool.class.getResource("/com/landwatch/ui/css/dark_theme.css");
            scene.getStylesheets().add(css.toExternalForm());
            Stage stage = new Stage();
            stage.setScene(scene);
            stage.setWidth(width);
            stage.setHeight(height);
            root.applyCss();
            root.layout();
            return new Object[]{scene, loader.getController(), stage};
        });
        Scene scene = (Scene) holder[0];
        DashboardController controller = (DashboardController) holder[1];

        files.add(save(scene, outDir, "dashboard_" + tag + "_0_initial.png"));

        // Click "Load Demo Pair", then "Analyze Changes" (real button handlers).
        onFx(() -> {
            ((Button) scene.lookup("#loadDemoButton")).fire();
            ((Button) scene.lookup("#analyzeButton")).fire();
            return null;
        });
        waitForAnalysis(controller);

        EngineResult result = onFx(controller::getLastEngineResult);
        if (result == null) {
            throw new IllegalStateException("Analysis produced no result");
        }
        System.out.printf("[analysis %s] engine=%s changed=%d px (%.2f %%) regions=%d otsu=%s latency=%d ms%n",
                tag, result.getEngineName(), result.getChangedPixels(), result.getChangePercentage(),
                result.getRegionCount(), result.getOtsuThreshold(), result.getLatencyMs());
        printEngineSelector(scene);

        // Top of page after loading (header, inputs, engine selector)
        onFx(() -> {
            ((ScrollPane) scene.lookup("#contentScrollPane")).setVvalue(0);
            return null;
        });
        files.add(save(scene, outDir, "dashboard_" + tag + "_1_top_after_analysis.png"));

        String[][] modes = {
                {"#modeMaskToggle", "2_mode_binary_mask"},
                {"#modeOverlayToggle", "3_mode_red_overlay"},
                {"#modeChangeMapToggle", "4_mode_change_map"}};
        for (String[] mode : modes) {
            onFx(() -> {
                ToggleButton toggle = (ToggleButton) scene.lookup(mode[0]);
                toggle.getToggleGroup().selectToggle(toggle);
                controller.scrollToResults();
                return null;
            });
            files.add(save(scene, outDir, "dashboard_" + tag + "_" + mode[1] + ".png"));
            reportTruncation(scene, tag + " " + mode[1]);
        }

        onFx(() -> {
            ((Stage) holder[2]).close();
            return null;
        });
        return files;
    }

    private static void waitForAnalysis(DashboardController controller) throws Exception {
        long deadline = System.currentTimeMillis() + 30_000;
        while (System.currentTimeMillis() < deadline) {
            boolean done = onFx(() -> !controller.isAnalysisRunning() && controller.getLastEngineResult() != null);
            if (done) {
                return;
            }
            Thread.sleep(100);
        }
        throw new IllegalStateException("Analysis did not finish within 30 s");
    }

    private static String save(Scene scene, File outDir, String name) throws Exception {
        WritableImage image = onFx(() -> {
            scene.getRoot().applyCss();
            scene.getRoot().layout();
            return scene.snapshot(null);
        });
        File file = new File(outDir, name);
        ImageIO.write(SwingFXUtils.fromFXImage(image, null), "png", file);
        return file.getPath();
    }

    /** Prints any visible Labeled whose rendered text is truncated (ellipsis), as a cross-check. */
    private static void reportTruncation(Scene scene, String context) throws Exception {
        List<String> truncated = onFx(() -> {
            List<String> out = new ArrayList<>();
            for (Node n : scene.getRoot().lookupAll("*")) {
                if (n instanceof Labeled && n.isVisible()) {
                    Labeled l = (Labeled) n;
                    Node text = l.lookup(".text");
                    if (text instanceof javafx.scene.text.Text && l.getText() != null && !l.getText().isEmpty()) {
                        String shown = ((javafx.scene.text.Text) text).getText();
                        boolean ellipsis = shown.endsWith("…") || shown.endsWith("...")
                                || shown.contains("...\n") || shown.contains("…\n");
                        if (!shown.equals(l.getText()) && ellipsis) {
                            out.add("'" + l.getText() + "' shown as '" + shown + "'");
                        }
                    }
                }
            }
            return out;
        });
        if (truncated.isEmpty()) {
            System.out.println("[truncation " + context + "] none");
        } else {
            truncated.forEach(t -> System.out.println("[truncation " + context + "] " + t));
        }
    }

    @SuppressWarnings("unchecked")
    private static void printEngineSelector(Scene scene) throws Exception {
        String info = onFx(() -> {
            ComboBox<ChangeDetectionEngine> combo = (ComboBox<ChangeDetectionEngine>) scene.lookup("#engineComboBox");
            StringBuilder sb = new StringBuilder("selected=" + combo.getValue().getDisplayName() + "; options=");
            for (ChangeDetectionEngine e : combo.getItems()) {
                sb.append('[').append(e.getDisplayName()).append(e.isAvailable() ? ", enabled" : ", DISABLED: " + e.getStatusNote()).append(']');
            }
            return sb.toString();
        });
        System.out.println("[engine selector] " + info);
    }

    private static <T> T onFx(Callable<T> callable) throws Exception {
        FutureTask<T> task = new FutureTask<>(callable);
        Platform.runLater(task);
        return task.get(30, TimeUnit.SECONDS);
    }
}
