package com.landwatch.ui;

import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.ui.controllers.DashboardController;
import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.engine.ChangeDetectionEngine;
import com.landwatch.ui.engine.EngineResult;
import com.landwatch.ui.engine.SiameseUNetEngine;
import javafx.application.Platform;
import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXMLLoader;
import javafx.scene.Group;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Labeled;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.WritableImage;
import javafx.stage.Stage;

import javax.imageio.ImageIO;
import java.io.File;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

/**
 * On-demand visual-evidence tool (NOT a unit test; it is not picked up by Surefire).
 *
 * <p>Loads dashboard.fxml with the real controller and stylesheet in an off-screen Stage and drives it
 * through the real handlers: the synthetic demo pair, a real LEVIR-CD test pair analysed by the deep
 * model and by the classical baseline (overlay, ground-truth and errors views), and the Benchmarks,
 * NN Fundamentals, Failure Analysis and About pages. Saves {@code scene.snapshot()} PNGs into
 * {@code results/ui_snapshots/} at 1280x840 (plus a few at the 1100x760 minimum-size check). No window
 * is shown; the tool exits with {@link Platform#exit()}.</p>
 *
 * <p>Run (after {@code mvn test-compile}) from the project root with target/classes, target/test-classes
 * and the dependency jars on the classpath:</p>
 * <pre>
 * java -Dlandwatch.model.dir=models/siamese_unet_levir -Dlandwatch.deeplearning.device=cpu \
 *      -Dlandwatch.dataset.dir=datasets/LEVIR-CD [-Dlandwatch.snapshot.pair=test_7.png] \
 *      -cp ... com.landwatch.ui.DashboardSnapshotTool [outputDir]
 * </pre>
 */
public final class DashboardSnapshotTool {

    private static final String PAIR_PROPERTY = "landwatch.snapshot.pair";

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
            written.addAll(capture(outDir));
        } catch (Throwable t) {
            t.printStackTrace();
            exitCode = 1;
        } finally {
            Platform.exit();
        }
        written.forEach(p -> System.out.println("[snapshot] " + p));
        System.exit(exitCode);
    }

    private static List<String> capture(File outDir) throws Exception {
        List<String> files = new ArrayList<>();
        int width = 1280;
        int height = 840;

        // Build stage + scene exactly like MainApp (same FXML, controller and stylesheet).
        Object[] holder = onFx(() -> {
            URL fxml = DashboardSnapshotTool.class.getResource("/com/landwatch/ui/views/dashboard.fxml");
            FXMLLoader loader = new FXMLLoader(fxml);
            Parent root = loader.load();
            Scene scene = new Scene(root, width, height);
            Stage stage = new Stage();
            stage.setScene(scene);
            stage.setWidth(width);
            stage.setHeight(height);
            styled(scene);
            root.applyCss();
            root.layout();
            return new Object[]{scene, loader.getController(), stage};
        });
        Scene scene = (Scene) holder[0];
        DashboardController controller = (DashboardController) holder[1];
        SiameseUNetEngine deep = controller.getDeepEngine();

        try {
            files.add(save(scene, outDir, "00_startup_model_loading.png"));
            waitFor(() -> !deep.isLoading(), 180_000, "deep model load");
            System.out.println("[deep model] state=" + deep.getState() + " note=" + deep.getStatusNote());
            printEngineSelector(scene);

            // 1. Synthetic demo pair, classical baseline (no ground truth available).
            onFx(() -> {
                ((Button) scene.lookup("#loadDemoButton")).fire();
                return null;
            });
            analyze(scene, controller, null);
            selectAndScroll(scene, controller, VisualizationMode.RED_OVERLAY);
            files.add(save(scene, outDir, "01_demo_synthetic_classical_overlay.png"));
            reportTruncation(scene, "01");

            // 2. Real LEVIR-CD test pair, deep model.
            Path pair = AppPaths.levirDatasetDir().resolve("test").resolve("A")
                    .resolve(System.getProperty(PAIR_PROPERTY, "test_7.png"));
            if (!Files.isRegularFile(pair)) {
                throw new IllegalStateException("LEVIR-CD pair not found: " + pair.toAbsolutePath()
                        + " (set -D" + AppPaths.DATASET_DIR_PROPERTY + ")");
            }
            boolean loaded = onFx(() -> controller.loadLevirPair(pair));
            if (!loaded) {
                throw new IllegalStateException("Could not load " + pair);
            }
            String tag = pair.getFileName().toString().replace(".png", "");
            if (deep.isAvailable()) {
                analyze(scene, controller, deep);
                report(controller, "deep");
                scrollTop(scene);
                files.add(save(scene, outDir, "02_levir_" + tag + "_deep_inputs.png"));
                for (Object[] v : new Object[][]{
                        {VisualizationMode.RED_OVERLAY, "03_levir_" + tag + "_deep_overlay.png"},
                        {GroundTruthView.GROUND_TRUTH, "04_levir_" + tag + "_deep_ground_truth.png"},
                        {GroundTruthView.ERRORS, "05_levir_" + tag + "_deep_errors.png"}}) {
                    selectAndScroll(scene, controller, (ChangeView) v[0]);
                    files.add(save(scene, outDir, (String) v[1]));
                    reportTruncation(scene, (String) v[1]);
                }
            } else {
                System.out.println("[deep model] NOT available, deep snapshots skipped: " + deep.getStatusNote());
            }

            // 3. Same pair, classical baseline.
            ChangeDetectionEngine classical = onFx(() -> engineCombo(scene).getItems().get(0));
            analyze(scene, controller, classical);
            report(controller, "classical");
            for (Object[] v : new Object[][]{
                    {VisualizationMode.RED_OVERLAY, "06_levir_" + tag + "_classical_overlay.png"},
                    {GroundTruthView.ERRORS, "07_levir_" + tag + "_classical_errors.png"}}) {
                selectAndScroll(scene, controller, (ChangeView) v[0]);
                files.add(save(scene, outDir, (String) v[1]));
                reportTruncation(scene, (String) v[1]);
            }

            // Deep model again so the minimum-size check shows the deep errors view.
            if (deep.isAvailable()) {
                analyze(scene, controller, deep);
                selectAndScroll(scene, controller, GroundTruthView.ERRORS);
            }

            // 4. Pages.
            fire(scene, "#navBenchmarksButton");
            scrollPage(scene, "#benchmarksPage", 0.0);
            files.add(save(scene, outDir, "08_page_model_benchmarks.png"));
            reportTruncation(scene, "08");

            fire(scene, "#navFundamentalsButton");
            fire(scene, "#runXorButton");
            fire(scene, "#runGradientCheckButton");
            fire(scene, "#runLearningRulesButton");
            waitFor(() -> !controller.getFundamentalsPage().isRunning(), 60_000, "fundamentals demos");
            Thread.sleep(300);
            scrollPage(scene, "#fundamentalsPage", 0.0);
            files.add(save(scene, outDir, "10_page_nn_fundamentals.png"));
            reportTruncation(scene, "10");
            scrollPage(scene, "#fundamentalsPage", 1.0);
            files.add(save(scene, outDir, "11_page_nn_fundamentals_learning_rules.png"));

            fire(scene, "#navFailureButton");
            files.add(save(scene, outDir, "12_page_failure_analysis.png"));
            reportTruncation(scene, "12");

            fire(scene, "#navAboutButton");
            files.add(save(scene, outDir, "13_page_about_runtime.png"));
            reportTruncation(scene, "13");

            // 5. Minimum window size (1100x760): the same root moved into a smaller scene.
            Scene small = onFx(() -> {
                Parent root = scene.getRoot();
                scene.setRoot(new Group());
                Scene s = new Scene(root, 1100, 760);
                styled(s);
                ((Stage) holder[2]).setScene(s);
                return s;
            });
            fire(small, "#navAnalysisButton");
            selectAndScroll(small, controller, GroundTruthView.ERRORS);
            files.add(save(small, outDir, "14_min_size_1100x760_levir_deep_errors.png"));
            reportTruncation(small, "14");
            fire(small, "#navBenchmarksButton");
            scrollPage(small, "#benchmarksPage", 0.0);
            files.add(save(small, outDir, "15_min_size_1100x760_benchmarks.png"));
            reportTruncation(small, "15");
        } finally {
            onFx(() -> {
                controller.shutdown();
                ((Stage) holder[2]).close();
                return null;
            });
        }
        return files;
    }

    private static void styled(Scene scene) {
        URL css = DashboardSnapshotTool.class.getResource("/com/landwatch/ui/css/dark_theme.css");
        scene.getStylesheets().add(css.toExternalForm());
    }

    @SuppressWarnings("unchecked")
    private static ComboBox<ChangeDetectionEngine> engineCombo(Scene scene) {
        return (ComboBox<ChangeDetectionEngine>) scene.lookup("#engineComboBox");
    }

    /** Selects {@code engine} (null = keep), clicks Analyze and waits until that run has finished. */
    private static void analyze(Scene scene, DashboardController controller, ChangeDetectionEngine engine) throws Exception {
        int before = onFx(controller::getCompletedAnalysisCount);
        onFx(() -> {
            if (engine != null) {
                engineCombo(scene).setValue(engine);
            }
            ((Button) scene.lookup("#analyzeButton")).fire();
            return null;
        });
        waitFor(() -> controller.getCompletedAnalysisCount() > before && !controller.isAnalysisRunning(),
                300_000, "analysis");
    }

    private static void report(DashboardController controller, String label) throws Exception {
        EngineResult r = onFx(controller::getLastEngineResult);
        BinaryChangeMetrics m = onFx(controller::getLastGroundTruthMetrics);
        System.out.printf(Locale.ROOT, "[analysis %s] engine=%s changed=%,d px (%.2f %%) regions=%d latency=%d ms%n",
                label, r.getEngineName(), r.getChangedPixels(), r.getChangePercentage(), r.getRegionCount(),
                r.getLatencyMs());
        if (m != null) {
            System.out.printf(Locale.ROOT, "[per-image vs ground truth %s] F1=%.4f IoU=%.4f precision=%.4f recall=%.4f "
                            + "TP=%d FP=%d FN=%d TN=%d%n", label, m.f1(), m.iou(), m.precision(), m.recall(),
                    m.getTruePositives(), m.getFalsePositives(), m.getFalseNegatives(), m.getTrueNegatives());
        }
    }

    /** Selects a view and scrolls so both result cards (change map + metrics) fill the viewport. */
    private static void selectAndScroll(Scene scene, DashboardController controller, ChangeView view) throws Exception {
        onFx(() -> {
            controller.selectView(view);
            scrollToNode(scene, "#contentScrollPane", "#resultsGrid");
            return null;
        });
    }

    /** Scrolls {@code scrollId} so that the top of {@code nodeId} (a child of its content) is at the top. */
    private static void scrollToNode(Scene scene, String scrollId, String nodeId) {
        ScrollPane sp = (ScrollPane) scene.lookup(scrollId);
        Node target = scene.lookup(nodeId);
        scene.getRoot().applyCss();
        scene.getRoot().layout();
        double scrollable = sp.getContent().getLayoutBounds().getHeight() - sp.getViewportBounds().getHeight();
        if (scrollable > 0) {
            double y = target.getBoundsInParent().getMinY() - 12;
            sp.setVvalue(sp.getVmin() + Math.max(0, Math.min(1, y / scrollable)) * (sp.getVmax() - sp.getVmin()));
        }
    }

    private static void scrollTop(Scene scene) throws Exception {
        onFx(() -> {
            ((ScrollPane) scene.lookup("#contentScrollPane")).setVvalue(0);
            return null;
        });
    }

    private static void scrollPage(Scene scene, String id, double v) throws Exception {
        onFx(() -> {
            scene.getRoot().applyCss();
            scene.getRoot().layout();
            ((ScrollPane) scene.lookup(id)).setVvalue(v);
            return null;
        });
    }

    private static void fire(Scene scene, String id) throws Exception {
        onFx(() -> {
            ((Button) scene.lookup(id)).fire();
            return null;
        });
    }

    private interface Condition {
        boolean holds() throws Exception;
    }

    private static void waitFor(Condition condition, long timeoutMs, String what) throws Exception {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            if (onFx(condition::holds)) {
                return;
            }
            Thread.sleep(100);
        }
        throw new IllegalStateException(what + " did not finish within " + timeoutMs / 1000 + " s");
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
                if (n instanceof Labeled && n.isVisible() && isTreeVisible(n)) {
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

    private static boolean isTreeVisible(Node n) {
        for (Node p = n; p != null; p = p.getParent()) {
            if (!p.isVisible()) {
                return false;
            }
        }
        return true;
    }

    private static void printEngineSelector(Scene scene) throws Exception {
        String info = onFx(() -> {
            ComboBox<ChangeDetectionEngine> combo = engineCombo(scene);
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
        return task.get(60, TimeUnit.SECONDS);
    }
}
