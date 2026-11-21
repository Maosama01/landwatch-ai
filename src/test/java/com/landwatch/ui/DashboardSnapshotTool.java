package com.landwatch.ui;

import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.ui.controllers.DashboardController;
import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.engine.ChangeDetectionEngine;
import com.landwatch.ui.engine.EngineResult;
import com.landwatch.ui.engine.SiameseUNetEngine;
import com.landwatch.ui.pages.EuroSatPage;
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
import javafx.scene.control.ToggleButton;
import javafx.scene.image.WritableImage;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyCodeCombination;
import javafx.scene.input.KeyCombination;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseButton;
import javafx.scene.input.MouseEvent;
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
 * model and by the classical baseline (compare, overlay, ground-truth and errors views, and the errors
 * view in the light theme), and the Benchmarks,
 * NN Fundamentals, Failure Analysis and About pages. Saves {@code scene.snapshot()} PNGs into
 * {@code results/ui_snapshots/} at 1280x840 (plus a few at the 1100x760 minimum-size check). No window
 * is shown; the tool exits with {@link Platform#exit()}.</p>
 *
 * <p>Run (after {@code mvn test-compile}) from the project root with target/classes, target/test-classes
 * and the dependency jars on the classpath:</p>
 * <pre>
 * java -Dlandwatch.model.dir=models/siamese_unet_levir -Dlandwatch.deeplearning.device=cpu \
 *      -Dlandwatch.dataset.dir=datasets/LEVIR-CD [-Dlandwatch.snapshot.pair=test_7.png] \
 *      [-Dlandwatch.eurosat.root=datasets/EuroSAT] [-Dlandwatch.snapshot.only=eurosat] \
 *      -cp ... com.landwatch.ui.DashboardSnapshotTool [outputDir]
 * </pre>
 *
 * <p>The Land use (EuroSAT) page is captured as well (17-23); {@code -Dlandwatch.snapshot.only=eurosat}
 * captures only that page and skips the LEVIR-CD flow.</p>
 */
public final class DashboardSnapshotTool {

    private static final String PAIR_PROPERTY = "landwatch.snapshot.pair";
    private static final String ONLY_PROPERTY = "landwatch.snapshot.only";

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
        }
        written.forEach(p -> System.out.println("[snapshot] " + p));
        // JavaFX's own shutdown can stall on macOS under heavy load (MacTimer._stop); every file is
        // already written, so give it 5 s and then exit regardless.
        Thread exiter = new Thread(Platform::exit, "fx-exit");
        exiter.setDaemon(true);
        exiter.start();
        exiter.join(5000);
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
            if ("eurosat".equals(System.getProperty(ONLY_PROPERTY))) {
                waitFor(() -> !deep.isLoading(), 180_000, "deep model load");
                captureEuroSat(scene, controller, outDir, files);
                return files;
            }
            files.add(save(scene, outDir, "00_startup_model_loading.png"));
            waitFor(() -> !deep.isLoading(), 180_000, "deep model load");
            System.out.println("[deep model] state=" + deep.getState() + " note=" + deep.getStatusNote());
            printEngineSelector(scene);

            // 1. Synthetic demo pair, classical baseline (no ground truth available). The deep model is
            //    selected automatically once loaded, so the baseline is chosen explicitly here.
            ChangeDetectionEngine classical = onFx(() -> engineCombo(scene).getItems().get(0));
            onFx(() -> {
                ((Button) scene.lookup("#loadDemoButton")).fire();
                return null;
            });
            analyze(scene, controller, classical);
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
                onFx(() -> {
                    controller.selectCompare();
                    return null;
                });
                scrollTop(scene);
                files.add(save(scene, outDir, "02_levir_" + tag + "_compare_before_after.png"));
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

            // Interaction check: real shortcuts, a drag on the swipe divider and a dropped LEVIR-CD file.
            interactionCheck(scene, controller, pair);

            // 3. Same pair, classical baseline.
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
                // The same view in the light theme (for a projector in a bright room).
                fire(scene, "#themeToggleButton");
                files.add(save(scene, outDir, "16_light_theme_levir_" + tag + "_deep_errors.png"));
                fire(scene, "#themeToggleButton");
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

            captureEuroSat(scene, controller, outDir, files);

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

    /**
     * Drives the controls the way a user does and prints what happened, so behaviour (not only layout) is
     * checked: Shortcut+6 opens the swipe comparison, a press at a quarter of the image moves the divider
     * there, Shortcut+1 returns to the overlay, and dropping one LEVIR-CD A/ file loads the pair with its label.
     */
    private static void interactionCheck(Scene scene, DashboardController controller, Path pair) throws Exception {
        String result = onFx(() -> {
            StringBuilder sb = new StringBuilder();
            KeyCombination compare = new KeyCodeCombination(KeyCode.DIGIT6, KeyCombination.SHORTCUT_DOWN);
            scene.getAccelerators().get(compare).run();
            sb.append("shortcut+6 -> compare=").append(controller.isCompareActive());
            scene.getRoot().applyCss();
            scene.getRoot().layout();
            SwipeCompareView swipe = controller.getSwipeView();
            double w = swipe.getWidth();
            double h = swipe.getHeight();
            double[] img = SwipeCompareView.fit(1024, 1024, w - 16, h - 16);
            double x = (w - img[0]) / 2 + img[0] * 0.25;
            javafx.geometry.Point2D at = swipe.localToScene(x, h / 2);   // fired events carry scene coordinates
            swipe.fireEvent(new MouseEvent(MouseEvent.MOUSE_PRESSED, at.getX(), at.getY(), 0, 0, MouseButton.PRIMARY, 1,
                    false, false, false, false, true, false, false, true, false, false, null));
            sb.append(String.format(Locale.ROOT, "; press at 25 %% -> divider %.3f", controller.getSwipePosition()));
            swipe.fireEvent(new KeyEvent(KeyEvent.KEY_PRESSED, "", "", KeyCode.RIGHT, false, false, false, false));
            sb.append(String.format(Locale.ROOT, "; right arrow -> %.3f", controller.getSwipePosition()));
            KeyCombination overlay = new KeyCodeCombination(KeyCode.DIGIT1, KeyCombination.SHORTCUT_DOWN);
            scene.getAccelerators().get(overlay).run();
            sb.append("; shortcut+1 -> compare=").append(controller.isCompareActive())
                    .append(" view=").append(controller.getCurrentView().getDisplayName());
            boolean dropped = controller.loadDroppedFiles(List.of(pair.toFile()), true);
            sb.append("; drop ").append(pair.getFileName()).append(" -> loaded=").append(dropped);
            return sb.toString();
        });
        System.out.println("[interaction] " + result);
        // Re-analyse so the views below show the deep result for the re-loaded pair.
        analyze(scene, controller, null);
        System.out.println("[interaction] after re-analysis: per-image F1 = "
                + String.format(Locale.ROOT, "%.4f", onFx(controller::getLastGroundTruthMetrics).f1()));
    }

    /**
     * Land use (EuroSAT) page: a random test image, the recorded-errors gallery with a misclassified tile
     * opened, the recorded test metrics, a non-64x64 input with its resize note, and the page when the
     * EuroSAT images are not installed.
     */
    private static void captureEuroSat(Scene scene, DashboardController controller, File outDir, List<String> files)
            throws Exception {
        EuroSatPage page = controller.getEuroSatPage();
        fire(scene, "#navLandUseButton");
        waitFor(() -> !page.isBusy(), 180_000, "EuroSAT classifier load and gallery");
        System.out.println("[eurosat] engine state=" + page.getEngine().getState() + " note="
                + page.getEngine().getStatusNote());
        boolean dataset = onFx(() -> !scene.lookup("#euroSatRandomButton").isDisabled());
        System.out.println("[eurosat] dataset images available: " + dataset);
        if (dataset) {
            fire(scene, "#euroSatRandomButton");
            waitFor(() -> !page.isBusy(), 60_000, "EuroSAT random test image");
        }
        scrollPage(scene, "#euroSatPage", 0.0);
        files.add(save(scene, outDir, "17_page_land_use_eurosat.png"));
        reportTruncation(scene, "17");

        if (dataset) {
            onFx(() -> {
                ((ToggleButton) scene.lookup("#euroSatGalleryErrorsToggle")).fire();
                return null;
            });
            waitFor(() -> !page.isBusy(), 60_000, "EuroSAT recorded-errors gallery");
            boolean opened = onFx(() -> {
                Node wrong = scene.getRoot().lookup(".gallery-tile-wrong");
                if (wrong == null) {
                    return false;
                }
                javafx.geometry.Point2D at = wrong.localToScene(10, 10);
                wrong.fireEvent(new MouseEvent(MouseEvent.MOUSE_CLICKED, at.getX(), at.getY(), 0, 0, MouseButton.PRIMARY,
                        1, false, false, false, false, true, false, false, true, false, false, null));
                return true;
            });
            System.out.println("[eurosat] opened a misclassified gallery tile: " + opened);
            scrollPage(scene, "#euroSatPage", 0.0);
            files.add(save(scene, outDir, "18_page_land_use_eurosat_wrong_prediction.png"));
            reportTruncation(scene, "18");
            scrollPage(scene, "#euroSatPage", 0.45);
            files.add(save(scene, outDir, "19_page_land_use_eurosat_recorded_errors_gallery.png"));
            reportTruncation(scene, "19");
        }
        scrollPage(scene, "#euroSatPage", 1.0);
        files.add(save(scene, outDir, "20_page_land_use_eurosat_test_metrics.png"));
        reportTruncation(scene, "20");

        // A non-64x64 input (the bundled synthetic demo image, not satellite imagery) shows the resize note.
        Path demo = Path.of("src", "main", "resources", "com", "landwatch", "sample_data", "siteA_pre.bmp");
        if (Files.isRegularFile(demo)) {
            onFx(() -> {
                page.classifyFile(demo);
                return null;
            });
            waitFor(() -> !page.isBusy(), 60_000, "EuroSAT resized input");
            scrollPage(scene, "#euroSatPage", 0.0);
            files.add(save(scene, outDir, "21_page_land_use_eurosat_resized_input_note.png"));
            reportTruncation(scene, "21");
        }

        // A LEVIR-CD pair image opened on this page is shown but deliberately not classified.
        Path levir = AppPaths.levirDatasetDir().resolve("test").resolve("A")
                .resolve(System.getProperty(PAIR_PROPERTY, "test_7.png"));
        if (Files.isRegularFile(levir)) {
            onFx(() -> {
                page.classifyFile(levir);
                return null;
            });
            waitFor(() -> !page.isBusy(), 60_000, "EuroSAT page with a LEVIR-CD image");
            scrollPage(scene, "#euroSatPage", 0.0);
            files.add(save(scene, outDir, "22_page_land_use_eurosat_levir_not_classified.png"));
            reportTruncation(scene, "22");
        }

        // Without the EuroSAT images the page still opens and says where they go (the default folder when
        // it has no images, as in a fresh clone; otherwise a folder that does not exist).
        String previous = System.getProperty(AppPaths.EUROSAT_DIR_PROPERTY);
        boolean defaultHasImages = Files.isDirectory(Path.of(AppPaths.DEFAULT_EUROSAT_DIR, AppPaths.EUROSAT_IMAGE_DIR));
        System.setProperty(AppPaths.EUROSAT_DIR_PROPERTY,
                defaultHasImages ? "datasets/EuroSAT-not-installed" : AppPaths.DEFAULT_EUROSAT_DIR);
        try {
            fire(scene, "#navLandUseButton");
            scrollPage(scene, "#euroSatPage", 0.0);
            files.add(save(scene, outDir, "23_page_land_use_eurosat_without_dataset.png"));
            reportTruncation(scene, "23");
        } finally {
            if (previous == null) {
                System.clearProperty(AppPaths.EUROSAT_DIR_PROPERTY);
            } else {
                System.setProperty(AppPaths.EUROSAT_DIR_PROPERTY, previous);
            }
        }
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

    /** Selects a view; inputs, result and measurements share the first screen, so the page stays at the top. */
    private static void selectAndScroll(Scene scene, DashboardController controller, ChangeView view) throws Exception {
        onFx(() -> {
            controller.selectView(view);
            ((ScrollPane) scene.lookup("#contentScrollPane")).setVvalue(0);
            return null;
        });
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
