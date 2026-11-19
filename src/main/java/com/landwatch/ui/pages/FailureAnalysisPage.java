package com.landwatch.ui.pages;

import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.data.PerImageResults;
import javafx.geometry.HPos;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * "Failure Analysis" page: the classical baseline's worst LEVIR-CD test images (from the per-image
 * benchmark CSV, read at runtime) with a button to open each pair in the analysis view, plus the
 * documented limitations of both engines and of the data.
 */
public final class FailureAnalysisPage {

    private static final int WORST = 10;

    private final ScrollPane root;
    private final Consumer<Path> openPair;
    private final Label summary = Ui.label("", "page-note");
    private final GridPane table = new GridPane();

    /** @param openPair called with {@code <dataset>/test/A/<image>} when the user opens a pair */
    public FailureAnalysisPage(Consumer<Path> openPair) {
        this.openPair = openPair;
        VBox content = Ui.pageContent("Failure analysis",
                "Where change detection goes wrong, and why. Per-image numbers come from the classical baseline's "
                        + "benchmark CSV, read at runtime; open any pair to compare both engines on it.",
                null);

        VBox worstCard = Ui.card("Classical baseline: the worst LEVIR-CD test images by F1", "per image", null);
        table.getStyleClass().add("bench-table");
        worstCard.getChildren().addAll(summary, table);

        VBox limitsCard = Ui.card("Known limitations", null, null);
        limitsCard.getChildren().add(Ui.bullets(
                "Illumination, season and sun angle: pixel differencing flags any brightness difference (shadows, "
                        + "vegetation cycles, haze), so it raises false alarms wherever brightness changed without a real "
                        + "change. The deep model was trained on labelled pairs that contain such differences, which makes it "
                        + "less sensitive to them (compare both engines on the same pair), but not immune.",
                "Registration: both engines compare A and B pixel by pixel and assume they are co-registered. A shift of a "
                        + "few pixels produces false change along every edge.",
                "Building-only labels: LEVIR-CD marks construction and demolition of buildings only. New roads, cleared soil "
                        + "or water changes are real but are scored as false positives, which lowers precision for both engines.",
                "Single domain: the deep model was trained only on LEVIR-CD (0.5 m/px aerial imagery). Other sensors, "
                        + "resolutions or regions are untested; its threshold was tuned on the LEVIR-CD val split.",
                "Synthetic demo pair: the bundled Site A pair is computer-generated, has no label and only exercises the "
                        + "pipeline. No accuracy claim is made from it.",
                "Areas are in pixels. Square metres are only shown for LEVIR-CD pairs, assuming its 0.5 m/px ground "
                        + "sample distance (0.25 m² per pixel)."));

        content.getChildren().addAll(worstCard, limitsCard);
        root = Ui.page(content);
        root.setId("failureAnalysisPage");
        refresh();
    }

    public Node getRoot() {
        return root;
    }

    /** Re-reads the CSV (called each time the page is shown). */
    public void refresh() {
        Path csv = AppPaths.baselinePerImage("test");
        PerImageResults results = PerImageResults.read(csv);
        table.getChildren().clear();
        table.getColumnConstraints().clear();
        if (!results.isOk()) {
            summary.setText("Not available: " + results.getProblem());
            return;
        }
        List<PerImageResults.Row> worst = results.worstByF1(WORST);
        long withChange = results.getRows().size() - results.imagesWithoutTrueChange();
        summary.setText(String.format(Locale.ROOT,
                "%d of %d test images contain labelled change; the %d with the lowest per-image F1 are listed "
                        + "(median per-image F1 over those %d: %.4f). %d image(s) contain no labelled change at all, so every "
                        + "pixel the baseline flags there is a false positive (F1 = 0 by convention). The worst images "
                        + "have little labelled change but many flagged pixels: brightness differences that are not building "
                        + "changes. Source: %s.",
                withChange, results.getRows().size(), worst.size(), withChange, results.medianF1WithChange(),
                results.imagesWithoutTrueChange(), BenchmarksPage.relative(csv)));

        String[] headers = {"Image", "True change px", "Predicted px", "Precision", "Recall", "F1", "IoU", ""};
        for (int i = 0; i < headers.length; i++) {
            ColumnConstraints c = new ColumnConstraints();
            if (i == 0) {
                c.setHgrow(Priority.ALWAYS);
                c.setMinWidth(110);
            } else {
                c.setMinWidth(i == headers.length - 1 ? 90 : 80);
                c.setHalignment(HPos.RIGHT);
            }
            table.getColumnConstraints().add(c);
            Label h = Ui.fixed(headers[i], "bench-header");
            h.setMaxWidth(Double.MAX_VALUE);
            h.setAlignment(i == 0 ? Pos.CENTER_LEFT : Pos.CENTER_RIGHT);
            table.add(h, i, 0);
        }

        Path testA = AppPaths.levirDatasetDir().resolve("test").resolve("A");
        int r = 1;
        for (PerImageResults.Row row : worst) {
            boolean shaded = r % 2 == 0;
            String[] values = {row.image(),
                    String.format(Locale.ROOT, "%,d", row.trueChangePx()),
                    String.format(Locale.ROOT, "%,d", row.predChangePx()),
                    String.format(Locale.ROOT, "%.4f", row.precision()),
                    String.format(Locale.ROOT, "%.4f", row.recall()),
                    String.format(Locale.ROOT, "%.4f", row.f1()),
                    String.format(Locale.ROOT, "%.4f", row.iou())};
            for (int i = 0; i < values.length; i++) {
                Label v = Ui.fixed(values[i], i == 5 ? "bench-value-primary" : (i == 0 ? "bench-model" : "bench-value"));
                table.add(cell(shaded, i == 0 ? Pos.CENTER_LEFT : Pos.CENTER_RIGHT, v), i, r);
            }
            Path picked = testA.resolve(row.image());
            Button open = new Button("Open pair");
            open.getStyleClass().add("btn-mini");
            open.setMinWidth(Region.USE_PREF_SIZE);
            if (Files.isRegularFile(picked)) {
                open.setOnAction(e -> openPair.accept(picked));
                open.setTooltip(new Tooltip("Load " + picked + " with its B image and label into Change Detection"));
            } else {
                open.setDisable(true);
                open.setTooltip(new Tooltip("Dataset image not found: " + picked));
            }
            table.add(cell(shaded, Pos.CENTER_RIGHT, open), headers.length - 1, r);
            r++;
        }
    }

    private static VBox cell(boolean shaded, Pos pos, Node... children) {
        VBox box = new VBox(children);
        box.setAlignment(pos);
        box.getStyleClass().add(shaded ? "bench-cell-shaded" : "bench-cell");
        box.setMaxWidth(Double.MAX_VALUE);
        box.setMaxHeight(Double.MAX_VALUE);
        return box;
    }
}
