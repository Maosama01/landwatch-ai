package com.landwatch.ui.pages;

import com.landwatch.ui.data.AppPaths;
import com.landwatch.ui.data.MetricsReport;
import com.landwatch.ui.data.MetricsReport.Metric;
import com.landwatch.ui.engine.SiameseUNetEngine;
import javafx.geometry.HPos;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.ColumnConstraints;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.OptionalDouble;

/**
 * "Model Benchmarks" page: dataset-level LEVIR-CD metrics of the classical baseline and the deep model,
 * plus EuroSAT classification, every value parsed at runtime from the results files. A missing file is
 * shown as "not available"; nothing is hard-coded.
 */
public final class BenchmarksPage {

    private static final Metric[] COLUMNS = {Metric.F1, Metric.IOU, Metric.PRECISION, Metric.RECALL, Metric.KAPPA};
    private static final String[] COLUMN_TITLES = {"F1", "IoU", "Precision", "Recall", "Kappa"};

    private final SiameseUNetEngine deepEngine;
    private final ScrollPane root;
    private final GridPane table = new GridPane();
    private final VBox euroSatBox = new VBox(8);
    private final VBox notesBox = new VBox(6);
    private final Label loadedModelLabel = Ui.label("", "page-note");

    public BenchmarksPage(SiameseUNetEngine deepEngine) {
        this.deepEngine = deepEngine;

        VBox content = Ui.pageContent("Model Benchmarks",
                "Dataset-level results over every pixel of a LEVIR-CD split, read at runtime from the results/ folder. "
                        + "These are DATASET benchmarks; the per-image numbers for one pair are on the Change Detection page.",
                "DATASET BENCHMARKS");

        VBox levirCard = Ui.card("LEVIR-CD BUILDING CHANGE DETECTION · PIXEL METRICS OF THE CHANGE CLASS",
                "ALL IMAGES OF A SPLIT", "badge-info");
        table.getStyleClass().add("bench-table");
        table.setHgap(0);
        table.setVgap(0);
        levirCard.getChildren().addAll(table, loadedModelLabel);

        VBox euroCard = Ui.card("EUROSAT LAND-COVER CLASSIFICATION · SEPARATE TASK", "TEST SPLIT", "badge-info");
        euroCard.getChildren().add(euroSatBox);

        VBox notesCard = Ui.card("HOW TO READ THESE NUMBERS", null, null);
        notesCard.getChildren().add(notesBox);

        content.getChildren().addAll(levirCard, euroCard, notesCard);
        root = Ui.page(content);
        root.setId("benchmarksPage");
        refresh();
    }

    public Node getRoot() {
        return root;
    }

    /** Re-reads every results file (called each time the page is shown). */
    public void refresh() {
        MetricsReport baseVal = MetricsReport.readChangeDetection(AppPaths.baselineMetrics("val"));
        MetricsReport baseTest = MetricsReport.readChangeDetection(AppPaths.baselineMetrics("test"));
        MetricsReport deepVal = MetricsReport.readChangeDetection(AppPaths.deepMetrics("val"));
        MetricsReport deepTest = MetricsReport.readChangeDetection(AppPaths.deepMetrics("test"));
        buildTable(List.of(
                new Row("Classical baseline", "Otsu differencing, no learning", "val", baseVal),
                new Row("Classical baseline", "Otsu differencing, no learning", "test", baseTest),
                new Row("Deep model", "FC-Siam-diff (Siamese U-Net)", "val", deepVal),
                new Row("Deep model", "FC-Siam-diff (Siamese U-Net)", "test", deepTest)));
        buildEuroSat(MetricsReport.readEuroSat(AppPaths.euroSatMetrics()));
        buildLoadedModelNote();
        buildNotes(baseTest);
    }

    private record Row(String model, String method, String split, MetricsReport report) {
    }

    private void buildTable(List<Row> rows) {
        table.getChildren().clear();
        table.getColumnConstraints().clear();
        ColumnConstraints model = new ColumnConstraints();
        model.setHgrow(Priority.ALWAYS);
        model.setMinWidth(Region.USE_PREF_SIZE);
        ColumnConstraints split = new ColumnConstraints();
        split.setMinWidth(130);
        split.setPrefWidth(150);
        table.getColumnConstraints().addAll(model, split);
        for (int i = 0; i < COLUMNS.length; i++) {
            ColumnConstraints c = new ColumnConstraints();
            c.setMinWidth(70);
            c.setPrefWidth(84);
            c.setHalignment(HPos.RIGHT);
            table.getColumnConstraints().add(c);
        }

        table.add(headerCell("Model · source file", Pos.CENTER_LEFT), 0, 0);
        table.add(headerCell("Split · size", Pos.CENTER_LEFT), 1, 0);
        for (int i = 0; i < COLUMNS.length; i++) {
            table.add(headerCell(COLUMN_TITLES[i], Pos.CENTER_RIGHT), 2 + i, 0);
        }

        int r = 1;
        for (Row row : rows) {
            boolean shaded = r % 2 == 0;
            Label name = Ui.fixed(row.model(), "bench-model");
            Label method = Ui.label(row.method(), "bench-sub");
            Label source = Ui.fixed(relative(row.report().getFile()), "bench-source");
            VBox modelCell = cell(shaded, Pos.CENTER_LEFT, name, method, source);
            table.add(modelCell, 0, r);

            String size = row.report().getScope().orElse(row.report().isOk() ? "size not stated" : "");
            Label splitLabel = Ui.fixed(row.split(), "bench-split");
            Label sizeLabel = Ui.label(size, "bench-sub");
            table.add(cell(shaded, Pos.CENTER_LEFT, splitLabel, sizeLabel), 1, r);

            if (row.report().isOk()) {
                for (int i = 0; i < COLUMNS.length; i++) {
                    Label v = Ui.fixed(row.report().format(COLUMNS[i]),
                            COLUMNS[i] == Metric.F1 ? "bench-value-primary" : "bench-value");
                    table.add(cell(shaded, Pos.CENTER_RIGHT, v), 2 + i, r);
                }
            } else {
                String text = row.report().getStatus() == MetricsReport.Status.MISSING
                        ? "not available: file not found"
                        : "not available: " + row.report().getProblem();
                Label na = Ui.label(text, "bench-na");
                VBox span = cell(shaded, Pos.CENTER_LEFT, na);
                table.add(span, 2, r, COLUMNS.length, 1);
            }
            r++;
        }
    }

    private static Label headerCell(String text, Pos pos) {
        Label l = Ui.fixed(text, "bench-header");
        l.setMaxWidth(Double.MAX_VALUE);
        l.setMaxHeight(Double.MAX_VALUE);
        l.setAlignment(pos);
        return l;
    }

    private static VBox cell(boolean shaded, Pos pos, Node... children) {
        VBox box = new VBox(1, children);
        box.setAlignment(pos);
        box.getStyleClass().add(shaded ? "bench-cell-shaded" : "bench-cell");
        box.setMaxWidth(Double.MAX_VALUE);
        box.setMaxHeight(Double.MAX_VALUE);
        return box;
    }

    private void buildEuroSat(MetricsReport euro) {
        euroSatBox.getChildren().clear();
        if (euro.isOk()) {
            HBox tiles = new HBox(10,
                    tile("OVERALL ACCURACY", euro.format(Metric.OVERALL_ACCURACY)),
                    tile("MACRO F1", euro.format(Metric.MACRO_F1)));
            euroSatBox.getChildren().add(tiles);
        } else {
            euroSatBox.getChildren().add(Ui.label(euro.getStatus() == MetricsReport.Status.MISSING
                    ? "Not available: no EuroSAT results file yet." : "Not available: " + euro.getProblem(), "bench-na"));
        }
        euroSatBox.getChildren().add(Ui.label("Source: " + relative(euro.getFile())
                + ". Scene classification is a different task from change detection, so these numbers are not "
                + "comparable with the LEVIR-CD table.", "bench-source"));
    }

    private static VBox tile(String caption, String value) {
        VBox t = new VBox(2, Ui.fixed(caption, "metric-caption"), Ui.fixed(value, "metric-value"));
        t.getStyleClass().add("metric-tile");
        t.setMinWidth(160);
        return t;
    }

    private void buildLoadedModelNote() {
        if (deepEngine == null || !deepEngine.isAvailable()) {
            String state = deepEngine == null ? "no deep engine" : deepEngine.getStatusNote();
            loadedModelLabel.setText("Deep model in the dashboard: " + state + ".");
            return;
        }
        StringBuilder sb = new StringBuilder("Deep model loaded in the dashboard (" + deepEngine.getModelDir() + "): its model.properties records");
        String[][] keys = {{"metrics.val.f1", "val F1"}, {"metrics.val.iou", "IoU"},
                {"metrics.val.precision", "precision"}, {"metrics.val.recall", "recall"}};
        boolean any = false;
        for (String[] k : keys) {
            String v = deepEngine.getModelProperty(k[0]);
            if (v != null) {
                sb.append(any ? ", " : " ").append(k[1]).append(' ').append(v);
                any = true;
            }
        }
        if (!any) {
            sb.append(" no validation metrics");
        }
        sb.append(" at checkpoint selection");
        String test = deepEngine.getModelProperty("metrics.test.f1");
        sb.append(test != null ? "; test F1 " + test + "." : "; no test metrics yet.");
        sb.append(" The table above only shows the evaluation report files.");
        loadedModelLabel.setText(sb.toString());
    }

    private void buildNotes(MetricsReport baseTest) {
        notesBox.getChildren().clear();
        List<String> notes = new ArrayList<>();
        notes.add("The test split is evaluated once. Every choice (deep-model checkpoint, decision threshold, "
                + "baseline settings) is made on val; test is scored a single time at the end, so it stays an honest "
                + "estimate of performance on unseen images.");
        String thr = "";
        if (deepEngine != null && deepEngine.getThreshold() != null) {
            thr = String.format(Locale.ROOT, " The loaded model uses threshold %.2f.", deepEngine.getThreshold());
        }
        notes.add("The deep model's decision threshold (probability ≥ threshold = change) is tuned on val to maximise "
                + "change F1, never on test." + thr);
        notes.add(pixelAccuracyNote(baseTest));
        notes.add("Metrics come from one global pixel confusion matrix per split (BinaryChangeMetrics). The baseline runs on "
                + "the full 1024×1024 images; the deep model on non-overlapping 256×256 tiles of the same images, so both "
                + "cover the same pixels.");
        notes.add("LEVIR-CD labels building changes only (construction and demolition). Other real changes, such as roads, "
                + "bare soil or vegetation, count as false positives for both engines.");
        notesBox.getChildren().add(Ui.bullets(notes.toArray(new String[0])));
    }

    private static String pixelAccuracyNote(MetricsReport baseTest) {
        String base = "Pixel accuracy is misleading on its own: change pixels are a small minority, so a model that never "
                + "predicts change still looks very accurate. Change F1 and IoU are the primary metrics.";
        if (!baseTest.isOk()) {
            return base;
        }
        OptionalDouble share = baseTest.getTrueChangePercent();
        OptionalDouble acc = baseTest.get(Metric.PIXEL_ACCURACY);
        OptionalDouble f1 = baseTest.get(Metric.F1);
        StringBuilder sb = new StringBuilder(base);
        if (share.isPresent()) {
            sb.append(String.format(Locale.ROOT,
                    " On LEVIR-CD test only %.2f %% of pixels are change, so predicting \"no change\" everywhere would score "
                            + "%.2f %% accuracy.", share.getAsDouble(), 100.0 - share.getAsDouble()));
        }
        if (acc.isPresent() && f1.isPresent()) {
            sb.append(String.format(Locale.ROOT, " The classical baseline reaches pixel accuracy %.4f but change F1 only %.4f.",
                    acc.getAsDouble(), f1.getAsDouble()));
        }
        sb.append(" (Figures from ").append(relative(baseTest.getFile())).append(".)");
        return sb.toString();
    }

    static String relative(Path p) {
        if (p == null) {
            return "?";
        }
        Path cwd = Path.of("").toAbsolutePath();
        Path abs = p.toAbsolutePath().normalize();
        return abs.startsWith(cwd) ? cwd.relativize(abs).toString() : p.toString();
    }
}
