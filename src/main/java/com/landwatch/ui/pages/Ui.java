package com.landwatch.ui.pages;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.layout.GridPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;

/** Small builders so the secondary pages use the same dark-theme classes as dashboard.fxml. */
final class Ui {

    private Ui() {
    }

    /** Scrollable page with the dashboard's padding, title, subtitle and an optional badge. */
    static ScrollPane page(VBox content) {
        ScrollPane scroll = new ScrollPane(content);
        scroll.setFitToWidth(true);
        scroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        scroll.getStyleClass().add("content-scroll-pane");
        return scroll;
    }

    static VBox pageContent(String title, String subtitle, String badge) {
        VBox content = new VBox(18);
        content.getStyleClass().add("main-content");
        content.setPadding(new Insets(22, 28, 24, 28));

        Label t = label(title, "page-title");
        Label s = label(subtitle, "page-subtitle");
        VBox titles = new VBox(3, t, s);
        HBox.setHgrow(titles, Priority.ALWAYS);
        HBox header = new HBox(12, titles);
        header.setAlignment(Pos.CENTER_LEFT);
        if (badge != null) {
            Label b = new Label(badge);
            b.getStyleClass().add("badge-academic");
            b.setMinWidth(Region.USE_PREF_SIZE);
            header.getChildren().add(b);
        }
        content.getChildren().add(header);
        return content;
    }

    /** A card with a section title and an optional badge on the right. */
    static VBox card(String title, String badge, String badgeStyle) {
        VBox card = new VBox(10);
        card.getStyleClass().add("card");
        card.setPadding(new Insets(16));
        Label t = label(title, "card-section-title");
        t.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(t, Priority.ALWAYS);
        HBox head = new HBox(8, t);
        head.setAlignment(Pos.CENTER_LEFT);
        if (badge != null) {
            Label b = new Label(badge);
            b.getStyleClass().add(badgeStyle);
            b.setMinWidth(Region.USE_PREF_SIZE);
            head.getChildren().add(b);
        }
        card.getChildren().add(head);
        return card;
    }

    /** A wrapping label with the given style classes. */
    static Label label(String text, String... styleClasses) {
        Label l = new Label(text);
        l.setWrapText(true);
        l.setMinHeight(Region.USE_PREF_SIZE);
        l.getStyleClass().addAll(styleClasses);
        return l;
    }

    /** A single-line label that never shrinks below its text width. */
    static Label fixed(String text, String... styleClasses) {
        Label l = new Label(text);
        l.setMinWidth(Region.USE_PREF_SIZE);
        l.getStyleClass().addAll(styleClasses);
        return l;
    }

    /** Bullet list of wrapping notes. */
    static VBox bullets(String... items) {
        VBox box = new VBox(6);
        for (String item : items) {
            Label dot = fixed("•", "note-bullet");
            Label text = label(item, "page-note");
            text.setMaxWidth(Double.MAX_VALUE);
            HBox.setHgrow(text, Priority.ALWAYS);
            HBox row = new HBox(8, dot, text);
            row.setAlignment(Pos.TOP_LEFT);
            box.getChildren().add(row);
        }
        return box;
    }

    /** Read-only monospace output area. */
    static TextArea monoArea(int rows) {
        TextArea area = new TextArea();
        area.setEditable(false);
        area.setWrapText(false);
        area.setPrefRowCount(rows);
        area.getStyleClass().addAll("log-area", "mono-output");
        return area;
    }

    /** Two-column key/value grid; returns the grid so rows can be added with {@link #kv}. */
    static GridPane kvGrid() {
        GridPane grid = new GridPane();
        grid.setHgap(14);
        grid.setVgap(6);
        javafx.scene.layout.ColumnConstraints k = new javafx.scene.layout.ColumnConstraints();
        k.setMinWidth(Region.USE_PREF_SIZE);
        javafx.scene.layout.ColumnConstraints v = new javafx.scene.layout.ColumnConstraints();
        v.setHgrow(Priority.ALWAYS);
        v.setFillWidth(true);
        grid.getColumnConstraints().addAll(k, v);
        return grid;
    }

    static void kv(GridPane grid, String key, String value, String... valueStyles) {
        int row = grid.getRowCount();
        Label k = fixed(key, "kv-key");
        Label v = label(value, "kv-value");
        v.getStyleClass().addAll(valueStyles);
        v.setMaxWidth(Double.MAX_VALUE);
        grid.add(k, 0, row);
        grid.add(v, 1, row);
        GridPane.setValignment(k, javafx.geometry.VPos.TOP);
    }

    static Node spacer() {
        Region r = new Region();
        HBox.setHgrow(r, Priority.ALWAYS);
        return r;
    }
}
