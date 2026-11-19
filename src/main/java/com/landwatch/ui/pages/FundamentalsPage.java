package com.landwatch.ui.pages;

import com.landwatch.ui.FundamentalsDemos;
import javafx.concurrent.Task;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ProgressIndicator;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.TextArea;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

/**
 * "NN fundamentals" page: buttons that RUN the project's pure-Java neural-network code on a background
 * thread and show its output. Nothing is pre-recorded.
 */
public final class FundamentalsPage {

    private final ScrollPane root;
    private final AtomicInteger running = new AtomicInteger();
    private final Consumer<String> log;

    public FundamentalsPage(Consumer<String> log) {
        this.log = log;
        VBox content = Ui.pageContent("NN fundamentals",
                "Pure-Java implementations in com.landwatch.fundamentals (no ML libraries). Each button runs the real "
                        + "computation on a background thread with a fixed seed; the output below is produced live.",
                "computed live in plain Java");

        content.getChildren().add(demoCard("runXorButton", "XOR with a multilayer perceptron and backpropagation",
                "Trains a 2-4-1 sigmoid network on the XOR truth table with backpropagation and plain SGD.",
                "▶  Run XOR training", 20, FundamentalsDemos::runXor));
        content.getChildren().add(demoCard("runGradientCheckButton", "Gradient check: backpropagation against finite differences",
                "Compares every analytical gradient of a 2-3-1 network with a central finite-difference estimate.",
                "▶  Run gradient check", 15, FundamentalsDemos::runGradientCheck));
        content.getChildren().add(demoCard("runLearningRulesButton", "Classic learning rules",
                "Hebbian, Oja, competitive (winner-take-all), perceptron, LMS / delta rule and Boltzmann learning on small "
                        + "synthetic data sets.",
                "▶  Run learning-rules demo", 22, FundamentalsDemos::runLearningRules));

        root = Ui.page(content);
        root.setId("fundamentalsPage");
    }

    public Node getRoot() {
        return root;
    }

    /** True while at least one demo is running. */
    public boolean isRunning() {
        return running.get() > 0;
    }

    private VBox demoCard(String buttonId, String title, String description, String buttonText, int rows,
                          Callable<String> computation) {
        VBox card = Ui.card(title, "computed live", null);
        Label desc = Ui.label(description, "page-note");
        desc.setMaxWidth(Double.MAX_VALUE);
        HBox.setHgrow(desc, Priority.ALWAYS);

        Button run = new Button(buttonText);
        run.setId(buttonId);
        run.getStyleClass().add("btn-secondary");
        run.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        ProgressIndicator spinner = new ProgressIndicator();
        spinner.setPrefSize(18, 18);
        spinner.setVisible(false);
        Label status = Ui.fixed("not run yet", "engine-note");

        HBox controls = new HBox(10, run, spinner, status);
        controls.setAlignment(Pos.CENTER_LEFT);
        TextArea output = Ui.monoArea(rows);
        output.setPromptText("Click the button to run the computation.");

        run.setOnAction(e -> {
            run.setDisable(true);
            spinner.setVisible(true);
            status.setText("running…");
            running.incrementAndGet();
            long t0 = System.nanoTime();
            Task<String> task = new Task<>() {
                @Override
                protected String call() throws Exception {
                    return computation.call();
                }
            };
            task.setOnSucceeded(ev -> {
                output.setText(task.getValue());
                output.positionCaret(0);
                output.setScrollTop(0);
                finish(run, spinner, status, String.format("finished in %d ms", (System.nanoTime() - t0) / 1_000_000));
                log.accept("[Fundamentals] " + buttonText.replace("▶", "").trim() + ": finished.");
            });
            task.setOnFailed(ev -> {
                Throwable t = task.getException();
                output.setText("ERROR: " + t);
                finish(run, spinner, status, "failed");
                log.accept("[Fundamentals] " + buttonText.replace("▶", "").trim() + " failed: " + t.getMessage());
            });
            Thread thread = new Thread(task, "landwatch-fundamentals");
            thread.setDaemon(true);
            thread.start();
        });

        card.getChildren().addAll(desc, controls, output);
        return card;
    }

    private void finish(Button run, ProgressIndicator spinner, Label status, String text) {
        run.setDisable(false);
        spinner.setVisible(false);
        status.setText(text);
        running.decrementAndGet();
    }
}
