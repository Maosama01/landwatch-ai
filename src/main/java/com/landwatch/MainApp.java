package com.landwatch;

import com.landwatch.ui.controllers.DashboardController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.IOException;
import java.net.URL;

/**
 * Main JavaFX Application for LandWatch AI.
 * Project Number: 50
 * Project Title: Satellite Land Use Change and Encroachment Detector
 */
public class MainApp extends Application {

    public static final String APP_TITLE = "LANDWATCH AI — Satellite Land Use Change & Encroachment Detector";
    public static final double DEFAULT_WIDTH = 1280;
    public static final double DEFAULT_HEIGHT = 840;
    public static final double MIN_WIDTH = 1080;
    public static final double MIN_HEIGHT = 700;

    private DashboardController controller;

    @Override
    public void start(Stage primaryStage) {
        try {
            URL fxmlLocation = getClass().getResource("/com/landwatch/ui/views/dashboard.fxml");
            if (fxmlLocation == null) {
                throw new IllegalStateException("Cannot find dashboard.fxml in resources at /com/landwatch/ui/views/dashboard.fxml");
            }

            FXMLLoader loader = new FXMLLoader(fxmlLocation);
            Parent root = loader.load();
            controller = loader.getController();

            Scene scene = new Scene(root, DEFAULT_WIDTH, DEFAULT_HEIGHT);

            URL cssLocation = getClass().getResource("/com/landwatch/ui/css/dark_theme.css");
            if (cssLocation != null) {
                scene.getStylesheets().add(cssLocation.toExternalForm());
            }

            primaryStage.setTitle(APP_TITLE);
            primaryStage.setScene(scene);
            primaryStage.setMinWidth(MIN_WIDTH);
            primaryStage.setMinHeight(MIN_HEIGHT);
            primaryStage.show();

        } catch (IOException e) {
            System.err.println("Failed to load application layout: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /** Releases the deep model's native resources when the application exits. */
    @Override
    public void stop() {
        if (controller != null) {
            controller.shutdown();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
