package com.landwatch;

/**
 * Application Entry Launcher.
 * <p>
 * Standard Java entry point that delegates to {@link MainApp}.
 * This avoids classpath module constraints when launching JavaFX applications.
 */
public class Launcher {
    public static void main(String[] args) {
        MainApp.main(args);
    }
}
