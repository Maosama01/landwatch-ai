package com.landwatch;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.net.URL;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Stage 1 Sanity Tests.
 * Verifies application structure, classpath resources, and runtime environment.
 */
class MainAppTest {

    @Test
    @DisplayName("Verify Java Runtime is version 17")
    void testJavaVersion() {
        String version = System.getProperty("java.version");
        assertNotNull(version, "Java version must not be null");
        assertTrue(version.startsWith("17"), "Application must run on Java 17 LTS. Detected: " + version);
    }

    @Test
    @DisplayName("Verify FXML resource exists on classpath")
    void testFxmlResourceExists() {
        URL fxml = getClass().getResource("/com/landwatch/ui/views/dashboard.fxml");
        assertNotNull(fxml, "dashboard.fxml must be available in classpath at /com/landwatch/ui/views/dashboard.fxml");
    }

    @Test
    @DisplayName("Verify CSS stylesheet exists on classpath")
    void testCssResourceExists() {
        URL css = getClass().getResource("/com/landwatch/ui/css/dark_theme.css");
        assertNotNull(css, "dark_theme.css must be available in classpath at /com/landwatch/ui/css/dark_theme.css");
    }

    @Test
    @DisplayName("Verify Launcher class is loadable")
    void testLauncherClassLoadable() {
        assertDoesNotThrow(() -> {
            Class<?> clazz = Class.forName("com.landwatch.Launcher");
            assertNotNull(clazz);
        });
    }
}
