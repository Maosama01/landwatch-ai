package com.landwatch.ui.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class AppPathsTest {

    private static final Path HOME = Path.of("").toAbsolutePath().getRoot().resolve("home-for-test").resolve("someone");

    @Test
    @DisplayName("Paths under the home directory are shown with ~ instead of the account name")
    void homeIsAbbreviated() {
        Path model = HOME.resolve("Projects").resolve("landwatch-AI").resolve("models").resolve("siamese_unet_levir");
        assertEquals("~/Projects/landwatch-AI/models/siamese_unet_levir", AppPaths.abbreviateHome(model, HOME.toString()));
        assertEquals("~", AppPaths.abbreviateHome(HOME, HOME.toString()));
    }

    @Test
    @DisplayName("Paths outside the home directory, or a sibling with the same prefix, are left unchanged")
    void otherPathsUnchanged() {
        Path sibling = HOME.getParent().resolve("someone-else").resolve("data");
        assertEquals(sibling.toString(), AppPaths.abbreviateHome(sibling, HOME.toString()));
        Path data = HOME.getRoot().resolve("data").resolve("LEVIR-CD");
        assertEquals(data.toString(), AppPaths.abbreviateHome(data, HOME.toString()));
        assertEquals(data.toString(), AppPaths.abbreviateHome(data, null));
    }

    @Test
    @DisplayName("forDisplay never shows the real home directory path")
    void forDisplayHidesRealHome() {
        String shown = AppPaths.forDisplay(Path.of("models"));
        assertTrue(shown.endsWith("models"));
        String home = System.getProperty("user.home");
        if (home != null && Path.of("").toAbsolutePath().startsWith(Path.of(home))) {
            assertFalse(shown.contains(home), shown);
            assertTrue(shown.startsWith("~"), shown);
        }
    }
}
