package com.landwatch.ui;

import com.landwatch.ui.engine.EngineResult;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that saving a run writes the B/W mask, red overlay, thematic map and an honest summary.
 */
class ResultExporterTest {

    private static final LocalDateTime WHEN = LocalDateTime.of(2026, 11, 20, 9, 30, 5);

    private static BufferedImage solid(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return img;
    }

    @Test
    @DisplayName("Export writes four clearly named files")
    void exportWritesAllFiles(@TempDir Path tmp) throws Exception {
        boolean[][] mask = new boolean[8][8];
        mask[3][3] = mask[3][4] = mask[4][3] = mask[4][4] = true;
        EngineResult r = new EngineResult("baseline-otsu", "Classical baseline (Otsu differencing)",
                mask, 12, 42, 10L, null);
        ChangeVisualizationSet visuals = ChangeVisualizationSet.build(r, solid(8, 8, Color.GREEN));

        File dir = tmp.resolve("results").toFile();
        List<File> files = ResultExporter.export(dir, r, visuals, "bundled SYNTHETIC demo pair", WHEN);

        String base = "landwatch_baseline-otsu_20261120_093005";
        assertEquals(base, ResultExporter.baseName(r, WHEN));
        assertEquals(List.of(
                base + "_binary_mask_bw.png",
                base + "_red_overlay.png",
                base + "_thematic_change_map.png",
                base + "_summary.txt"),
                files.stream().map(File::getName).toList());
        for (File f : files) {
            assertTrue(f.isFile() && f.length() > 0, f + " must exist");
        }

        String summary = Files.readString(files.get(3).toPath(), StandardCharsets.UTF_8);
        assertTrue(summary.contains("Engine:           Classical baseline (Otsu differencing)"));
        assertTrue(summary.contains("bundled SYNTHETIC demo pair"));
        assertTrue(summary.contains("Changed pixels:   4 px"));
        assertTrue(summary.contains("Otsu threshold:   42 / 255"));
        assertTrue(summary.contains("Engine latency:   12 ms"));
        assertTrue(summary.contains("No ground-sample-distance calibration"));
    }

    @Test
    @DisplayName("Two runs in the same second do not overwrite each other")
    void secondRunInSameSecondGetsUniquePrefix(@TempDir Path tmp) throws Exception {
        boolean[][] mask = new boolean[4][4];
        mask[1][1] = true;
        EngineResult r = EngineResult.fromMask("baseline-otsu", "Classical baseline (Otsu differencing)", mask, 1);
        ChangeVisualizationSet visuals = ChangeVisualizationSet.build(r, solid(4, 4, Color.GREEN));
        File dir = tmp.toFile();

        List<File> first = ResultExporter.export(dir, r, visuals, "in", WHEN);
        List<File> second = ResultExporter.export(dir, r, visuals, "in", WHEN);

        assertEquals("landwatch_baseline-otsu_20261120_093005_binary_mask_bw.png", first.get(0).getName());
        assertEquals("landwatch_baseline-otsu_20261120_093005-2_binary_mask_bw.png", second.get(0).getName());
        assertEquals(8, dir.listFiles().length);
    }

    @Test
    @DisplayName("Summary marks Otsu as n/a for engines that do not use it")
    void summaryWithoutOtsu() {
        EngineResult r = EngineResult.fromMask("siamese-unet", "Deep model – Siamese U-Net", new boolean[4][4], 7);
        String summary = ResultExporter.buildSummary(r, "test input", WHEN);
        assertTrue(summary.contains("Engine:           Deep model – Siamese U-Net"));
        assertTrue(summary.contains("Otsu threshold:   n/a"));
        assertFalse(summary.contains("Raw changed px"));
        assertFalse(summary.toLowerCase().contains("hectare:"));
    }
}
