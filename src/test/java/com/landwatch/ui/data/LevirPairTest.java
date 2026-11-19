package com.landwatch.ui.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/** Locating and loading a LEVIR-CD pair (A, B, label) from one picked file. */
class LevirPairTest {

    private static Path png(Path dir, String name, int w, int h, int rgb) throws IOException {
        Files.createDirectories(dir);
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.setRGB(x, y, rgb);
            }
        }
        Path p = dir.resolve(name);
        ImageIO.write(img, "png", p.toFile());
        return p;
    }

    private static Path label(Path dir, String name, int w, int h) throws IOException {
        Files.createDirectories(dir);
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
        img.getRaster().setSample(1, 1, 0, 255);
        img.getRaster().setSample(2, 1, 0, 156);   // stray LEVIR value, still change (> 0)
        Path p = dir.resolve(name);
        ImageIO.write(img, "png", p.toFile());
        return p;
    }

    @Test
    @DisplayName("Picking A/, B/ or label/ finds the same triplet; the label is loaded as value > 0 = change")
    void locatesAndLoadsTriplet(@TempDir Path tmp) throws Exception {
        Path split = tmp.resolve("LEVIR-CD").resolve("test");
        Path a = png(split.resolve("A"), "test_7.png", 4, 3, 0x336633);
        Path b = png(split.resolve("B"), "test_7.png", 4, 3, 0x996633);
        Path l = label(split.resolve("label"), "test_7.png", 4, 3);

        for (Path picked : new Path[]{a, b, l}) {
            LevirPair pair = LevirPair.locate(picked);
            assertEquals("test_7.png", pair.name());
            assertEquals("test", pair.split());
            assertEquals(a.toAbsolutePath(), pair.pre());
            assertEquals(b.toAbsolutePath(), pair.post());
            assertEquals(l.toAbsolutePath(), pair.label());
            assertTrue(pair.hasLabel());
            assertEquals("LEVIR-CD test/test_7.png", pair.shortName());
        }

        LevirPair.Loaded loaded = LevirPair.locate(a).load();
        assertTrue(loaded.hasGroundTruth());
        assertEquals(4, loaded.pre().getWidth());
        assertEquals(3, loaded.post().getHeight());
        boolean[][] gt = loaded.groundTruth();
        assertTrue(gt[1][1]);
        assertTrue(gt[1][2]);
        assertFalse(gt[0][0]);
        int count = 0;
        for (boolean[] row : gt) {
            for (boolean v : row) {
                count += v ? 1 : 0;
            }
        }
        assertEquals(2, count);
    }

    @Test
    @DisplayName("Missing label: the pair still loads, without ground truth")
    void missingLabel(@TempDir Path tmp) throws Exception {
        Path split = tmp.resolve("val");
        Path a = png(split.resolve("A"), "val_3.png", 5, 5, 0x101010);
        png(split.resolve("B"), "val_3.png", 5, 5, 0x202020);
        LevirPair pair = LevirPair.locate(a);
        assertNull(pair.label());
        assertFalse(pair.hasLabel());
        assertFalse(pair.load().hasGroundTruth());
    }

    @Test
    @DisplayName("Files outside A/B/label, missing counterparts and size mismatches are explained")
    void invalidPicks(@TempDir Path tmp) throws Exception {
        Path loose = png(tmp.resolve("photos"), "x.png", 2, 2, 0);
        IOException notLevir = assertThrows(IOException.class, () -> LevirPair.locate(loose));
        assertTrue(notLevir.getMessage().contains("A/ or B/"), notLevir.getMessage());

        Path onlyA = png(tmp.resolve("s1").resolve("A"), "p.png", 2, 2, 0);
        IOException noB = assertThrows(IOException.class, () -> LevirPair.locate(onlyA));
        assertTrue(noB.getMessage().contains("Missing after image (B)"), noB.getMessage());

        Path onlyB = png(tmp.resolve("s2").resolve("B"), "p.png", 2, 2, 0);
        IOException noA = assertThrows(IOException.class, () -> LevirPair.locate(onlyB));
        assertTrue(noA.getMessage().contains("Missing before image (A)"), noA.getMessage());

        Path s3 = tmp.resolve("s3");
        Path a = png(s3.resolve("A"), "q.png", 4, 4, 0);
        png(s3.resolve("B"), "q.png", 4, 4, 0);
        label(s3.resolve("label"), "q.png", 5, 4);
        IOException sizes = assertThrows(IOException.class, () -> LevirPair.locate(a).load());
        assertTrue(sizes.getMessage().contains("does not match"), sizes.getMessage());

        assertThrows(IOException.class, () -> LevirPair.locate(null));
    }

    @Test
    @DisplayName("Area in m² assumes 0.5 m/px (0.25 m² per pixel)")
    void areaAssumption() {
        assertEquals(0.5, LevirPair.LEVIR_GSD_METRES);
        assertEquals(1.0, LevirPair.areaSquareMetres(4), 1e-12);
        assertEquals(262_144.0, LevirPair.areaSquareMetres(1024L * 1024L), 1e-9);
    }
}
