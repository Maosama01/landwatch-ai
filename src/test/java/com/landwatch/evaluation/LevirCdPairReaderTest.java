package com.landwatch.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.IndexColorModel;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LevirCdPairReader} using small synthetic images only (never the real dataset).
 */
class LevirCdPairReaderTest {

    @TempDir
    Path root;

    private static BufferedImage rgb(int w, int h, int gray) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                img.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
            }
        }
        return img;
    }

    /** 8-bit grayscale image whose raw samples are exactly {@code values[y][x]}. */
    private static BufferedImage gray(int[][] values) {
        BufferedImage img = new BufferedImage(values[0].length, values.length, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster r = img.getRaster();
        for (int y = 0; y < values.length; y++) {
            for (int x = 0; x < values[y].length; x++) {
                r.setSample(x, y, 0, values[y][x]);
            }
        }
        return img;
    }

    private static void writePng(BufferedImage img, Path file) throws IOException {
        Files.createDirectories(file.getParent());
        assertTrue(ImageIO.write(img, "png", file.toFile()), "PNG writer must be available");
    }

    private void writeTriplet(String split, String name, int w, int h) throws IOException {
        Path s = root.resolve(split);
        writePng(rgb(w, h, 40), s.resolve("A").resolve(name));
        writePng(rgb(w, h, 80), s.resolve("B").resolve(name));
        writePng(gray(new int[h][w]), s.resolve("label").resolve(name));
    }

    @Test
    @DisplayName("Triplets are paired by name, in natural order, ignoring ._* and other hidden files")
    void testPairsAndIgnoresHiddenFiles() throws IOException {
        for (String name : List.of("test_10.png", "test_2.png", "test_1.png")) {
            writeTriplet("test", name, 8, 8);
        }
        for (String sub : List.of("A", "B", "label")) {
            Path dir = root.resolve("test").resolve(sub);
            // macOS AppleDouble files: not valid PNGs, and only present in some folders
            Files.write(dir.resolve("._test_1.png"), new byte[]{0, 5, 22, 7});
            Files.write(dir.resolve(".DS_Store"), new byte[]{1, 2, 3});
        }
        Files.write(root.resolve("test").resolve("A").resolve("._test_99.png"), new byte[]{0, 5, 22, 7});
        Files.writeString(root.resolve("test").resolve("A").resolve("notes.txt"), "not an image");

        List<LevirCdPairReader.Triplet> triplets = new LevirCdPairReader(root).listTriplets("test");

        List<String> names = triplets.stream().map(LevirCdPairReader.Triplet::name).toList();
        assertEquals(List.of("test_1.png", "test_2.png", "test_10.png"), names);
        for (LevirCdPairReader.Triplet t : triplets) {
            assertEquals(root.resolve("test").resolve("A").resolve(t.name()), t.pre());
            assertEquals(root.resolve("test").resolve("B").resolve(t.name()), t.post());
            assertEquals(root.resolve("test").resolve("label").resolve(t.name()), t.label());
        }
    }

    @Test
    @DisplayName("Natural order sorts embedded numbers by value")
    void testNaturalOrder() {
        List<String> names = new ArrayList<>(List.of("test_100.png", "test_10.png", "test_2.png",
                "val_1.png", "test_1.png", "test_20.png"));
        names.sort(LevirCdPairReader.NATURAL_ORDER);
        assertEquals(List.of("test_1.png", "test_2.png", "test_10.png", "test_20.png", "test_100.png", "val_1.png"),
                names);
        assertEquals(0, LevirCdPairReader.compareNatural("test_7.png", "test_7.png"));
        assertTrue(LevirCdPairReader.compareNatural("test_007.png", "test_7.png") != 0,
                "Different names must never compare equal");
    }

    @Test
    @DisplayName("Grayscale masks are binarised with value > 0, including stray 156 and 254 values")
    void testGrayscaleMaskBinarisation() throws IOException {
        int[][] values = {
                {0, 1, 156, 254, 255},
                {0, 0, 0, 128, 0}
        };
        Path file = root.resolve("mask.png");
        writePng(gray(values), file);

        BufferedImage reread = ImageIO.read(file.toFile());
        assertEquals(156, reread.getRaster().getSample(2, 0, 0), "PNG round trip must keep raw values");
        assertEquals(254, reread.getRaster().getSample(3, 0, 0), "PNG round trip must keep raw values");

        boolean[][] mask = LevirCdPairReader.toChangeMask(reread);
        boolean[][] expected = {
                {false, true, true, true, true},
                {false, false, false, true, false}
        };
        assertTrue(Arrays.deepEquals(expected, mask), "Every value > 0 must be change, 0 must be no change");
    }

    @Test
    @DisplayName("Palette masks are binarised by palette colour, not by raw index")
    void testPaletteMaskBinarisation() throws IOException {
        // index 0 = black (no change), 1 = white, 2 = gray 156, 3 = red: all non-black entries are change
        byte[] r = {0, (byte) 255, (byte) 156, (byte) 255};
        byte[] g = {0, (byte) 255, (byte) 156, 0};
        byte[] b = {0, (byte) 255, (byte) 156, 0};
        IndexColorModel icm = new IndexColorModel(2, 4, r, g, b);
        BufferedImage palette = new BufferedImage(4, 2, BufferedImage.TYPE_BYTE_BINARY, icm);
        WritableRaster raster = palette.getRaster();
        int[][] indices = {{0, 1, 2, 3}, {3, 0, 0, 1}};
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 4; x++) {
                raster.setSample(x, y, 0, indices[y][x]);
            }
        }
        Path file = root.resolve("palette.png");
        writePng(palette, file);
        BufferedImage reread = ImageIO.read(file.toFile());
        assertInstanceOf(IndexColorModel.class, reread.getColorModel(), "Expected a palette PNG");

        boolean[][] expected = {{false, true, true, true}, {true, false, false, true}};
        assertTrue(Arrays.deepEquals(expected, LevirCdPairReader.toChangeMask(reread)));
        assertTrue(Arrays.deepEquals(expected, LevirCdPairReader.toChangeMask(palette)));
    }

    @Test
    @DisplayName("16-bit grayscale and RGB masks are also binarised with value > 0")
    void testOtherMaskFormats() {
        BufferedImage gray16 = new BufferedImage(3, 1, BufferedImage.TYPE_USHORT_GRAY);
        gray16.getRaster().setSample(1, 0, 0, 1);
        gray16.getRaster().setSample(2, 0, 0, 65535);
        assertTrue(Arrays.deepEquals(new boolean[][]{{false, true, true}}, LevirCdPairReader.toChangeMask(gray16)));

        BufferedImage colour = new BufferedImage(3, 1, BufferedImage.TYPE_INT_RGB);
        colour.setRGB(0, 0, 0x000000);
        colour.setRGB(1, 0, 0x000100);
        colour.setRGB(2, 0, 0xFFFFFF);
        assertTrue(Arrays.deepEquals(new boolean[][]{{false, true, true}}, LevirCdPairReader.toChangeMask(colour)));
    }

    @Test
    @DisplayName("Loading a triplet returns RGB images and the binarised label")
    void testLoadTriplet() throws IOException {
        Path s = root.resolve("val");
        writePng(rgb(5, 2, 10), s.resolve("A").resolve("val_1.png"));
        writePng(rgb(5, 2, 200), s.resolve("B").resolve("val_1.png"));
        writePng(gray(new int[][]{{0, 255, 254, 156, 0}, {0, 0, 0, 0, 255}}), s.resolve("label").resolve("val_1.png"));

        LevirCdPairReader reader = new LevirCdPairReader(root);
        LevirCdPairReader.Sample sample = reader.load(reader.listTriplets("val").get(0));

        assertEquals("val_1.png", sample.name());
        assertEquals(5, sample.width());
        assertEquals(2, sample.height());
        assertEquals(BufferedImage.TYPE_INT_RGB, sample.pre().getType());
        assertEquals(10, sample.pre().getRGB(0, 0) & 0xFF);
        assertEquals(200, sample.post().getRGB(0, 0) & 0xFF);
        assertTrue(Arrays.deepEquals(new boolean[][]{{false, true, true, true, false}, {false, false, false, false, true}},
                sample.changeMask()));
    }

    @Test
    @DisplayName("A missing counterpart file is reported with its name and the folders it is missing from")
    void testMissingCounterpartReported() throws IOException {
        writeTriplet("test", "test_1.png", 4, 4);
        writeTriplet("test", "test_2.png", 4, 4);
        Files.delete(root.resolve("test").resolve("label").resolve("test_2.png"));
        writePng(rgb(4, 4, 0), root.resolve("test").resolve("B").resolve("test_3.png"));

        LevirCdPairReader reader = new LevirCdPairReader(root);
        LevirCdPairReader.IncompleteSplitException ex =
                assertThrows(LevirCdPairReader.IncompleteSplitException.class, () -> reader.listTriplets("test"));

        assertEquals(List.of("test_2.png: present in A, B; missing in label",
                "test_3.png: present in B; missing in A, label"), ex.getProblems());
        assertTrue(ex.getMessage().contains("test_2.png: present in A, B; missing in label"), ex.getMessage());
        assertTrue(ex.getMessage().contains("2 file name(s)"), ex.getMessage());
    }

    @Test
    @DisplayName("A missing split or sub-folder is reported clearly")
    void testMissingFolders() throws IOException {
        LevirCdPairReader reader = new LevirCdPairReader(root);
        IOException noSplit = assertThrows(IOException.class, () -> reader.listTriplets("test"));
        assertTrue(noSplit.getMessage().contains("Split folder not found"), noSplit.getMessage());

        Files.createDirectories(root.resolve("test").resolve("A"));
        Files.createDirectories(root.resolve("test").resolve("B"));
        IOException noLabel = assertThrows(IOException.class, () -> reader.listTriplets("test"));
        assertTrue(noLabel.getMessage().contains("Missing folder 'label'"), noLabel.getMessage());
    }

    @Test
    @DisplayName("Images of different sizes within a triplet are rejected with a clear message")
    void testSizeMismatch() throws IOException {
        Path s = root.resolve("test");
        writePng(rgb(4, 4, 0), s.resolve("A").resolve("test_1.png"));
        writePng(rgb(4, 4, 0), s.resolve("B").resolve("test_1.png"));
        writePng(gray(new int[3][4]), s.resolve("label").resolve("test_1.png"));

        LevirCdPairReader reader = new LevirCdPairReader(root);
        LevirCdPairReader.Triplet t = reader.listTriplets("test").get(0);
        IOException ex = assertThrows(IOException.class, () -> reader.load(t));
        assertTrue(ex.getMessage().contains("label=4x3"), ex.getMessage());
    }
}
