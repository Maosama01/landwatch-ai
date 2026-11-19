package com.landwatch.deeplearning.eurosat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The stratified split on a tiny synthetic folder tree (never the real dataset). */
class EuroSatSplitTest {

    @TempDir
    Path root;

    /** Creates EuroSAT_RGB/<Class>/<Class>_<n>.jpg placeholders (the split only lists file names). */
    private void createTree() throws IOException {
        createClass("Beta", 20);
        createClass("Alpha", 10);
        createClass("Gamma", 7);
        // must be ignored: a non-JPEG file and a hidden macOS file
        Files.writeString(root.resolve("EuroSAT_RGB/Alpha/readme.txt"), "x");
        Files.writeString(root.resolve("EuroSAT_RGB/Alpha/._Alpha_1.jpg"), "x");
    }

    private void createClass(String name, int n) throws IOException {
        Path dir = Files.createDirectories(root.resolve("EuroSAT_RGB").resolve(name));
        for (int i = 1; i <= n; i++) {
            Files.write(dir.resolve(name + "_" + i + ".jpg"), new byte[] {1});
        }
    }

    private static Set<String> paths(List<EuroSatSample> samples) {
        Set<String> s = new HashSet<>();
        for (EuroSatSample x : samples) {
            s.add(x.relativePath());
        }
        return s;
    }

    @Test
    void classesAreAlphabeticalAndIndexedFromZero() throws IOException {
        createTree();
        EuroSatSplit.Split split = EuroSatSplit.create(root, "EuroSAT_RGB", 42);
        assertEquals(List.of("Alpha", "Beta", "Gamma"), split.classNames());
        for (String name : EuroSatSplit.SPLIT_NAMES) {
            for (EuroSatSample s : split.get(name)) {
                assertEquals(split.classNames().get(s.classIndex()), s.className());
                assertTrue(s.relativePath().startsWith("EuroSAT_RGB/" + s.className() + "/" + s.className() + "_"),
                        s.relativePath());
            }
        }
    }

    @Test
    void eachClassIsSplit70_15_15() throws IOException {
        createTree();
        EuroSatSplit.Split split = EuroSatSplit.create(root, "EuroSAT_RGB", 42);
        int[][] counts = split.countsPerClass();
        // Alpha 10 -> 7 / 2 / 1 (round(1.5) = 2), Beta 20 -> 14 / 3 / 3, Gamma 7 -> 5 / 1 / 1
        assertArrayEquals(new int[] {7, 14, 5}, counts[0]);
        assertArrayEquals(new int[] {2, 3, 1}, counts[1]);
        assertArrayEquals(new int[] {1, 3, 1}, counts[2]);
        // the real EuroSAT class sizes divide exactly
        assertArrayEquals(new int[] {2100, 450, 450}, EuroSatSplit.splitSizes(3000));
        assertArrayEquals(new int[] {1750, 375, 375}, EuroSatSplit.splitSizes(2500));
        assertArrayEquals(new int[] {1400, 300, 300}, EuroSatSplit.splitSizes(2000));
    }

    @Test
    void splitsDoNotOverlapAndCoverEveryImage() throws IOException {
        createTree();
        EuroSatSplit.Split split = EuroSatSplit.create(root, "EuroSAT_RGB", 42);
        Set<String> train = paths(split.train());
        Set<String> val = paths(split.val());
        Set<String> test = paths(split.test());
        assertEquals(split.train().size(), train.size());
        assertTrue(train.stream().noneMatch(val::contains));
        assertTrue(train.stream().noneMatch(test::contains));
        assertTrue(val.stream().noneMatch(test::contains));
        Set<String> all = new HashSet<>(train);
        all.addAll(val);
        all.addAll(test);
        assertEquals(37, all.size());
        EuroSatSplit.verifyDisjointAndComplete(split, EuroSatSplit.listImages(root, "EuroSAT_RGB"));

        List<EuroSatSample> leaked = new ArrayList<>(split.val());
        leaked.add(split.train().get(0));
        EuroSatSplit.Split bad = new EuroSatSplit.Split(split.classNames(), split.train(), leaked, split.test());
        assertThrows(IllegalStateException.class, () -> EuroSatSplit.verifyDisjointAndComplete(bad, null));
    }

    @Test
    void fixedSeedIsReproducibleAndSeedMatters() throws IOException {
        createTree();
        EuroSatSplit.Split a = EuroSatSplit.create(root, "EuroSAT_RGB", 42);
        EuroSatSplit.Split b = EuroSatSplit.create(root, "EuroSAT_RGB", 42);
        EuroSatSplit.Split c = EuroSatSplit.create(root, "EuroSAT_RGB", 7);
        assertEquals(a.train(), b.train());
        assertEquals(a.val(), b.val());
        assertEquals(a.test(), b.test());
        assertNotEquals(paths(a.train()), paths(c.train()));
    }

    @Test
    void csvRoundTripAndMainWritesFiles() throws IOException {
        createTree();
        Path out = root.resolve("out");
        EuroSatSplit.main(new String[] {"--root", root.toString(), "--out", out.toString(), "--seed", "42"});
        EuroSatSplit.Split expected = EuroSatSplit.create(root, "EuroSAT_RGB", 42);
        EuroSatSplit.Split read = EuroSatSplit.read(out.resolve("splits"));
        assertEquals(expected, read);
        List<String> lines = Files.readAllLines(out.resolve("splits/val.csv"), StandardCharsets.UTF_8);
        assertEquals(EuroSatSplit.CSV_HEADER, lines.get(0));
        assertTrue(lines.get(1).matches("EuroSAT_RGB/Alpha/Alpha_\\d+\\.jpg,Alpha,0"), lines.get(1));
        String summary = Files.readString(out.resolve("dataset_summary.txt"));
        assertTrue(summary.lines().anyMatch(l -> l.matches("\\s*1\\s+Beta\\s+14\\s+3\\s+3\\s+20")), summary);
        assertTrue(summary.lines().anyMatch(l -> l.matches("\\s+total\\s+26\\s+6\\s+5\\s+37")), summary);
    }

    @Test
    void readRejectsInconsistentClassIndices() throws IOException {
        Path dir = Files.createDirectories(root.resolve("splits"));
        EuroSatSplit.writeCsv(dir.resolve("train.csv"), List.of(new EuroSatSample("I/A/A_1.jpg", "A", 0),
                new EuroSatSample("I/B/B_1.jpg", "B", 1)));
        EuroSatSplit.writeCsv(dir.resolve("val.csv"), List.of(new EuroSatSample("I/A/A_2.jpg", "A", 1)));
        EuroSatSplit.writeCsv(dir.resolve("test.csv"), List.of());
        assertThrows(IOException.class, () -> EuroSatSplit.read(dir));
    }

    @Test
    void filesAreOrderedByTheirNumber() {
        List<String> names = new ArrayList<>(List.of("Forest_10.jpg", "Forest_2.jpg", "Forest_1.jpg", "notes.jpg"));
        names.sort(EuroSatSplit.FILE_ORDER);
        assertEquals(List.of("Forest_1.jpg", "Forest_2.jpg", "Forest_10.jpg", "notes.jpg"), names);
    }
}
