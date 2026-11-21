package com.landwatch.ui.data;

import com.landwatch.deeplearning.eurosat.EuroSatSample;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** The EuroSAT test split as the Land use page reads it: CSV, picking, true-label lookup, recorded errors. */
class EuroSatTestSetTest {

    private static final String HEADER = "path,class_name,class_index";

    /** Three classes, 4 test images each (Forest_1..4, River_1..4, SeaLake_1..4) plus train/val rows. */
    private static Path writeSplits(Path root) throws IOException {
        Path splits = Files.createDirectories(root.resolve("splits"));
        List<String> test = new ArrayList<>(List.of(HEADER));
        String[] classes = {"Forest", "River", "SeaLake"};
        for (int c = 0; c < classes.length; c++) {
            for (int i = 1; i <= 4; i++) {
                test.add("EuroSAT_RGB/" + classes[c] + "/" + classes[c] + "_" + i + ".jpg," + classes[c] + "," + c);
            }
        }
        Files.write(splits.resolve("test.csv"), test, StandardCharsets.UTF_8);
        Files.write(splits.resolve("train.csv"), List.of(HEADER,
                "EuroSAT_RGB/Forest/Forest_10.jpg,Forest,0",
                "EuroSAT_RGB/River/River_10.jpg,River,1"), StandardCharsets.UTF_8);
        Files.write(splits.resolve("val.csv"), List.of(HEADER,
                "EuroSAT_RGB/SeaLake/SeaLake_20.jpg,SeaLake,2"), StandardCharsets.UTF_8);
        return splits;
    }

    @Test
    @DisplayName("The test CSV is read with class names in index order; images are reported missing until EuroSAT_RGB exists")
    void readsSplitAndChecksImages(@TempDir Path root) throws Exception {
        EuroSatTestSet set = EuroSatTestSet.load(root, writeSplits(root));
        assertTrue(set.isOk(), set.getProblem());
        assertEquals(12, set.size());
        assertEquals(List.of("Forest", "River", "SeaLake"), set.getClassNames());
        assertEquals(root.resolve("EuroSAT_RGB/River/River_2.jpg"), set.resolve(set.getSamples().get(5)));

        assertFalse(set.imagesAvailable());
        String msg = set.missingImagesMessage();
        assertTrue(msg.contains("EuroSAT_RGB/<ClassName>/*.jpg"), msg);
        assertTrue(msg.contains("EuroSAT_RGB.zip") && msg.contains("Zenodo record 7711810"), msg);
        assertTrue(msg.contains("-D" + AppPaths.EUROSAT_DIR_PROPERTY + "="), msg);
        assertTrue(msg.contains("still classify any image"), msg);

        Files.createDirectories(root.resolve("EuroSAT_RGB"));
        assertTrue(set.imagesAvailable(), "checked live, no reload needed");
    }

    @Test
    @DisplayName("A missing or inconsistent test CSV gives an empty set with the reason, never invented rows")
    void missingOrBadCsv(@TempDir Path root) throws Exception {
        EuroSatTestSet none = EuroSatTestSet.load(root, root.resolve("splits"));
        assertFalse(none.isOk());
        assertTrue(none.getProblem().startsWith("test split not found"), none.getProblem());
        assertEquals(0, none.size());
        assertThrows(IllegalStateException.class, () -> none.pickOne(new Random(1)));
        assertTrue(none.pickRandom(5, new Random(1)).isEmpty());
        assertTrue(none.lookup(root.resolve("EuroSAT_RGB/Forest/Forest_1.jpg")).isEmpty());

        Path splits = Files.createDirectories(root.resolve("bad"));
        Files.write(splits.resolve("test.csv"), List.of(HEADER,
                "EuroSAT_RGB/Forest/Forest_1.jpg,Forest,0",
                "EuroSAT_RGB/River/River_1.jpg,River,0"), StandardCharsets.UTF_8);
        EuroSatTestSet clash = EuroSatTestSet.load(root, splits);
        assertFalse(clash.isOk());
        assertTrue(clash.getProblem().contains("both Forest and River"), clash.getProblem());

        Files.write(splits.resolve("test.csv"), List.of(HEADER,
                "EuroSAT_RGB/Forest/Forest_1.jpg,Forest,0",
                "EuroSAT_RGB/River/River_1.jpg,River,2"), StandardCharsets.UTF_8);
        EuroSatTestSet gap = EuroSatTestSet.load(root, splits);
        assertFalse(gap.isOk());
        assertTrue(gap.getProblem().contains("not 0..2"), gap.getProblem());

        Files.write(splits.resolve("test.csv"), List.of("wrong,header"), StandardCharsets.UTF_8);
        assertFalse(EuroSatTestSet.load(root, splits).isOk());
    }

    @Test
    @DisplayName("Random picks are distinct test rows, reproducible with a seed, and capped at the split size")
    void randomPicks(@TempDir Path root) throws Exception {
        EuroSatTestSet set = EuroSatTestSet.load(root, writeSplits(root));
        List<EuroSatSample> a = set.pickRandom(5, new Random(42));
        List<EuroSatSample> b = set.pickRandom(5, new Random(42));
        assertEquals(a, b);
        assertEquals(5, new HashSet<>(a).size());
        assertTrue(set.getSamples().containsAll(a));

        List<EuroSatSample> all = set.pickRandom(100, new Random(3));
        assertEquals(12, all.size());
        assertEquals(new HashSet<>(set.getSamples()), new HashSet<>(all));
        assertTrue(set.pickRandom(0, new Random(3)).isEmpty());
        assertThrows(IllegalArgumentException.class, () -> set.pickRandom(-1, new Random(3)));
        assertTrue(set.getSamples().contains(set.pickOne(new Random(9))));

        // every row can be picked: over many draws each of the 12 rows shows up
        Set<EuroSatSample> seen = new HashSet<>();
        Random rnd = new Random(7);
        for (int i = 0; i < 500; i++) {
            seen.add(set.pickOne(rnd));
        }
        assertEquals(12, seen.size());
    }

    @Test
    @DisplayName("One per class gives one row of every class in class order; pickAmong keeps only listed test rows")
    void perClassAndAmong(@TempDir Path root) throws Exception {
        EuroSatTestSet set = EuroSatTestSet.load(root, writeSplits(root));
        List<EuroSatSample> perClass = set.pickPerClass(1, new Random(5));
        assertEquals(List.of(0, 1, 2), perClass.stream().map(EuroSatSample::classIndex).toList());
        assertEquals(6, set.pickPerClass(2, new Random(5)).size());

        List<String> wanted = List.of("EuroSAT_RGB/River/River_3.jpg", "EuroSAT_RGB/SeaLake/SeaLake_1.jpg",
                "EuroSAT_RGB/Forest/Forest_10.jpg" /* a train image: ignored */, "EuroSAT_RGB/Nope/x.jpg");
        List<EuroSatSample> among = set.pickAmong(wanted, 10, new Random(1));
        assertEquals(Set.of("EuroSAT_RGB/River/River_3.jpg", "EuroSAT_RGB/SeaLake/SeaLake_1.jpg"),
                new HashSet<>(among.stream().map(EuroSatSample::relativePath).toList()));
        assertEquals(1, set.pickAmong(wanted, 1, new Random(1)).size());
    }

    @Test
    @DisplayName("The true label of a dataset file comes from its class folder, its split from the split CSVs")
    void lookup(@TempDir Path root) throws Exception {
        EuroSatTestSet set = EuroSatTestSet.load(root, writeSplits(root));
        Optional<EuroSatTestSet.DatasetMatch> test = set.lookup(root.resolve("EuroSAT_RGB/River/River_2.jpg"));
        assertTrue(test.isPresent());
        assertEquals("River", test.get().trueClass());
        assertEquals(1, test.get().trueIndex());
        assertEquals("test", test.get().split());
        assertTrue(test.get().isTestImage());

        assertEquals("train", set.lookup(root.resolve("EuroSAT_RGB/Forest/Forest_10.jpg")).orElseThrow().split());
        assertEquals("val", set.lookup(root.resolve("EuroSAT_RGB/SeaLake/SeaLake_20.jpg")).orElseThrow().split());

        // a file in a class folder that no split lists: class from the folder, split unknown
        EuroSatTestSet.DatasetMatch unlisted = set.lookup(root.resolve("EuroSAT_RGB/Forest/Forest_999.jpg")).orElseThrow();
        assertEquals("Forest", unlisted.trueClass());
        assertNull(unlisted.split());
        assertFalse(unlisted.inKnownSplit());

        // a copy of the dataset somewhere else is recognised by its EuroSAT_RGB/<Class>/<file> path
        Path elsewhere = root.resolve("downloads").resolve("EuroSAT_RGB").resolve("River").resolve("River_4.jpg");
        assertEquals("test", set.lookup(elsewhere).orElseThrow().split());

        // not a EuroSAT file: no true label is claimed
        assertTrue(set.lookup(root.resolve("photos/River/River_2.jpg")).isEmpty());
        assertTrue(set.lookup(root.resolve("EuroSAT_RGB/Desert/Desert_1.jpg")).isEmpty());
        assertTrue(set.lookup(root.resolve("test_7.png")).isEmpty());
        assertTrue(set.lookup(null).isEmpty());
    }

    @Test
    @DisplayName("Recorded errors are the rows with correct = no, columns found by header name")
    void recordedErrors(@TempDir Path tmp) throws Exception {
        Path csv = tmp.resolve("test_predictions.csv");
        Files.write(csv, List.of(
                "path,actual_class,actual_index,predicted_class,predicted_index,predicted_probability,correct",
                "EuroSAT_RGB/Forest/Forest_1.jpg,Forest,0,Forest,0,0.99,yes",
                "EuroSAT_RGB/Forest/Forest_2.jpg,Forest,0,River,1,0.51,no",
                "broken",
                "EuroSAT_RGB/River/River_9.jpg,River,1,SeaLake,2,0.70,no",
                ""), StandardCharsets.UTF_8);
        assertEquals(List.of("EuroSAT_RGB/Forest/Forest_2.jpg", "EuroSAT_RGB/River/River_9.jpg"),
                EuroSatTestSet.readRecordedErrors(csv));

        Files.write(csv, List.of("correct,path", "no,a.jpg", "yes,b.jpg"), StandardCharsets.UTF_8);
        assertEquals(List.of("a.jpg"), EuroSatTestSet.readRecordedErrors(csv));

        Files.write(csv, List.of("image,ok", "a.jpg,no"), StandardCharsets.UTF_8);
        assertThrows(IOException.class, () -> EuroSatTestSet.readRecordedErrors(csv));
        assertThrows(IOException.class, () -> EuroSatTestSet.readRecordedErrors(tmp.resolve("missing.csv")));
    }

    @Test
    @DisplayName("Committed files: the test split has 10 classes and its recorded errors match the metrics file")
    void committedSplitAndResults() throws Exception {
        Path splits = Path.of(AppPaths.DEFAULT_EUROSAT_DIR, "splits");
        Assumptions.assumeTrue(Files.isRegularFile(splits.resolve("test.csv")), "split CSVs not present");
        EuroSatTestSet set = EuroSatTestSet.load(Path.of(AppPaths.DEFAULT_EUROSAT_DIR), splits);
        assertTrue(set.isOk(), set.getProblem());
        assertEquals(10, set.getClassNames().size());
        assertEquals("AnnualCrop", set.getClassNames().get(0));
        assertEquals("SeaLake", set.getClassNames().get(9));

        Path predictions = Path.of(AppPaths.DEFAULT_RESULTS_DIR, "eurosat", "test_predictions.csv");
        Path metrics = Path.of(AppPaths.DEFAULT_RESULTS_DIR, "eurosat", "test_metrics.txt");
        Assumptions.assumeTrue(Files.isRegularFile(predictions) && Files.isRegularFile(metrics), "results not present");
        List<String> errors = EuroSatTestSet.readRecordedErrors(predictions);
        EuroSatTestReport report = EuroSatTestReport.read(metrics);
        assertTrue(report.getWrong().isPresent());
        assertEquals(report.getWrong().getAsInt(), errors.size());
        assertEquals(report.getImages().getAsInt(), set.size());
        // every recorded error is a test image, so the "Recorded errors" gallery can pick all of them
        assertEquals(errors.size(), set.pickAmong(errors, Integer.MAX_VALUE, new Random(1)).size());
    }
}
