package com.landwatch.ui.data;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Reading results/eurosat/test_metrics.txt for the Land use page; nothing may be filled in. */
class EuroSatTestReportTest {

    /** The layout EvaluateEuroSatClassifier writes, shortened to three classes. */
    private static final List<String> REPORT = List.of(
            "LandWatch AI - EuroSAT land-use / land-cover classification (ResNet), evaluation on the TEST split",
            "Generated: 2026-11-19T22:13:15.666683",
            "Protocol: TEST evaluated once, with the final model selected on VAL; TEST was not used for training or selection.",
            "Inference device: cpu(), DJL 0.40.0 / PyTorch 2.7.1 (training device: mps(0))",
            "",
            "Overall accuracy : 0.9500",
            "Macro F1         : 0.9400",
            "Macro precision  : 0.9450",
            "Images           : 100 (95 correct, 5 wrong)",
            "",
            "Per-class metrics (one-vs-rest from the confusion matrix):",
            "idx  class                 precision     recall         F1  support",
            "  0  Forest                  0.9800     0.9600     0.9700       50",
            "  1  River                   0.9000     0.9333     0.9164       30",
            "  2  SeaLake                 0.9500     0.9500     0.9500       20",
            "     macro average            0.9433     0.9478     0.9455      100",
            "",
            "Confusion matrix (rows = actual class, columns = predicted class, in index order):",
            "     actual \\ predicted      0    1    2",
            "  0  Forest                48    2    0",
            "  1  River                  1   28    1",
            "  2  SeaLake                0    1   19",
            "",
            "Most frequent confusions (actual -> predicted: count):",
            "  Forest -> River: 2",
            "  River -> Forest: 1",
            "",
            "Inference: 1.0 s for 100 images in batches of 64 on cpu() (10.00 ms/image)");

    private static Path write(Path dir, List<String> lines) throws Exception {
        return Files.write(dir.resolve("test_metrics.txt"), lines, StandardCharsets.UTF_8);
    }

    @Test
    @DisplayName("Headline, image counts, per-class rows, confusions and protocol are read from the file")
    void parsesFullReport(@TempDir Path tmp) throws Exception {
        EuroSatTestReport r = EuroSatTestReport.read(write(tmp, REPORT));
        assertTrue(r.isOk());
        assertEquals(0.95, r.getHeadline().get(MetricsReport.Metric.OVERALL_ACCURACY).getAsDouble(), 1e-12);
        assertEquals("0.9400", r.getHeadline().format(MetricsReport.Metric.MACRO_F1));
        assertEquals(100, r.getImages().getAsInt());
        assertEquals(95, r.getCorrect().getAsInt());
        assertEquals(5, r.getWrong().getAsInt());

        List<EuroSatTestReport.ClassRow> rows = r.getPerClass();
        assertEquals(3, rows.size(), "the macro-average line and the confusion matrix are not class rows");
        assertEquals(new EuroSatTestReport.ClassRow(1, "River", 0.9, 0.9333, 0.9164, 30), rows.get(1));
        assertNull(r.getPerClassProblem());

        assertEquals(List.of(new EuroSatTestReport.Confusion("Forest", "River", 2),
                new EuroSatTestReport.Confusion("River", "Forest", 1)), r.getConfusions());
        assertTrue(r.getProtocol().orElseThrow().startsWith("TEST evaluated once"));
        assertEquals("2026-11-19T22:13:15.666683", r.getGenerated().orElseThrow());
    }

    @Test
    @DisplayName("A missing file is MISSING with no values at all")
    void missingFile(@TempDir Path tmp) {
        EuroSatTestReport r = EuroSatTestReport.read(tmp.resolve("eurosat").resolve("test_metrics.txt"));
        assertFalse(r.isOk());
        assertEquals(MetricsReport.Status.MISSING, r.getHeadline().getStatus());
        assertTrue(r.getImages().isEmpty());
        assertTrue(r.getPerClass().isEmpty());
        assertTrue(r.getConfusions().isEmpty());
        assertTrue(r.getProtocol().isEmpty());
    }

    @Test
    @DisplayName("Without a macro F1 line the file is MALFORMED and nothing else is read")
    void malformedHeadline(@TempDir Path tmp) throws Exception {
        EuroSatTestReport r = EuroSatTestReport.read(write(tmp, List.of("Overall accuracy : 0.9",
                "Images : 10 (9 correct, 1 wrong)")));
        assertFalse(r.isOk());
        assertEquals(MetricsReport.Status.MALFORMED, r.getHeadline().getStatus());
        assertTrue(r.getImages().isEmpty());
    }

    @Test
    @DisplayName("Absent or out-of-range per-class rows and an inconsistent count line are dropped, not repaired")
    void partsAreDroppedWhenUnusable(@TempDir Path tmp) throws Exception {
        EuroSatTestReport noTable = EuroSatTestReport.read(write(tmp, List.of(
                "Overall accuracy : 0.9", "Macro F1 : 0.8", "Images : 10 (9 correct, 2 wrong)")));
        assertTrue(noTable.isOk());
        assertTrue(noTable.getPerClass().isEmpty());
        assertEquals("no per-class table in the file", noTable.getPerClassProblem());
        assertTrue(noTable.getImages().isEmpty(), "9 + 2 != 10, so the count line is not shown");
        assertTrue(noTable.getCorrect().isEmpty());

        EuroSatTestReport badRow = EuroSatTestReport.read(write(tmp, List.of(
                "Overall accuracy : 0.9", "Macro F1 : 0.8",
                "Per-class metrics (one-vs-rest from the confusion matrix):",
                "  0  Forest                  0.9800     1.2000     0.9700       50",
                "  1  River                   0.9000     0.9333     0.9164       30")));
        assertTrue(badRow.isOk());
        assertTrue(badRow.getPerClass().isEmpty());
        assertTrue(badRow.getPerClassProblem().contains("out of range"), badRow.getPerClassProblem());
    }

    @Test
    @DisplayName("Committed results/eurosat/test_metrics.txt: 10 classes whose supports add up to the image count")
    void committedReportIsConsistent() {
        Path file = Path.of(AppPaths.DEFAULT_RESULTS_DIR, "eurosat", "test_metrics.txt");
        Assumptions.assumeTrue(Files.isRegularFile(file), "EuroSAT results not present");
        EuroSatTestReport r = EuroSatTestReport.read(file);
        assertTrue(r.isOk(), r.getHeadline().getProblem());
        assertEquals(10, r.getPerClass().size());
        int support = r.getPerClass().stream().mapToInt(EuroSatTestReport.ClassRow::support).sum();
        assertEquals(r.getImages().getAsInt(), support);
        double accuracy = (double) r.getCorrect().getAsInt() / r.getImages().getAsInt();
        assertEquals(r.getHeadline().get(MetricsReport.Metric.OVERALL_ACCURACY).getAsDouble(), accuracy, 5e-5 + 1e-9,
                "the accuracy line agrees with correct / images");
        assertFalse(r.getConfusions().isEmpty());
        assertTrue(r.getProtocol().isPresent());
    }
}
