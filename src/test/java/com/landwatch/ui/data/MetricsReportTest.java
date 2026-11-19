package com.landwatch.ui.data;

import com.landwatch.evaluation.BinaryChangeMetrics;
import com.landwatch.ui.data.MetricsReport.Metric;
import com.landwatch.ui.data.MetricsReport.Status;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/** Runtime parsing of the benchmark / evaluation report files shown on the Model Benchmarks page. */
class MetricsReportTest {

    /** Excerpt in the exact layout of results/benchmarks/baseline_test_metrics.txt. */
    private static final String BASELINE = String.join("\n",
            "================================================================================",
            " LANDWATCH AI - Classical baseline change detection on LEVIR-CD",
            "================================================================================",
            "Dataset        : LEVIR-CD (datasets/LEVIR-CD)",
            "Split          : test",
            "Images         : 128 image pairs, full resolution (1024x1024), no tiling, no resizing",
            "Parameters     : minRegionSize = 12 (DEFAULT_MIN_REGION_SIZE, untouched default; not tuned)",
            "",
            "Global metrics (test split)",
            "Confusion matrix (pixels): TP=1695603  FP=25101718  FN=5141801  TN=102278606  (total 134217728)",
            "Change precision : 0.0633",
            "Change recall    : 0.2480",
            "Change F1        : 0.1008",
            "Change IoU       : 0.0531",
            "Macro F1         : 0.4860",
            "Mean IoU         : 0.4124",
            "Cohen's kappa    : 0.0214",
            "Pixel accuracy   : 0.7747  (misleading alone: classes are imbalanced)",
            "",
            "True change pixels      : 6837404 (5.0943% of all pixels)",
            "Images with no true change pixels      : 10 of 128",
            "");

    private static Path write(Path dir, String name, String text) throws Exception {
        return Files.writeString(dir.resolve(name), text, StandardCharsets.UTF_8);
    }

    @Test
    @DisplayName("Good baseline file: every metric, the image count and the change share are parsed")
    void parsesBaselineReport(@TempDir Path tmp) throws Exception {
        MetricsReport r = MetricsReport.readChangeDetection(write(tmp, "baseline_test_metrics.txt", BASELINE));
        assertEquals(Status.OK, r.getStatus());
        assertNull(r.getProblem());
        assertEquals(0.1008, r.get(Metric.F1).getAsDouble(), 1e-12);
        assertEquals(0.0531, r.get(Metric.IOU).getAsDouble(), 1e-12);
        assertEquals(0.0633, r.get(Metric.PRECISION).getAsDouble(), 1e-12);
        assertEquals(0.2480, r.get(Metric.RECALL).getAsDouble(), 1e-12);
        assertEquals(0.0214, r.get(Metric.KAPPA).getAsDouble(), 1e-12);
        assertEquals(0.7747, r.get(Metric.PIXEL_ACCURACY).getAsDouble(), 1e-12);
        assertEquals(0.4860, r.get(Metric.MACRO_F1).getAsDouble(), 1e-12);
        assertEquals("128 image pairs", r.getScope().orElseThrow());
        assertEquals(5.0943, r.getTrueChangePercent().getAsDouble(), 1e-12);
        assertEquals("0.1008", r.format(Metric.F1));
        assertEquals("n/a", r.format(Metric.OVERALL_ACCURACY));
    }

    @Test
    @DisplayName("Deep-model report: patch scope parsed, and the untuned-threshold reference line is not taken as F1")
    void deepReportUsesFirstMetricLines(@TempDir Path tmp) throws Exception {
        BinaryChangeMetrics m = BinaryChangeMetrics.of(700, 150, 210, 9000);
        String text = "LandWatch AI - FC-Siam-diff change detection, evaluation on LEVIR-CD TEST split\n"
                + "Patches: 2048 of 256x256 (datasets/LEVIR-CD/processed_256/metadata/test_patches.csv)\n"
                + "Decision threshold: 0.4 (tuned on VAL during training)\n\n"
                + m.toReport()
                + "\nFor reference only, at the untuned threshold 0.50: precision 0.1111, recall 0.2222, F1 0.3333, IoU 0.4444\n";
        MetricsReport r = MetricsReport.readChangeDetection(write(tmp, "test_metrics.txt", text));
        assertTrue(r.isOk(), r.getProblem());
        assertEquals(m.f1(), r.get(Metric.F1).getAsDouble(), 5e-5);
        assertEquals(m.iou(), r.get(Metric.IOU).getAsDouble(), 5e-5);
        assertEquals(m.precision(), r.get(Metric.PRECISION).getAsDouble(), 5e-5);
        assertEquals(m.recall(), r.get(Metric.RECALL).getAsDouble(), 5e-5);
        assertEquals(m.kappa(), r.get(Metric.KAPPA).getAsDouble(), 5e-5);
        assertEquals("2048 patches of 256x256", r.getScope().orElseThrow());
    }

    @Test
    @DisplayName("Missing file: MISSING and 'not available', never a number")
    void missingFile(@TempDir Path tmp) {
        MetricsReport r = MetricsReport.readChangeDetection(tmp.resolve("deeplearning").resolve("test_metrics.txt"));
        assertEquals(Status.MISSING, r.getStatus());
        assertFalse(r.isOk());
        assertTrue(r.get(Metric.F1).isEmpty());
        assertEquals("not available", r.format(Metric.F1));
        assertTrue(r.getProblem().contains("file not found"));

        assertEquals(Status.MISSING, MetricsReport.readChangeDetection(tmp).getStatus(), "a directory is not a report");
        assertEquals(Status.MISSING, MetricsReport.readChangeDetection(null).getStatus());
    }

    @Test
    @DisplayName("Malformed files: non-numeric value, missing F1 line, out-of-range value, empty file")
    void malformedFiles(@TempDir Path tmp) throws Exception {
        MetricsReport notNumber = MetricsReport.readChangeDetection(write(tmp, "a.txt",
                "Change precision : 0.5\nChange F1        : abc\n"));
        assertEquals(Status.MALFORMED, notNumber.getStatus());
        assertTrue(notNumber.getProblem().contains("not a number"), notNumber.getProblem());
        assertTrue(notNumber.getProblem().contains("line 2"), notNumber.getProblem());
        assertTrue(notNumber.get(Metric.PRECISION).isEmpty(), "no partial values from a malformed file");
        assertEquals("malformed file", notNumber.format(Metric.F1));

        MetricsReport noF1 = MetricsReport.readChangeDetection(write(tmp, "b.txt", "Change IoU : 0.4\n"));
        assertEquals(Status.MALFORMED, noF1.getStatus());
        assertTrue(noF1.getProblem().contains("Change F1"), noF1.getProblem());

        MetricsReport outOfRange = MetricsReport.readChangeDetection(write(tmp, "c.txt", "Change F1 : 1.7\n"));
        assertEquals(Status.MALFORMED, outOfRange.getStatus());
        assertTrue(outOfRange.getProblem().contains("out of range"), outOfRange.getProblem());

        assertEquals(Status.MALFORMED, MetricsReport.readChangeDetection(write(tmp, "d.txt", "")).getStatus());
        assertEquals(Status.MALFORMED,
                MetricsReport.readChangeDetection(write(tmp, "e.txt", "\u0000\u0001 binary junk")).getStatus());
    }

    @Test
    @DisplayName("Negative kappa is valid; other metrics must be in [0, 1]")
    void kappaMayBeNegative(@TempDir Path tmp) throws Exception {
        MetricsReport r = MetricsReport.readChangeDetection(write(tmp, "k.txt",
                "Change F1        : 0.0000\nCohen's kappa    : -0.0123\n"));
        assertTrue(r.isOk(), r.getProblem());
        assertEquals(-0.0123, r.get(Metric.KAPPA).getAsDouble(), 1e-12);
        assertEquals("-0.0123", r.format(Metric.KAPPA));
    }

    @Test
    @DisplayName("EuroSAT report: overall accuracy and macro F1, fractions or percentages")
    void euroSat(@TempDir Path tmp) throws Exception {
        MetricsReport ok = MetricsReport.readEuroSat(write(tmp, "eurosat.txt",
                "EuroSAT test\nOverall accuracy : 0.9512\nMacro F1         : 0.9487\n"));
        assertTrue(ok.isOk(), ok.getProblem());
        assertEquals(0.9512, ok.get(Metric.OVERALL_ACCURACY).getAsDouble(), 1e-12);
        assertEquals(0.9487, ok.get(Metric.MACRO_F1).getAsDouble(), 1e-12);

        MetricsReport percent = MetricsReport.readEuroSat(write(tmp, "eurosat_pct.txt",
                "Overall accuracy : 95.12 %\nMacro F1         : 94.87%\n"));
        assertTrue(percent.isOk(), percent.getProblem());
        assertEquals(0.9512, percent.get(Metric.OVERALL_ACCURACY).getAsDouble(), 1e-12);
        assertEquals(0.9487, percent.get(Metric.MACRO_F1).getAsDouble(), 1e-12);

        MetricsReport noMacro = MetricsReport.readEuroSat(write(tmp, "eurosat_bad.txt", "Overall accuracy : 0.9\n"));
        assertEquals(Status.MALFORMED, noMacro.getStatus());

        assertEquals(Status.MISSING, MetricsReport.readEuroSat(tmp.resolve("eurosat").resolve("test_metrics.txt")).getStatus());
    }
}
