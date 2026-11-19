package com.landwatch.ui.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Parsing of the per-image benchmark CSV used by the Failure Analysis page. */
class PerImageResultsTest {

    private static final String HEADER =
            "image,width,height,true_change_px,pred_change_px,tp,fp,fn,tn,precision,recall,f1,iou,otsu_threshold";

    @Test
    @DisplayName("Rows are parsed by header name; worst-by-F1 skips images without labelled change")
    void parsesAndRanks(@TempDir Path tmp) throws Exception {
        Path csv = Files.writeString(tmp.resolve("baseline_test_per_image.csv"), String.join("\n",
                HEADER,
                "test_1.png,1024,1024,13189,212724,5976,206748,7213,828639,0.028093,0.453105,0.052905,0.027171,45",
                "test_2.png,1024,1024,50321,172915,11103,161812,39218,836443,0.064211,0.220643,0.099473,0.052340,52",
                "test_3.png,1024,1024,0,178009,0,178009,0,870567,0.000000,0.000000,0.000000,0.000000,65",
                "test_4.png,1024,1024,12197,249337,3204,246133,8993,790246,0.012850,0.262688,0.024502,0.012403,56",
                "broken,row",
                ""), StandardCharsets.UTF_8);
        PerImageResults r = PerImageResults.read(csv);
        assertTrue(r.isOk(), r.getProblem());
        assertEquals(4, r.getRows().size());
        assertEquals(1, r.getSkippedRows());
        assertEquals(1, r.imagesWithoutTrueChange());

        List<PerImageResults.Row> worst = r.worstByF1(2);
        assertEquals(List.of("test_4.png", "test_1.png"), worst.stream().map(PerImageResults.Row::image).toList());
        assertEquals(0.024502, worst.get(0).f1(), 1e-12);
        assertEquals(12197, worst.get(0).trueChangePx());
        assertEquals(0.052905, r.medianF1WithChange(), 1e-12);
    }

    @Test
    @DisplayName("Missing file and malformed header are reported, never filled in")
    void missingAndMalformed(@TempDir Path tmp) throws Exception {
        PerImageResults missing = PerImageResults.read(tmp.resolve("nope.csv"));
        assertEquals(MetricsReport.Status.MISSING, missing.getStatus());
        assertTrue(missing.getRows().isEmpty());

        Path bad = Files.writeString(tmp.resolve("bad.csv"), "image,f1\ntest_1.png,0.1\n", StandardCharsets.UTF_8);
        PerImageResults malformed = PerImageResults.read(bad);
        assertEquals(MetricsReport.Status.MALFORMED, malformed.getStatus());
        assertTrue(malformed.getProblem().contains("true_change_px"), malformed.getProblem());

        Path empty = Files.writeString(tmp.resolve("empty.csv"), "", StandardCharsets.UTF_8);
        assertEquals(MetricsReport.Status.MALFORMED, PerImageResults.read(empty).getStatus());
    }
}
