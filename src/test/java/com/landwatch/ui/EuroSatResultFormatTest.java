package com.landwatch.ui;

import com.landwatch.ui.EuroSatResultFormat.ProbabilityRow;
import com.landwatch.ui.data.EuroSatTestSet.DatasetMatch;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** Ordering and wording on the Land use page. */
class EuroSatResultFormatTest {

    private static final List<String> NAMES = List.of("AnnualCrop", "Forest", "River", "SeaLake");

    @Test
    @DisplayName("All classes are listed, most likely first; the arg-max is the prediction, the true class is flagged")
    void sortedRows() {
        float[] p = {0.10f, 0.60f, 0.05f, 0.25f};
        List<ProbabilityRow> rows = EuroSatResultFormat.sortedRows(p, NAMES, 3);
        assertEquals(List.of("Forest", "SeaLake", "AnnualCrop", "River"),
                rows.stream().map(ProbabilityRow::className).toList());
        assertTrue(rows.get(0).predicted());
        assertEquals(1, rows.stream().filter(ProbabilityRow::predicted).count());
        assertTrue(rows.get(1).trueClass());
        assertEquals(1, rows.stream().filter(ProbabilityRow::trueClass).count());
        assertEquals(0.60f, rows.get(0).probability(), 1e-7);

        List<ProbabilityRow> unknown = EuroSatResultFormat.sortedRows(p, NAMES, -1);
        assertTrue(unknown.stream().noneMatch(ProbabilityRow::trueClass));

        // ties keep class order, and the first of the tied maxima is the prediction (as arg-max does)
        List<ProbabilityRow> ties = EuroSatResultFormat.sortedRows(new float[] {0.25f, 0.25f, 0.25f, 0.25f}, NAMES, -1);
        assertEquals(List.of(0, 1, 2, 3), ties.stream().map(ProbabilityRow::classIndex).toList());
        assertTrue(ties.get(0).predicted());

        assertThrows(IllegalArgumentException.class,
                () -> EuroSatResultFormat.sortedRows(new float[] {1f}, NAMES, -1));
    }

    @Test
    @DisplayName("Percentages never show a non-zero value as 0 % or a value below 1 as 100 %")
    void percent() {
        assertEquals("0.00 %", EuroSatResultFormat.percent(0));
        assertEquals("100.00 %", EuroSatResultFormat.percent(1));
        assertEquals("50.00 %", EuroSatResultFormat.percent(0.5));
        assertEquals("12.35 %", EuroSatResultFormat.percent(0.123456));
        assertEquals("<0.01 %", EuroSatResultFormat.percent(0.00001));
        assertEquals("0.01 %", EuroSatResultFormat.percent(0.0001));
        assertEquals(">99.99 %", EuroSatResultFormat.percent(0.99999));
        assertEquals("99.99 %", EuroSatResultFormat.percent(0.9999));
        assertEquals("n/a", EuroSatResultFormat.percent(Double.NaN));
    }

    @Test
    @DisplayName("Bar widths are the probability, clamped to 0..1")
    void barFraction() {
        assertEquals(0.42, EuroSatResultFormat.barFraction(0.42), 1e-12);
        assertEquals(0, EuroSatResultFormat.barFraction(-0.1));
        assertEquals(0, EuroSatResultFormat.barFraction(Double.NaN));
        assertEquals(1, EuroSatResultFormat.barFraction(1.5));
    }

    @Test
    @DisplayName("Verdicts and source sentences say where the label comes from and warn about train / val images")
    void verdictAndSource() {
        assertEquals("Correct: the true class is Forest", EuroSatResultFormat.verdict("Forest", "Forest"));
        assertEquals("Wrong: the true class is River", EuroSatResultFormat.verdict("Forest", "River"));
        assertEquals("", EuroSatResultFormat.verdict("Forest", null));

        String test = EuroSatResultFormat.sourceSentence(new DatasetMatch("Forest", 1, "test"));
        assertTrue(test.startsWith("EuroSAT test split · true label Forest"), test);
        assertTrue(test.contains("never saw this image"), test);
        String train = EuroSatResultFormat.sourceSentence(new DatasetMatch("River", 2, "train"));
        assertTrue(train.contains("train split") && train.contains("trained on this image"), train);
        String val = EuroSatResultFormat.sourceSentence(new DatasetMatch("River", 2, "val"));
        assertTrue(val.contains("validation split") && val.contains("not an unseen test image"), val);
        String folder = EuroSatResultFormat.sourceSentence(new DatasetMatch("SeaLake", 3, null));
        assertTrue(folder.contains("folder name") && folder.contains("do not list"), folder);
        String own = EuroSatResultFormat.sourceSentence(null);
        assertTrue(own.contains("no true label is known"), own);
    }

    @Test
    @DisplayName("LEVIR-CD pair images are refused with the reason; other files are not")
    void levirCdImagesAreNotClassified(@TempDir Path tmp) throws Exception {
        Path a = Files.createDirectories(tmp.resolve("LEVIR-CD/test/A")).resolve("test_7.png");
        Path b = Files.createDirectories(tmp.resolve("LEVIR-CD/test/B")).resolve("test_7.png");
        Files.write(a, new byte[] {1});
        Files.write(b, new byte[] {1});
        String reason = EuroSatResultFormat.levirCdRefusal(a).orElseThrow();
        assertTrue(reason.startsWith("Not classified: this is a LEVIR-CD image (LEVIR-CD test/test_7.png)"), reason);
        assertTrue(reason.contains("0.5 m per pixel") && reason.contains("Change detection page"), reason);
        assertTrue(EuroSatResultFormat.levirCdRefusal(b).isPresent());

        Path tile = Files.createDirectories(tmp.resolve("EuroSAT_RGB/Forest")).resolve("Forest_1.jpg");
        Files.write(tile, new byte[] {1});
        assertTrue(EuroSatResultFormat.levirCdRefusal(tile).isEmpty());
        // an A/ folder without its B/ counterpart is not a LEVIR-CD pair
        Path lonely = Files.createDirectories(tmp.resolve("other/A")).resolve("x.png");
        Files.write(lonely, new byte[] {1});
        assertTrue(EuroSatResultFormat.levirCdRefusal(lonely).isEmpty());
    }

    @Test
    @DisplayName("Gallery summary, sizes, latency and device names")
    void smallTexts() {
        String s = EuroSatResultFormat.sampleSummary(11, 12, 0, "Random sample.");
        assertTrue(s.startsWith("11 of 12 correct in this sample (classified live just now). Random sample."), s);
        assertFalse(s.contains("could not be read"));
        assertTrue(EuroSatResultFormat.sampleSummary(3, 4, 2, "x").contains("2 listed image(s) could not be read"));
        assertEquals("64×64 px", EuroSatResultFormat.sizeText(64, 64));
        assertEquals("12 ms on CPU", EuroSatResultFormat.latencyText(12, "cpu()"));
        assertEquals("Apple GPU (MPS)", EuroSatResultFormat.deviceText("mps(0)"));
        assertEquals("GPU", EuroSatResultFormat.deviceText("gpu(0)"));
        assertEquals("?", EuroSatResultFormat.deviceText(null));
        assertEquals("tpu", EuroSatResultFormat.deviceText("tpu"));
    }
}
