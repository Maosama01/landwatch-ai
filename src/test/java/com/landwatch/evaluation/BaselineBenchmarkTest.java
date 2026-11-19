package com.landwatch.evaluation;

import com.landwatch.geospatial.BaselineChangeDetector;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end tests of {@link BaselineBenchmark} and {@link BaselineMinRegionSweep} on a tiny
 * synthetic LEVIR-CD style split (never the real dataset).
 */
class BaselineBenchmarkTest {

    private static final int SIZE = 48;
    private static final int SQUARE_X = 10;
    private static final int SQUARE_Y = 14;
    private static final int SQUARE = 16;

    @TempDir
    Path tmp;

    private static BufferedImage solid(int gray) {
        BufferedImage img = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                img.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
            }
        }
        return img;
    }

    private static BufferedImage withSquare(BufferedImage base, int gray) {
        BufferedImage img = solid(0);
        img.setData(base.getData());
        for (int y = SQUARE_Y; y < SQUARE_Y + SQUARE; y++) {
            for (int x = SQUARE_X; x < SQUARE_X + SQUARE; x++) {
                img.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
            }
        }
        return img;
    }

    private static BufferedImage label(boolean square, int value) {
        BufferedImage img = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_BYTE_GRAY);
        if (square) {
            WritableRaster r = img.getRaster();
            for (int y = SQUARE_Y; y < SQUARE_Y + SQUARE; y++) {
                for (int x = SQUARE_X; x < SQUARE_X + SQUARE; x++) {
                    r.setSample(x, y, 0, value);
                }
            }
        }
        return img;
    }

    private static void write(BufferedImage img, Path file) throws IOException {
        Files.createDirectories(file.getParent());
        assertTrue(ImageIO.write(img, "png", file.toFile()));
    }

    /**
     * Synthetic split "tiny": two pairs with a bright 16x16 new structure (labelled 255 and 156)
     * and one pair with no change at all. Plus an AppleDouble file that must be ignored.
     */
    private Path createTinySplit() throws IOException {
        Path dataset = tmp.resolve("LEVIR-CD");
        Path split = dataset.resolve("tiny");
        BufferedImage pre = solid(90);
        BufferedImage post = withSquare(pre, 220);

        write(pre, split.resolve("A").resolve("tiny_1.png"));
        write(post, split.resolve("B").resolve("tiny_1.png"));
        write(label(true, 255), split.resolve("label").resolve("tiny_1.png"));

        write(pre, split.resolve("A").resolve("tiny_2.png"));
        write(pre, split.resolve("B").resolve("tiny_2.png"));
        write(label(false, 0), split.resolve("label").resolve("tiny_2.png"));

        write(pre, split.resolve("A").resolve("tiny_10.png"));
        write(post, split.resolve("B").resolve("tiny_10.png"));
        write(label(true, 156), split.resolve("label").resolve("tiny_10.png"));

        Files.write(split.resolve("label").resolve("._tiny_1.png"), new byte[]{0, 5, 22, 7});
        return dataset;
    }

    @Test
    @DisplayName("End-to-end benchmark on a tiny synthetic split produces exact metrics and both output files")
    void testEndToEndOnSyntheticSplit() throws IOException {
        Path dataset = createTinySplit();

        BaselineBenchmark.Result result = BaselineBenchmark.evaluate("tiny", dataset, null);

        assertEquals(3, result.images().size());
        assertEquals(List.of("tiny_1.png", "tiny_2.png", "tiny_10.png"),
                result.images().stream().map(BaselineBenchmark.ImageResult::name).toList());

        // The new structure is detected exactly; the unchanged pair yields an empty mask.
        BinaryChangeMetrics g = result.global();
        long square = (long) SQUARE * SQUARE;
        assertEquals(2 * square, g.getTruePositives());
        assertEquals(0, g.getFalsePositives());
        assertEquals(0, g.getFalseNegatives());
        assertEquals(3L * SIZE * SIZE, g.getTotal());
        assertEquals(1.0, g.f1(), 1e-12);
        assertEquals(1.0, g.iou(), 1e-12);

        BaselineBenchmark.ImageResult unchanged = result.images().get(1);
        assertEquals(0, unchanged.trueChangePixels());
        assertEquals(0, unchanged.predictedChangePixels());
        assertEquals(square, result.images().get(2).trueChangePixels(), "Label value 156 must count as change");
        for (BaselineBenchmark.ImageResult r : result.images()) {
            assertTrue(r.analyzeMs() >= 0 && r.loadMs() >= 0);
        }

        Path out = tmp.resolve("results").resolve("benchmarks");
        List<Path> written = BaselineBenchmark.writeOutputs(result, out);
        Path txt = out.resolve("baseline_tiny_metrics.txt");
        Path csv = out.resolve("baseline_tiny_per_image.csv");
        assertEquals(List.of(txt, csv), written);

        String report = Files.readString(txt, StandardCharsets.UTF_8);
        assertTrue(report.contains(g.toReport()), "Report must contain the global BinaryChangeMetrics report");
        assertTrue(report.contains("Split          : tiny"));
        assertTrue(report.contains("Images         : 3 image pairs"));
        assertTrue(report.contains("minRegionSize = " + BaselineChangeDetector.DEFAULT_MIN_REGION_SIZE));
        assertTrue(report.contains("os.arch=" + System.getProperty("os.arch")));
        assertTrue(report.contains("Limitations"));

        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        assertEquals(4, lines.size());
        assertEquals(BaselineBenchmark.CSV_HEADER, lines.get(0));
        assertTrue(lines.get(1).startsWith("tiny_1.png,48,48,256,256,256,0,0,2048,1.000000,1.000000,1.000000,1.000000,"),
                lines.get(1));
        assertTrue(lines.get(2).startsWith("tiny_2.png,48,48,0,0,0,0,0,2304,0.000000,0.000000,0.000000,0.000000,0,"),
                lines.get(2));
        assertTrue(lines.get(3).startsWith("tiny_10.png,"), lines.get(3));
    }

    @Test
    @DisplayName("The CLI runs end to end and reports a missing dataset with a non-zero exit code")
    void testCli() throws IOException {
        Path dataset = createTinySplit();
        Path out = tmp.resolve("cli-out");
        ByteArrayOutputStream stdout = new ByteArrayOutputStream();
        ByteArrayOutputStream stderr = new ByteArrayOutputStream();

        int ok = BaselineBenchmark.runCli(new String[]{"tiny", dataset.toString(), out.toString()},
                new PrintStream(stdout, true, StandardCharsets.UTF_8), new PrintStream(stderr, true, StandardCharsets.UTF_8));
        assertEquals(0, ok, stderr.toString(StandardCharsets.UTF_8));
        assertTrue(Files.exists(out.resolve("baseline_tiny_metrics.txt")));
        assertTrue(Files.exists(out.resolve("baseline_tiny_per_image.csv")));

        ByteArrayOutputStream err2 = new ByteArrayOutputStream();
        int fail = BaselineBenchmark.runCli(new String[]{"tiny", tmp.resolve("nope").toString(), out.toString()},
                new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8),
                new PrintStream(err2, true, StandardCharsets.UTF_8));
        assertEquals(1, fail);
        assertTrue(err2.toString(StandardCharsets.UTF_8).contains("Split folder not found"));
    }

    @Test
    @DisplayName("Latency statistics use the true median for odd and even counts")
    void testStats() {
        BaselineBenchmark.Stats odd = BaselineBenchmark.Stats.of(new double[]{5, 1, 3});
        assertEquals(3.0, odd.mean(), 1e-12);
        assertEquals(3.0, odd.median(), 1e-12);
        assertEquals(1.0, odd.min(), 1e-12);
        assertEquals(5.0, odd.max(), 1e-12);
        assertEquals(2.5, BaselineBenchmark.Stats.of(new double[]{4, 1, 2, 3}).median(), 1e-12);
    }

    @Test
    @DisplayName("The minRegionSize sweep refuses the test split and runs on a synthetic val split")
    void testSweepIsValOnly() throws IOException {
        Path dataset = createTinySplit();
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> BaselineMinRegionSweep.evaluate("test", dataset, new int[]{12}, null));
        assertTrue(ex.getMessage().contains("test split"), ex.getMessage());

        BaselineMinRegionSweep.Result sweep = BaselineMinRegionSweep.evaluate("tiny", dataset, new int[]{12, 300}, null);
        assertEquals(2, sweep.variants().size());
        // 16x16 = 256 px squares survive minRegionSize 12 but are removed by 300.
        assertEquals(512, sweep.variants().get(0).metrics().getTruePositives());
        assertTrue(sweep.variants().get(0).isDefault());
        assertEquals(0, sweep.variants().get(1).metrics().getTruePositives());
        assertEquals(512, sweep.variants().get(1).metrics().getFalseNegatives());

        List<Path> written = BaselineMinRegionSweep.writeOutputs(sweep, tmp.resolve("sweep"));
        String report = Files.readString(written.get(0), StandardCharsets.UTF_8);
        assertTrue(report.contains("NOT used to"));
        assertTrue(report.contains("12 (default)"));
    }
}
