package com.landwatch.deeplearning.eurosat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end smoke test of the command-line tools on a tiny synthetic EuroSAT-like folder
 * (3 classes x 14 JPEGs of 64 x 64): split -> train (2 epochs, tiny network, CPU) -> evaluate VAL
 * and TEST. Checks that every output file is written in the documented format; the numbers
 * themselves are meaningless here.
 */
class EuroSatPipelineTest {

    @TempDir
    Path root;

    private void writeImages(String className, int channel, int count, long seed) throws IOException {
        Path dir = Files.createDirectories(root.resolve("EuroSAT_RGB").resolve(className));
        Random rnd = new Random(seed);
        for (int i = 1; i <= count; i++) {
            BufferedImage img = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB);
            for (int y = 0; y < 64; y++) {
                for (int x = 0; x < 64; x++) {
                    int[] rgb = new int[3];
                    for (int c = 0; c < 3; c++) {
                        int base = c == channel ? 190 : 60;
                        rgb[c] = Math.max(0, Math.min(255, base + (int) (25 * rnd.nextGaussian())));
                    }
                    img.setRGB(x, y, (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]);
                }
            }
            ImageIO.write(img, "jpg", dir.resolve(className + "_" + i + ".jpg").toFile());
        }
    }

    @Test
    void splitTrainEvaluateWritesAllOutputs() throws Exception {
        writeImages("Forest", 1, 14, 1);
        writeImages("Industrial", 0, 14, 2);
        writeImages("SeaLake", 2, 14, 3);
        Path model = root.resolve("model");
        Path results = root.resolve("results");

        EuroSatSplit.main(new String[] {"--root", root.toString()});
        assertTrue(Files.exists(root.resolve("splits/train.csv")));
        assertTrue(Files.exists(root.resolve("dataset_summary.txt")));

        TrainEuroSatClassifier.main(new String[] {"--data", root.toString(), "--out", model.toString(),
                "--results", results.toString(), "--epochs", "2", "--batch", "8", "--widths", "4,8",
                "--blocks", "1", "--device", "cpu", "--workers", "2", "--lr", "3e-3"});

        List<String> log = Files.readAllLines(results.resolve("training_log.csv"), StandardCharsets.UTF_8);
        assertEquals(3, log.size(), String.join("\n", log));
        assertTrue(log.get(0).startsWith("epoch,learning_rate_end,train_loss,train_accuracy,val_loss,val_accuracy,val_macro_f1"));
        EuroSatModelConfig config = EuroSatModelConfig.load(model);
        assertEquals(List.of("Forest", "Industrial", "SeaLake"), config.getClassNames());
        assertEquals("2", config.get("training.epochs_completed"));
        assertEquals("30", config.get("data.train_images"));
        assertEquals("6", config.get("data.val_images"));
        assertEquals("6", config.get("data.test_images"));
        assertEquals("cpu()", config.get("training.device"));
        int bestEpoch = Integer.parseInt(config.get("training.best_epoch"));
        assertTrue(Files.exists(model.resolve(String.format("eurosat_resnet-%04d.params", bestEpoch))));

        for (String split : List.of("val", "test")) {
            EvaluateEuroSatClassifier.main(new String[] {split, "--data", root.toString(), "--model", model.toString(),
                    "--results", results.toString(), "--device", "cpu"});
            List<String> report = Files.readAllLines(results.resolve(split + "_metrics.txt"), StandardCharsets.UTF_8);
            assertTrue(report.stream().anyMatch(l -> l.matches("Overall accuracy : [01]\\.\\d{4}")), String.join("\n", report));
            assertTrue(report.stream().anyMatch(l -> l.matches("Macro F1         : [01]\\.\\d{4}")), String.join("\n", report));
            List<String> csv = Files.readAllLines(results.resolve("confusion_matrix_" + split + ".csv"));
            assertEquals(4, csv.size());
            assertEquals("actual\\predicted,Forest,Industrial,SeaLake", csv.get(0));
            assertEquals(7, Files.readAllLines(results.resolve(split + "_predictions.csv")).size());
        }
        EuroSatModelConfig evaluated = EuroSatModelConfig.load(model);
        assertEquals("6", evaluated.get("metrics.test.images"));
        assertTrue(evaluated.get("metrics.val.accuracy").matches("[01]\\.\\d{4}"));
        String valReport = Files.readString(results.resolve("val_metrics.txt"));
        assertTrue(valReport.contains("Consistency check"), valReport);
    }
}
