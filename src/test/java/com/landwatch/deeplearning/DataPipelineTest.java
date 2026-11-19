package com.landwatch.deeplearning;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Tests for the data side: CSV index, image conversion, normalisation and augmentation. */
class DataPipelineTest {

    @TempDir
    Path tempDir;

    @Test
    void readsMetadataCsv() throws Exception {
        Path meta = Files.createDirectories(tempDir.resolve("metadata"));
        Files.writeString(meta.resolve("val_patches.csv"),
                "patch_filename,relative_path_A,relative_path_B,relative_path_label,original_filename,split,patch_idx,row,col,"
                        + "x1,y1,x2,y2,patch_width,patch_height,has_change,change_pixel_count,change_ratio,mask_values\n"
                        + "val_1_patch_00.png,val/A/val_1_patch_00.png,val/B/val_1_patch_00.png,val/label/val_1_patch_00.png,"
                        + "val_1.png,val,0,0,0,0,0,256,256,256,256,1,1234,0.018829,0;255\n",
                StandardCharsets.UTF_8);
        List<LevirCdPatch> patches = LevirCdPatch.readSplit(tempDir, "val");
        assertEquals(1, patches.size());
        LevirCdPatch p = patches.get(0);
        assertEquals("val_1_patch_00.png", p.name());
        assertEquals(tempDir.resolve("val/B/val_1_patch_00.png"), p.imageB());
        assertEquals(1234, p.changePixelCount());
        assertEquals(65536, p.pixelCount());
        assertThrows(IllegalArgumentException.class, () -> LevirCdPatch.readSplit(tempDir, "holdout"));
    }

    @Test
    void realSplitsHaveTheDocumentedSizesAndDoNotOverlap() {
        Path root = LevirCdPatch.locateDatasetRoot();
        assumeTrue(root != null, "LEVIR-CD processed_256 not found");
        List<LevirCdPatch> train = LevirCdPatch.readSplit(root, "train");
        List<LevirCdPatch> val = LevirCdPatch.readSplit(root, "val");
        List<LevirCdPatch> test = LevirCdPatch.readSplit(root, "test");
        assertEquals(7120, train.size());
        assertEquals(1024, val.size());
        assertEquals(2048, test.size());
        Set<String> names = new HashSet<>();
        for (List<LevirCdPatch> split : List.of(train, val, test)) {
            for (LevirCdPatch p : split) {
                assertTrue(names.add(p.name()), "patch listed twice: " + p.name());
            }
        }
        assertTrue(Files.isRegularFile(train.get(0).imageA()));
    }

    @Test
    void imageToChwKeepsChannelOrder() {
        BufferedImage img = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, 0xFF0000);
        img.setRGB(1, 0, 0x00FF80);
        float[] chw = ImageTensors.toChw01(img);
        assertArrayEquals(new float[] {1f, 0f, 0f, 1f, 0f, 128 / 255f}, chw, 1e-6f);
    }

    @Test
    void anyNonZeroLabelValueIsChange() {
        BufferedImage label = new BufferedImage(4, 1, BufferedImage.TYPE_BYTE_GRAY);
        int[] values = {0, 156, 254, 255};
        for (int x = 0; x < 4; x++) {
            label.getRaster().setSample(x, 0, 0, values[x]);
        }
        assertArrayEquals(new float[] {0f, 1f, 1f, 1f}, ImageTensors.toBinaryMask(label));
        boolean[][] mask = ImageTensors.toBooleanMask(ImageTensors.toBinaryMask(label), 4, 1);
        assertArrayEquals(new boolean[] {false, true, true, true}, mask[0]);
    }

    @Test
    void normalisationStandardisesEachChannel() {
        Normalization n = new Normalization(new float[] {0.5f, 0.25f, 0f}, new float[] {0.5f, 0.25f, 2f});
        float[] chw = {1f, 0f, 0.5f, 0.25f, 2f, 4f};
        assertArrayEquals(new float[] {1f, -1f, 1f, 0f, 1f, 2f}, n.applyInPlace(chw), 1e-6f);
        assertThrows(IllegalArgumentException.class, () -> new Normalization(new float[] {0, 0, 0}, new float[] {1, 0, 1}));
    }

    @Test
    void d4TransformsArePermutationsAndAllDistinct() {
        int size = 5;
        float[] img = new float[size * size];
        for (int i = 0; i < img.length; i++) {
            img[i] = i;
        }
        Set<String> seen = new HashSet<>();
        for (int id = 0; id < D4Augmentation.COUNT; id++) {
            float[] t = D4Augmentation.apply(img, 1, size, id);
            float[] sorted = t.clone();
            Arrays.sort(sorted);
            assertArrayEquals(img, sorted, 0f, "transform " + id + " must be a permutation");
            seen.add(Arrays.toString(t));
        }
        assertEquals(8, seen.size(), "the 8 D4 transforms must all differ");
        assertArrayEquals(img, D4Augmentation.apply(img, 1, size, 0), 0f);
    }

    @Test
    void quarterTurnFourTimesIsIdentityAndFlipTwiceIsIdentity() {
        int size = 4;
        float[] img = new float[3 * size * size];
        for (int i = 0; i < img.length; i++) {
            img[i] = i * 0.5f;
        }
        float[] r = img;
        for (int k = 0; k < 4; k++) {
            r = D4Augmentation.apply(r, 3, size, 1);
        }
        assertArrayEquals(img, r, 0f);
        float[] flipped = D4Augmentation.apply(img, 3, size, 4);
        assertNotEquals(Arrays.toString(img), Arrays.toString(flipped));
        assertArrayEquals(img, D4Augmentation.apply(flipped, 3, size, 4), 0f);
    }

    @Test
    void sameTransformKeepsImageAndLabelAligned() {
        // label marks exactly the pixels where channel 0 of the image is bright
        int size = 6;
        float[] image = new float[3 * size * size];
        float[] label = new float[size * size];
        for (int i : new int[] {1, 8, 9, 22, 35}) {
            image[i] = 1f;
            label[i] = 1f;
        }
        for (int id = 0; id < D4Augmentation.COUNT; id++) {
            float[] ti = D4Augmentation.apply(image, 3, size, id);
            float[] tl = D4Augmentation.apply(label, 1, size, id);
            assertArrayEquals(tl, Arrays.copyOfRange(ti, 0, size * size), 0f, "misaligned for transform " + id);
        }
    }
}
