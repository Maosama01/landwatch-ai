package com.landwatch.fundamentals.math;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

/**
 * SVD demonstration: low-rank approximation of a satellite image tile, with nothing but the JDK.
 *
 * <p>Usage: {@code SvdImageDemo [image] [--out results/fundamentals]}. Without an image argument it
 * uses the first Residential image of the EuroSAT TEST split list ({@code datasets/EuroSAT/splits/test.csv})
 * when the dataset is present, otherwise the bundled synthetic demo image. It writes
 * {@code svd_demo_output.txt} and {@code svd_low_rank.png} into the output folder.</p>
 *
 * <p>Steps: (1) a 4x3 worked example printing U, the singular values and V^T and checking
 * A = U S V^T; (2) the image as a grey-level matrix X (BT.601 luminance in [0, 1]); (3) X = U S V^T;
 * (4) for k = 1, 2, 4, ..., the rank-k approximation X_k = sum_{i<=k} s_i u_i v_i^T, its share of the
 * "energy" sum s_i^2, its relative Frobenius error and PSNR, and how many numbers it needs,
 * k (m + n + 1), against m n for the full image.</p>
 */
public final class SvdImageDemo {

    private static final int SCALE = 3;

    private SvdImageDemo() {
    }

    /** One row of the rank table. */
    public record RankResult(int rank, double energy, double relativeError, double psnr, int numbers) {
    }

    /** Grey-level matrix of an image: BT.601 luminance divided by 255. */
    public static Matrix toGrey(BufferedImage img) {
        double[][] x = new double[img.getHeight()][img.getWidth()];
        for (int y = 0; y < img.getHeight(); y++) {
            for (int c = 0; c < img.getWidth(); c++) {
                int rgb = img.getRGB(c, y);
                double r = (rgb >> 16) & 0xFF;
                double g = (rgb >> 8) & 0xFF;
                double b = rgb & 0xFF;
                x[y][c] = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
            }
        }
        return new Matrix(x);
    }

    /** Rank-k results for the given ranks (ranks above the matrix rank limit are skipped). */
    public static RankResult[] rankTable(Matrix x, SingularValueDecomposition svd, int[] ranks) {
        double[] s = svd.getSingularValues();
        double sumSquares = 0;
        for (double v : s) {
            sumSquares += v * v;
        }
        final double total = sumSquares;
        double norm = x.frobeniusNorm();
        int m = x.getRows();
        int n = x.getCols();
        return java.util.Arrays.stream(ranks).filter(k -> k >= 1 && k <= s.length).mapToObj(k -> {
            double kept = 0;
            for (int i = 0; i < k; i++) {
                kept += s[i] * s[i];
            }
            double err = x.subtract(svd.reconstruct(k)).frobeniusNorm();
            double mse = err * err / (m * (double) n);
            double psnr = mse == 0 ? Double.POSITIVE_INFINITY : 10 * Math.log10(1 / mse);
            return new RankResult(k, kept / total, err / norm, psnr, k * (m + n + 1));
        }).toArray(RankResult[]::new);
    }

    public static void main(String[] args) throws IOException {
        Path out = Path.of("results/fundamentals");
        String imageArg = null;
        for (int i = 0; i < args.length; i++) {
            if ("--out".equals(args[i]) && i + 1 < args.length) {
                out = Path.of(args[++i]);
            } else {
                imageArg = args[i];
            }
        }
        StringBuilder sb = new StringBuilder();
        sb.append("LandWatch AI - SVD demonstration (one-sided Jacobi, plain Java)\n\n");

        // (1) worked example
        Matrix a = new Matrix(new double[][] {{3, 1, 1}, {-1, 3, 1}, {1, 1, 3}, {0, 2, 2}});
        SingularValueDecomposition small = new SingularValueDecomposition(a);
        sb.append("[1] Worked example: A (4 x 3) = U S V^T\n");
        sb.append("A =\n").append(format(a));
        sb.append("U (4 x 3) =\n").append(format(small.getU()));
        sb.append("singular values S = ").append(formatRow(small.getSingularValues())).append('\n');
        sb.append("V^T (3 x 3) =\n").append(format(small.getV().transpose()));
        sb.append(String.format(Locale.ROOT, "max |A - U S V^T| = %.2e; rank-1 approximation keeps %.1f %% of sum s_i^2%n%n",
                maxAbs(a.subtract(small.reconstruct())), 100 * sq(small.getSingularValues()[0]) / sqSum(small.getSingularValues())));

        // (2)-(4) image
        String source;
        BufferedImage img;
        Path image = imageArg != null ? Path.of(imageArg) : defaultEuroSatImage();
        if (image != null) {
            img = ImageIO.read(image.toFile());
            source = image.toString().replace('\\', '/');
        } else {
            try (InputStream in = SvdImageDemo.class.getResourceAsStream("/com/landwatch/sample_data/siteA_post.bmp")) {
                img = ImageIO.read(in);
            }
            img = img.getSubimage(0, 0, Math.min(96, img.getWidth()), Math.min(96, img.getHeight()));
            source = "bundled synthetic demo image siteA_post.bmp, top-left 96x96 crop (NOT satellite imagery)";
        }
        Matrix x = toGrey(img);
        long t0 = System.nanoTime();
        SingularValueDecomposition svd = new SingularValueDecomposition(x);
        double ms = (System.nanoTime() - t0) / 1e6;
        double[] s = svd.getSingularValues();
        sb.append("[2] Image: ").append(source).append('\n');
        sb.append(String.format(Locale.ROOT, "Grey-level matrix X: %d x %d (BT.601 luminance / 255); SVD in %.0f ms, %d Jacobi sweeps%n",
                x.getRows(), x.getCols(), ms, svd.getSweeps()));
        sb.append(String.format(Locale.ROOT, "max |X - U S V^T| = %.2e%n", maxAbs(x.subtract(svd.reconstruct()))));
        sb.append("Largest singular values: ");
        for (int i = 0; i < Math.min(10, s.length); i++) {
            sb.append(String.format(Locale.ROOT, i == 0 ? "%.3f" : ", %.3f", s[i]));
        }
        sb.append("\n\n[3] Rank-k approximations X_k (best possible for each k, Eckart-Young)\n");
        sb.append("   k   energy kept   relative error   PSNR      numbers stored (of ").append(x.getRows() * x.getCols()).append(")\n");
        int[] ranks = {1, 2, 4, 8, 16, 32, 64};
        RankResult[] table = rankTable(x, svd, ranks);
        for (RankResult r : table) {
            String psnr = r.relativeError() < 1e-9 ? "  exact  " : String.format(Locale.ROOT, "%6.2f dB", r.psnr());
            sb.append(String.format(Locale.ROOT, "%4d   %9.2f %%   %12.4f   %s   %6d (%.0f %%)%n", r.rank(), 100 * r.energy(),
                    r.relativeError(), psnr, r.numbers(), 100.0 * r.numbers() / (x.getRows() * x.getCols())));
        }
        sb.append("\nReading: the first singular value holds the overall brightness; a few more add the large\n");
        sb.append("structures (fields, roads); fine texture needs many. Storing k (m + n + 1) numbers saves space\n");
        sb.append("only while k is well below m n / (m + n + 1).\n");

        Files.createDirectories(out);
        Files.writeString(out.resolve("svd_demo_output.txt"), sb.toString(), StandardCharsets.UTF_8);
        writePanel(x, svd, table, out.resolve("svd_low_rank.png"));
        System.out.print(sb);
        System.out.println("Wrote " + out.resolve("svd_demo_output.txt").toString().replace('\\', '/') + " and svd_low_rank.png");
    }

    /** Original followed by rank 1, 4, 16 and 32 approximations, side by side. */
    private static void writePanel(Matrix x, SingularValueDecomposition svd, RankResult[] table, Path file) throws IOException {
        int[] show = {1, 4, 16, 32};
        int m = x.getRows();
        int n = x.getCols();
        int count = 1;
        for (int k : show) {
            if (k <= svd.getSingularValues().length) {
                count++;
            }
        }
        BufferedImage panel = new BufferedImage(count * n * SCALE + (count - 1) * 6, m * SCALE, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = panel.createGraphics();
        g.setColor(java.awt.Color.WHITE);
        g.fillRect(0, 0, panel.getWidth(), panel.getHeight());
        g.dispose();
        int ox = 0;
        paint(panel, x, ox);
        for (int k : show) {
            if (k <= svd.getSingularValues().length) {
                ox += n * SCALE + 6;
                paint(panel, svd.reconstruct(k), ox);
            }
        }
        ImageIO.write(panel, "png", file.toFile());
    }

    private static void paint(BufferedImage dst, Matrix x, int ox) {
        for (int y = 0; y < x.getRows(); y++) {
            for (int c = 0; c < x.getCols(); c++) {
                int v = (int) Math.round(Math.max(0, Math.min(1, x.get(y, c))) * 255);
                int rgb = (v << 16) | (v << 8) | v;
                for (int dy = 0; dy < SCALE; dy++) {
                    for (int dx = 0; dx < SCALE; dx++) {
                        dst.setRGB(ox + c * SCALE + dx, y * SCALE + dy, rgb);
                    }
                }
            }
        }
    }

    /** First Residential image listed in the EuroSAT test split, if the dataset is present. */
    static Path defaultEuroSatImage() throws IOException {
        Path root = Path.of("datasets/EuroSAT");
        Path csv = root.resolve("splits/test.csv");
        if (!Files.isRegularFile(csv)) {
            return null;
        }
        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        for (String line : lines.subList(1, lines.size())) {
            String[] f = line.split(",");
            if (f.length >= 2 && "Residential".equals(f[1].trim())) {
                Path p = root.resolve(f[0].trim());
                return Files.isRegularFile(p) ? p : null;
            }
        }
        return null;
    }

    private static String format(Matrix a) {
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < a.getRows(); r++) {
            sb.append("  ");
            for (int c = 0; c < a.getCols(); c++) {
                sb.append(String.format(Locale.ROOT, "%8.4f", a.get(r, c)));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    private static String formatRow(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(String.format(Locale.ROOT, i == 0 ? "%.4f" : ", %.4f", v[i]));
        }
        return sb.append(']').toString();
    }

    private static double maxAbs(Matrix a) {
        double m = 0;
        for (int r = 0; r < a.getRows(); r++) {
            for (int c = 0; c < a.getCols(); c++) {
                m = Math.max(m, Math.abs(a.get(r, c)));
            }
        }
        return m;
    }

    private static double sq(double v) {
        return v * v;
    }

    private static double sqSum(double[] v) {
        double t = 0;
        for (double x : v) {
            t += x * x;
        }
        return t;
    }
}
