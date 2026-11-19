package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

/**
 * Compares training speed and numerical behaviour of the CPU and the Apple MPS GPU.
 *
 * <p>For each device it trains a fresh FC-Siam-diff on one fixed synthetic batch (a square that is
 * present in B but not in A must be detected), reports milliseconds per training step and whether
 * the loss falls and stays finite. It then checks that a CPU-initialised model gives the same
 * inference output when loaded on MPS.</p>
 *
 * <p>Usage: {@code DeviceBenchmark [batchSize] [steps] [patchSize]} (defaults 8, 12, 256).</p>
 */
public final class DeviceBenchmark {

    private DeviceBenchmark() {
    }

    public static void main(String[] args) throws Exception {
        int batch = args.length > 0 ? Integer.parseInt(args[0]) : 8;
        int steps = args.length > 1 ? Integer.parseInt(args[1]) : 12;
        int size = args.length > 2 ? Integer.parseInt(args[2]) : 256;
        System.out.println(DjlRuntime.describe());
        System.out.printf("batch=%d steps=%d patch=%dx%d%n", batch, steps, size, size);
        boolean mps = DjlRuntime.isMpsUsable();
        System.out.println("MPS usable: " + mps);

        SyntheticBatch data = SyntheticBatch.create(batch, size, 42L);
        benchmark(Device.cpu(), data, steps);
        if (mps) {
            benchmark(DjlRuntime.mpsDevice(), data, steps);
            crossDeviceCheck(data, size);
        }
    }

    private static void benchmark(Device device, SyntheticBatch data, int steps) {
        try (Model model = TrainingSupport.newModel(FcSiamDiff.DEFAULT_WIDTHS, device);
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(4.5f, 1f), 1e-3f,
                     data.batch, data.size)) {
            NDManager m = trainer.getManager();
            NDList inputs = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            NDList labels = new NDList(m.create(data.y, data.labelShape()));
            long totalNs = 0;
            float first = Float.NaN;
            float last = Float.NaN;
            StringBuilder trace = new StringBuilder();
            for (int i = 0; i < steps; i++) {
                long t0 = System.nanoTime();
                float loss = TrainingSupport.trainStep(trainer, inputs, labels);
                long dt = System.nanoTime() - t0;
                if (i >= 2) {
                    totalNs += dt; // skip warm-up steps
                }
                if (i == 0) {
                    first = loss;
                }
                last = loss;
                trace.append(String.format("%.4f ", loss));
            }
            NDList out = trainer.evaluate(inputs);
            float[] logits = TrainingSupport.toCpuFloats(out.singletonOrThrow());
            boolean finite = true;
            for (float v : logits) {
                if (!Float.isFinite(v)) {
                    finite = false;
                    break;
                }
            }
            System.out.printf("[%s] ms/step (excluding 2 warm-up) = %.1f | loss first=%.4f last=%.4f | outputs finite=%s%n",
                    device, totalNs / 1e6 / Math.max(1, steps - 2), first, last, finite);
            System.out.println("[" + device + "] loss trace: " + trace);
        }
    }

    private static void crossDeviceCheck(SyntheticBatch data, int size) throws Exception {
        Path dir = Files.createTempDirectory("fcsiam-xdev");
        float[] cpuOut;
        try (Model model = TrainingSupport.newModel(FcSiamDiff.DEFAULT_WIDTHS, Device.cpu());
             Trainer trainer = TrainingSupport.newTrainer(model, new BceDiceLoss(4.5f, 1f), 1e-3f, data.batch, size)) {
            NDManager m = trainer.getManager();
            NDList inputs = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            NDList labels = new NDList(m.create(data.y, data.labelShape()));
            for (int i = 0; i < 3; i++) {
                TrainingSupport.trainStep(trainer, inputs, labels); // non-trivial BatchNorm statistics
            }
            model.save(dir, TrainingSupport.MODEL_NAME);
            cpuOut = TrainingSupport.toCpuFloats(trainer.evaluate(inputs).singletonOrThrow());
        }
        Device mps = DjlRuntime.mpsDevice();
        try (Model model = TrainingSupport.newModel(FcSiamDiff.DEFAULT_WIDTHS, mps)) {
            model.load(dir, TrainingSupport.MODEL_NAME);
            NDManager m = model.getNDManager();
            NDList inputs = new NDList(m.create(data.a, data.inputShape()), m.create(data.b, data.inputShape()));
            NDArray out = model.getBlock().forward(new ParameterStore(m, false), inputs, false).singletonOrThrow();
            float[] mpsOut = TrainingSupport.toCpuFloats(out);
            double maxAbs = 0;
            for (int i = 0; i < cpuOut.length; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(cpuOut[i] - mpsOut[i]));
            }
            System.out.printf("Cross-device check (CPU-trained weights, inference on CPU vs MPS): max |diff| of logits = %.6g%n", maxAbs);
        }
    }

    /** A fixed batch where B equals A except for one bright square per sample (the "change"). */
    static final class SyntheticBatch {
        final int batch;
        final int size;
        final float[] a;
        final float[] b;
        final float[] y;

        private SyntheticBatch(int batch, int size) {
            this.batch = batch;
            this.size = size;
            this.a = new float[batch * 3 * size * size];
            this.b = new float[batch * 3 * size * size];
            this.y = new float[batch * size * size];
        }

        Shape inputShape() {
            return new Shape(batch, 3, size, size);
        }

        Shape labelShape() {
            return new Shape(batch, 1, size, size);
        }

        static SyntheticBatch create(int batch, int size, long seed) {
            Random rnd = new Random(seed);
            SyntheticBatch s = new SyntheticBatch(batch, size);
            int plane = size * size;
            for (int i = 0; i < s.a.length; i++) {
                s.a[i] = (float) rnd.nextGaussian() * 0.5f;
            }
            System.arraycopy(s.a, 0, s.b, 0, s.a.length);
            for (int n = 0; n < batch; n++) {
                int side = size / 8 + rnd.nextInt(size / 4);
                int y0 = rnd.nextInt(size - side);
                int x0 = rnd.nextInt(size - side);
                for (int yy = y0; yy < y0 + side; yy++) {
                    for (int xx = x0; xx < x0 + side; xx++) {
                        s.y[n * plane + yy * size + xx] = 1f;
                        for (int c = 0; c < 3; c++) {
                            s.b[(n * 3 + c) * plane + yy * size + xx] = 1.5f;
                        }
                    }
                }
            }
            return s;
        }
    }
}
