package com.landwatch.deeplearning;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Small helpers around the DJL engine: version strings and device selection.
 *
 * <p>DJL (Deep Java Library) is a framework-neutral Java API. The actual tensor maths is done by a
 * native "engine"; this project uses the PyTorch engine, whose native libtorch library DJL downloads
 * into {@code ~/.djl.ai} the first time it is used.</p>
 *
 * <p>Device names accepted by {@link #resolveDevice(String)}:</p>
 * <ul>
 *   <li>{@code cpu} - always available;</li>
 *   <li>{@code mps} - Apple-silicon GPU through PyTorch's Metal Performance Shaders backend;</li>
 *   <li>{@code gpu} - an NVIDIA CUDA GPU (not available on a Mac);</li>
 *   <li>{@code auto} - MPS if it works on this machine, otherwise CPU.</li>
 * </ul>
 */
public final class DjlRuntime {

    private DjlRuntime() {
    }

    /** The DJL release in use, for example {@code 0.40.0}. */
    public static String djlVersion() {
        return Engine.getDjlVersion();
    }

    /** Engine name, for example {@code PyTorch}. */
    public static String engineName() {
        return Engine.getInstance().getEngineName();
    }

    /** Version of the native engine, for example {@code 2.7.1} for PyTorch. */
    public static String engineVersion() {
        return Engine.getInstance().getVersion();
    }

    /**
     * The Apple-silicon GPU device. PyTorch reports MPS tensors as device index 0, so the index must
     * be 0 (not -1): DJL's parameter store looks devices up by equality.
     */
    public static Device mpsDevice() {
        return Device.of("mps", 0);
    }

    /** One-line summary such as {@code DJL 0.40.0 / PyTorch 2.7.1}. */
    public static String describe() {
        return "DJL " + djlVersion() + " / " + engineName() + " " + engineVersion();
    }

    /**
     * Turns a device name into a DJL {@link Device}.
     *
     * @param name {@code cpu}, {@code mps}, {@code gpu} or {@code auto}
     * @return the device; {@code auto} falls back to CPU when MPS is not usable
     */
    public static Device resolveDevice(String name) {
        String n = name == null ? "auto" : name.trim().toLowerCase();
        switch (n) {
            case "cpu":
                return Device.cpu();
            case "mps":
                return mpsDevice();
            case "gpu":
            case "cuda":
                return Device.gpu();
            case "auto":
                return isMpsUsable() ? mpsDevice() : Device.cpu();
            default:
                throw new IllegalArgumentException("Unknown device '" + name + "' (use cpu, mps, gpu or auto)");
        }
    }

    /**
     * Checks whether the Apple MPS device works by running a tiny computation on it and
     * comparing the answer with the CPU.
     */
    public static boolean isMpsUsable() {
        if (!System.getProperty("os.name", "").toLowerCase().contains("mac")
                || !"aarch64".equals(System.getProperty("os.arch"))) {
            return false;
        }
        try (NDManager manager = NDManager.newBaseManager(mpsDevice())) {
            NDArray a = manager.create(new float[] {1f, 2f, 3f, 4f});
            float[] result = a.mul(a).sum().toDevice(Device.cpu(), true).toFloatArray();
            return result.length == 1 && Math.abs(result[0] - 30f) < 1e-4f;
        } catch (RuntimeException | UnsatisfiedLinkError e) {
            return false;
        }
    }
}
