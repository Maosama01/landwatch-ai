package com.landwatch.deeplearning;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;

/**
 * The {@code model.properties} file stored next to the trained weights. It records everything
 * needed to rebuild and use the model (architecture, widths, normalisation constants, decision
 * threshold) plus how it was trained (hyper-parameters, device, versions, data sizes, timings).
 *
 * <p>It is a plain Java properties file so it can be read in any text editor.</p>
 */
public final class ModelConfig {

    /** File name inside the model directory. */
    public static final String FILE_NAME = "model.properties";

    public static final String ARCHITECTURE = "architecture";
    public static final String WIDTHS = "widths";
    public static final String PATCH_SIZE = "patch_size";
    public static final String NORM_MEAN = "normalization.mean";
    public static final String NORM_STD = "normalization.std";
    public static final String THRESHOLD = "decision_threshold";
    public static final String PARAMS_NAME = "params_name";

    private final Map<String, String> values = new TreeMap<>();

    public ModelConfig set(String key, Object value) {
        values.put(key, String.valueOf(value));
        return this;
    }

    public String get(String key) {
        String v = values.get(key);
        if (v == null) {
            throw new IllegalStateException("Missing key '" + key + "' in " + FILE_NAME);
        }
        return v;
    }

    public String get(String key, String defaultValue) {
        return values.getOrDefault(key, defaultValue);
    }

    public Map<String, String> asMap() {
        return new TreeMap<>(values);
    }

    public int[] getWidths() {
        return parseInts(get(WIDTHS));
    }

    public int getPatchSize() {
        return Integer.parseInt(get(PATCH_SIZE));
    }

    public float getThreshold() {
        return Float.parseFloat(get(THRESHOLD));
    }

    public Normalization getNormalization() {
        return new Normalization(parseFloats(get(NORM_MEAN)), parseFloats(get(NORM_STD)));
    }

    public ModelConfig setNormalization(Normalization n) {
        set(NORM_MEAN, join(n.getMean()));
        set(NORM_STD, join(n.getStd()));
        return this;
    }

    public ModelConfig setWidths(int[] widths) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < widths.length; i++) {
            sb.append(i == 0 ? "" : ",").append(widths[i]);
        }
        return set(WIDTHS, sb.toString());
    }

    /** Writes the config to {@code dir/model.properties}. */
    public void save(Path dir) throws IOException {
        Files.createDirectories(dir);
        Properties p = new Properties() {
            @Override
            public synchronized java.util.Set<Map.Entry<Object, Object>> entrySet() {
                // write keys in sorted order for readability
                java.util.Set<Map.Entry<Object, Object>> sorted = new java.util.TreeSet<>(
                        java.util.Comparator.comparing(e -> e.getKey().toString()));
                sorted.addAll(super.entrySet());
                return java.util.Collections.synchronizedSet(sorted);
            }
        };
        p.putAll(values);
        try (Writer w = Files.newBufferedWriter(dir.resolve(FILE_NAME), StandardCharsets.UTF_8)) {
            p.store(w, "LandWatch AI - FC-Siam-diff change detector (trained with DJL)");
        }
    }

    /** Reads {@code dir/model.properties}. */
    public static ModelConfig load(Path dir) throws IOException {
        Path file = dir.resolve(FILE_NAME);
        if (!Files.isRegularFile(file)) {
            throw new IOException("No " + FILE_NAME + " in " + dir);
        }
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        }
        ModelConfig c = new ModelConfig();
        for (String k : p.stringPropertyNames()) {
            c.values.put(k, p.getProperty(k));
        }
        return c;
    }

    private static String join(float[] v) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < v.length; i++) {
            sb.append(i == 0 ? "" : ",").append(v[i]);
        }
        return sb.toString();
    }

    private static int[] parseInts(String s) {
        String[] parts = s.split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Integer.parseInt(parts[i].trim());
        }
        return out;
    }

    private static float[] parseFloats(String s) {
        String[] parts = s.split(",");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Float.parseFloat(parts[i].trim());
        }
        return out;
    }
}
