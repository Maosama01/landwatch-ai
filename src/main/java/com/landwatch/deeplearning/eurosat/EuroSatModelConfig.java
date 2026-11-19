package com.landwatch.deeplearning.eurosat;

import com.landwatch.deeplearning.Normalization;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * The {@code model.properties} file stored next to the EuroSAT classifier weights: everything needed
 * to rebuild and use the model (architecture, widths, class names in index order, normalisation)
 * and a record of how it was trained and how well it did (hyper-parameters, device, versions, split
 * sizes, validation and test metrics). A plain Java properties file, readable in any text editor.
 */
public final class EuroSatModelConfig {

    public static final String FILE_NAME = "model.properties";

    public static final String ARCHITECTURE = "architecture";
    public static final String WIDTHS = "widths";
    public static final String BLOCKS_PER_STAGE = "blocks_per_stage";
    public static final String NUM_CLASSES = "num_classes";
    public static final String CLASS_NAMES = "class_names";
    public static final String INPUT_SIZE = "input_size";
    public static final String NORM_MEAN = "normalization.mean";
    public static final String NORM_STD = "normalization.std";
    public static final String PARAMS_NAME = "params_name";

    private final Map<String, String> values = new TreeMap<>();

    public EuroSatModelConfig set(String key, Object value) {
        values.put(key, String.valueOf(value));
        return this;
    }

    /** Sets a number formatted with 4 decimals (Locale.ROOT). */
    public EuroSatModelConfig setMetric(String key, double value) {
        return set(key, String.format(Locale.ROOT, "%.4f", value));
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
        String[] parts = get(WIDTHS).split(",");
        int[] out = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Integer.parseInt(parts[i].trim());
        }
        return out;
    }

    public EuroSatModelConfig setWidths(int[] widths) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < widths.length; i++) {
            sb.append(i == 0 ? "" : ",").append(widths[i]);
        }
        return set(WIDTHS, sb.toString());
    }

    public int getBlocksPerStage() {
        return Integer.parseInt(get(BLOCKS_PER_STAGE));
    }

    public int getInputSize() {
        return Integer.parseInt(get(INPUT_SIZE));
    }

    /** Class names in class-index order. */
    public List<String> getClassNames() {
        List<String> names = new ArrayList<>();
        for (String s : get(CLASS_NAMES).split(",")) {
            names.add(s.trim());
        }
        int declared = Integer.parseInt(get(NUM_CLASSES));
        if (declared != names.size()) {
            throw new IllegalStateException(NUM_CLASSES + "=" + declared + " but " + names.size() + " class names");
        }
        return List.copyOf(names);
    }

    public EuroSatModelConfig setClassNames(List<String> names) {
        set(NUM_CLASSES, names.size());
        return set(CLASS_NAMES, String.join(",", names));
    }

    public Normalization getNormalization() {
        return new Normalization(parseFloats(get(NORM_MEAN)), parseFloats(get(NORM_STD)));
    }

    public EuroSatModelConfig setNormalization(Normalization n) {
        set(NORM_MEAN, join(n.getMean()));
        return set(NORM_STD, join(n.getStd()));
    }

    /** Writes {@code dir/model.properties} with keys in sorted order. */
    public void save(Path dir) throws IOException {
        Files.createDirectories(dir);
        Properties p = new Properties() {
            @Override
            public synchronized Set<Map.Entry<Object, Object>> entrySet() {
                Set<Map.Entry<Object, Object>> sorted = new TreeSet<>(Comparator.comparing(e -> e.getKey().toString()));
                sorted.addAll(super.entrySet());
                return Collections.synchronizedSet(sorted);
            }
        };
        p.putAll(values);
        try (Writer w = Files.newBufferedWriter(dir.resolve(FILE_NAME), StandardCharsets.UTF_8)) {
            p.store(w, "LandWatch AI - EuroSAT land-use/land-cover ResNet classifier (trained with DJL)");
        }
    }

    /** Reads {@code dir/model.properties}. */
    public static EuroSatModelConfig load(Path dir) throws IOException {
        Path file = dir.resolve(FILE_NAME);
        if (!Files.isRegularFile(file)) {
            throw new IOException("No " + FILE_NAME + " in " + dir);
        }
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        }
        EuroSatModelConfig c = new EuroSatModelConfig();
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

    private static float[] parseFloats(String s) {
        String[] parts = s.split(",");
        float[] out = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            out[i] = Float.parseFloat(parts[i].trim());
        }
        return out;
    }
}
