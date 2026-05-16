package com.github.gb.moonshot;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

/**
 * In-memory bank of reference vectors + fraud labels, loaded once at startup from a gzipped JSON dump.
 * <p>
 * Vectors are packed into a single flat {@code float[]} with stride {@value STRIDE} per record the {@value DIMS}
 * real dimensions followed by 2 zero-pad floats. The 16-float stride aligns each vector to one 64-byte L1 cache
 * line: every distance computation issues one line load and the SIMD path runs two AVX2 ops with no tail branch.
 * <p>
 * The pad floats are never written the loader leaves them at the zero-initialized default. Code that mutates
 * {@link #vectors()} must preserve that invariant.
 * <p>
 * Heap cost at the contest's 3M-record dataset: 3M × 16 × 4 B ≈ 192 MB.
 */
public final class Dataset {

    public static final int DIMS = 14;
    public static final int STRIDE = 16;

    private static final int EXPECTED_RECORDS = 3_000_000;
    private static final String FIELD_VECTOR = "vector";
    private static final String FIELD_LABEL = "label";
    private static final String LABEL_FRAUD = "fraud";

    private final float[] vectors;
    private final boolean[] fraudLabels;
    private final int size;

    private Dataset(float[] vectors, boolean[] fraudLabels, int size) {
        this.vectors = vectors;
        this.fraudLabels = fraudLabels;
        this.size = size;
    }

    /**
     * Bench-only: loads the first {@code n} records to compare backends at sub-3M sizes without the full 12 s load.
     */
    public static Dataset loadFirst(Path datasetPath, int n) throws IOException {
        if (n <= 0) throw new IllegalArgumentException("n must be positive, got " + n);
        float[] vectors = new float[n * STRIDE];
        boolean[] labels = new boolean[n];
        int count = 0;

        try (InputStream input = new GZIPInputStream(Files.newInputStream(datasetPath));
             JsonParser parser = new JsonFactory().createParser(input)) {

            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected JSON array at top level of " + datasetPath);
            }
            while (count < n && parser.nextToken() == JsonToken.START_OBJECT) {
                labels[count] = parseRecord(parser, vectors, count * STRIDE);
                count++;
            }
        }

        if (count < n) {
            throw new IOException("requested n=" + n + " but dataset has only " + count + " records");
        }

        return new Dataset(vectors, labels, count);
    }

    public static Dataset load(Path datasetPath) throws IOException {
        // Pad lanes (positions DIMS..STRIDE-1) stay at the zero-initialized default parseRecord never writes them.
        float[] vectors = new float[EXPECTED_RECORDS * STRIDE];
        boolean[] labels = new boolean[EXPECTED_RECORDS];
        int count = 0;

        try (InputStream input = new GZIPInputStream(Files.newInputStream(datasetPath));
             JsonParser parser = new JsonFactory().createParser(input)) {

            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected JSON array at top level of " + datasetPath);
            }

            while (parser.nextToken() == JsonToken.START_OBJECT) {
                if (count == labels.length) {
                    int newCapacity = labels.length + (labels.length >> 1);
                    vectors = Arrays.copyOf(vectors, newCapacity * STRIDE);
                    labels = Arrays.copyOf(labels, newCapacity);
                }
                labels[count] = parseRecord(parser, vectors, count * STRIDE);
                count++;
            }
        }

        warnIfUnexpectedSize(count);
        return buildTrimmed(vectors, labels, count);
    }

    private static boolean parseRecord(JsonParser parser, float[] vectors, int offset) throws IOException {
        boolean isFraud = false;

        while (parser.nextToken() != JsonToken.END_OBJECT) {
            String name = parser.currentName();
            parser.nextToken();
            if (FIELD_VECTOR.equals(name)) {
                parseVectorArray(parser, vectors, offset);
            } else if (FIELD_LABEL.equals(name)) {
                isFraud = LABEL_FRAUD.equals(parser.getText());
            } else {
                parser.skipChildren();
            }
        }

        return isFraud;
    }

    /**
     * Reads the {@value DIMS} floats into {@code vectors[offset..offset+DIMS]}; extra floats are dropped.
     */
    private static void parseVectorArray(JsonParser parser, float[] vectors, int offset) throws IOException {
        int dim = 0;

        while (parser.nextToken() != JsonToken.END_ARRAY) {
            if (dim < DIMS) {
                vectors[offset + dim++] = parser.getFloatValue();
            }
        }
    }

    private static void warnIfUnexpectedSize(int count) {
        if (count != EXPECTED_RECORDS) {
            System.err.println("[dataset] warning: loaded " + count + " records, expected " + EXPECTED_RECORDS);
        }
    }

    private static Dataset buildTrimmed(float[] vectors, boolean[] labels, int count) {
        if (labels.length != count) {
            vectors = Arrays.copyOf(vectors, count * STRIDE);
            labels = Arrays.copyOf(labels, count);
        }

        return new Dataset(vectors, labels, count);
    }

    public float[] vectors() {
        return vectors;
    }

    public boolean[] fraudLabels() {
        return fraudLabels;
    }

    public int size() {
        return size;
    }
}
