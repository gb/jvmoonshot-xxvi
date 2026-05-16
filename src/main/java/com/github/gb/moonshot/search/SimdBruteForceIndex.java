package com.github.gb.moonshot.search;

import com.github.gb.moonshot.Dataset;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

/**
 * Brute-force KNN; ground-truth reference for the approximate indexes. Stride-16 layout means
 * each vector is exactly two AVX2 lanes (8 floats each); the 2 zero-pad floats per vector
 * contribute 0 to the squared distance, making the SIMD load tail-free.
 */
public final class SimdBruteForceIndex implements VectorIndex {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;

    private final float[] vectors;
    private final boolean[] fraudLabels;
    private final int n;

    public SimdBruteForceIndex(Dataset dataset) {
        if (Dataset.STRIDE != 16) {
            throw new IllegalStateException("SimdBruteForceIndex assumes STRIDE=16, got " + Dataset.STRIDE);
        }
        if (SPECIES.length() != 8) {
            throw new IllegalStateException("SimdBruteForceIndex assumes 8-lane SPECIES_256");
        }
        this.vectors = dataset.vectors();
        this.fraudLabels = dataset.fraudLabels();
        this.n = dataset.size();
    }

    @Override
    public int size() {
        return n;
    }

    @Override
    public int countFraudsInTopK(float[] query, int k) {
        int[] topIds = topK(query, k);
        int fraudCount = 0;
        boolean[] labels = this.fraudLabels;
        for (int id : topIds) {
            if (id >= 0 && labels[id]) fraudCount++;
        }
        return fraudCount;
    }

    /**
     * Top-k ids in distance-ascending order. Diagnostic / recall path.
     */
    public int[] topK(float[] query, int k) {
        if (query.length < Dataset.STRIDE) {
            throw new IllegalArgumentException("query must be at least STRIDE long");
        }

        FloatVector q0 = FloatVector.fromArray(SPECIES, query, 0);
        FloatVector q1 = FloatVector.fromArray(SPECIES, query, 8);

        float[] topDists = new float[k];
        int[] topIds = new int[k];
        Arrays.fill(topDists, Float.POSITIVE_INFINITY);
        Arrays.fill(topIds, -1);
        float cutoff = Float.POSITIVE_INFINITY;

        final float[] vecs = this.vectors;
        final int n = this.n;

        for (int i = 0; i < n; i++) {
            int off = i << 4;
            FloatVector v0 = FloatVector.fromArray(SPECIES, vecs, off);
            FloatVector v1 = FloatVector.fromArray(SPECIES, vecs, off + 8);
            FloatVector d0 = v0.sub(q0);
            FloatVector d1 = v1.sub(q1);
            float dist = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);

            if (dist < cutoff) {
                int pos = k - 1;
                while (pos > 0 && topDists[pos - 1] > dist) {
                    topDists[pos] = topDists[pos - 1];
                    topIds[pos] = topIds[pos - 1];
                    pos--;
                }
                topDists[pos] = dist;
                topIds[pos] = i;
                cutoff = topDists[k - 1];
            }
        }
        return topIds;
    }
}
