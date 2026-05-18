package com.github.gb.moonshot.search;

/**
 * No-op VectorIndex for HTTP transport benchmarking. Returns a fixed fraud count of 2 (score=0.4,
 * approved) with zero dataset load so the server boots in seconds and measures the pure
 * parse→vectorize→HTTP path without kd-tree search cost.
 *
 * Activate with {@code INDEX_KIND=stub}.
 */
public final class StubVectorIndex implements VectorIndex {

    /** fraud_score = 2/5 = 0.4 → approved: realistic stub response. */
    private static final int STUB_COUNT = 2;

    /** Must be ≥ 1 to avoid division-by-zero in WarmupDriver.warmSearch's modulo. */
    @Override
    public int size() {
        return 1;
    }

    @Override
    public int countFraudsInTopK(float[] query, int k) {
        return STUB_COUNT;
    }

    @Override
    public int countFraudsInTop5(float[] query) {
        return STUB_COUNT;
    }
}
