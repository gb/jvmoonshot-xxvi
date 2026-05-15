package com.github.gb.moonshot.search;

import static com.github.gb.moonshot.search.KdTree.DIMS;
import static com.github.gb.moonshot.search.KdTree.DIM_PERMUTATION;
import static com.github.gb.moonshot.search.KdTree.STRIDE;

/**
 * Bench-only entry points into {@link KdTree} internals. Used by {@code BboxMicroBench} and
 * {@code NavReadMicroBench} to measure sub-operations of the descent (sequential vs random
 * throughput on the bbox-distance kernel and packed-nav int reads) without running a full
 * search. Never invoked on the request hot path.
 */
public final class KdTreeProbes {

    private KdTreeProbes() {}

    /**
     * Memory-vs-compute probe. Returns {@code [elapsedNanos, dceSink]} caller-visible sink
     * keeps the JIT from eliminating the work. Big seq/rand throughput ratio = memory-bound;
     * ratio near 1 = compute-bound.
     *
     * <p>{@code permutedQuery} must be 16 floats in variance-descending lane order
     * ({@link KdTree#DIM_PERMUTATION}); lanes 14, 15 must be 0 to match the stride-16
     * convention.
     */
    public static long[] bboxThroughputNanos(KdTree tree, float[] permutedQuery, int[] slotOrder, int reps) {
        if (permutedQuery.length != STRIDE) {
            throw new IllegalArgumentException("permutedQuery length must be " + STRIDE);
        }
        KdTree.Scratch scratch = new KdTree.Scratch();
        // bboxDistSquared reads from scratch.permutedQuery directly mirror the caller buffer in.
        System.arraycopy(permutedQuery, 0, scratch.permutedQuery, 0, STRIDE);
        int sink = 0;
        long start = System.nanoTime();
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < slotOrder.length; i++) {
                sink ^= Float.floatToRawIntBits(tree.bboxDistSquared(scratch, slotOrder[i]));
            }
        }
        long elapsed = System.nanoTime() - start;
        return new long[] { elapsed, (long) sink };
    }

    /**
     * Mirrors descent's nav-int read shape (two ints leftAndDim + right per node, exactly
     * what {@code descend}/{@code prime}/{@code plunge} do per visit). Returns
     * {@code [elapsedNanos, dceSink]}.
     */
    public static long[] navReadThroughputNanos(KdTree tree, int[] idxOrder, int reps) {
        int sink = 0;
        long start = System.nanoTime();
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < idxOrder.length; i++) {
                int idx = idxOrder[i];
                sink ^= tree.leftAndDimAt(idx);
                sink ^= tree.rightAt(idx);
            }
        }
        long elapsed = System.nanoTime() - start;
        return new long[] { elapsed, (long) sink };
    }

    /**
     * Diagnostic / recall-benchmark path: returns the orig IDs of the top-k nearest neighbors in distance-ascending
     * order. Mirrors {@link KdTree#countFraudsInTop5}'s descent but drains the heap to ids instead of folding into a
     * fraud count. Allocates per call never use on the request hot path.
     */
    public static int[] queryRaw(KdTree tree, float[] query, int k) {
        KdTree.Scratch scratch = KdTree.scratchTL.get();
        scratch.results.ensureCapacity(k);
        scratch.ensureTopKCapacity(k);
        KdTree.prepareSearch(query, scratch);
        float[] q = scratch.permutedQuery;
        tree.prime(q, scratch, k);
        tree.descendBBF(q, scratch, k);
        int[] treeIdxs = scratch.topKBuf;
        int n = scratch.results.drainAscending(treeIdxs);
        int[] origs = new int[n];
        for (int i = 0; i < n; i++) origs[i] = tree.origIdAt(treeIdxs[i]);
        return origs;
    }

    /** Per-thread node-visit count from the last {@link KdTree#countFraudsInTop5} (or {@link #queryRaw}) call. */
    public static int lastQueryNodesVisited() {
        return KdTree.scratchTL.get().visits;
    }

    /** Per-thread top-bbox prune-check count from the last search call. */
    public static int lastQueryBboxChecks() {
        return KdTree.scratchTL.get().bboxChecks;
    }

    /**
     * Rotate a caller's semantic 14-float vector into the variance-descending lane order used
     * by pts/topBbox. Returns a fresh 16-float array (lanes 14, 15 = 0).
     */
    public static float[] permuteQueryForBench(float[] semantic) {
        if (semantic.length != DIMS) {
            throw new IllegalArgumentException("semantic length must be " + DIMS);
        }
        float[] out = new float[STRIDE];
        for (int i = 0; i < DIMS; i++) out[i] = semantic[DIM_PERMUTATION[i]];
        return out;
    }
}
