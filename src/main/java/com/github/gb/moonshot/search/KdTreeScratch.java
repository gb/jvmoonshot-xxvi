package com.github.gb.moonshot.search;

/** Per-thread mutable state for a single {@link KdTree} query. */
final class KdTreeScratch {

    static final ThreadLocal<KdTreeScratch> scratchTL = ThreadLocal.withInitial(KdTreeScratch::new);

    final TopKSortedArray results         = new TopKSortedArray();
    final int[]           slab            = new int[KdTree.DIMS];
    /** Lanes 14-15 stay zero; permuted query in i16 units. */
    final short[]         permutedQueryI16 = new short[KdTree.STRIDE];
    int[] topKBuf                          = new int[8];
    final int[]   fanOutBuf                = new int[KdTree.PRIME_FANOUT_COUNT];
    final float[] fanOutScores             = new float[KdTree.PRIME_FANOUT_COUNT];
    int fanOutCount;
    int visits;
    int bboxChecks;
    int bbfPushes;
    int bbfHeapMax;
    int slabPrunes;
    int bboxPrunes;
    int topKFilledAt;
    int topKReplaced;
    float finalPeekDist;

    /** Max concurrent entries in the BBF min-heap. Measured peak = 153 across 54100 contest queries. */
    static final int BBF_HEAP_CAP = 256;
    /**
     * Pool slots are never recycled: each heap push allocates one slot even if entries are later popped.
     * Must be ≥ the maximum total pushes in one query. Measured over 270 k contest queries: max
     * pushes ≤ 256 (p99 = 139), so 256 gives exactly zero pool overflows with zero DFS fallback.
     * Pool size: 256 × 14 × 4 = 14 KB; combined with heap arrays (4 KB) the full BBF scratch fits
     * in Haswell L1D (32 KB), while the old 4096-slot pool was 229 KB (deep L2 / L3).
     */
    static final int BBF_POOL_CAP = 256;
    final int[]   bbfTreeIdx  = new int[BBF_HEAP_CAP];
    final int[]   bbfSlabSum  = new int[BBF_HEAP_CAP];
    final int[]   bbfDepth    = new int[BBF_HEAP_CAP];
    final int[]   bbfSlabIdx  = new int[BBF_HEAP_CAP];
    final int[]   bbfSlabPool = new int[BBF_POOL_CAP * KdTree.DIMS];
    int bbfSize;
    int bbfSlabNext;

    void ensureTopKCapacity(int k) {
        if (topKBuf.length < k) topKBuf = new int[k];
    }
}
