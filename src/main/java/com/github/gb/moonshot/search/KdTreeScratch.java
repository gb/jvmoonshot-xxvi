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
    int bboxPrunesLo;   // subset of bboxPrunes: pruned by dims 0-7 alone (lo early-exit)
    int topKFilledAt;
    int topKReplaced;
    float finalPeekDist;

    /** Max concurrent entries in the BBF min-heap. Measured peak = 153 across 54100 contest queries. */
    static final int BBF_HEAP_CAP = 256;
    /**
     * Pool slots are never recycled: each heap push allocates one slot even if entries are later popped.
     * Must be ≥ the maximum total pushes in one query. Measured over 270 k contest queries: max
     * pushes ≤ 256 (p99 = 139), so 256 gives exactly zero pool overflows with zero DFS fallback.
     * Pool size: 256 × 14 × 4 = 14 KB; combined with heap arrays (3 KB, depth removed) the full
     * BBF scratch fits in Haswell L1D (32 KB), while the old 4096-slot pool was 229 KB (deep L2/L3).
     */
    static final int BBF_POOL_CAP = 256;
    /**
     * Fully packed heap entry, 64 bits:
     * <pre>
     *   bits 63..32: slabSum (32 bits, non-negative, sort key)
     *   bits 31..24: slabIdx (8 bits, 0..BBF_POOL_CAP-1)
     *   bits 23..0:  treeIdx (24 bits, 0..16M; current n is 3M)
     * </pre>
     * Min-heap order by signed long compare is correct (slabSum's sign bit stays clear).
     * Single array eliminates the parallel {@code bbfSlabIdx int[]}, saving one cache line per
     * swap. The compare-on-slabSum form (extract high 32 bits) is still required to preserve the
     * old FIFO tiebreak across equal-slabSum siblings.
     */
    static final int  BBF_TREEIDX_BITS = 24;
    static final long BBF_TREEIDX_MASK = (1L << BBF_TREEIDX_BITS) - 1L;     // 0x00FFFFFF
    static final int  BBF_SLABIDX_BITS = 8;
    static final long BBF_SLABIDX_MASK = (1L << BBF_SLABIDX_BITS) - 1L;     // 0xFF
    static final int  BBF_SLABIDX_SHIFT = BBF_TREEIDX_BITS;                 // 24
    final long[]  bbfHeap = new long[BBF_HEAP_CAP];
    final int[]   bbfEnd = new int[BBF_HEAP_CAP];
    final byte[]  bbfDepth = new byte[BBF_HEAP_CAP];

    static long packBbfEntry(int slabSum, int slabIdx, int treeIdx) {
        return ((long) slabSum << 32)
                | ((slabIdx & BBF_SLABIDX_MASK) << BBF_SLABIDX_SHIFT)
                | (treeIdx & BBF_TREEIDX_MASK);
    }
    static int bbfSlabSum(long entry)  { return (int) (entry >>> 32); }
    static int bbfSlabIdx(long entry)  { return (int) ((entry >>> BBF_SLABIDX_SHIFT) & BBF_SLABIDX_MASK); }
    static int bbfTreeIdx(long entry)  { return (int) (entry & BBF_TREEIDX_MASK); }
    final int[]   bbfSlabPool = new int[BBF_POOL_CAP * KdTree.DIMS];
    int bbfSize;
    int bbfSlabNext;
    boolean approximateStopped;

    /** G2 thresholdSum cache: invalidated when peekSum changes or when visits crosses a bucket
     *  boundary (16 visits/bucket). The threshold value cached is the one computed at the FIRST
     *  visit of the bucket — i.e. the *loosest* threshold in that bucket — so we never over-prune.
     *  Recall-safe by construction. */
    int cachedThresholdPeekSum;
    int cachedThresholdBucket;
    int cachedThresholdSum;

}
