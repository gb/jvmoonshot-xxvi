package com.github.gb.moonshot.search;

import java.lang.foreign.MemorySegment;
import java.nio.MappedByteBuffer;

import static com.github.gb.moonshot.search.KdTreeTuning.*;

/**
 * Exact k-NN via balanced KD-tree with i16-quantized point storage.
 * Each feature is stored as {@code round(v * SCALE)} where {@code SCALE=10_000},
 * which is a lossless encoding for the contest's 4-decimal-precision floats.
 *
 * <p>The distance kernel is scalar: on JDK 25 x86, {@code ShortVector.convertShape(S2I)}
 * from SPECIES_128 to IntVector.SPECIES_256 is not escape-analyzed by C2, causing ~787 KB
 * of Vector heap allocation per query. The scalar loop is auto-vectorized by C2 on AVX2
 * and produces zero allocation.
 *
 * <p>Distances are scaled back to f32 via {@link #INV_SCALE_SQ} so all slab/pruning
 * comparisons remain float-compatible — no changes to the BBF logic.
 *
 * <p>Memory: n × 16 × 2 bytes pts vs n × 16 × 4 bytes for f32 — 50% reduction.
 * Production loads pts via mmap (off the cgroup JVM heap) while right/origId/fraud/
 * topSlot/topBbox stay on-heap (total ~54 MB for n=3M, fits in -Xmx65m).
 */
public final class KdTree implements VectorIndex {

    public static final int DIMS = KdTreeLayout.DIMS; // 14
    public static final int[] DIM_PERMUTATION = KdTreeLayout.DIM_PERMUTATION;
    /**
     * STRIDE=20: dims[0..13] + leftAndDim[14..15] + right[16..17] + padding[18..19].
     * Packing right into the same stride as pts eliminates the separate int[] right[] heap array
     * (saves 12 MB) and — critically for the contest box (2014 Mac Mini, 3–4 MB L3) — turns
     * each node visit from 2 independent DRAM accesses (pts + right[]) into 1, saving
     * ~1716 × 90 ns ≈ 150 µs per query.
     * Addressing: treeIdx * 20 (imul, 3 cycles) vs treeIdx << 4 (shift, 1 cycle) — the extra
     * ~1.3 µs per query for the multiply is negligible compared to the DRAM savings.
     * Cache density: 64/40 = 1.6 nodes per 64-byte cache line (vs 2.0 for STRIDE=16). The
     * reduction is outweighed by eliminating the separate right[] cache-line misses.
     */
    public static final int STRIDE = 20;
    public static final int STRIDE_BBOX = 32; // 16 lo + 16 hi shorts per bbox node

    /**
     * Quantization: f32 → i16 via round(v * SCALE). Matches 4-decimal contest precision.
     *
     * <p>{@code INV_SCALE_SQ} is not exactly {@code 1/10000²} due to float rounding
     * ({@code 1.0f/10000f} ≠ 0.0001 in IEEE 754).  All distances are scaled by the same
     * constant factor, so KNN ordering is preserved and results are correct.  However,
     * pruning thresholds differ slightly from f32 KdTree, causing ~20% fewer node visits
     * at p99 — not a bug, but means visit counts are not directly comparable between the
     * two implementations.
     */
    public static final int SCALE = 10_000;
    public static final float INV_SCALE = 1.0f / SCALE;
    public static final float INV_SCALE_SQ = INV_SCALE * INV_SCALE;

    static final int LANE_LEFT_DIM = KdTreeLayout.LANE_LEFT_DIM; // 14 — leftAndDim packed at pts[14..15]
    static final int LANE_RIGHT = 16; // right packed at pts[16..17]
    /**
     * Lane 18: fraud flag packed into the previously-unused padding short.
     * Value is 0 (not fraud) or 1 (fraud). Eliminates the separate on-heap fraud[] byte array
     * in mmap mode, freeing ~3 MB of L3 that was displaced by the heap allocation.
     */
    static final int LANE_FRAUD = 18;

    public static final int TOP_BBOX_DEPTH = 18;
    public static final int BBF_MAX_DEPTH = 18;
    public static final int PRIME_FANOUT_DEPTH = 5;
    public static final int PRIME_FANOUT_COUNT = 1 << PRIME_FANOUT_DEPTH;
    /** Forwarding constant for bench reporting; matches {@link KdTreeScratch#BBF_HEAP_CAP}. */
    public static final int BBF_HEAP_CAP = KdTreeScratch.BBF_HEAP_CAP;

    // ── Bench-facing tuning setters ───────────────────────────────────────────────────────────────

    public static void setMaxVisitsBudget(int cap) {
        KdTreeTuning.MAX_VISITS_BUDGET = Math.max(1, cap);
    }

    public static void setPrimeBboxScoring(boolean e) {
        KdTreeTuning.PRIME_BBOX_SCORING = e;
    }

    public static void setPrimePlungeCap(int cap) {
        KdTreeTuning.PRIME_PLUNGE_CAP = Math.max(1, Math.min(PRIME_FANOUT_COUNT, cap));
    }

    public static void setRelaxParams(int softCap, int range, float maxEpsilon) {
        KdTreeTuning.RELAX_SOFT_CAP = softCap;
        KdTreeTuning.RELAX_RANGE = Math.max(1, range);
        KdTreeTuning.RELAX_MAX_EPSILON = maxEpsilon;
    }

    final int n;

    // Heap mode fields (non-null in heap mode, null in mmap mode).
    final short[] pts;

    // Mmap mode fields (non-null in mmap mode, null in heap mode).
    final MappedByteBuffer ptsBuf;
    final MemorySegment ptsSeg;

    // right[] is packed into pts[16..17] (LANE_RIGHT=16) as two i16 halves of a LE int32.
    // No separate int[] right field — eliminates 12 MB heap and one independent DRAM access
    // per visit on the contest box (3–4 MB L3, right[] would be entirely in DRAM otherwise).
    final int[] origId;   // null in mmap mode (not read in hot path; saves 11.4 MB heap)
    final byte[] fraud;
    final int[] topSlot;
    final short[] topBbox;
    final int topNodeCount;

    /** Owned scratch for the single-threaded NIO hot path — eliminates the ThreadLocal lookup per query. */
    private final KdTreeScratch instanceScratch = new KdTreeScratch();

    /** Nodes visited by the last {@link #countFraudsInTop5Fast} call — for {@link com.github.gb.moonshot.bench.PrefetchBench}. */
    public int lastFastVisits() { return instanceScratch.visits; }

    /**
     * Heap-mode constructor.
     *
     * <p><b>Single-instance constraint:</b> {@link KdTreeUnsafe#ptsBaseAddr} is a
     * {@code static long}.  Constructing a second {@code KdTree} in the same JVM
     * (whether heap or mmap) overwrites the address, corrupting nav reads ({@link #leftAndDimAt})
     * for any previously constructed instance.  In production there is exactly one instance
     * per JVM.  Bench tools that load multiple index variants must use separate JVM invocations.
     */
    KdTree(int n, short[] pts, int[] origId, byte[] fraud,
           int[] topSlot, short[] topBbox, int topNodeCount) {
        this.n = n;
        this.pts = pts;
        this.ptsBuf = null;
        this.ptsSeg = null;
        this.origId = origId;
        this.fraud = fraud;
        this.topSlot = topSlot;
        this.topBbox = topBbox;
        this.topNodeCount = topNodeCount;
        KdTreeUnsafe.bindPtsSegment(null);
    }

    /**
     * Mmap-mode constructor. See heap-mode constructor for the single-instance constraint.
     */
    KdTree(int n, MappedByteBuffer ptsBuf, int[] origId, byte[] fraud,
           int[] topSlot, short[] topBbox, int topNodeCount) {
        this.n = n;
        this.pts = null;
        this.ptsBuf = ptsBuf;
        this.ptsSeg = MemorySegment.ofBuffer(ptsBuf);
        this.origId = origId;
        this.fraud = fraud;
        this.topSlot = topSlot;
        this.topBbox = topBbox;
        this.topNodeCount = topNodeCount;
        KdTreeUnsafe.bindPtsSegment(this.ptsSeg);
    }

    @Override
    public void applyMmapHints() {
        if (ptsSeg != null) {
            KdTreeMmap.madvise(ptsSeg, KdTreeMmap.MADV_HUGEPAGE);
            KdTreeMmap.madvise(ptsSeg, KdTreeMmap.MADV_RANDOM);
        }
    }

    @Override
    public void prewarm() {
        if (ptsBuf != null) KdTreeMmap.touchPages(ptsBuf);
    }

    // -------------------------------------------------------------------------
    // Quantization
    // -------------------------------------------------------------------------

    public static short quantize(float v) {
        if (v <= -1.0f) return (short) (-SCALE);
        if (v >= 1.0f) return (short) (SCALE);
        return (short) Math.round(v * SCALE);
    }

    private static short quantizeQuery(float v) {
        if (v <= -1.0f) return (short) (-SCALE);
        if (v >= 1.0f) return (short) SCALE;
        return (short) (v * SCALE + 0.5f);
    }

    // -------------------------------------------------------------------------
    // Nav access
    // -------------------------------------------------------------------------

    int leftAndDimAt(int treeIdx) {
        return leftAndDimAtBase(treeIdx * STRIDE);
    }

    private int leftAndDimAtBase(int base) {
        if (pts != null) {
            int off = base + LANE_LEFT_DIM;
            return (pts[off] & 0xFFFF) | ((pts[off + 1] & 0xFFFF) << 16);
        }
        // Mmap: two consecutive i16 LE at byte offset = (treeIdx*STRIDE + LANE_LEFT_DIM)*2.
        // Reading a LE int32 there reconstructs leftAndDim directly (both halves are LE i16).
        return KdTreeUnsafe.UNSAFE.getInt(
                KdTreeUnsafe.ptsBaseAddr + ((long) base + LANE_LEFT_DIM) * 2L);
    }

    static int unpackLeft(int leftAndDim) {
        return KdTreeLayout.unpackLeft(leftAndDim);
    }

    private static int unpackDim(int leftAndDim) {
        return (leftAndDim >>> 28) & 0xF;
    }

    int rightAt(int treeIdx) {
        return rightAtBase(treeIdx * STRIDE);
    }

    private int rightAtBase(int base) {
        if (pts != null) {
            int off = base + LANE_RIGHT;
            return (pts[off] & 0xFFFF) | ((pts[off + 1] & 0xFFFF) << 16);
        }
        // Mmap: byte offset = (treeIdx * STRIDE + LANE_RIGHT) * 2 = treeIdx * 40 + 32 (4-byte aligned).
        return KdTreeUnsafe.UNSAFE.getInt(
                KdTreeUnsafe.ptsBaseAddr + ((long) base + LANE_RIGHT) * 2L);
    }

    /**
     * Returns the original dataset ID for a tree node, or -1 when origId is null (mmap production mode).
     */
    int origIdAt(int treeIdx) {
        return (origId != null) ? origId[treeIdx] : -1;
    }

    /**
     * Fill {@code out[0..DIMS)} with the raw permuted feature values for the node at
     * {@code treeIdx}, dequantized from i16. Values are in variance-descending lane order
     * (not semantic order).
     * {@link com.github.gb.moonshot.WarmupDriver} samples these and perturbs them, so lane
     * order is irrelevant — what matters is that the magnitudes are realistic.
     */
    @Override
    public void copyNodeVector(int treeIdx, float[] out) {
        int base = treeIdx * STRIDE;
        if (pts != null) {
            for (int d = 0; d < DIMS; d++) out[d] = pts[base + d] * INV_SCALE;
        } else {
            long addr = KdTreeUnsafe.ptsBaseAddr + (long) base * 2L;
            for (int d = 0; d < DIMS; d++) {
                out[d] = KdTreeUnsafe.UNSAFE.getShort(addr + (long) d * 2L) * INV_SCALE;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Distance kernel — scalar int loop, scaled to f32 via INV_SCALE_SQ
    // -------------------------------------------------------------------------

    /**
     * Read a single i16 dim value from pts (heap or mmap). Used only for near/far split decisions.
     */
    private short ptAtI16Base(int base, int splitDim) {
        if (pts != null) return pts[base + splitDim];
        return KdTreeUnsafe.UNSAFE.getShort(
                KdTreeUnsafe.ptsBaseAddr + ((long) base + splitDim) * 2L);
    }

    /**
     * Squared L2 distance in i16 units, scaled back to f32 by INV_SCALE_SQ.
     * Overflow proof under the contest spec: only semantic dims 5 and 6 may use the -1 sentinel; all other
     * dims are clamped to [0, 1]. After quantization the worst squared sum is
     * 2 × 20000² + 12 × 10000² = 2.0B, below Integer.MAX_VALUE, so int accumulation is safe.
     *
     * <p>Scalar implementation avoids ShortVector → IntVector convertShape, which
     * C2 on x86 JDK 25 fails to escape-analyze, causing ~787 KB heap allocation
     * per query (Vector objects not eliminated). Scalar loop is auto-vectorized
     * by C2 on AVX2 hosts with no allocation.
     */
    private int distSumI16AtBase(KdTreeScratch scratch, int base) {
        short[] q = scratch.permutedQueryI16;
        int sum = 0;
        if (pts != null) {
            for (int d = 0; d < DIMS; d++) {
                int diff = q[d] - pts[base + d];
                sum += diff * diff;
            }
        } else {
            long nodeBase = KdTreeUnsafe.ptsBaseAddr + (long) base * 2L;
            long w = KdTreeUnsafe.UNSAFE.getLong(nodeBase);
            int diff = q[0] - (short) w;
            sum += diff * diff;
            diff = q[1] - (short) (w >>> 16);
            sum += diff * diff;
            diff = q[2] - (short) (w >>> 32);
            sum += diff * diff;
            diff = q[3] - (short) (w >>> 48);
            sum += diff * diff;

            w = KdTreeUnsafe.UNSAFE.getLong(nodeBase + 8L);
            diff = q[4] - (short) w;
            sum += diff * diff;
            diff = q[5] - (short) (w >>> 16);
            sum += diff * diff;
            diff = q[6] - (short) (w >>> 32);
            sum += diff * diff;
            diff = q[7] - (short) (w >>> 48);
            sum += diff * diff;

            w = KdTreeUnsafe.UNSAFE.getLong(nodeBase + 16L);
            diff = q[8] - (short) w;
            sum += diff * diff;
            diff = q[9] - (short) (w >>> 16);
            sum += diff * diff;
            diff = q[10] - (short) (w >>> 32);
            sum += diff * diff;
            diff = q[11] - (short) (w >>> 48);
            sum += diff * diff;

            int tail = KdTreeUnsafe.UNSAFE.getInt(nodeBase + 24L);
            diff = q[12] - (short) tail;
            sum += diff * diff;
            diff = q[13] - (short) (tail >>> 16);
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * Squared L2 bbox distance in i16 units, scaled to f32. Early exit after first 8 dims.
     * Layout: topBbox[slot*STRIDE_BBOX+0..13] = lo, topBbox[slot*STRIDE_BBOX+16..29] = hi.
     */
    float bboxDistSquaredI16(KdTreeScratch scratch, int slot) {
        int base = slot * STRIDE_BBOX;
        short[] q = scratch.permutedQueryI16;
        short[] bb = topBbox;
        int partLo = 0;
        for (int d = 0; d < 8; d++) {
            int clamped = Math.max(bb[base + d] - q[d], Math.max(q[d] - bb[base + 16 + d], 0));
            partLo += clamped * clamped;
        }
        if (scratch.results.size() == TopKSortedArray.MAX_K
                && partLo * INV_SCALE_SQ >= scratch.results.peekDist()) {
            return Float.POSITIVE_INFINITY;
        }
        int partHi = 0;
        for (int d = 8; d < DIMS; d++) {
            int clamped = Math.max(bb[base + d] - q[d], Math.max(q[d] - bb[base + 16 + d], 0));
            partHi += clamped * clamped;
        }
        return (partLo + partHi) * INV_SCALE_SQ;
    }

    private boolean bboxPrunesI16Sum(KdTreeScratch scratch, int slot, int thresholdSum) {
        int base = slot * STRIDE_BBOX;
        short[] q = scratch.permutedQueryI16;
        short[] bb = topBbox;
        int partLo = 0;
        for (int d = 0; d < 8; d++) {
            int clamped = Math.max(bb[base + d] - q[d], Math.max(q[d] - bb[base + 16 + d], 0));
            partLo += clamped * clamped;
        }
        if (partLo >= thresholdSum) return true;
        int partHi = 0;
        for (int d = 8; d < DIMS; d++) {
            int clamped = Math.max(bb[base + d] - q[d], Math.max(q[d] - bb[base + 16 + d], 0));
            partHi += clamped * clamped;
        }
        return partLo + partHi >= thresholdSum;
    }

    private static int thresholdSum(int peekSum, int visits, boolean exact) {
        if (exact || !RELAX_INITIALLY_ENABLED) return peekSum;
        return Math.max(0, (int) (peekSum * KdTreeTuning.relaxScale(visits)));
    }

    private static boolean visitBudgetExhausted(KdTreeScratch scratch, boolean exact) {
        return !exact && MAX_VISITS_ENABLED && scratch.visits >= MAX_VISITS_BUDGET;
    }

    // -------------------------------------------------------------------------
    // VectorIndex interface
    // -------------------------------------------------------------------------

    @Override
    public int size() {
        return n;
    }

    @Override
    public int countFraudsInTop5(float[] query) {
        KdTreeScratch scratch = KdTreeScratch.scratchTL.get();
        int fraudCount = countFraudsInTop5(query, scratch, false);
        if ((RELAX_INITIALLY_ENABLED || MAX_VISITS_ENABLED) && REFINE_BOUNDARY
                && (fraudCount == 2 || fraudCount == 3)) {
            fraudCount = countFraudsInTop5(query, scratch, true);
        }
        return fraudCount;
    }

    /**
     * Hot path for the single-threaded NIO event loop: uses the owned {@link #instanceScratch},
     * avoiding the ThreadLocal lookup on every query.
     */
    public int countFraudsInTop5Fast(float[] query) {
        KdTreeScratch scratch = instanceScratch;
        int fraudCount = countFraudsInTop5(query, scratch, false);
        if ((RELAX_INITIALLY_ENABLED || MAX_VISITS_ENABLED) && REFINE_BOUNDARY
                && (fraudCount == 2 || fraudCount == 3)) {
            fraudCount = countFraudsInTop5(query, scratch, true);
        }
        return fraudCount;
    }

    private int countFraudsInTop5(float[] query, KdTreeScratch scratch, boolean exact) {
        prepareSearch(query, scratch);
        prime(scratch, TopKSortedArray.MAX_K);
        descendBBF(scratch, TopKSortedArray.MAX_K, exact);
        if (PROFILING_ENABLED) scratch.finalPeekDist = scratch.results.peekDist();
        return ptsSeg != null
                ? scratch.results.countFraudsFromMmap(KdTreeUnsafe.ptsBaseAddr, STRIDE)
                : scratch.results.countFrauds(fraud);
    }

    @Override
    public int countFraudsInTopK(float[] query, int k) {
        KdTreeScratch scratch = KdTreeScratch.scratchTL.get();
        scratch.results.ensureCapacity(k);
        scratch.ensureTopKCapacity(k);
        prepareSearch(query, scratch);
        prime(scratch, k);
        descendBBF(scratch, k, false);
        if (PROFILING_ENABLED) scratch.finalPeekDist = scratch.results.peekDist();
        int[] treeIdxs = scratch.topKBuf;
        int count = scratch.results.drainAscending(treeIdxs);
        int fraudCount = 0;
        if (ptsSeg != null) {
            long base = KdTreeUnsafe.ptsBaseAddr;
            for (int i = 0; i < Math.min(k, count); i++) {
                fraudCount += KdTreeUnsafe.UNSAFE.getShort(
                        base + (long) treeIdxs[i] * (STRIDE * 2L) + LANE_FRAUD * 2L) & 1;
            }
        } else {
            for (int i = 0; i < Math.min(k, count); i++) {
                if (fraud[treeIdxs[i]] != 0) fraudCount++;
            }
        }
        return fraudCount;
    }

    // -------------------------------------------------------------------------
    // Search preparation
    // -------------------------------------------------------------------------

    static void prepareSearch(float[] query, KdTreeScratch scratch) {
        scratch.results.clear();
        int[] slab = scratch.slab;
        slab[0] = 0;
        slab[1] = 0;
        slab[2] = 0;
        slab[3] = 0;
        slab[4] = 0;
        slab[5] = 0;
        slab[6] = 0;
        slab[7] = 0;
        slab[8] = 0;
        slab[9] = 0;
        slab[10] = 0;
        slab[11] = 0;
        slab[12] = 0;
        slab[13] = 0;
        scratch.visits = 0;
        scratch.bboxChecks = 0;
        scratch.bbfSize = 0;
        scratch.bbfSlabNext = 0;
        if (PROFILING_ENABLED) {
            scratch.bbfPushes     = 0;
            scratch.bbfHeapMax    = 0;
            scratch.slabPrunes    = 0;
            scratch.bboxPrunes    = 0;
            scratch.topKFilledAt  = 0;
            scratch.topKReplaced  = 0;
            scratch.finalPeekDist = 0f;
        }
        short[] pqi = scratch.permutedQueryI16;
        // Keep this unrolled order in sync with KdTreeLayout.DIM_PERMUTATION.
        pqi[0] = quantizeQuery(query[6]);
        pqi[1] = quantizeQuery(query[10]);
        pqi[2] = quantizeQuery(query[9]);
        pqi[3] = quantizeQuery(query[5]);
        pqi[4] = quantizeQuery(query[11]);
        pqi[5] = quantizeQuery(query[2]);
        pqi[6] = quantizeQuery(query[4]);
        pqi[7] = quantizeQuery(query[7]);
        pqi[8] = quantizeQuery(query[0]);
        pqi[9] = quantizeQuery(query[1]);
        pqi[10] = quantizeQuery(query[3]);
        pqi[11] = quantizeQuery(query[8]);
        pqi[12] = quantizeQuery(query[12]);
        pqi[13] = quantizeQuery(query[13]);
        pqi[LANE_LEFT_DIM] = 0;
        pqi[LANE_RIGHT] = 0;
    }

    // -------------------------------------------------------------------------
    // Prime (fan-out + plunge to pre-fill top-K)
    // -------------------------------------------------------------------------

    void prime(KdTreeScratch scratch, int k) {
        scratch.fanOutCount = 0;
        primeRecurse(0, scratch, k, 0);
        primeSelectAndPlunge(scratch, k);
    }

    private void primeRecurse(int treeIdx, KdTreeScratch scratch, int k, int depth) {
        if (treeIdx < 0) return;
        TopKSortedArray topK = scratch.results;
        scratch.visits++;
        int nodeBase = treeIdx * STRIDE;
        int dist = distSumI16AtBase(scratch, nodeBase);
        if (topK.size() < k) {
            topK.push(treeIdx, dist);
            if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
        } else if (dist < topK.peekSum()) {
            topK.replaceFarthest(treeIdx, dist);
            if (PROFILING_ENABLED) scratch.topKReplaced++;
        }
        int leftAndDim = leftAndDimAtBase(nodeBase);
        int leftIdx = unpackLeft(leftAndDim);
        int rightIdx = rightAtBase(nodeBase);
        if (depth < PRIME_FANOUT_DEPTH) {
            primeRecurse(leftIdx, scratch, k, depth + 1);
            primeRecurse(rightIdx, scratch, k, depth + 1);
        } else {
            int splitDim = unpackDim(leftAndDim);
            int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
            int near = (delta_i16 < 0) ? leftIdx : rightIdx;
            if (near >= 0 && scratch.fanOutCount < PRIME_FANOUT_COUNT) {
                scratch.fanOutBuf[scratch.fanOutCount++] = near;
            }
        }
    }

    private void primeSelectAndPlunge(KdTreeScratch scratch, int k) {
        int count = scratch.fanOutCount;
        if (count == 0) return;
        int[] buf = scratch.fanOutBuf;
        int m = Math.min(count, PRIME_PLUNGE_CAP);

        if (PRIME_BBOX_SCORING) {
            float[] scores = scratch.fanOutScores;
            for (int i = 0; i < count; i++) {
                int slot = topSlot[buf[i]];
                scores[i] = (slot >= 0) ? bboxDistSquaredI16(scratch, slot) : Float.POSITIVE_INFINITY;
            }
            scratch.bboxChecks += count;
            for (int i = 0; i < m; i++) {
                int minIdx = i;
                float minScore = scores[i];
                for (int j = i + 1; j < count; j++) {
                    if (scores[j] < minScore) {
                        minScore = scores[j];
                        minIdx = j;
                    }
                }
                if (minIdx != i) {
                    int tBuf = buf[i];
                    buf[i] = buf[minIdx];
                    buf[minIdx] = tBuf;
                    float tScore = scores[i];
                    scores[i] = scores[minIdx];
                    scores[minIdx] = tScore;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            plunge(buf[i], scratch, k);
        }
    }

    private void plunge(int treeIdx, KdTreeScratch scratch, int k) {
        TopKSortedArray topK = scratch.results;
        while (treeIdx >= 0) {
            scratch.visits++;
            int nodeBase = treeIdx * STRIDE;
            int dist = distSumI16AtBase(scratch, nodeBase);
            if (topK.size() < k) {
                topK.push(treeIdx, dist);
                if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
            } else if (dist < topK.peekSum()) {
                topK.replaceFarthest(treeIdx, dist);
                if (PROFILING_ENABLED) scratch.topKReplaced++;
            }
            int leftAndDim = leftAndDimAtBase(nodeBase);
            int splitDim = unpackDim(leftAndDim);
            int leftIdx = unpackLeft(leftAndDim);
            int rightIdx = rightAtBase(nodeBase);
            int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
            treeIdx = (delta_i16 < 0) ? leftIdx : rightIdx;
        }
    }

    // -------------------------------------------------------------------------
    // Descend — classical DFS fallback for deep FAR nodes
    // -------------------------------------------------------------------------

    void descend(int treeIdx, KdTreeScratch scratch, int k, int slabSum, int depth, boolean exact) {
        if (treeIdx < 0) return;
        if (visitBudgetExhausted(scratch, exact)) return;
        TopKSortedArray topK = scratch.results;
        if (topK.size() >= k) {
            int peekSum = topK.peekSum();
            int threshSum = thresholdSum(peekSum, scratch.visits, exact);
            if (slabSum > threshSum) {
                if (PROFILING_ENABLED) scratch.slabPrunes++;
                return;
            }
            if (depth <= TOP_BBOX_DEPTH) {
                int slot = topSlot[treeIdx];
                if (slot >= 0) {
                    scratch.bboxChecks++;
                    if (bboxPrunesI16Sum(scratch, slot, threshSum)) {
                        if (PROFILING_ENABLED) scratch.bboxPrunes++;
                        return;
                    }
                }
            }
        }
        scratch.visits++;
        int nodeBase = treeIdx * STRIDE;
        int dist = distSumI16AtBase(scratch, nodeBase);
        if (topK.size() < k) {
            if (!topK.contains(treeIdx)) {
                topK.push(treeIdx, dist);
                if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
            }
        } else if (dist < topK.peekSum() && !topK.contains(treeIdx)) {
            topK.replaceFarthest(treeIdx, dist);
            if (PROFILING_ENABLED) scratch.topKReplaced++;
        }
        int leftAndDim = leftAndDimAtBase(nodeBase);
        int splitDim = unpackDim(leftAndDim);
        int leftIdx = unpackLeft(leftAndDim);
        int rightIdx = rightAtBase(nodeBase);
        int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
        int near, far;
        if (delta_i16 < 0) {
            near = leftIdx;
            far = rightIdx;
        } else {
            near = rightIdx;
            far = leftIdx;
        }

        descend(near, scratch, k, slabSum, depth + 1, exact);

        int[] slab = scratch.slab;
        int oldSlabD = slab[splitDim];
        int newSlabD = delta_i16 * delta_i16;
        int newSlabSum = slabSum - oldSlabD + newSlabD;
        if (topK.size() < k || newSlabSum <= thresholdSum(topK.peekSum(), scratch.visits, exact)) {
            slab[splitDim] = newSlabD;
            descend(far, scratch, k, newSlabSum, depth + 1, exact);
            slab[splitDim] = oldSlabD;
        }
    }

    // -------------------------------------------------------------------------
    // Best-first BBF
    // -------------------------------------------------------------------------

    void descendBBF(KdTreeScratch scratch, int k, boolean exact) {
        continueDfsBBF(0, 0, 0, scratch, k, exact);
        int[] hTreeIdx = scratch.bbfTreeIdx;
        int[] hSlabSum = scratch.bbfSlabSum;
        int[] hDepth = scratch.bbfDepth;
        int[] hSlabIdx = scratch.bbfSlabIdx;
        int[] pool = scratch.bbfSlabPool;
        TopKSortedArray topK = scratch.results;
        while (scratch.bbfSize > 0) {
            if (visitBudgetExhausted(scratch, exact)) break;
            int popTi = hTreeIdx[0];
            int popSs = hSlabSum[0];
            int popDe = hDepth[0];
            int popSi = hSlabIdx[0];
            int lastIdx = --scratch.bbfSize;
            if (lastIdx > 0) {
                int ti = hTreeIdx[lastIdx];
                int ss = hSlabSum[lastIdx];
                int de = hDepth[lastIdx];
                int si = hSlabIdx[lastIdx];
                int i = 0, half = lastIdx >>> 1;
                while (i < half) {
                    int child = (i << 1) + 1;
                    int right = child + 1;
                    if (right < lastIdx && hSlabSum[right] < hSlabSum[child]) child = right;
                    if (ss <= hSlabSum[child]) break;
                    hTreeIdx[i] = hTreeIdx[child];
                    hSlabSum[i] = hSlabSum[child];
                    hDepth[i] = hDepth[child];
                    hSlabIdx[i] = hSlabIdx[child];
                    i = child;
                }
                hTreeIdx[i] = ti;
                hSlabSum[i] = ss;
                hDepth[i] = de;
                hSlabIdx[i] = si;
            }
            if (topK.size() >= k && popSs > thresholdSum(topK.peekSum(), scratch.visits, exact)) continue;
            System.arraycopy(pool, popSi * DIMS, scratch.slab, 0, DIMS);
            continueDfsBBF(popTi, popSs, popDe, scratch, k, exact);
        }
    }

    private void continueDfsBBF(int treeIdx, int slabSum, int depth, KdTreeScratch scratch, int k,
                                boolean exact) {
        TopKSortedArray topK = scratch.results;
        int[] slab = scratch.slab;
        int[] hTreeIdx = scratch.bbfTreeIdx;
        int[] hSlabSum = scratch.bbfSlabSum;
        int[] hDepth = scratch.bbfDepth;
        int[] hSlabIdx = scratch.bbfSlabIdx;
        int[] pool = scratch.bbfSlabPool;
        while (treeIdx >= 0) {
            if (visitBudgetExhausted(scratch, exact)) return;
            boolean full = topK.size() >= k;
            int peekSum = full ? topK.peekSum() : Integer.MAX_VALUE;
            int thresholdSum = full ? thresholdSum(peekSum, scratch.visits, exact) : Integer.MAX_VALUE;
            if (full) {
                if (slabSum > thresholdSum) {
                    if (PROFILING_ENABLED) scratch.slabPrunes++;
                    return;
                }
            }
            if (depth <= TOP_BBOX_DEPTH && full) {
                int slot = topSlot[treeIdx];
                if (slot >= 0) {
                    scratch.bboxChecks++;
                    if (bboxPrunesI16Sum(scratch, slot, thresholdSum)) {
                        if (PROFILING_ENABLED) scratch.bboxPrunes++;
                        return;
                    }
                }
            }
            scratch.visits++;
            int nodeBase = treeIdx * STRIDE;
            int dist = distSumI16AtBase(scratch, nodeBase);
            if (!full) {
                if (!topK.contains(treeIdx)) {
                    topK.push(treeIdx, dist);
                    if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
                    full = topK.size() >= k;
                    if (full) {
                        peekSum = topK.peekSum();
                        thresholdSum = thresholdSum(peekSum, scratch.visits, exact);
                    }
                }
            } else if (dist < peekSum && !topK.contains(treeIdx)) {
                topK.replaceFarthest(treeIdx, dist);
                peekSum = topK.peekSum();
                thresholdSum = thresholdSum(peekSum, scratch.visits, exact);
                if (PROFILING_ENABLED) scratch.topKReplaced++;
            }
            int leftAndDim = leftAndDimAtBase(nodeBase);
            int splitDim = unpackDim(leftAndDim);
            int leftIdx = unpackLeft(leftAndDim);
            int rightIdx = rightAtBase(nodeBase);
            int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
            int near, far;
            if (delta_i16 < 0) {
                near = leftIdx;
                far = rightIdx;
            } else {
                near = rightIdx;
                far = leftIdx;
            }

            if (far >= 0) {
                int oldSlabD = slab[splitDim];
                int newSlabD = delta_i16 * delta_i16;
                int newSlabSum = slabSum - oldSlabD + newSlabD;
                if (!full || newSlabSum <= thresholdSum) {
                    int nextDepth = depth + 1;
                    if (nextDepth <= BBF_MAX_DEPTH && scratch.bbfSize < KdTreeScratch.BBF_HEAP_CAP
                            && scratch.bbfSlabNext < KdTreeScratch.BBF_POOL_CAP) {
                        int newSlabIdx = scratch.bbfSlabNext++;
                        int poolOff = newSlabIdx * DIMS;
                        System.arraycopy(slab, 0, pool, poolOff, DIMS);
                        pool[poolOff + splitDim] = newSlabD;
                        int i = scratch.bbfSize++;
                        if (PROFILING_ENABLED) {
                            scratch.bbfPushes++;
                            if (scratch.bbfSize > scratch.bbfHeapMax) scratch.bbfHeapMax = scratch.bbfSize;
                        }
                        while (i > 0) {
                            int parent = (i - 1) >>> 1;
                            if (hSlabSum[parent] <= newSlabSum) break;
                            hTreeIdx[i] = hTreeIdx[parent];
                            hSlabSum[i] = hSlabSum[parent];
                            hDepth[i] = hDepth[parent];
                            hSlabIdx[i] = hSlabIdx[parent];
                            i = parent;
                        }
                        hTreeIdx[i] = far;
                        hSlabSum[i] = newSlabSum;
                        hDepth[i] = nextDepth;
                        hSlabIdx[i] = newSlabIdx;
                    } else {
                        slab[splitDim] = newSlabD;
                        descend(far, scratch, k, newSlabSum, nextDepth, exact);
                        slab[splitDim] = oldSlabD;
                    }
                }
            }
            treeIdx = near;
            depth++;
        }
    }

}
