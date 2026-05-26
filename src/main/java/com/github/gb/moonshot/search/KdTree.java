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
     * STRIDE=16 shorts = 32 bytes: dims[0..13] + packed nav i32 at lanes 14..15.
     * The preorder builder makes the left child implicit ({@code treeIdx + 1} when present);
     * right child, split dim, and fraud bit live in the nav word. This keeps two complete nodes
     * per 64-byte cache line and removes the second nav load from the 40-byte layout.
     */
    public static final int STRIDE = KdTreeLayout.STRIDE;
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

    static final int LANE_NAV = KdTreeLayout.LANE_NAV; // 14 — packed nav int at pts[14..15]
    static final int LANE_NAV_HI = LANE_NAV + 1;

    /**
     * Maximum tree depth at which a node gets a {@code topBbox} slot. 524 k slots × 64 B = 32 MB
     * topBbox footprint. Tested at 17 (16 MB, smaller L3 footprint): regressed with 1 false
     * negative on the 54100-query contest decision fixture — the shallower bbox loses prune
     * coverage on at least one edge-case query that subsequently crosses the deny/approve
     * boundary. Don't shrink without retesting the full fixture, not just verify-recall.
     */
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

    // Right child, split dim, fraud bit, and left-present bit are packed into pts[14..15].
    final int[] origId;   // null in mmap mode (not read in hot path; saves 11.4 MB heap)
    final byte[] fraud;
    final int[] topSlot;
    final short[] topBbox;
    final int topNodeCount;

    /** Owned scratch for the single-threaded NIO hot path — eliminates the ThreadLocal lookup per query. */
    private final KdTreeScratch instanceScratch = new KdTreeScratch();

    /** Nodes visited by the last {@link #countFraudsInTop5Fast} call. Read by FraudScoreHandler's
     *  StageTimer + several bench/diag tools. */
    public int lastFastVisits() { return instanceScratch.visits; }

    /** Top-bbox prune checks from the last {@link #countFraudsInTop5Fast} call. */
    public int lastFastBboxChecks() { return instanceScratch.bboxChecks; }

    /** Slab-prune count from the last fast call; non-zero only when {@code KDTREE_PROFILING=1}.
     *  Read by {@code BboxPruneRateBench}. */
    public int lastFastSlabPrunes() { return instanceScratch.slabPrunes; }

    /** Bbox-prune count from the last fast call; non-zero only when {@code KDTREE_PROFILING=1}.
     *  Read by {@code BboxPruneRateBench}. */
    public int lastFastBboxPrunes() { return instanceScratch.bboxPrunes; }

    /**
     * Heap-mode constructor.
     *
     * <p><b>Single-instance constraint:</b> {@link KdTreeUnsafe#ptsBaseAddr} is a
     * {@code static long}.  Constructing a second {@code KdTree} in the same JVM
     * (whether heap or mmap) overwrites the address, corrupting nav reads ({@link #navAt})
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
        if (ptsBuf == null) return;
        // Prefer the single-syscall synchronous prewarm on modern kernels (Linux 5.14+ for
        // MADV_POPULATE_READ, 6.1+ for MADV_COLLAPSE). Falls back to the per-page Java
        // touchPages loop if the kernel is too old or the syscalls fail. On HDD-backed
        // page cache the syscall path is materially faster because the kernel issues large
        // readahead instead of N userspace round-trips.
        if (!KdTreeMmap.prewarmSync(ptsSeg)) KdTreeMmap.touchPages(ptsBuf);

        // Lock top-level hot pages (16 MB ≈ depth ≤ 18 nodes) in physical memory to protect
        // them from kernel eviction under cgroup memory pressure.
        long topHotBytes = 16L * 1024 * 1024;
        int locked = KdTreeMmap.mlock(ptsSeg, topHotBytes);
        System.out.println("[kdtree] mlock top 16 MB: rc=" + locked);
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

    int navAt(int treeIdx) {
        return navAtBase(treeIdx * STRIDE);
    }

    private int navAtBase(int base) {
        if (pts != null) {
            int off = base + LANE_NAV;
            return (pts[off] & 0xFFFF) | ((pts[off + 1] & 0xFFFF) << 16);
        }
        // Mmap: two consecutive i16 LE lanes hold the packed nav int.
        return KdTreeUnsafe.UNSAFE.getInt(
                KdTreeUnsafe.ptsBaseAddr + ((long) base + LANE_NAV) * 2L);
    }

    static int unpackLeft(int treeIdx, int nav) {
        return KdTreeLayout.unpackLeft(treeIdx, nav);
    }

    private static int unpackDim(int nav) {
        return KdTreeLayout.unpackDim(nav);
    }

    int rightAt(int treeIdx) {
        return KdTreeLayout.unpackRight(navAt(treeIdx));
    }

    /**
     * Returns the original dataset ID for a tree node, or -1 when origId is null (mmap production mode).
     */
    int origIdAt(int treeIdx) {
        return (origId != null) ? origId[treeIdx] : -1;
    }

    /**
     * Fill {@code out[0..DIMS)} with the semantic feature values for the node at {@code treeIdx},
     * dequantized from i16. Stored KD-tree lanes are permuted; warmup feeds this vector back into
     * the normal query path, so output must be de-permuted to the request/vectorizer order.
     */
    @Override
    public void copyNodeVector(int treeIdx, float[] out) {
        int base = treeIdx * STRIDE;
        int[] perm = DIM_PERMUTATION;
        if (pts != null) {
            for (int d = 0; d < DIMS; d++) out[perm[d]] = pts[base + d] * INV_SCALE;
        } else {
            long addr = KdTreeUnsafe.ptsBaseAddr + (long) base * 2L;
            for (int d = 0; d < DIMS; d++) {
                out[perm[d]] = KdTreeUnsafe.UNSAFE.getShort(addr + (long) d * 2L) * INV_SCALE;
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
     * Squared L2 distance in i16 units (low 32 bits) packed with the node nav word (high 32 bits).
     * Overflow proof under the contest spec: only semantic dims 5 and 6 may use the -1 sentinel; all other
     * dims are clamped to [0, 1]. After quantization the worst squared sum is
     * 2 × 20000² + 12 × 10000² = 2.0B, below Integer.MAX_VALUE, so int accumulation is safe.
     * Per-accumulator max is ~0.5B (four lanes × ~12.5 % of total), comfortably within int.
     *
     * <p>Scalar implementation avoids the Vector API path. As of Temurin 25.0.3,
     * {@code DistKernelBench} measured the Vector API form at ~34 B/op steady-state
     * allocation (down from ~1124 B/op on JDK 21) with wall-time identical to scalar
     * — C2 emits real VPMULLD/VPSUBW/VPMOVSXWD/VPHADDD but escape analysis still leaks
     * one or two wrappers per call, and the SIMD speedup is cancelled by setup +
     * horizontal-reduction overhead on a 14-dim kernel. At 700 visits × 900 RPS the
     * leak is ~21 MB/s sustained allocation → real GC pressure on the 65 MB heap.
     * Note: C2 does NOT auto-vectorize this scalar form either — verified via
     * PrintAssembly: 14 scalar IMULs, zero VPMADDWD/VPMULLD. The 4-lane accumulator
     * split below targets the scalar critical path (port-0 PMULLD chain on Haswell).
     *
     * <p><b>4-lane accumulator split:</b> the 14 squared-diffs are distributed round-robin
     * across {@code sumA/B/C/D}, then summed pairwise at the end. Breaks the single-register
     * dependency chain that was 14-deep on the IADD/PMULLD critical path — on Haswell (port 0
     * PMULLD, 5-cycle latency, 1/cycle throughput) the chain dominated. With 4 independent
     * accumulators the ROB can dispatch consecutive MULs to port 0 every cycle instead of
     * waiting for the previous sum. Final combine is one IADD-tree: {@code (A+B)+(C+D)}.
     *
     * <p>Returns a packed {@code long}: high 32 bits = nav word, low 32 bits = distance sum.
     * Keeping nav in a register (rather than writing then reading {@code scratch.lastNodeNav})
     * eliminates the store-load forwarding stall on the nav → splitDim → delta_i16 critical path.
     * In mmap mode the nav is already in a register as {@code (int)(tail8 >>> 32)};
     * in heap mode it shares the same 32-byte cache region as the feature shorts.
     */
    private long distSumI16AtBase(KdTreeScratch scratch, int base) {
        short[] q = scratch.permutedQueryI16;
        if (pts != null) {
            int sumA = 0, sumB = 0, sumC = 0, sumD = 0;
            int diff;
            diff = q[0]  - pts[base];      sumA += diff * diff;
            diff = q[1]  - pts[base + 1];  sumB += diff * diff;
            diff = q[2]  - pts[base + 2];  sumC += diff * diff;
            diff = q[3]  - pts[base + 3];  sumD += diff * diff;
            diff = q[4]  - pts[base + 4];  sumA += diff * diff;
            diff = q[5]  - pts[base + 5];  sumB += diff * diff;
            diff = q[6]  - pts[base + 6];  sumC += diff * diff;
            diff = q[7]  - pts[base + 7];  sumD += diff * diff;
            diff = q[8]  - pts[base + 8];  sumA += diff * diff;
            diff = q[9]  - pts[base + 9];  sumB += diff * diff;
            diff = q[10] - pts[base + 10]; sumC += diff * diff;
            diff = q[11] - pts[base + 11]; sumD += diff * diff;
            diff = q[12] - pts[base + 12]; sumA += diff * diff;
            diff = q[13] - pts[base + 13]; sumB += diff * diff;
            int sum = (sumA + sumB) + (sumC + sumD);
            int off = base + LANE_NAV;
            int nav = (pts[off] & 0xFFFF) | ((pts[off + 1] & 0xFFFF) << 16);
            return ((long) nav << 32) | Integer.toUnsignedLong(sum);
        } else {
            long nodeBase = KdTreeUnsafe.ptsBaseAddr + (long) base * 2L;
            // F10: read q[] as 4 longs via Unsafe.getLong (one 8-byte load per 4 i16 lanes)
            // instead of 14 individual short[] loads. Saves array-base-load + per-element
            // bounds-check sequences vs the q[d] loop form.
            long qBase = sun.misc.Unsafe.ARRAY_SHORT_BASE_OFFSET;
            long qw0 = KdTreeUnsafe.UNSAFE.getLong(q, qBase);
            long qw1 = KdTreeUnsafe.UNSAFE.getLong(q, qBase + 8);
            long qw2 = KdTreeUnsafe.UNSAFE.getLong(q, qBase + 16);
            long qw3 = KdTreeUnsafe.UNSAFE.getLong(q, qBase + 24);

            int sumA = 0, sumB = 0, sumC = 0, sumD = 0;
            int diff;

            long w = KdTreeUnsafe.UNSAFE.getLong(nodeBase);
            diff = (short) qw0 - (short) w;                       sumA += diff * diff;
            diff = (short) (qw0 >>> 16) - (short) (w >>> 16);     sumB += diff * diff;
            diff = (short) (qw0 >>> 32) - (short) (w >>> 32);     sumC += diff * diff;
            diff = (short) (qw0 >>> 48) - (short) (w >>> 48);     sumD += diff * diff;

            w = KdTreeUnsafe.UNSAFE.getLong(nodeBase + 8L);
            diff = (short) qw1 - (short) w;                       sumA += diff * diff;
            diff = (short) (qw1 >>> 16) - (short) (w >>> 16);     sumB += diff * diff;
            diff = (short) (qw1 >>> 32) - (short) (w >>> 32);     sumC += diff * diff;
            diff = (short) (qw1 >>> 48) - (short) (w >>> 48);     sumD += diff * diff;

            w = KdTreeUnsafe.UNSAFE.getLong(nodeBase + 16L);
            diff = (short) qw2 - (short) w;                       sumA += diff * diff;
            diff = (short) (qw2 >>> 16) - (short) (w >>> 16);     sumB += diff * diff;
            diff = (short) (qw2 >>> 32) - (short) (w >>> 32);     sumC += diff * diff;
            diff = (short) (qw2 >>> 48) - (short) (w >>> 48);     sumD += diff * diff;

            // getLong at +24 captures dims 12-13 (low 32 bits) AND the nav word (high 32 bits).
            long tail8 = KdTreeUnsafe.UNSAFE.getLong(nodeBase + 24L);
            diff = (short) qw3 - (short) tail8;                   sumA += diff * diff;
            diff = (short) (qw3 >>> 16) - (short) (tail8 >>> 16); sumB += diff * diff;
            int sum = (sumA + sumB) + (sumC + sumD);
            // nav is already in a register; combine with sum without a memory round-trip.
            return (tail8 & 0xFFFFFFFF00000000L) | Integer.toUnsignedLong(sum);
        }
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

    /**
     * Bbox lower bound as an i16 squared-sum, without thresholding. Computed at BBF push time
     * to drive dual-bound heap ordering (Arya et al. 1998): the resulting value is strictly ≥
     * the slab-LB used to be the heap key, so ordering by it pops tighter-bound candidates
     * first. Layout: see {@link #bboxPrunesI16Sum} — same access pattern, but returns the sum
     * instead of a boolean.
     */
    private int bboxLowerBoundI16Sum(KdTreeScratch scratch, int slot) {
        int base = slot * STRIDE_BBOX;
        short[] q = scratch.permutedQueryI16;
        short[] bb = topBbox;
        int sum = 0;
        for (int d = 0; d < DIMS; d++) {
            int clamped = Math.max(bb[base + d] - q[d], Math.max(q[d] - bb[base + 16 + d], 0));
            sum += clamped * clamped;
        }
        return sum;
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
        if (partLo >= thresholdSum) {
            if (PROFILING_ENABLED) scratch.bboxPrunesLo++;
            return true;
        }
        int partHi = 0;
        for (int d = 8; d < DIMS; d++) {
            int clamped = Math.max(bb[base + d] - q[d], Math.max(q[d] - bb[base + 16 + d], 0));
            partHi += clamped * clamped;
        }
        return partLo + partHi >= thresholdSum;
    }

    private static int thresholdSum(int peekSum, int visits, boolean exact) {
        if (exact || !RELAX_INITIALLY_ENABLED) return peekSum;
        // Integer Q16 path avoids the int→float→int round-trip and the float divide; also avoids the
        // 24-bit mantissa precision loss for peekSum > ~16.7M (peekSum can reach ~2 × 10⁹).
        return (int) (((long) peekSum * KdTreeTuning.relaxScaleQ16(visits)) >>> 16);
    }

    /**
     * G4: Software prefetch of the pts mmap line for {@code treeIdx}, issued as an
     * {@code Unsafe.getByte} whose result is XOR'd into a static sink so C2 can't elide it.
     * The byte read brings the full 32-byte node (one cache line in the 64-byte-line layout)
     * into L1 ahead of the next iteration's distSumI16AtBase. Single instruction issue, no
     * stall, no recall side-effect — it's just a load whose return we throw away.
     * <p>Only meaningful in mmap mode (heap mode pts is JVM-managed and accessed via
     * regular array load that the compiler can speculate). Caller checks {@code treeIdx >= 0}.
     */
    private void prefetchPtsLine(int treeIdx) {
        if (KdTreeUnsafe.ptsBaseAddr == 0L) return;
        // The XOR-into-sink prevents C2 from dropping the load if it sees no use.
        PREFETCH_SINK ^= KdTreeUnsafe.UNSAFE.getByte(
                KdTreeUnsafe.ptsBaseAddr + (long) treeIdx * STRIDE * 2L);
    }

    /** DCE-defeat sink for {@link #prefetchPtsLine}. Volatile-free: a single writer. */
    private static byte PREFETCH_SINK;

    /**
     * Bucketed threshold cache: returns the threshold computed at the first visit of the current
     * bucket ({@code visits >>> 4}). Within a bucket the relax-scale changes by ~28/65536 ≈ 0.04 %,
     * so the cached threshold is at most that much looser than the per-visit truth — recall-safe
     * (looser threshold = less pruning = at-or-above-true recall). Cuts the per-visit threshold
     * recompute (~5 ns) by ~93 % on the hot path, with re-evaluation only on bucket cross or
     * peekSum change.
     */
    private static int thresholdSumCached(int peekSum, int visits, boolean exact, KdTreeScratch scratch) {
        if (exact || !RELAX_INITIALLY_ENABLED) return peekSum;
        int bucket = visits >>> 4;
        if (peekSum == scratch.cachedThresholdPeekSum && bucket == scratch.cachedThresholdBucket) {
            return scratch.cachedThresholdSum;
        }
        int t = (int) (((long) peekSum * KdTreeTuning.relaxScaleQ16(visits)) >>> 16);
        scratch.cachedThresholdPeekSum = peekSum;
        scratch.cachedThresholdBucket = bucket;
        scratch.cachedThresholdSum = t;
        return t;
    }

    private static boolean visitBudgetExhausted(KdTreeScratch scratch, boolean exact) {
        return !exact && MAX_VISITS_ENABLED && scratch.visits >= MAX_VISITS_BUDGET;
    }

    private static boolean earlyDone(TopKSortedArray topK, int k, boolean exact) {
        return !exact && EARLY_DONE_ENABLED && topK.size() >= k && topK.peekSum() <= EARLY_DONE_SUM;
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
        if (shouldRefineBoundary(fraudCount, scratch)) {
            int approxVisits = scratch.visits;
            int approxBboxChecks = scratch.bboxChecks;
            if (PROFILING_ENABLED) {
                int approxBbfPushes = scratch.bbfPushes;
                int approxBbfHeapMax = scratch.bbfHeapMax;
                int approxSlabPrunes = scratch.slabPrunes;
                int approxBboxPrunes = scratch.bboxPrunes;
                int approxBboxPrunesLo = scratch.bboxPrunesLo;
                int approxTopKFilledAt = scratch.topKFilledAt;
                int approxTopKReplaced = scratch.topKReplaced;
                fraudCount = countFraudsInTop5(query, scratch, true);
                mergeRefineCounters(scratch, approxVisits, approxBboxChecks, approxBbfPushes,
                        approxBbfHeapMax, approxSlabPrunes, approxBboxPrunes, approxBboxPrunesLo,
                        approxTopKFilledAt, approxTopKReplaced);
            } else {
                fraudCount = countFraudsInTop5(query, scratch, true);
                mergeRefineWorkCounters(scratch, approxVisits, approxBboxChecks);
            }
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
        if (shouldRefineBoundary(fraudCount, scratch)) {
            int approxVisits = scratch.visits;
            int approxBboxChecks = scratch.bboxChecks;
            if (PROFILING_ENABLED) {
                int approxBbfPushes = scratch.bbfPushes;
                int approxBbfHeapMax = scratch.bbfHeapMax;
                int approxSlabPrunes = scratch.slabPrunes;
                int approxBboxPrunes = scratch.bboxPrunes;
                int approxBboxPrunesLo = scratch.bboxPrunesLo;
                int approxTopKFilledAt = scratch.topKFilledAt;
                int approxTopKReplaced = scratch.topKReplaced;
                fraudCount = countFraudsInTop5(query, scratch, true);
                mergeRefineCounters(scratch, approxVisits, approxBboxChecks, approxBbfPushes,
                        approxBbfHeapMax, approxSlabPrunes, approxBboxPrunes, approxBboxPrunesLo,
                        approxTopKFilledAt, approxTopKReplaced);
            } else {
                fraudCount = countFraudsInTop5(query, scratch, true);
                mergeRefineWorkCounters(scratch, approxVisits, approxBboxChecks);
            }
        }
        return fraudCount;
    }

    private static boolean shouldRefineBoundary(int fraudCount, KdTreeScratch scratch) {
        return (RELAX_INITIALLY_ENABLED || MAX_VISITS_ENABLED || EARLY_DONE_ENABLED) && REFINE_BOUNDARY
                && scratch.approximateStopped
                && (fraudCount == 2 || fraudCount == 3);
    }

    private static void mergeRefineWorkCounters(KdTreeScratch scratch, int approxVisits, int approxBboxChecks) {
        scratch.visits += approxVisits;
        scratch.bboxChecks += approxBboxChecks;
        scratch.approximateStopped = true;
    }

    private static void mergeRefineCounters(KdTreeScratch scratch, int approxVisits, int approxBboxChecks,
                                            int approxBbfPushes, int approxBbfHeapMax,
                                            int approxSlabPrunes, int approxBboxPrunes, int approxBboxPrunesLo,
                                            int approxTopKFilledAt, int approxTopKReplaced) {
        mergeRefineWorkCounters(scratch, approxVisits, approxBboxChecks);
        if (PROFILING_ENABLED) {
            int exactTopKFilledAt = scratch.topKFilledAt;
            scratch.bbfPushes += approxBbfPushes;
            scratch.bbfHeapMax = Math.max(approxBbfHeapMax, scratch.bbfHeapMax);
            scratch.slabPrunes += approxSlabPrunes;
            scratch.bboxPrunes += approxBboxPrunes;
            scratch.bboxPrunesLo += approxBboxPrunesLo;
            scratch.topKReplaced += approxTopKReplaced;
            if (approxTopKFilledAt != 0) {
                scratch.topKFilledAt = approxTopKFilledAt;
            } else if (exactTopKFilledAt != 0) {
                scratch.topKFilledAt = approxVisits + exactTopKFilledAt;
            }
        }
    }

    private int countFraudsInTop5(float[] query, KdTreeScratch scratch, boolean exact) {
        prepareSearch(query, scratch);
        prime(scratch, TopKSortedArray.MAX_K);
        if (earlyDone(scratch.results, TopKSortedArray.MAX_K, exact)) {
            scratch.approximateStopped = true;
        } else {
            descendBBF(scratch, TopKSortedArray.MAX_K, exact);
        }
        if (PROFILING_ENABLED) scratch.finalPeekDist = scratch.results.peekDist();
        return ptsSeg != null
                ? scratch.results.countFraudsFromMmap(KdTreeUnsafe.ptsBaseAddr)
                : scratch.results.countFrauds(fraud);
    }

    @Override
    public int countFraudsInTopK(float[] query, int k) {
        KdTreeScratch scratch = KdTreeScratch.scratchTL.get();
        scratch.results.ensureCapacity(k);
        prepareSearch(query, scratch);
        prime(scratch, k);
        descendBBF(scratch, k, false);
        if (PROFILING_ENABLED) scratch.finalPeekDist = scratch.results.peekDist();
        int[] treeIdxs = scratch.topKBuf.length >= k ? scratch.topKBuf : (scratch.topKBuf = new int[k]);
        int count = scratch.results.drainAscending(treeIdxs);
        int limit = Math.min(k, count);
        int fraudCount = 0;
        if (ptsSeg != null) {
            long base = KdTreeUnsafe.ptsBaseAddr;
            final long pointStrideBytes = STRIDE * 2L;
            final long navByteOffset = LANE_NAV * 2L;
            for (int i = 0; i < limit; i++) {
                int nav = KdTreeUnsafe.UNSAFE.getInt(base + (long) treeIdxs[i] * pointStrideBytes + navByteOffset);
                fraudCount += KdTreeLayout.unpackFraud(nav);
            }
        } else {
            for (int i = 0; i < limit; i++) {
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
        scratch.approximateStopped = false;
        // G2: invalidate thresholdSum cache. peekSum=-1 is unreachable (peekSum is non-negative).
        scratch.cachedThresholdPeekSum = -1;
        scratch.cachedThresholdBucket = -1;
        if (PROFILING_ENABLED) {
            scratch.bbfPushes     = 0;
            scratch.bbfHeapMax    = 0;
            scratch.slabPrunes    = 0;
            scratch.bboxPrunes    = 0;
            scratch.bboxPrunesLo  = 0;
            scratch.topKFilledAt  = 0;
            scratch.topKReplaced  = 0;
            scratch.finalPeekDist = 0f;
        }
        short[] pqi = scratch.permutedQueryI16;
        // Keep this unrolled order in sync with KdTreeLayout.DIM_PERMUTATION.
        pqi[0] = quantizeQuery(query[6]);
        pqi[1] = quantizeQuery(query[1]);
        pqi[2] = quantizeQuery(query[9]);
        pqi[3] = quantizeQuery(query[5]);
        pqi[4] = quantizeQuery(query[11]);
        pqi[5] = quantizeQuery(query[13]);
        pqi[6] = quantizeQuery(query[2]);
        pqi[7] = quantizeQuery(query[7]);
        pqi[8] = quantizeQuery(query[0]);
        pqi[9] = quantizeQuery(query[10]);
        pqi[10] = quantizeQuery(query[3]);
        pqi[11] = quantizeQuery(query[8]);
        pqi[12] = quantizeQuery(query[12]);
        pqi[13] = quantizeQuery(query[4]);
        // Lanes 14-15 (LANE_NAV) are zero-initialized at allocation and never written by the
        // quantizer, so no explicit zero-write needed here.
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
        long distNav = distSumI16AtBase(scratch, nodeBase);
        int dist = (int) distNav;
        if (topK.size() < k) {
            topK.push(treeIdx, dist);
            if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
        } else if (dist < topK.peekSum()) {
            topK.replaceFarthest(treeIdx, dist);
            if (PROFILING_ENABLED) scratch.topKReplaced++;
        }
        int nav = (int) (distNav >>> 32);
        int leftIdx = unpackLeft(treeIdx, nav);
        int rightIdx = KdTreeLayout.unpackRight(nav);
        if (depth < PRIME_FANOUT_DEPTH) {
            primeRecurse(leftIdx, scratch, k, depth + 1);
            primeRecurse(rightIdx, scratch, k, depth + 1);
        } else {
            int splitDim = unpackDim(nav);
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
            long distNav = distSumI16AtBase(scratch, nodeBase);
            int dist = (int) distNav;
            if (topK.size() < k) {
                topK.push(treeIdx, dist);
                if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
            } else if (dist < topK.peekSum()) {
                topK.replaceFarthest(treeIdx, dist);
                if (PROFILING_ENABLED) scratch.topKReplaced++;
            }
            int nav = (int) (distNav >>> 32);
            int splitDim = unpackDim(nav);
            int leftIdx = unpackLeft(treeIdx, nav);
            int rightIdx = KdTreeLayout.unpackRight(nav);
            int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
            int parentIdx = treeIdx;
            treeIdx = (delta_i16 < 0) ? leftIdx : rightIdx;
            // Skip the prefetch when the child is the sequential next address (already on the
            // same cache line / next-line that hardware prefetcher will fetch anyway).
            if (treeIdx >= 0 && treeIdx != parentIdx + 1) prefetchPtsLine(treeIdx);
        }
    }

    private void scanBucketLeaf(int treeIdx, int subtreeEnd, KdTreeScratch scratch, int k) {
        TopKSortedArray topK = scratch.results;
        int end = Math.min(subtreeEnd, n);
        for (int ti = treeIdx; ti >= 0 && ti < end; ti++) {
            scratch.visits++;
            long distNav = distSumI16AtBase(scratch, ti * STRIDE);
            int dist = (int) distNav;
            if (topK.size() < k) {
                if (!topK.contains(ti)) {
                    topK.push(ti, dist);
                    if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
                }
            } else if (dist < topK.peekSum() && !topK.contains(ti)) {
                topK.replaceFarthest(ti, dist);
                if (PROFILING_ENABLED) scratch.topKReplaced++;
            }
        }
    }

    private static boolean shouldScanBucketLeaf(int treeIdx, int subtreeEnd, int depth) {
        return depth >= BUCKET_LEAF_DEPTH && subtreeEnd - treeIdx <= BUCKET_LEAF_MAX_NODES;
    }

    // -------------------------------------------------------------------------
    // Descend — classical DFS fallback for deep FAR nodes
    // -------------------------------------------------------------------------

    void descend(int treeIdx, int subtreeEnd, int depth,
                 KdTreeScratch scratch, int k, int slabSum, boolean exact) {
        if (treeIdx < 0) return;
        if (visitBudgetExhausted(scratch, exact)) {
            scratch.approximateStopped = true;
            return;
        }
        if (shouldScanBucketLeaf(treeIdx, subtreeEnd, depth)) {
            scanBucketLeaf(treeIdx, subtreeEnd, scratch, k);
            return;
        }
        TopKSortedArray topK = scratch.results;
        int nodeBase = treeIdx * STRIDE;
        if (topK.size() >= k) {
            int peekSum = topK.peekSum();
            int threshSum = thresholdSumCached(peekSum, scratch.visits, exact, scratch);
            if (slabSum > threshSum) {
                if (PROFILING_ENABLED) scratch.slabPrunes++;
                return;
            }
            // hasBbox bit replaces `topSlot[treeIdx] >= 0` — skips the 12 MB topSlot probe
            // entirely for deep nodes (depth > TOP_BBOX_DEPTH).
            int navEarly = navAtBase(nodeBase);
            if ((navEarly & KdTreeLayout.HAS_BBOX_MASK) != 0) {
                int slot = topSlot[treeIdx];
                scratch.bboxChecks++;
                if (bboxPrunesI16Sum(scratch, slot, threshSum)) {
                    if (PROFILING_ENABLED) scratch.bboxPrunes++;
                    return;
                }
            }
        }
        scratch.visits++;
        long distNav = distSumI16AtBase(scratch, nodeBase);
        int dist = (int) distNav;
        if (topK.size() < k) {
            if (!topK.contains(treeIdx)) {
                topK.push(treeIdx, dist);
                if (PROFILING_ENABLED && topK.size() == k) scratch.topKFilledAt = scratch.visits;
            }
        } else if (dist < topK.peekSum() && !topK.contains(treeIdx)) {
            topK.replaceFarthest(treeIdx, dist);
            if (PROFILING_ENABLED) scratch.topKReplaced++;
        }
        int nav = (int) (distNav >>> 32);
        int splitDim = unpackDim(nav);
        int leftIdx = unpackLeft(treeIdx, nav);
        int rightIdx = KdTreeLayout.unpackRight(nav);
        int leftEnd = rightIdx >= 0 ? rightIdx : subtreeEnd;
        int rightEnd = subtreeEnd;
        int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
        int near, far, nearEnd, farEnd;
        if (delta_i16 < 0) {
            near = leftIdx;
            far = rightIdx;
            nearEnd = leftEnd;
            farEnd = rightEnd;
        } else {
            near = rightIdx;
            far = leftIdx;
            nearEnd = rightEnd;
            farEnd = leftEnd;
        }

        descend(near, nearEnd, depth + 1, scratch, k, slabSum, exact);

        int[] slab = scratch.slab;
        int oldSlabD = slab[splitDim];
        int newSlabD = delta_i16 * delta_i16;
        int newSlabSum = slabSum - oldSlabD + newSlabD;
        if (topK.size() < k || newSlabSum <= thresholdSumCached(topK.peekSum(), scratch.visits, exact, scratch)) {
            slab[splitDim] = newSlabD;
            descend(far, farEnd, depth + 1, scratch, k, newSlabSum, exact);
            slab[splitDim] = oldSlabD;
        }
    }

    // -------------------------------------------------------------------------
    // Best-first BBF
    // -------------------------------------------------------------------------

    void descendBBF(KdTreeScratch scratch, int k, boolean exact) {
        continueDfsBBF(0, n, 0, 0, scratch, k, exact);
        // Heap order: slabSum (FIFO tiebreak preserved). bboxLb is recorded in the parallel
        // array at push time and used ONLY to drive the push-time early-reject; the pop order
        // is unchanged from baseline. Heap-key-on-bboxLb was tried and broke approx-mode
        // recall (fn=1 on the 54100 fixture) because visit-count trajectories diverged per
        // subtree, shifting relax thresholds. See notes/2026-05-26-T2-dual-bound-bbf.md.
        long[] hEntry  = scratch.bbfHeap;
        int[] hEnd = scratch.bbfEnd;
        byte[] hDepth = scratch.bbfDepth;
        int[]  pool = scratch.bbfSlabPool;
        TopKSortedArray topK = scratch.results;
        while (scratch.bbfSize > 0) {
            if (earlyDone(topK, k, exact)) {
                scratch.approximateStopped = true;
                break;
            }
            if (visitBudgetExhausted(scratch, exact)) {
                scratch.approximateStopped = true;
                break;
            }
            long popEntry = hEntry[0];
            int popTi = KdTreeScratch.bbfTreeIdx(popEntry);
            int popSs = KdTreeScratch.bbfSlabSum(popEntry);
            int popSi = KdTreeScratch.bbfSlabIdx(popEntry);
            int popEnd = hEnd[0];
            int popDepth = hDepth[0] & 0xFF;
            int lastIdx = --scratch.bbfSize;
            if (lastIdx > 0) {
                long entry = hEntry[lastIdx];
                int entrySs = KdTreeScratch.bbfSlabSum(entry);
                int entryEnd = hEnd[lastIdx];
                byte entryDepth = hDepth[lastIdx];
                int i = 0, half = lastIdx >>> 1;
                while (i < half) {
                    int child = (i << 1) + 1;
                    int right = child + 1;
                    long leftE = hEntry[child];
                    int leftSs = KdTreeScratch.bbfSlabSum(leftE);
                    if (right < lastIdx) {
                        long rightE = hEntry[right];
                        int rightSs = KdTreeScratch.bbfSlabSum(rightE);
                        if (rightSs < leftSs) { child = right; leftE = rightE; leftSs = rightSs; }
                    }
                    if (entrySs <= leftSs) break;
                    hEntry[i]   = leftE;
                    hEnd[i] = hEnd[child];
                    hDepth[i] = hDepth[child];
                    i = child;
                }
                hEntry[i]   = entry;
                hEnd[i] = entryEnd;
                hDepth[i] = entryDepth;
            }
            if (topK.size() >= k && popSs > thresholdSumCached(topK.peekSum(), scratch.visits, exact, scratch)) continue;
            System.arraycopy(pool, popSi * DIMS, scratch.slab, 0, DIMS);
            // G4: warm L1 with popTi's pts line before continueDfsBBF's first iteration touches
            // it via distSumI16AtBase. On contest's 3 MB L3 + 96 MB mmap pts, hides DRAM-miss
            // latency for the popped subtree root.
            prefetchPtsLine(popTi);
            continueDfsBBF(popTi, popEnd, popDepth, popSs, scratch, k, exact);
        }
    }

    /**
     * Continues a DFS from {@code treeIdx} along the near path, pushing far subtrees onto the BBF
     * min-heap ordered by slab distance.
     *
     * <p>Invariants assumed on entry (both initial call and heap pop):
     * <ul>
     *   <li>{@code topK.size() == k} — prime always fills the result set before BBF starts, so the
     *       "not full" branch from the old code is unreachable and has been removed.</li>
     *   <li>{@code topSlot[treeIdx] >= 0} iff {@code depth <= TOP_BBOX_DEPTH} — KdTreeBuilder
     *       initialises all topSlot entries to -1 and only writes non-negative slots for shallow nodes,
     *       so the old {@code depth <= TOP_BBOX_DEPTH} guard is redundant and has been removed.</li>
     * </ul>
     */
    private void continueDfsBBF(int treeIdx, int subtreeEnd, int depth, int slabSum,
                                KdTreeScratch scratch, int k, boolean exact) {
        TopKSortedArray topK = scratch.results;
        int[]  slab     = scratch.slab;
        long[] hEntry   = scratch.bbfHeap;
        int[] hEnd      = scratch.bbfEnd;
        byte[] hDepth   = scratch.bbfDepth;
        int[]  pool     = scratch.bbfSlabPool;
        // Cache thresholdSum across iterations. peekSum only changes when topK is replaced;
        // when relax is on, threshold also changes per visit-bucket (visits >>> 4). Skip
        // the thresholdSumCached call when neither input has changed.
        boolean relaxEnabled = !exact && RELAX_INITIALLY_ENABLED;
        int localPeekSum = topK.peekSum();
        int localBucket = scratch.visits >>> 4;
        int localThresholdSum = relaxEnabled
                ? thresholdSumCached(localPeekSum, scratch.visits, exact, scratch)
                : localPeekSum;
        while (treeIdx >= 0) {
            if (visitBudgetExhausted(scratch, exact)) {
                scratch.approximateStopped = true;
                return;
            }
            if (shouldScanBucketLeaf(treeIdx, subtreeEnd, depth)) {
                scanBucketLeaf(treeIdx, subtreeEnd, scratch, k);
                return;
            }
            // topK is always full here (prime guarantees size == k before BBF is called).
            int peekSum = topK.peekSum();
            if (peekSum != localPeekSum || (relaxEnabled && (scratch.visits >>> 4) != localBucket)) {
                localPeekSum = peekSum;
                localBucket = scratch.visits >>> 4;
                localThresholdSum = relaxEnabled
                        ? (int) (((long) localPeekSum * KdTreeTuning.relaxScaleQ16(scratch.visits)) >>> 16)
                        : localPeekSum;
            }
            if (slabSum > localThresholdSum) {
                if (PROFILING_ENABLED) scratch.slabPrunes++;
                return;
            }
            // Pre-load nav so we can gate the topSlot[] probe on the hasBbox bit. The pts mmap
            // line at nodeBase+24..31 holds both dims 12-13 and the nav word; reading nav here
            // brings that line into L1 ahead of distSumI16AtBase's final getLong (cheap prefetch).
            int nodeBase = treeIdx * STRIDE;
            int navEarly = navAtBase(nodeBase);
            if ((navEarly & KdTreeLayout.HAS_BBOX_MASK) != 0) {
                int slot = topSlot[treeIdx];
                scratch.bboxChecks++;
                if (bboxPrunesI16Sum(scratch, slot, localThresholdSum)) {
                    if (PROFILING_ENABLED) scratch.bboxPrunes++;
                    return;
                }
            }
            scratch.visits++;
            long distNav = distSumI16AtBase(scratch, nodeBase);
            int dist = (int) distNav;
            if (dist < localPeekSum && !topK.contains(treeIdx)) {
                topK.replaceFarthest(treeIdx, dist);
                localPeekSum = topK.peekSum();
                localBucket = scratch.visits >>> 4;
                localThresholdSum = relaxEnabled
                        ? (int) (((long) localPeekSum * KdTreeTuning.relaxScaleQ16(scratch.visits)) >>> 16)
                        : localPeekSum;
                if (PROFILING_ENABLED) scratch.topKReplaced++;
            }
            int nav = (int) (distNav >>> 32);
            int splitDim = unpackDim(nav);
            int leftIdx = unpackLeft(treeIdx, nav);
            int rightIdx = KdTreeLayout.unpackRight(nav);
            int leftEnd = rightIdx >= 0 ? rightIdx : subtreeEnd;
            int rightEnd = subtreeEnd;
            int delta_i16 = scratch.permutedQueryI16[splitDim] - ptAtI16Base(nodeBase, splitDim);
            int near, far, nearEnd, farEnd;
            if (delta_i16 < 0) {
                near = leftIdx;
                far = rightIdx;
                nearEnd = leftEnd;
                farEnd = rightEnd;
            } else {
                near = rightIdx;
                far = leftIdx;
                nearEnd = rightEnd;
                farEnd = leftEnd;
            }

            if (far >= 0) {
                int oldSlabD = slab[splitDim];
                int newSlabD = delta_i16 * delta_i16;
                int newSlabSum = slabSum - oldSlabD + newSlabD;
                if (newSlabSum <= localThresholdSum) {
                    // childrenHaveBbox bit (set in parent's nav when depth < TOP_BBOX_DEPTH) is the
                    // cache-resident equivalent of `topSlot[far] >= 0` — replaces a random probe
                    // into the 12 MB topSlot[] array with a single bit-test on the nav we already
                    // loaded via distSumI16AtBase.
                    if ((nav & KdTreeLayout.CHILDREN_HAVE_BBOX_MASK) != 0
                            && scratch.bbfSize < KdTreeScratch.BBF_HEAP_CAP
                            && scratch.bbfSlabNext < KdTreeScratch.BBF_POOL_CAP) {
                        // T-2: push-time bbox early-reject. Compute the actual bbox-LB now;
                        // if it already exceeds threshold the subtree would be pruned at
                        // descend time anyway (line 928), so skip the push entirely. Threshold
                        // monotonically tightens (topK only gets closer, relax only sharpens
                        // with visits), so this is recall-safe by construction in both exact
                        // and approx modes — see notes/2026-05-26-T2-dual-bound-bbf.md.
                        if (bboxLowerBoundI16Sum(scratch, topSlot[far]) > localThresholdSum) {
                            if (PROFILING_ENABLED) scratch.bboxPrunes++;
                        } else {
                            int newSlabIdx = scratch.bbfSlabNext++;
                            int poolOff = newSlabIdx * DIMS;
                            System.arraycopy(slab, 0, pool, poolOff, DIMS);
                            pool[poolOff + splitDim] = newSlabD;
                            int i = scratch.bbfSize++;
                            if (PROFILING_ENABLED) {
                                scratch.bbfPushes++;
                                if (scratch.bbfSize > scratch.bbfHeapMax) scratch.bbfHeapMax = scratch.bbfSize;
                            }
                            long newEntry = KdTreeScratch.packBbfEntry(newSlabSum, newSlabIdx, far);
                            while (i > 0) {
                                int parent = (i - 1) >>> 1;
                                long parentEntry = hEntry[parent];
                                if (KdTreeScratch.bbfSlabSum(parentEntry) <= newSlabSum) break;
                                hEntry[i]   = parentEntry;
                                hEnd[i] = hEnd[parent];
                                hDepth[i] = hDepth[parent];
                                i = parent;
                            }
                            hEntry[i]   = newEntry;
                            hEnd[i] = farEnd;
                            hDepth[i] = (byte) (depth + 1);
                        }
                    } else {
                        slab[splitDim] = newSlabD;
                        descend(far, farEnd, depth + 1, scratch, k, newSlabSum, exact);
                        slab[splitDim] = oldSlabD;
                        // descend can mutate visits and peekSum; invalidate the cache.
                        localPeekSum = topK.peekSum();
                        localBucket = scratch.visits >>> 4;
                        localThresholdSum = relaxEnabled
                                ? (int) (((long) localPeekSum * KdTreeTuning.relaxScaleQ16(scratch.visits)) >>> 16)
                                : localPeekSum;
                    }
                }
            }
            int parentIdx = treeIdx;
            treeIdx = near;
            subtreeEnd = nearEnd;
            depth++;
            // G4: prefetch near's pts line. The dependent use is the next iteration's
            // distSumI16AtBase ~30-50 cycles down the loop, plenty of time for a DRAM miss to
            // resolve. On contest hardware (3 MB L3, LPDDR3-1600 ~150 ns/miss) this hides the
            // pts-load latency for most BBF visits. Skip when child is parent+1 (already
            // adjacent in tree-layout — hardware prefetcher covers it).
            if (treeIdx >= 0 && treeIdx != parentIdx + 1) prefetchPtsLine(treeIdx);
        }
    }

}
