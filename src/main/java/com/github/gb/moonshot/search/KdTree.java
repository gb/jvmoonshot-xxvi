package com.github.gb.moonshot.search;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;

/**
 * Exact k-NN via balanced KD-tree with hybrid best-first branch-and-bound (BBF at depth ≤ {@link #BBF_MAX_DEPTH},
 * DFS below) + slab/bbox pruning. Distance kernel is AVX2 FloatVector over a stride-16 layout; nav fields
 * (leftAndDim, right) ride in lanes 14, 15 of {@code pts} so each descend visit reads one 64-byte cache line.
 * Production loads pts/origId/topBbox via mmap to keep them off the cgroup heap.
 */
public final class KdTree implements VectorIndex {

    public static final int DIMS = 14;
    public static final int STRIDE = 16;

    /**
     * Variance-descending lane order. Drives {@link KdTreeBuilder#chooseSplitDim}'s tie-break toward the
     * highest-discrimination dim, yielding a more balanced tree and fewer visits per query.
     */
    public static final int[] DIM_PERMUTATION = { 6, 10, 9, 5, 11, 2, 4, 7, 0, 1, 3, 8, 12, 13 };

    public static final int TOP_BBOX_DEPTH = 18;

    /**
     * Best-first heap only covers FAR pushes at depths ≤ BBF_MAX_DEPTH. Below, FARs use DFS recursion (call descend).
     * Most of the re-ordering benefit comes from shallow interleaving where peekDist is still being tightened; deep
     * splits have slabSums too close to each other for reordering to matter much. Caps heap size.
     */
    static final int BBF_MAX_DEPTH = 18;

    static final int LANE_LEFT_DIM = 14;
    static final int LANE_RIGHT = 15;

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;

    /**
     * Masked LOAD (not masked reduce): lanes 6, 7 receive 0.0f directly. The packed int-bits we store there
     * reinterpret as subnormal floats; letting them enter FP arithmetic triggers x86 microcode assists that
     * dominate the hot path.
     */
    private static final VectorMask<Float> TAIL_MASK = VectorMask.fromLong(SPECIES, 0b00111111L);

    /** Pre-materialized so {@code .max(ZERO)} folds to a register-resident vxorps instead of a per-call broadcast. */
    private static final FloatVector ZERO = FloatVector.zero(SPECIES);

    /**
     * Hard cap on per-query node visits. Default Integer.MAX_VALUE (disabled).
     * Set {@code KDTREE_MAX_VISITS=N} only as a last-resort safety net; prefer
     * {@link #RELAX_MAX_EPSILON} which reduces visits without hard accuracy cuts.
     * <p>{@code MAX_VISITS_ENABLED} is {@code static final}: C2 folds the guard in {@link #descend}
     * to dead code when the env var is absent, eliminating the per-visit field read.
     */
    static int MAX_VISITS_BUDGET;
    static final boolean MAX_VISITS_ENABLED;
    static {
        String value = System.getenv("KDTREE_MAX_VISITS");
        int parsed = Integer.MAX_VALUE;
        if (value != null && !value.isEmpty()) {
            try { parsed = Integer.parseInt(value); }
            catch (NumberFormatException ignored) { }
        }
        MAX_VISITS_BUDGET = Math.max(1, parsed);
        MAX_VISITS_ENABLED = MAX_VISITS_BUDGET != Integer.MAX_VALUE;
    }
    public static void setMaxVisitsBudget(int cap) { MAX_VISITS_BUDGET = cap; }

    /**
     * Epsilon-relaxed pruning: once {@code visits > RELAX_SOFT_CAP}, prune thresholds shrink
     * by a factor that ramps linearly from 0 to {@code RELAX_MAX_EPSILON} over {@code RELAX_RANGE}
     * additional visits.  A subtree pruned under relaxation may contain a neighbor whose true distance
     * is within {@code epsilon × peekDist} of the k-th current candidate — so the returned top-K is
     * approximately optimal, not exactly.  Default epsilon=0 (exact); set
     * {@code KDTREE_RELAX_EPSILON=0.01} to enable 1% relaxation.
     *
     * <p>SOFT_CAP should be below visits_p90 (so typical queries are unaffected).
     * RANGE determines how quickly full epsilon is reached above SOFT_CAP.
     */
    static int   RELAX_SOFT_CAP    = 1_500;
    static int   RELAX_RANGE       = 1_500;
    static float RELAX_MAX_EPSILON = 0f;
    /**
     * Captured once at class load from {@code KDTREE_RELAX_EPSILON}. Being {@code static final},
     * C2 constant-folds {@link #relaxScale}: when false (epsilon=0 at startup) the entire body
     * collapses to {@code return 1.0f} — zero overhead on the bench / exact-search path. When true
     * (production: epsilon=0.01), the dead branch is eliminated and only the relaxation arithmetic
     * remains. {@link #setRelaxParams} can still mutate {@link #RELAX_MAX_EPSILON} for experiment
     * sweeps; the explicit {@code RELAX_MAX_EPSILON == 0f} guard in {@link #relaxScale} handles the
     * sweep's epsilon=0 baseline correctly when {@code RELAX_INITIALLY_ENABLED} is true.
     * <p>{@code RELAX_SOFT_CAP_INIT} mirrors {@link #RELAX_SOFT_CAP} at class load as a
     * {@code static final} so C2 inlines the literal 1500 into {@link #relaxScale}'s fast exit,
     * eliminating the per-call mutable field read. {@link EpsilonRelaxSweep} sweeps only epsilon
     * (not softCap), so correctness is preserved for the standard {@code --soft-cap 1500} case.
     */
    static final boolean RELAX_INITIALLY_ENABLED;
    static final int     RELAX_SOFT_CAP_INIT;
    static {
        String sc = System.getenv("KDTREE_RELAX_SOFT_CAP");
        if (sc != null && !sc.isEmpty()) { try { RELAX_SOFT_CAP = Integer.parseInt(sc); } catch (NumberFormatException ignored) {} }
        String rng = System.getenv("KDTREE_RELAX_RANGE");
        if (rng != null && !rng.isEmpty()) { try { RELAX_RANGE = Math.max(1, Integer.parseInt(rng)); } catch (NumberFormatException ignored) {} }
        String eps = System.getenv("KDTREE_RELAX_EPSILON");
        if (eps != null && !eps.isEmpty()) { try { RELAX_MAX_EPSILON = Float.parseFloat(eps); } catch (NumberFormatException ignored) {} }
        RELAX_INITIALLY_ENABLED = RELAX_MAX_EPSILON != 0f;
        RELAX_SOFT_CAP_INIT     = RELAX_SOFT_CAP;
    }
    public static void setRelaxParams(int softCap, int range, float maxEpsilon) {
        RELAX_SOFT_CAP    = softCap;
        RELAX_RANGE       = Math.max(1, range);
        RELAX_MAX_EPSILON = maxEpsilon;
    }

    /**
     * Multiplicative scale for prune thresholds at a given visit count. Returns 1.0 when exact
     * (epsilon=0) or visits ≤ soft cap; < 1.0 when relaxing. {@code RELAX_INITIALLY_ENABLED} is
     * {@code static final}: C2 folds {@code !RELAX_INITIALLY_ENABLED} to dead code at class-load
     * epsilon=0, making every call a straight {@code return 1.0f} with no field loads.
     */
    static float relaxScale(int visits) {
        if (!RELAX_INITIALLY_ENABLED || visits <= RELAX_SOFT_CAP_INIT) return 1.0f;
        if (RELAX_MAX_EPSILON == 0f) return 1.0f;   // sweep baseline: setRelaxParams(_, _, 0f)
        float t = Math.min(1.0f, (float)(visits - RELAX_SOFT_CAP) / RELAX_RANGE);
        return 1.0f - RELAX_MAX_EPSILON * t;
    }

    /**
     * When true (default), {@link #primeSelectAndPlunge} ranks the {@link #PRIME_FANOUT_COUNT} fanout
     * endpoints by bbox distance and plunges only the closest {@link #PRIME_PLUNGE_CAP}.
     * When false, skip the bbox scoring and plunge the first PRIME_PLUNGE_CAP endpoints unsorted;
     * visits increase (worse initial peekDist) but the upfront cost of 32 topBbox mmap lookups is avoided.
     * Set {@code KDTREE_PRIME_BBOX=0} to disable.
     */
    public static boolean PRIME_BBOX_SCORING;
    static {
        String value = System.getenv("KDTREE_PRIME_BBOX");
        PRIME_BBOX_SCORING = value == null || !("0".equals(value) || "false".equalsIgnoreCase(value));
    }
    /** Bench-only switch. */
    public static void setPrimeBboxScoring(boolean enabled) { PRIME_BBOX_SCORING = enabled; }

    private static final ValueLayout.OfInt INT_LE =
        ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    final int n;

    // Heap mode (build/load) vs mmap mode (loadMmap): hot paths dispatch on whether the heap field is null.
    final float[] pts;
    final int[] origId;
    final byte[] fraud;

    final MappedByteBuffer ptsBuf;
    final FloatBuffer ptsFloats;
    final MemorySegment ptsSeg;

    final MappedByteBuffer origIdBuf;
    final MemorySegment origIdSeg;

    final int[] topSlot;
    final float[] topBbox;
    final MappedByteBuffer topBboxBuf;
    final MemorySegment topBboxSeg;
    final int topNodeCount;

    KdTree(int n,
           float[] pts, MappedByteBuffer ptsBuf, FloatBuffer ptsFloats,
           int[] origId, MappedByteBuffer origIdBuf,
           byte[] fraud,
           int[] topSlot, float[] topBbox, MappedByteBuffer topBboxBuf, int topNodeCount) {
        this.n = n;
        this.pts = pts; this.ptsBuf = ptsBuf; this.ptsFloats = ptsFloats;
        this.ptsSeg = (ptsBuf != null) ? MemorySegment.ofBuffer(ptsBuf) : null;
        this.origId = origId;
        this.origIdBuf = origIdBuf;
        this.origIdSeg = (origIdBuf != null) ? MemorySegment.ofBuffer(origIdBuf) : null;
        this.fraud = fraud;
        this.topSlot = topSlot;
        this.topBbox = topBbox;
        this.topBboxBuf = topBboxBuf;
        this.topBboxSeg = (topBboxBuf != null) ? MemorySegment.ofBuffer(topBboxBuf) : null;
        this.topNodeCount = topNodeCount;
        KdTreeUnsafe.bindPtsSegment(this.ptsSeg);
    }

    static int packLeftAndDim(int left, int dim) {
        return (left & 0x0FFFFFFF) | ((dim & 0xF) << 28);
    }

    int leftAndDimAt(int treeIdx) {
        if (pts != null) {
            return Float.floatToRawIntBits(pts[treeIdx * STRIDE + LANE_LEFT_DIM]);
        }
        return KdTreeUnsafe.UNSAFE.getInt(
            KdTreeUnsafe.ptsBaseAddr + ((long) treeIdx * STRIDE + LANE_LEFT_DIM) * 4L);
    }

    /** Sign-extend the 28-bit signed left index preserves the -1 absent-child sentinel. */
    static int unpackLeft(int leftAndDim) {
        return (leftAndDim << 4) >> 4;
    }

    private static int unpackDim(int leftAndDim) {
        return (leftAndDim >>> 28) & 0xF;
    }

    @Override
    public void applyMmapHints() {
        int ptsHugepageRc    = KdTreeMmap.madvise(ptsSeg, KdTreeMmap.MADV_HUGEPAGE);
        int ptsRandomRc      = KdTreeMmap.madvise(ptsSeg, KdTreeMmap.MADV_RANDOM);
        int origIdHugepageRc = (origIdSeg != null) ? KdTreeMmap.madvise(origIdSeg, KdTreeMmap.MADV_HUGEPAGE) : 0;
        int origIdRandomRc   = (origIdSeg != null) ? KdTreeMmap.madvise(origIdSeg, KdTreeMmap.MADV_RANDOM)   : 0;
        int topBboxHugepageRc = (topBboxSeg != null) ? KdTreeMmap.madvise(topBboxSeg, KdTreeMmap.MADV_HUGEPAGE) : 0;
        int topBboxRandomRc  = (topBboxSeg != null) ? KdTreeMmap.madvise(topBboxSeg, KdTreeMmap.MADV_RANDOM)   : 0;

        boolean ok = ptsHugepageRc == 0 && ptsRandomRc == 0
                  && origIdHugepageRc == 0 && origIdRandomRc == 0
                  && topBboxHugepageRc == 0 && topBboxRandomRc == 0;
        if (ok) {
            System.out.println("[kdtree] madvise: ok (" + KdTreeMmap.MADVISE_DIAG + ")");
        } else {
            System.out.println("[kdtree] madvise: lookup=" + KdTreeMmap.MADVISE_DIAG
                + " pts(hp=" + ptsHugepageRc + ",rd=" + ptsRandomRc + ")"
                + " origId(hp=" + origIdHugepageRc + ",rd=" + origIdRandomRc + ")"
                + " topBbox(hp=" + topBboxHugepageRc + ",rd=" + topBboxRandomRc + ")");
        }

    }

    @Override
    public void prewarm() {
        int sink = 0;
        if (ptsBuf != null) {
            sink ^= KdTreeMmap.touchPages(ptsBuf);
        }
        if (origIdBuf != null) {
            sink ^= KdTreeMmap.touchPages(origIdBuf);
        }
        if (topBboxBuf != null) {
            sink ^= KdTreeMmap.touchPages(topBboxBuf);
        }
        prewarmSink = sink;
    }

    @SuppressWarnings("unused")
    private static volatile int prewarmSink;

    @Override
    public int size() { return n; }

    private float ptAt(int treeIdx, int d) {
        return (pts != null) ? pts[treeIdx * STRIDE + d] : ptsFloats.get(treeIdx * STRIDE + d);
    }

    public void copyNodeVector(int treeIdx, float[] out) {
        for (int d = 0; d < DIMS; d++) out[d] = ptAt(treeIdx, d);
    }

    int rightAt(int treeIdx) {
        if (pts != null) {
            return Float.floatToRawIntBits(pts[treeIdx * STRIDE + LANE_RIGHT]);
        }
        return KdTreeUnsafe.UNSAFE.getInt(
            KdTreeUnsafe.ptsBaseAddr + ((long) treeIdx * STRIDE + LANE_RIGHT) * 4L);
    }

    int origIdAt(int treeIdx) {
        if (origId != null) return origId[treeIdx];
        return origIdSeg.get(INT_LE, (long) treeIdx * 4L);
    }

    /**
     * Joint-reduce over both 8-lane chunks, no early exit. Chunk 2's lanes 6, 7 cover the packed nav int-bits;
     * {@link #TAIL_MASK} zeros them at load time so they never enter FP arithmetic.
     */
    private float distSquared(Scratch scratch, int treeIdx) {
        int baseOffset = treeIdx << 4;
        FloatVector q0 = FloatVector.fromArray(SPECIES, scratch.permutedQuery, 0);
        FloatVector q1 = FloatVector.fromArray(SPECIES, scratch.permutedQuery, 8);
        FloatVector v0, v1;
        if (pts != null) {
            v0 = FloatVector.fromArray(SPECIES, pts, baseOffset);
            v1 = FloatVector.fromArray(SPECIES, pts, baseOffset + 8, TAIL_MASK);
        } else {
            long byteOff = (long) baseOffset * 4L;
            v0 = FloatVector.fromMemorySegment(SPECIES, ptsSeg, byteOff,         ByteOrder.LITTLE_ENDIAN);
            v1 = FloatVector.fromMemorySegment(SPECIES, ptsSeg, byteOff + 32L,   ByteOrder.LITTLE_ENDIAN, TAIL_MASK);
        }
        FloatVector d0 = q0.sub(v0);
        FloatVector d1 = q1.sub(v1);
        return d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
    }

    /** Production hot path skips the variable-k capacity checks and id-projection of {@link #countFraudsInTopK}. */
    @Override
    public int countFraudsInTop5(float[] query) {
        Scratch scratch = scratchTL.get();
        prepareSearch(query, scratch);
        float[] q = scratch.permutedQuery;
        prime(q, scratch, TopKSortedArray.MAX_K);
        descendBBF(q, scratch, TopKSortedArray.MAX_K);
        return scratch.results.countFrauds(fraud);
    }

    @Override
    public int countFraudsInTopK(float[] query, int k) {
        Scratch scratch = scratchTL.get();
        scratch.results.ensureCapacity(k);
        scratch.ensureTopKCapacity(k);
        prepareSearch(query, scratch);
        float[] q = scratch.permutedQuery;
        prime(q, scratch, k);
        descendBBF(q, scratch, k);
        int[] treeIdxs = scratch.topKBuf;
        int n = scratch.results.drainAscending(treeIdxs);
        int fraudCount = 0;
        int limit = Math.min(k, n);
        for (int i = 0; i < limit; i++) {
            if (fraud[treeIdxs[i]] != 0) fraudCount++;
        }
        return fraudCount;
    }


    static void prepareSearch(float[] query, Scratch scratch) {
        scratch.results.clear();
        Arrays.fill(scratch.slab, 0f);
        scratch.visits = 0;
        scratch.bboxChecks = 0;
        scratch.bbfSize = 0;
        scratch.bbfSlabNext = 0;
        permuteForSearch(query, scratch.permutedQuery);
    }

    private static void permuteForSearch(float[] semantic, float[] permuted) {
        for (int i = 0; i < DIMS; i++) {
            permuted[i] = semantic[DIM_PERMUTATION[i]];
        }
    }

    /**
     * At depths 0..FANOUT_DEPTH-1 visit both children; below, greedy-plunge. Fan-out spreads the pre-fill across
     * diverse subtrees so queries near multiple splitting walls don't pre-load far from the true top-K.
     */
    static final int PRIME_FANOUT_DEPTH = 5;

    public static final int PRIME_FANOUT_COUNT = 1 << PRIME_FANOUT_DEPTH;

    /**
     * Only the bbox-closest {@code PRIME_PLUNGE_CAP} of {@link #PRIME_FANOUT_COUNT} fan-out endpoints get plunged;
     * the rest would waste visits on bbox-far subtrees. Deterministic A/B confirms: plunge-from-all regresses
     * visits_p99 + total_work_p99 by a real margin. Do not remove.
     */
    static int PRIME_PLUNGE_CAP = 16;
    static {
        String value = System.getenv("KDTREE_PRIME_PLUNGE_CAP");
        if (value != null && !value.isEmpty()) {
            try { PRIME_PLUNGE_CAP = Math.max(1, Math.min(PRIME_FANOUT_COUNT, Integer.parseInt(value))); }
            catch (NumberFormatException ignored) { }
        }
    }
    public static void setPrimePlungeCap(int cap) { PRIME_PLUNGE_CAP = Math.max(1, Math.min(PRIME_FANOUT_COUNT, cap)); }

    void prime(float[] query, Scratch scratch, int k) {
        scratch.fanOutCount = 0;
        primeRecurse(query, 0, scratch, k, 0);
        primeSelectAndPlunge(query, scratch, k);
    }

    private void primeRecurse(float[] query, int treeIdx, Scratch scratch, int k, int depth) {
        if (treeIdx < 0) return;
        TopKSortedArray topK = scratch.results;
        scratch.visits++;
        float dist = distSquared(scratch, treeIdx);
        if (topK.size() < k) {
            topK.push(treeIdx, dist);
        } else if (dist < topK.peekDist()) {
            topK.replaceFarthest(treeIdx, dist);
        }
        int leftAndDim = leftAndDimAt(treeIdx);
        int leftIdx = unpackLeft(leftAndDim);
        int rightIdx = rightAt(treeIdx);
        if (depth < PRIME_FANOUT_DEPTH) {
            primeRecurse(query, leftIdx, scratch, k, depth + 1);
            primeRecurse(query, rightIdx, scratch, k, depth + 1);
        } else {
            int splitDim = unpackDim(leftAndDim);
            float ptVal = ptAt(treeIdx, splitDim);
            float delta = query[splitDim] - ptVal;
            int near = (delta < 0f) ? leftIdx : rightIdx;
            if (near >= 0 && scratch.fanOutCount < PRIME_FANOUT_COUNT) {
                scratch.fanOutBuf[scratch.fanOutCount++] = near;
            }
        }
    }

    private void primeSelectAndPlunge(float[] query, Scratch scratch, int k) {
        int count = scratch.fanOutCount;
        if (count == 0) return;

        int[] buf = scratch.fanOutBuf;
        int m = Math.min(count, PRIME_PLUNGE_CAP);

        if (PRIME_BBOX_SCORING) {
            float[] scores = scratch.fanOutScores;
            for (int i = 0; i < count; i++) {
                int slot = topSlot[buf[i]];
                scores[i] = (slot >= 0) ? bboxDistSquared(scratch, slot) : Float.POSITIVE_INFINITY;
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
                    int   tBuf   = buf[i];    buf[i]    = buf[minIdx];    buf[minIdx]    = tBuf;
                    float tScore = scores[i]; scores[i] = scores[minIdx]; scores[minIdx] = tScore;
                }
            }
        }

        for (int i = 0; i < m; i++) {
            plunge(query, buf[i], scratch, k);
        }
    }

    private void plunge(float[] query, int treeIdx, Scratch scratch, int k) {
        TopKSortedArray topK = scratch.results;
        while (treeIdx >= 0) {
            scratch.visits++;
            float dist = distSquared(scratch, treeIdx);
            if (topK.size() < k) {
                topK.push(treeIdx, dist);
            } else if (dist < topK.peekDist()) {
                topK.replaceFarthest(treeIdx, dist);
            }
            int leftAndDim = leftAndDimAt(treeIdx);
            int splitDim = unpackDim(leftAndDim);
            int leftIdx = unpackLeft(leftAndDim);
            int rightIdx = rightAt(treeIdx);
            float ptVal = ptAt(treeIdx, splitDim);
            float delta = query[splitDim] - ptVal;
            treeIdx = (delta < 0f) ? leftIdx : rightIdx;
        }
    }

    /**
     * Classical DFS descend with NEAR-FAR ordering — retained as the deep-FAR fallback path called from
     * {@link #continueDfsBBF} when a FAR push would exceed {@link #BBF_MAX_DEPTH}. Also the entry shape preserved
     * for A/B flips back to pure-DFS. {@code slab[d]} = squared wall-distance on dim d; sum over dims is the lower
     * bound on squared distance to anything in the subtree.
     */
    void descend(float[] query, int treeIdx, Scratch scratch, int k, float slabSum, int depth) {
        if (treeIdx < 0) return;
        if (MAX_VISITS_ENABLED && scratch.visits >= MAX_VISITS_BUDGET) return;
        TopKSortedArray topK = scratch.results;
        if (topK.size() >= k) {
            float thresh = topK.peekDist() * relaxScale(scratch.visits);
            if (slabSum >= thresh) return;

            if (depth <= TOP_BBOX_DEPTH) {
                int slot = topSlot[treeIdx];
                if (slot >= 0) {
                    scratch.bboxChecks++;
                    if (bboxDistSquared(scratch, slot) >= thresh) return;
                }
            }
        }

        scratch.visits++;
        float dist = distSquared(scratch, treeIdx);
        // contains() is a K-scan; only run when an actual push would otherwise happen.
        if (topK.size() < k) {
            if (!topK.contains(treeIdx)) topK.push(treeIdx, dist);
        } else if (dist < topK.peekDist() && !topK.contains(treeIdx)) {
            topK.replaceFarthest(treeIdx, dist);
        }

        int leftAndDim = leftAndDimAt(treeIdx);
        int splitDim = unpackDim(leftAndDim);
        int leftIdx = unpackLeft(leftAndDim);
        int rightIdx = rightAt(treeIdx);
        float ptVal = ptAt(treeIdx, splitDim);
        float delta = query[splitDim] - ptVal;
        int near, far;
        if (delta < 0f) {
            near = leftIdx;
            far  = rightIdx;
        } else {
            near = rightIdx;
            far  = leftIdx;
        }

        descend(query, near, scratch, k, slabSum, depth + 1);

        float[] slab = scratch.slab;
        float oldSlabD = slab[splitDim];
        float newSlabD = delta * delta;
        float newSlabSum = slabSum - oldSlabD + newSlabD;
        // After near-side recursion, visits have grown; recompute scale for the far-side gate.
        if (topK.size() < k || newSlabSum < topK.peekDist() * relaxScale(scratch.visits)) {
            slab[splitDim] = newSlabD;
            descend(query, far, scratch, k, newSlabSum, depth + 1);
            slab[splitDim] = oldSlabD;
        }
    }

    /**
     * Best-first branch-and-bound (H16-A). Hybrid: NEAR children stay inline (they inherit parent's slabSum so DFS
     * ordering is equivalent to popping them by slabSum). FAR children are pushed to a min-heap keyed by slabSum,
     * so we globally explore the tightest-bound pending subtree first — converts dark slab on tail queries into fired
     * prunes.
     */
    void descendBBF(float[] query, Scratch scratch, int k) {
        // First DFS starts at root with all-zero slab from prepareSearch().
        continueDfsBBF(query, 0, 0f, 0, scratch, k);
        // Inlined outer loop: pop min, stale-prune, restore slab, continueDfs.
        int[]   hTreeIdx = scratch.bbfTreeIdx;
        float[] hSlabSum = scratch.bbfSlabSum;
        int[]   hDepth   = scratch.bbfDepth;
        int[]   hSlabIdx = scratch.bbfSlabIdx;
        float[] pool     = scratch.bbfSlabPool;
        TopKSortedArray topK = scratch.results;
        while (scratch.bbfSize > 0) {
            // Pop min (inlined): read slot 0, then sift down last-element.
            int   popTi = hTreeIdx[0];
            float popSs = hSlabSum[0];
            int   popDe = hDepth[0];
            int   popSi = hSlabIdx[0];
            int   lastIdx = --scratch.bbfSize;
            if (lastIdx > 0) {
                int   ti = hTreeIdx[lastIdx];
                float ss = hSlabSum[lastIdx];
                int   de = hDepth[lastIdx];
                int   si = hSlabIdx[lastIdx];
                int i = 0;
                int half = lastIdx >>> 1;
                while (i < half) {
                    int child = (i << 1) + 1;
                    int right = child + 1;
                    if (right < lastIdx && hSlabSum[right] < hSlabSum[child]) child = right;
                    if (ss <= hSlabSum[child]) break;
                    hTreeIdx[i] = hTreeIdx[child];
                    hSlabSum[i] = hSlabSum[child];
                    hDepth[i]   = hDepth[child];
                    hSlabIdx[i] = hSlabIdx[child];
                    i = child;
                }
                hTreeIdx[i] = ti;
                hSlabSum[i] = ss;
                hDepth[i]   = de;
                hSlabIdx[i] = si;
            }
            if (topK.size() >= k && popSs >= topK.peekDist()) continue;
            System.arraycopy(pool, popSi * DIMS, scratch.slab, 0, DIMS);
            continueDfsBBF(query, popTi, popSs, popDe, scratch, k);
        }
    }

    /** Inner DFS-NEAR loop. Pushes each visited node's FAR child to the BBF heap if slab prune doesn't fire. */
    private void continueDfsBBF(float[] query, int treeIdx, float slabSum, int depth, Scratch scratch, int k) {
        TopKSortedArray topK = scratch.results;
        float[] slab = scratch.slab;
        int[]   hTreeIdx = scratch.bbfTreeIdx;
        float[] hSlabSum = scratch.bbfSlabSum;
        int[]   hDepth   = scratch.bbfDepth;
        int[]   hSlabIdx = scratch.bbfSlabIdx;
        float[] pool     = scratch.bbfSlabPool;
        while (treeIdx >= 0) {
            if (topK.size() >= k && slabSum >= topK.peekDist()) return;

            if (depth <= TOP_BBOX_DEPTH && topK.size() >= k) {
                int slot = topSlot[treeIdx];
                if (slot >= 0) {
                    scratch.bboxChecks++;
                    float bboxDist = bboxDistSquared(scratch, slot);
                    if (bboxDist >= topK.peekDist()) return;
                }
            }

            scratch.visits++;
            float dist = distSquared(scratch, treeIdx);
            if (topK.size() < k) {
                if (!topK.contains(treeIdx)) topK.push(treeIdx, dist);
            } else if (dist < topK.peekDist() && !topK.contains(treeIdx)) {
                topK.replaceFarthest(treeIdx, dist);
            }

            int leftAndDim = leftAndDimAt(treeIdx);
            int splitDim = unpackDim(leftAndDim);
            int leftIdx = unpackLeft(leftAndDim);
            int rightIdx = rightAt(treeIdx);
            float ptVal = ptAt(treeIdx, splitDim);
            float delta = query[splitDim] - ptVal;
            int near, far;
            if (delta < 0f) { near = leftIdx;  far = rightIdx; }
            else            { near = rightIdx; far = leftIdx;  }

            if (far >= 0) {
                float oldSlabD = slab[splitDim];
                float newSlabD = delta * delta;
                float newSlabSum = slabSum - oldSlabD + newSlabD;
                if (topK.size() < k || newSlabSum < topK.peekDist()) {
                    int nextDepth = depth + 1;
                    if (nextDepth <= BBF_MAX_DEPTH) {
                        // Shallow FAR: push to best-first heap for reordering.
                        int newSlabIdx = scratch.bbfSlabNext++;
                        int poolOff = newSlabIdx * DIMS;
                        System.arraycopy(slab, 0, pool, poolOff, DIMS);
                        pool[poolOff + splitDim] = newSlabD;
                        int i = scratch.bbfSize++;
                        while (i > 0) {
                            int parent = (i - 1) >>> 1;
                            if (hSlabSum[parent] <= newSlabSum) break;
                            hTreeIdx[i] = hTreeIdx[parent];
                            hSlabSum[i] = hSlabSum[parent];
                            hDepth[i]   = hDepth[parent];
                            hSlabIdx[i] = hSlabIdx[parent];
                            i = parent;
                        }
                        hTreeIdx[i] = far;
                        hSlabSum[i] = newSlabSum;
                        hDepth[i]   = nextDepth;
                        hSlabIdx[i] = newSlabIdx;
                    } else {
                        // Deep FAR: DFS recursion; slab mutation + restore, same as classical descend.
                        slab[splitDim] = newSlabD;
                        descend(query, far, scratch, k, newSlabSum, nextDepth);
                        slab[splitDim] = oldSlabD;
                    }
                }
            }

            treeIdx = near;
            depth++;
            // slab unchanged for NEAR: same side of the splitting wall on splitDim.
        }
    }

    /**
     * Squared L2 distance from the query to the axis-aligned bbox: per dim {@code max(0, lo-q, q-hi)^2}, summed.
     * SIMD vectors are loaded method-locally so escape analysis scalar-replaces them into the same registers
     * {@link #distSquared} uses.
     */
    float bboxDistSquared(Scratch scratch, int slot) {
        int base = slot * DIMS * 2;

        FloatVector q0 = FloatVector.fromArray(SPECIES, scratch.permutedQuery, 0);
        FloatVector q1 = FloatVector.fromArray(SPECIES, scratch.permutedQuery, 8);
        FloatVector lo0;
        FloatVector hi0;
        FloatVector lo1;
        FloatVector hi1;
        if (topBbox != null) {
            lo0 = FloatVector.fromArray(SPECIES, topBbox, base);
            hi0 = FloatVector.fromArray(SPECIES, topBbox, base + DIMS);
            lo1 = FloatVector.fromArray(SPECIES, topBbox, base + 8,        TAIL_MASK);
            hi1 = FloatVector.fromArray(SPECIES, topBbox, base + DIMS + 8, TAIL_MASK);
        } else {
            long byteBase = (long) base * Float.BYTES;
            long dimBytes = (long) DIMS * Float.BYTES;
            lo0 = FloatVector.fromMemorySegment(SPECIES, topBboxSeg, byteBase, ByteOrder.LITTLE_ENDIAN);
            hi0 = FloatVector.fromMemorySegment(SPECIES, topBboxSeg, byteBase + dimBytes, ByteOrder.LITTLE_ENDIAN);
            // Masking is mandatory on the mmap path topBboxSeg has exactly the file bytes, no tail padding.
            lo1 = FloatVector.fromMemorySegment(SPECIES, topBboxSeg, byteBase + 8L * Float.BYTES, ByteOrder.LITTLE_ENDIAN, TAIL_MASK);
            hi1 = FloatVector.fromMemorySegment(SPECIES, topBboxSeg, byteBase + dimBytes + 8L * Float.BYTES, ByteOrder.LITTLE_ENDIAN, TAIL_MASK);
        }

        FloatVector diff0 = lo0.sub(q0).max(q0.sub(hi0)).max(ZERO);
        float partial0 = diff0.mul(diff0).reduceLanes(VectorOperators.ADD);
        // partial0 ≥ peekDist means total bbox dist ≥ peekDist (partial1 ≥ 0). Skip second chunk.
        if (scratch.results.size() == TopKSortedArray.MAX_K && partial0 >= scratch.results.peekDist())
            return Float.POSITIVE_INFINITY;

        FloatVector diff1 = lo1.sub(q1).max(q1.sub(hi1)).max(ZERO);
        float partial1 = diff1.mul(diff1).reduceLanes(VectorOperators.ADD);

        return partial0 + partial1;
    }

    public int topNodeCount() { return topNodeCount; }

    static final ThreadLocal<Scratch> scratchTL = ThreadLocal.withInitial(Scratch::new);

    static final class Scratch {
        final TopKSortedArray results = new TopKSortedArray();
        final float[] slab = new float[DIMS];
        /** Lanes 14, 15 stay zero; never written past lane 13. Reloaded per call so SIMD vectors stay method-local. */
        final float[] permutedQuery = new float[STRIDE];
        int[] topKBuf = new int[8];
        final int[]   fanOutBuf    = new int[PRIME_FANOUT_COUNT];
        final float[] fanOutScores = new float[PRIME_FANOUT_COUNT];
        int fanOutCount;
        int visits;
        int bboxChecks;

        // Best-first FAR heap (H16-A). Stored SoA to keep hot reads dense.
        // Pushed: one FAR per visit when slab-prune doesn't fire. Popped: one per DFS-restart in outer loop.
        // Heap size is bounded by visits-with-far-not-pruned, < visits_p99 (~2600) comfortably.
        static final int BBF_HEAP_CAP = 4096;
        final int[]   bbfTreeIdx  = new int[BBF_HEAP_CAP];
        final float[] bbfSlabSum  = new float[BBF_HEAP_CAP];
        final int[]   bbfDepth    = new int[BBF_HEAP_CAP];
        final int[]   bbfSlabIdx  = new int[BBF_HEAP_CAP];
        /** Flat pool: slab i occupies indices [i*DIMS, (i+1)*DIMS). Avoids aaload indirection. */
        final float[] bbfSlabPool = new float[BBF_HEAP_CAP * DIMS];
        int bbfSize;
        int bbfSlabNext;

        void ensureTopKCapacity(int k) {
            if (topKBuf.length < k) topKBuf = new int[k];
        }
    }

    /**
     * Fixed-capacity k=5 set kept sorted ascending {@code dists[size-1]} is the eviction threshold. At k=5 a
     * linear-scan over a hard-sized array beats a true heap whose sift-down branches mispredict more than a tight
     * counter-bound loop. Previously named {@code MaxDistHeap}.
     */
    static final class TopKSortedArray {
        private static final int MAX_K = 5;

        private final int[]   ids   = new int[MAX_K];
        private final float[] dists = new float[MAX_K];
        private int size;

        void clear() { size = 0; }
        int size() { return size; }

        float peekDist() { return dists[size - 1]; }

        boolean contains(int id) {
            for (int i = 0; i < size; i++) if (ids[i] == id) return true;
            return false;
        }

        void ensureCapacity(int n) {
            if (n > MAX_K) {
                throw new IllegalArgumentException("TopKSortedArray fixed at k=" + MAX_K + " but caller requested " + n);
            }
        }

        void push(int id, float dist) {
            int pos = size;
            while (pos > 0 && dists[pos - 1] > dist) {
                dists[pos] = dists[pos - 1];
                ids[pos]   = ids[pos - 1];
                pos--;
            }
            dists[pos] = dist;
            ids[pos]   = id;
            size++;
        }

        void replaceFarthest(int id, float dist) {
            int pos = MAX_K - 1;
            while (pos > 0 && dists[pos - 1] > dist) {
                dists[pos] = dists[pos - 1];
                ids[pos]   = ids[pos - 1];
                pos--;
            }
            dists[pos] = dist;
            ids[pos]   = id;
        }

        int countFrauds(byte[] fraud) {
            int count = 0;
            for (int i = 0; i < size; i++) {
                count += fraud[ids[i]];
            }
            return count;
        }

        int drainAscending(int[] out) {
            int n = size;
            System.arraycopy(ids, 0, out, 0, n);
            size = 0;
            return n;
        }
    }

}
