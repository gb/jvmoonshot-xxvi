package com.github.gb.moonshot.search;

/**
 * Runtime-configurable search parameters. All mutable state lives here; public read/write access
 * is via {@link KdTree}'s setters and forwarding constants. {@code static final} booleans/ints
 * are captured once at class load so C2 can constant-fold the hot-path guards to dead code when
 * the relevant env var is absent.
 */
final class KdTreeTuning {

    // ── MAX_VISITS ────────────────────────────────────────────────────────────────────────────────
    // Hard cap on per-query node visits. Default = disabled (Integer.MAX_VALUE).
    // MAX_VISITS_ENABLED is static final: C2 folds the descend guard to dead code when absent.

    static int MAX_VISITS_BUDGET;
    static final boolean MAX_VISITS_ENABLED;

    static {
        String v = System.getenv("KDTREE_MAX_VISITS");
        int parsed = Integer.MAX_VALUE;
        if (v != null && !v.isEmpty()) {
            try {
                parsed = Integer.parseInt(v);
            } catch (NumberFormatException ignored) {
            }
        }
        MAX_VISITS_BUDGET = Math.max(1, parsed);
        MAX_VISITS_ENABLED = MAX_VISITS_BUDGET != Integer.MAX_VALUE;
    }

    // ── EARLY_DONE ───────────────────────────────────────────────────────────────────────────────
    // Confidence shortcut after the prime/plunge seed: if the current kth candidate is already
    // within this L2 distance, skip the remaining BBF search. Default disabled.

    static final boolean EARLY_DONE_ENABLED;
    static final int EARLY_DONE_SUM;

    static {
        String v = System.getenv("KDTREE_EARLY_DIST_MILLI");
        int parsedMilli = 0;
        if (v != null && !v.isEmpty()) {
            try {
                parsedMilli = Integer.parseInt(v);
            } catch (NumberFormatException ignored) {
            }
        }
        EARLY_DONE_ENABLED = parsedMilli > 0;
        int scaled = Math.max(0, parsedMilli) * (KdTree.SCALE / 1000);
        long sum = (long) scaled * scaled;
        EARLY_DONE_SUM = sum > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) sum;
    }

    // ── RELAX ─────────────────────────────────────────────────────────────────────────────────────
    // Epsilon-relaxed pruning: once visits > RELAX_SOFT_CAP, prune thresholds shrink linearly from
    // 1.0 to (1 − epsilon) over RELAX_RANGE additional visits. A subtree pruned under relaxation
    // may omit a true neighbor whose distance is within epsilon × peekDist of the k-th candidate.
    // Default epsilon = 0 (exact). Set KDTREE_RELAX_EPSILON=0.01 to enable 1 % relaxation.

    static int RELAX_SOFT_CAP = 1_500;
    static int RELAX_RANGE = 1_500;
    static float RELAX_MAX_EPSILON = 0f;

    /**
     * Captured once at class load. Being {@code static final}, C2 constant-folds
     * {@link #relaxScaleQ16}: when false (epsilon=0 at startup) the entire body collapses to
     * {@code return 65536}. When true, only the relaxation arithmetic remains.
     */
    static final boolean RELAX_INITIALLY_ENABLED;
    /**
     * Mirrors {@link #RELAX_SOFT_CAP} at class load as a {@code static final} so C2 inlines the
     * literal 1500 into {@link #relaxScaleQ16}'s fast exit, eliminating the per-call field read.
     */
    static final int RELAX_SOFT_CAP_INIT;

    static {
        String sc = System.getenv("KDTREE_RELAX_SOFT_CAP");
        String rng = System.getenv("KDTREE_RELAX_RANGE");
        String eps = System.getenv("KDTREE_RELAX_EPSILON");
        if (sc != null && !sc.isEmpty()) {
            try {
                RELAX_SOFT_CAP = Integer.parseInt(sc);
            } catch (NumberFormatException ignored) {
            }
        }
        if (rng != null && !rng.isEmpty()) {
            try {
                RELAX_RANGE = Math.max(1, Integer.parseInt(rng));
            } catch (NumberFormatException ignored) {
            }
        }
        if (eps != null && !eps.isEmpty()) {
            try {
                RELAX_MAX_EPSILON = Float.parseFloat(eps);
            } catch (NumberFormatException ignored) {
            }
        }
        RELAX_INITIALLY_ENABLED = RELAX_MAX_EPSILON != 0f;
        RELAX_SOFT_CAP_INIT = RELAX_SOFT_CAP;
    }

    // ── PRIME ─────────────────────────────────────────────────────────────────────────────────────
    // Fan-out + greedy-plunge seed strategy at query start (see KdTree.prime).

    static int PRIME_PLUNGE_CAP = 1;

    static {
        String v = System.getenv("KDTREE_PRIME_PLUNGE_CAP");
        if (v != null && !v.isEmpty()) {
            try {
                PRIME_PLUNGE_CAP = Math.max(1, Math.min(KdTree.PRIME_FANOUT_COUNT, Integer.parseInt(v)));
            } catch (NumberFormatException ignored) {
            }
        }
    }

    static boolean PRIME_BBOX_SCORING;

    static {
        String v = System.getenv("KDTREE_PRIME_BBOX");
        // Default false: bbox scoring 32 fanOut entries adds cache-unfriendly topBbox reads
        // that cost more than the marginal improvement in plunge ordering. BBF handles the rest
        // efficiently with its own priority queue.
        PRIME_BBOX_SCORING = "1".equals(v) || "true".equalsIgnoreCase(v);
    }

    // ── Boundary refinement ──────────────────────────────────────────────────────────────────────
    // The API decision boundary is fraudCount >= 3. When relaxed pruning is enabled, only fast-path
    // counts 2 and 3 can change approved/denied after an exact rerun. Disabled by default for this
    // KdTree because a rerun repeats most of the expensive tree walk; it remains useful for sweeps.

    static final boolean REFINE_BOUNDARY;

    static {
        String v = System.getenv("KDTREE_REFINE_BOUNDARY");
        REFINE_BOUNDARY = "1".equals(v) || "true".equalsIgnoreCase(v);
    }

    // ── Bucket-leaf prototype ───────────────────────────────────────────────────────────────────
    // Exact experiment: once traversal reaches this depth, linearly scan the contiguous preorder
    // subtree range instead of continuing recursive/BBF descent. Disabled by default.

    static final int BUCKET_LEAF_DEPTH;
    static final int BUCKET_LEAF_MAX_NODES;

    static {
        String depth = System.getenv("KDTREE_BUCKET_LEAF_DEPTH");
        int parsed = Integer.MAX_VALUE;
        if (depth != null && !depth.isEmpty()) {
            try {
                parsed = Integer.parseInt(depth);
            } catch (NumberFormatException ignored) {
            }
        }
        BUCKET_LEAF_DEPTH = parsed <= 0 ? Integer.MAX_VALUE : parsed;

        String maxNodes = System.getenv("KDTREE_BUCKET_LEAF_MAX_NODES");
        parsed = Integer.MAX_VALUE;
        if (maxNodes != null && !maxNodes.isEmpty()) {
            try {
                parsed = Integer.parseInt(maxNodes);
            } catch (NumberFormatException ignored) {
            }
        }
        BUCKET_LEAF_MAX_NODES = parsed <= 0 ? Integer.MAX_VALUE : parsed;
    }

    // ── Derived ───────────────────────────────────────────────────────────────────────────────────

    /**
     * Integer Q16 relax scale: returns {@code round(scale * 65536)}. Hot path uses this to
     * avoid the int→float→int round-trip and a float divide. Production tuning (epsilon,
     * soft-cap, range) is fixed at class load so the per-visit slope folds into a single
     * 64-bit integer multiply.
     *
     * <p>Slope is stored at Q32 precision (not Q16) so that {@code delta * slope} produces a
     * subtractor accurate to ~10⁻⁹ relative to a float-domain scale — recall byte-identical
     * to the float form (which lived here before commit 16afab3) on the contest fixture.
     */
    static final long RELAX_EPSILON_Q32_PER_VISIT;
    static final int  RELAX_MAX_SUB_Q16;
    static final int  RELAX_RANGE_INIT;
    /**
     * Precomputed Q16 scale per (visits − softCap − 1), capped at RELAX_RANGE_INIT − 1.
     * Replaces a 64-bit multiply + shift on the hot path with a single int[] load.
     * Size: range_init × 4 bytes (~2.8 KB for default 700) — fits comfortably in L1d.
     */
    static final int[] RELAX_SCALE_Q16_TABLE;

    static {
        RELAX_RANGE_INIT = Math.max(1, RELAX_RANGE);
        if (RELAX_INITIALLY_ENABLED) {
            // Slope at Q32 (8 fractional digits of decimal precision).
            RELAX_EPSILON_Q32_PER_VISIT = Math.round(
                    (double) RELAX_MAX_EPSILON * 4294967296.0 / RELAX_RANGE_INIT);
            RELAX_MAX_SUB_Q16 = Math.round(RELAX_MAX_EPSILON * 65536f);
            int[] tbl = new int[RELAX_RANGE_INIT];
            for (int d = 0; d < RELAX_RANGE_INIT; d++) {
                long subQ32 = (long)(d + 1) * RELAX_EPSILON_Q32_PER_VISIT;
                tbl[d] = 65536 - (int) (subQ32 >>> 16);
            }
            RELAX_SCALE_Q16_TABLE = tbl;
        } else {
            RELAX_EPSILON_Q32_PER_VISIT = 0L;
            RELAX_MAX_SUB_Q16 = 0;
            RELAX_SCALE_Q16_TABLE = new int[0];
        }
    }

    static int relaxScaleQ16(int visits) {
        if (!RELAX_INITIALLY_ENABLED) return 65536;
        int delta = visits - RELAX_SOFT_CAP_INIT;
        if (delta <= 0) return 65536;
        if (delta >= RELAX_RANGE_INIT) return 65536 - RELAX_MAX_SUB_Q16;
        return RELAX_SCALE_Q16_TABLE[delta - 1];
    }

    // ── PROFILING ─────────────────────────────────────────────────────────────────────────────────
    // When false (default), C2 eliminates every counter-write site as dead code — zero hot-path
    // overhead in production. Set KDTREE_PROFILING=1 before running KdTreeSearchBench.

    static final boolean PROFILING_ENABLED;

    static {
        String v = System.getenv("KDTREE_PROFILING");
        PROFILING_ENABLED = v != null && !v.isEmpty() && !"0".equals(v) && !"false".equalsIgnoreCase(v);
    }

    private KdTreeTuning() {
    }
}
