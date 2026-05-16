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
     * {@link #relaxScale}: when false (epsilon=0 at startup) the entire body collapses to
     * {@code return 1.0f}. When true, only the relaxation arithmetic remains.
     */
    static final boolean RELAX_INITIALLY_ENABLED;
    /**
     * Mirrors {@link #RELAX_SOFT_CAP} at class load as a {@code static final} so C2 inlines the
     * literal 1500 into {@link #relaxScale}'s fast exit, eliminating the per-call field read.
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

    static int PRIME_PLUNGE_CAP = 4;

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

    // ── Derived ───────────────────────────────────────────────────────────────────────────────────

    /**
     * Multiplicative scale applied to prune thresholds at {@code visits} visited nodes.
     * Returns 1.0 (exact) when disabled or visits ≤ soft cap; < 1.0 during relaxation.
     * {@link #RELAX_INITIALLY_ENABLED} is {@code static final}: C2 folds the fast-path to a
     * single {@code return 1.0f} with no field reads when epsilon was 0 at class load.
     */
    static float relaxScale(int visits) {
        if (!RELAX_INITIALLY_ENABLED || visits <= RELAX_SOFT_CAP_INIT) return 1.0f;
        if (RELAX_MAX_EPSILON == 0f) return 1.0f;
        float t = Math.min(1.0f, (float) (visits - RELAX_SOFT_CAP) / RELAX_RANGE);
        return 1.0f - RELAX_MAX_EPSILON * t;
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
