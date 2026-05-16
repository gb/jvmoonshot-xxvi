package com.github.gb.moonshot.search;

import static com.github.gb.moonshot.search.KdTreeTuning.PROFILING_ENABLED;

/**
 * Read-only probes for the last {@link KdTree} query on the current thread.
 */
public final class KdTreeProbes {

    private KdTreeProbes() {
    }

    /**
     * True when {@code KDTREE_PROFILING} env var is set; bench should fail-fast when false.
     */
    public static boolean isProfilingEnabled() {
        return PROFILING_ENABLED;
    }

    public static int lastQueryNodesVisited() {
        return KdTreeScratch.scratchTL.get().visits;
    }

    public static int lastQueryBboxChecks() {
        return KdTreeScratch.scratchTL.get().bboxChecks;
    }

    public static int lastBbfPushes() {
        return KdTreeScratch.scratchTL.get().bbfPushes;
    }

    public static int lastBbfHeapMax() {
        return KdTreeScratch.scratchTL.get().bbfHeapMax;
    }

    public static int lastSlabPrunes() {
        return KdTreeScratch.scratchTL.get().slabPrunes;
    }

    public static int lastBboxPrunes() {
        return KdTreeScratch.scratchTL.get().bboxPrunes;
    }

    public static int lastTopKFilledAt() {
        return KdTreeScratch.scratchTL.get().topKFilledAt;
    }

    public static int lastTopKReplaced() {
        return KdTreeScratch.scratchTL.get().topKReplaced;
    }

    public static float lastFinalPeekDist() {
        return KdTreeScratch.scratchTL.get().finalPeekDist;
    }
}
