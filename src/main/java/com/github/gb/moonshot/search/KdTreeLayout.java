package com.github.gb.moonshot.search;

/**
 * Data-geometry constants, packed-nav encode/decode for the KD-tree. No mutable state, no env
 * reads. All fields are compile-time constants or final at class load.
 *
 * <p>An earlier Vector API SIMD path (SPECIES_256, TAIL_MASK, LANES_0_5_KEEP_INTS, ZERO) lived
 * here but was removed once {@code DistKernelBench} confirmed that on JDK 25 the auto-vectorized
 * scalar form in {@code KdTree.distSumI16AtBase} is 20 % faster than the explicit Vector API
 * (which still allocates ~34 B/op through {@code ShortVector.convertShape}).
 */
public final class KdTreeLayout {

    /**
     * Number of semantic feature dimensions.
     */
    public static final int DIMS = 14;
    /**
     * Short array stride per KD-tree node: 14 feature lanes plus one packed i32 nav word.
     */
    public static final int STRIDE = 16;

    /**
     * Legacy non-PCA lane order. Drives split-dim tie-breaking toward dimensions that reduce
     * visit p99 for the current query distribution. Re-tune with {@code DimPermTuner} whenever
     * fixture generation changes materially, because randomized dates / last-transaction ranges
     * can shift which dimensions dominate tail search work.
     *
     * <p>Production PCA builds bypass this permutation (PCA emits its own axis order). Treat this
     * constant as relevant only for {@link KdTreeBuilder#build(com.github.gb.moonshot.Dataset)}
     * and non-PCA comparison tools; PCA-specific changes need their own validation.
     */
    public static final int[] DIM_PERMUTATION = {6, 1, 9, 5, 11, 13, 2, 7, 0, 10, 3, 8, 12, 4};

    /** Low short lane of the packed nav int. High short lane is {@code LANE_NAV + 1}. */
    static final int LANE_NAV = 14;

    private static final int RIGHT_BITS = 23;
    private static final int RIGHT_MASK = (1 << RIGHT_BITS) - 1;
    private static final int DIM_SHIFT = RIGHT_BITS;
    private static final int FRAUD_SHIFT = DIM_SHIFT + 4;
    private static final int HAS_LEFT_SHIFT = FRAUD_SHIFT + 1;
    /**
     * Set iff this node's depth ≤ {@link com.github.gb.moonshot.search.KdTree#TOP_BBOX_DEPTH},
     * i.e. {@code topSlot[treeIdx] >= 0}. Lets the bbox-prune site short-circuit the 12 MB
     * {@code topSlot[]} random probe for deep nodes.
     */
    private static final int HAS_BBOX_SHIFT = HAS_LEFT_SHIFT + 1;
    /**
     * Set iff this node's depth &lt; {@link com.github.gb.moonshot.search.KdTree#TOP_BBOX_DEPTH},
     * i.e. {@code topSlot[child] >= 0} for both children. Lets the BBF-push site replace the
     * {@code topSlot[far]} random probe with a single bit test on the already-loaded parent nav.
     */
    private static final int CHILDREN_HAVE_BBOX_SHIFT = HAS_BBOX_SHIFT + 1;
    public static final int HAS_BBOX_MASK = 1 << HAS_BBOX_SHIFT;
    public static final int CHILDREN_HAVE_BBOX_MASK = 1 << CHILDREN_HAVE_BBOX_SHIFT;

    static int packNav(int right, int dim, boolean fraud, boolean hasLeft,
                       boolean hasBbox, boolean childrenHaveBbox) {
        int rightPlusOne = right + 1;
        if ((rightPlusOne & ~RIGHT_MASK) != 0) {
            throw new IllegalArgumentException("right child out of packed-nav range: " + right);
        }
        return rightPlusOne
                | ((dim & 0xF) << DIM_SHIFT)
                | (fraud ? (1 << FRAUD_SHIFT) : 0)
                | (hasLeft ? (1 << HAS_LEFT_SHIFT) : 0)
                | (hasBbox ? HAS_BBOX_MASK : 0)
                | (childrenHaveBbox ? CHILDREN_HAVE_BBOX_MASK : 0);
    }

    static int unpackLeft(int treeIdx, int nav) {
        return ((nav >>> HAS_LEFT_SHIFT) & 1) != 0 ? treeIdx + 1 : -1;
    }

    static int unpackRight(int nav) {
        int rightPlusOne = nav & RIGHT_MASK;
        return rightPlusOne == 0 ? -1 : rightPlusOne - 1;
    }

    static int unpackDim(int nav) {
        return (nav >>> DIM_SHIFT) & 0xF;
    }

    static int unpackFraud(int nav) {
        return (nav >>> FRAUD_SHIFT) & 1;
    }

    private KdTreeLayout() {
    }
}
