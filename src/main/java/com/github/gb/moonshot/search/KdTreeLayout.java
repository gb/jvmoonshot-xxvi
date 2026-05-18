package com.github.gb.moonshot.search;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

/**
 * Data-geometry constants, SIMD species/masks, and packed-nav encode/decode for the KD-tree.
 * No mutable state, no env reads. All fields are compile-time constants or final at class load.
 */
public final class KdTreeLayout {

    /**
     * Number of semantic feature dimensions.
     */
    public static final int DIMS = 14;
    /**
     * Float array stride per node (DIMS + 2 nav lanes).
     */
    public static final int STRIDE = 16;

    /**
     * Variance-descending lane order. Drives split-dim tie-breaking toward the highest-variance
     * dimension, yielding a more balanced tree and fewer visits per query. Tuned offline via
     * {@code DimPermTuner} against vectorized transaction payload banks.
     */
    public static final int[] DIM_PERMUTATION = {6, 10, 9, 5, 11, 2, 4, 7, 0, 1, 3, 8, 12, 13};

    /**
     * Lane within a stride-16 node that carries the packed {@code (leftIdx | splitDim)} nav int.
     */
    static final int LANE_LEFT_DIM = 14;
    /**
     * Lane within a stride-16 node that carries the right child index.
     */
    static final int LANE_RIGHT = 15;

    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;

    /**
     * Masks lanes 6–7 of the second 8-float chunk at load time — used only in bboxDistSquared
     * where the last topBbox slot may be OOB and the mask prevents the unsafe read.
     * <p>For distSquared we use {@link #LANES_0_5_KEEP_INTS} instead: an unmasked load + integer
     * AND avoids the per-call broadcast through AbstractMask.checkIndexByLane (was ~20% CPU).
     */
    static final VectorMask<Float> TAIL_MASK = VectorMask.fromLong(SPECIES, 0b00111111L);

    /**
     * Integer bit-mask for {@code distSquared}'s second SIMD chunk: keeps lanes 0–5 (real dims
     * 8–13) and zeros lanes 6–7 (packed nav int-bits). Unmasked float or int load + AND bypasses
     * AbstractMask.checkIndexByLane's per-call broadcast that dominated profiles at peak load.
     * OOB-safe: the pts segment is sized n*STRIDE*4 bytes; every node has all 16 lanes written.
     */
    static final IntVector LANES_0_5_KEEP_INTS =
            IntVector.fromArray(IntVector.SPECIES_256, new int[]{-1, -1, -1, -1, -1, -1, 0, 0}, 0);

    /**
     * Pre-materialised zero vector; {@code .max(ZERO)} folds to a register-resident vxorps.
     */
    static final FloatVector ZERO = FloatVector.zero(SPECIES);

    /**
     * Packs a left-child index (28 bits, signed) and split dimension (4 bits) into one int.
     */
    static int packLeftAndDim(int left, int dim) {
        return (left & 0x0FFFFFFF) | ((dim & 0xF) << 28);
    }

    /**
     * Sign-extends the 28-bit signed left index, preserving the {@code -1} absent-child sentinel.
     */
    static int unpackLeft(int leftAndDim) {
        return (leftAndDim << 4) >> 4;
    }

    static int unpackDim(int leftAndDim) {
        return (leftAndDim >>> 28) & 0xF;
    }

    private KdTreeLayout() {
    }
}
