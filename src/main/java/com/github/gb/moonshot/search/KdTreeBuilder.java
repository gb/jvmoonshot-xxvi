package com.github.gb.moonshot.search;

import com.github.gb.moonshot.Dataset;

import java.util.Arrays;
import java.util.Random;

import static com.github.gb.moonshot.search.KdTree.*;
import static com.github.gb.moonshot.search.KdTreeLayout.packLeftAndDim;
import static com.github.gb.moonshot.search.KdTreeLayout.unpackLeft;

/**
 * One-shot {@link KdTree} construction from a {@link Dataset}. Sliding-midpoint splits,
 * 3-way quickselect, range-descending lane permutation. Feature vectors are quantized
 * to i16 ({@link KdTree#quantize}) at write time; split decisions use the original floats.
 */
public final class KdTreeBuilder {

    private KdTreeBuilder() {
    }

    public static KdTree build(Dataset dataset) {
        return build(dataset, DIM_PERMUTATION);
    }

    public static KdTree build(Dataset dataset, int[] dimPerm) {
        int nodeCount = dataset.size();
        float[] srcVecsSemantic = dataset.vectors();
        boolean[] srcFraud = dataset.fraudLabels();

        // Float source in permuted order (for split decisions).
        // srcVecsFloat/srcVecsShort use STRIDE (=20) so split code reads them with the same
        // stride as it reads pts[].  dataset.vectors() uses Dataset.STRIDE (=16): use
        // separate source and dest bases to avoid the mismatch.
        float[] srcVecsFloat = new float[nodeCount * STRIDE];
        short[] srcVecsShort = new short[nodeCount * STRIDE];
        for (int i = 0; i < nodeCount; i++) {
            int srcBase = i * KdTreeLayout.STRIDE; // Dataset uses STRIDE=16 (f32 layout)
            int dstBase = i * STRIDE;         // our arrays use STRIDE=20
            for (int d = 0; d < DIMS; d++) {
                float v = srcVecsSemantic[srcBase + dimPerm[d]];
                srcVecsFloat[dstBase + d] = v;
                srcVecsShort[dstBase + d] = KdTree.quantize(v);
            }
        }

        int[] indices = new int[nodeCount];
        for (int i = 0; i < nodeCount; i++) indices[i] = i;

        short[] pts = new short[nodeCount * STRIDE];
        int[] origId = new int[nodeCount];
        byte[] fraud = new byte[nodeCount];

        Random rng = new Random(42L);
        int[] nextIdx = {0};
        int rootIdx = buildRecursive(srcVecsFloat, srcVecsShort, srcFraud, indices,
                0, nodeCount, 0, pts, origId, fraud, nextIdx, rng);
        if (rootIdx != 0) throw new IllegalStateException("root not at 0: " + rootIdx);
        if (nextIdx[0] != nodeCount) throw new IllegalStateException("node count mismatch");

        // Build top-level bboxes in i16 space.
        int[] topSlot = new int[nodeCount];
        Arrays.fill(topSlot, -1);
        int maxTopNodes = 1 << (TOP_BBOX_DEPTH + 1);
        short[] tmpBbox = new short[maxTopNodes * STRIDE_BBOX];
        int[] topNodeCountArr = {0};
        short[] rootOut = new short[DIMS * 2];
        computeTopBboxesRec(pts, 0, 0, topSlot, tmpBbox, topNodeCountArr, rootOut);

        int topNodeCount = topNodeCountArr[0];
        short[] topBbox = new short[topNodeCount * STRIDE_BBOX];
        System.arraycopy(tmpBbox, 0, topBbox, 0, topNodeCount * STRIDE_BBOX);

        return new KdTree(nodeCount, pts, origId, fraud, topSlot, topBbox, topNodeCount);
    }

    /**
     * Compute bbox bounds recursively. {@code out[0..DIMS-1]} = lo, {@code out[DIMS..2*DIMS-1]} = hi.
     * Stores into {@code tmpBbox} in STRIDE_BBOX layout: lo at [0..13], hi at [16..29]; pads stay 0.
     * right index is read from pts[LANE_RIGHT..LANE_RIGHT+1] (STRIDE=20 layout).
     */
    private static void computeTopBboxesRec(
            short[] pts, int treeIdx, int depth,
            int[] topSlot, short[] tmpBbox, int[] nextSlot, short[] out
    ) {
        int ptsOffset = treeIdx * STRIDE;
        int leftAndDim = (pts[ptsOffset + LANE_LEFT_DIM] & 0xFFFF)
                | ((pts[ptsOffset + LANE_LEFT_DIM + 1] & 0xFFFF) << 16);
        int leftIdx = unpackLeft(leftAndDim);
        // right is now packed in pts[LANE_RIGHT..LANE_RIGHT+1]
        int rightIdx = (pts[ptsOffset + LANE_RIGHT] & 0xFFFF)
                | ((pts[ptsOffset + LANE_RIGHT + 1] & 0xFFFF) << 16);

        for (int d = 0; d < DIMS; d++) {
            short v = pts[ptsOffset + d];
            out[d] = v;
            out[DIMS + d] = v;
        }
        if (leftIdx >= 0) {
            short[] leftOut = new short[DIMS * 2];
            computeTopBboxesRec(pts, leftIdx, depth + 1, topSlot, tmpBbox, nextSlot, leftOut);
            for (int d = 0; d < DIMS; d++) {
                if (leftOut[d] < out[d]) out[d] = leftOut[d];
                if (leftOut[DIMS + d] > out[DIMS + d]) out[DIMS + d] = leftOut[DIMS + d];
            }
        }
        if (rightIdx >= 0) {
            short[] rightOut = new short[DIMS * 2];
            computeTopBboxesRec(pts, rightIdx, depth + 1, topSlot, tmpBbox, nextSlot, rightOut);
            for (int d = 0; d < DIMS; d++) {
                if (rightOut[d] < out[d]) out[d] = rightOut[d];
                if (rightOut[DIMS + d] > out[DIMS + d]) out[DIMS + d] = rightOut[DIMS + d];
            }
        }

        if (depth <= TOP_BBOX_DEPTH) {
            int slot = nextSlot[0]++;
            topSlot[treeIdx] = slot;
            int base = slot * STRIDE_BBOX;
            // lo at [base..base+13], zero-pad at 14-15 (stays 0 from array init).
            // hi at [base+16..base+29], zero-pad at 30-31 (stays 0).
            System.arraycopy(out, 0, tmpBbox, base, DIMS);
            System.arraycopy(out, DIMS, tmpBbox, base + 16, DIMS);
        }
    }

    private static int buildRecursive(
            float[] srcVecsFloat, short[] srcVecsShort, boolean[] srcFraud,
            int[] indices, int from, int to, int depth,
            short[] pts, int[] origId, byte[] fraud, int[] nextIdx, Random rng
    ) {
        if (from >= to) return -1;
        int splitDim = chooseSplitDim(srcVecsFloat, indices, from, to);
        int splitPos = slidingMidpointTarget(srcVecsFloat, indices, from, to, splitDim);
        quickselect(indices, from, to, splitPos, srcVecsFloat, splitDim, rng);

        int treeIdx = nextIdx[0]++;
        int srcId = indices[splitPos];

        // Copy quantized feature dims (NOT nav slots — those come after left/right build).
        System.arraycopy(srcVecsShort, srcId * STRIDE, pts, treeIdx * STRIDE, DIMS);
        origId[treeIdx] = srcId;
        fraud[treeIdx] = (byte) (srcFraud[srcId] ? 1 : 0);

        int leftIdx = buildRecursive(srcVecsFloat, srcVecsShort, srcFraud, indices,
                from, splitPos, depth + 1, pts, origId, fraud, nextIdx, rng);
        int rightIdx = buildRecursive(srcVecsFloat, srcVecsShort, srcFraud, indices,
                splitPos + 1, to, depth + 1, pts, origId, fraud, nextIdx, rng);

        // Pack leftAndDim into pts shorts[14..15].
        int packed = packLeftAndDim(leftIdx, splitDim);
        pts[treeIdx * STRIDE + LANE_LEFT_DIM] = (short) (packed & 0xFFFF);
        pts[treeIdx * STRIDE + LANE_LEFT_DIM + 1] = (short) ((packed >>> 16) & 0xFFFF);
        // Pack right into pts shorts[16..17].
        pts[treeIdx * STRIDE + LANE_RIGHT] = (short) (rightIdx & 0xFFFF);
        pts[treeIdx * STRIDE + LANE_RIGHT + 1] = (short) ((rightIdx >>> 16) & 0xFFFF);

        return treeIdx;
    }

    private static final int IMBALANCE_CAP_DENOM = 9;

    private static int slidingMidpointTarget(float[] srcVecs, int[] indices, int from, int to, int splitDim) {
        int sz = to - from;
        if (sz <= 2) return from + (sz >>> 1);
        float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
        for (int i = from; i < to; i++) {
            float v = srcVecs[indices[i] * STRIDE + splitDim];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        if (max == min) return from + (sz >>> 1);
        float mid = (min + max) * 0.5f;
        int below = 0;
        for (int i = from; i < to; i++) {
            if (srcVecs[indices[i] * STRIDE + splitDim] < mid) below++;
        }
        int minSide = Math.max(1, sz / IMBALANCE_CAP_DENOM);
        if (below < minSide) below = minSide;
        else if (below > sz - minSide) below = sz - minSide;
        return from + below;
    }

    private static int chooseSplitDim(float[] srcVecs, int[] indices, int from, int to) {
        int sz = to - from, step = Math.max(1, sz / 256);
        float[] mins = new float[DIMS], maxs = new float[DIMS];
        Arrays.fill(mins, Float.POSITIVE_INFINITY);
        Arrays.fill(maxs, Float.NEGATIVE_INFINITY);
        for (int i = from; i < to; i += step) {
            int off = indices[i] * STRIDE;
            for (int d = 0; d < DIMS; d++) {
                float v = srcVecs[off + d];
                if (v < mins[d]) mins[d] = v;
                if (v > maxs[d]) maxs[d] = v;
            }
        }
        int best = 0;
        float bestRange = -1f;
        for (int d = 0; d < DIMS; d++) {
            float r = maxs[d] - mins[d];
            if (r > bestRange) {
                bestRange = r;
                best = d;
            }
        }
        return best;
    }

    private static void quickselect(int[] indices, int from, int to, int target,
                                    float[] srcVecs, int splitDim, Random rng) {
        while (to - from > 1) {
            float pivot = srcVecs[indices[from + rng.nextInt(to - from)] * STRIDE + splitDim];
            int less = from, cur = from, greater = to;
            while (cur < greater) {
                float v = srcVecs[indices[cur] * STRIDE + splitDim];
                if (v < pivot) {
                    int t = indices[less];
                    indices[less++] = indices[cur];
                    indices[cur++] = t;
                } else if (v > pivot) {
                    int t = indices[--greater];
                    indices[greater] = indices[cur];
                    indices[cur] = t;
                } else {
                    cur++;
                }
            }
            if (target < less) to = less;
            else if (target >= greater) from = greater;
            else return;
        }
    }
}
