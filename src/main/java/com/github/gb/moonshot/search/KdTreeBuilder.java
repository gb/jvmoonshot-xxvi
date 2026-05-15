package com.github.gb.moonshot.search;

import com.github.gb.moonshot.Dataset;

import java.util.Arrays;
import java.util.Random;

import static com.github.gb.moonshot.search.KdTree.DIMS;
import static com.github.gb.moonshot.search.KdTree.DIM_PERMUTATION;
import static com.github.gb.moonshot.search.KdTree.LANE_LEFT_DIM;
import static com.github.gb.moonshot.search.KdTree.LANE_RIGHT;
import static com.github.gb.moonshot.search.KdTree.STRIDE;
import static com.github.gb.moonshot.search.KdTree.TOP_BBOX_DEPTH;
import static com.github.gb.moonshot.search.KdTree.packLeftAndDim;
import static com.github.gb.moonshot.search.KdTree.unpackLeft;

/**
 * One-shot {@link KdTree} construction. Sliding-midpoint split with imbalance cap; 3-way
 * quickselect partition; split-dim = largest sampled range. Vectors are permuted into
 * variance-descending lane order ({@link KdTree#DIM_PERMUTATION}) so the packed split-dim
 * matches what descend reads after permuteForSearch.
 */
public final class KdTreeBuilder {

    private KdTreeBuilder() {}

    public static KdTree build(Dataset dataset) {
        return build(dataset, DIM_PERMUTATION);
    }

    /** Build with a custom dimension permutation (for offline tuning). */
    public static KdTree build(Dataset dataset, int[] dimPerm) {
        int nodeCount = dataset.size();
        float[] srcVecsSemantic = dataset.vectors();
        boolean[] srcFraud = dataset.fraudLabels();

        // Permute into target lane order so packed split-dim at lane 14 matches descend.
        float[] srcVecs = new float[nodeCount * STRIDE];
        for (int i = 0; i < nodeCount; i++) {
            int srcBase = i * STRIDE;
            int dstBase = i * STRIDE;
            for (int d = 0; d < DIMS; d++) {
                srcVecs[dstBase + d] = srcVecsSemantic[srcBase + dimPerm[d]];
            }
        }

        int[] indices = new int[nodeCount];
        for (int i = 0; i < nodeCount; i++) indices[i] = i;

        float[] pts = new float[nodeCount * STRIDE];
        int[] origId = new int[nodeCount];
        byte[] fraud = new byte[nodeCount];

        Random rng = new Random(42L);
        int[] nextIdx = { 0 };
        int rootIdx = buildRecursive(srcVecs, srcFraud, indices, 0, nodeCount, 0,
            pts, origId, fraud, nextIdx, rng);
        if (rootIdx != 0) throw new IllegalStateException("expected root at idx 0, got " + rootIdx);
        if (nextIdx[0] != nodeCount) throw new IllegalStateException("built " + nextIdx[0] + " nodes, expected " + nodeCount);

        int[] topSlot = new int[nodeCount];
        Arrays.fill(topSlot, -1);
        int maxTopNodes = (1 << (TOP_BBOX_DEPTH + 1));
        float[] tmpBbox = new float[maxTopNodes * DIMS * 2];
        int[] topNodeCountArr = { 0 };
        float[] rootOut = new float[DIMS * 2];
        computeTopBboxesRec(pts, 0, 0, topSlot, tmpBbox, topNodeCountArr, rootOut);
        int topNodeCount = topNodeCountArr[0];
        float[] topBbox = new float[topNodeCount * DIMS * 2];
        System.arraycopy(tmpBbox, 0, topBbox, 0, topNodeCount * DIMS * 2);

        return new KdTree(nodeCount, pts, null, null, origId, null, fraud, topSlot, topBbox, null, topNodeCount);
    }

    private static void computeTopBboxesRec(
        float[] pts, int treeIdx, int depth,
        int[] topSlot, float[] topBbox, int[] nextSlot, float[] out
    ) {
        int ptsOffset = treeIdx * STRIDE;
        int leftAndDim = Float.floatToRawIntBits(pts[ptsOffset + LANE_LEFT_DIM]);
        int leftIdx = unpackLeft(leftAndDim);
        int rightIdx = Float.floatToRawIntBits(pts[ptsOffset + LANE_RIGHT]);

        for (int d = 0; d < DIMS; d++) {
            float dimValue = pts[ptsOffset + d];
            out[d] = dimValue;
            out[DIMS + d] = dimValue;
        }
        if (leftIdx >= 0) {
            float[] leftOut = new float[DIMS * 2];
            computeTopBboxesRec(pts, leftIdx, depth + 1, topSlot, topBbox, nextSlot, leftOut);
            for (int d = 0; d < DIMS; d++) {
                if (leftOut[d] < out[d]) out[d] = leftOut[d];
                if (leftOut[DIMS + d] > out[DIMS + d]) out[DIMS + d] = leftOut[DIMS + d];
            }
        }
        if (rightIdx >= 0) {
            float[] rightOut = new float[DIMS * 2];
            computeTopBboxesRec(pts, rightIdx, depth + 1, topSlot, topBbox, nextSlot, rightOut);
            for (int d = 0; d < DIMS; d++) {
                if (rightOut[d] < out[d]) out[d] = rightOut[d];
                if (rightOut[DIMS + d] > out[DIMS + d]) out[DIMS + d] = rightOut[DIMS + d];
            }
        }
        if (depth <= TOP_BBOX_DEPTH) {
            int slot = nextSlot[0]++;
            topSlot[treeIdx] = slot;
            int base = slot * DIMS * 2;
            System.arraycopy(out, 0, topBbox, base, DIMS * 2);
        }
    }

    private static int buildRecursive(
        float[] srcVecs, boolean[] srcFraud, int[] indices, int from, int to, int depth,
        float[] pts, int[] origId, byte[] fraud, int[] nextIdx, Random rng
    ) {
        if (from >= to) return -1;
        int splitDim = chooseSplitDim(srcVecs, indices, from, to);
        int splitPos = slidingMidpointTarget(srcVecs, indices, from, to, splitDim);
        quickselect(indices, from, to, splitPos, srcVecs, splitDim, rng);

        int treeIdx = nextIdx[0]++;
        int srcId = indices[splitPos];
        System.arraycopy(srcVecs, srcId * STRIDE, pts, treeIdx * STRIDE, STRIDE);
        origId[treeIdx] = srcId;
        fraud[treeIdx] = (byte) (srcFraud[srcId] ? 1 : 0);

        int leftIdx = buildRecursive(srcVecs, srcFraud, indices, from, splitPos, depth + 1,
            pts, origId, fraud, nextIdx, rng);
        int rightIdx = buildRecursive(srcVecs, srcFraud, indices, splitPos + 1, to, depth + 1,
            pts, origId, fraud, nextIdx, rng);

        pts[treeIdx * STRIDE + LANE_LEFT_DIM] = Float.intBitsToFloat(packLeftAndDim(leftIdx, splitDim));
        pts[treeIdx * STRIDE + LANE_RIGHT]   = Float.intBitsToFloat(rightIdx);
        return treeIdx;
    }

    /** Chosen by sweep; both visits_p99 and warm-p99 dip at 9 vs neighbors. Do not change without re-sweep. */
    private static final int IMBALANCE_CAP_DENOM = 9;

    /**
     * Sliding-midpoint: wall at geometric midpoint of (min, max), not the data median. Median
     * splits cluster the wall where queries cluster too, so |delta| stays tiny and slab pruning
     * rarely fires; midpoint splits put the wall away from clusters, |delta| grows, slab pruning
     * fires earlier. "Sliding" handles all-points-on-one-side by sliding past the data extreme.
     */
    private static int slidingMidpointTarget(float[] srcVecs, int[] indices, int from, int to, int splitDim) {
        int subtreeSize = to - from;
        if (subtreeSize <= 2) return from + (subtreeSize >>> 1);

        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int i = from; i < to; i++) {
            float dimValue = srcVecs[indices[i] * STRIDE + splitDim];
            if (dimValue < min) min = dimValue;
            if (dimValue > max) max = dimValue;
        }
        if (max == min) return from + (subtreeSize >>> 1);

        float midpoint = (min + max) * 0.5f;

        int countBelow = 0;
        for (int i = from; i < to; i++) {
            if (srcVecs[indices[i] * STRIDE + splitDim] < midpoint) countBelow++;
        }

        // Imbalance cap: pure sliding-midpoint degenerates into slide-by-1 chains in dense clusters.
        int minSide = Math.max(1, subtreeSize / IMBALANCE_CAP_DENOM);
        if (countBelow < minSide) countBelow = minSide;
        else if (countBelow > subtreeSize - minSide) countBelow = subtreeSize - minSide;

        return from + countBelow;
    }

    /** Largest sampled range across subtree points. Sampling cap (256) keeps build O(N log N). */
    private static int chooseSplitDim(float[] srcVecs, int[] indices, int from, int to) {
        int subtreeSize = to - from;
        int sampleStep = Math.max(1, subtreeSize / 256);
        float[] mins = new float[DIMS];
        float[] maxs = new float[DIMS];
        Arrays.fill(mins, Float.POSITIVE_INFINITY);
        Arrays.fill(maxs, Float.NEGATIVE_INFINITY);
        for (int i = from; i < to; i += sampleStep) {
            int srcOffset = indices[i] * STRIDE;
            for (int d = 0; d < DIMS; d++) {
                float dimValue = srcVecs[srcOffset + d];
                if (dimValue < mins[d]) mins[d] = dimValue;
                if (dimValue > maxs[d]) maxs[d] = dimValue;
            }
        }
        int bestDim = 0;
        float bestRange = -1f;
        for (int d = 0; d < DIMS; d++) {
            float range = maxs[d] - mins[d];
            if (range > bestRange) {
                bestRange = range;
                bestDim = d;
            }
        }
        return bestDim;
    }

    /** 3-way partition quickselect O(N) per level, robust to duplicates. */
    private static void quickselect(
        int[] indices, int from, int to, int targetPos,
        float[] srcVecs, int splitDim, Random rng
    ) {
        while (to - from > 1) {
            int pivotPos = from + rng.nextInt(to - from);
            float pivotVal = srcVecs[indices[pivotPos] * STRIDE + splitDim];

            int lessEnd = from;
            int cursor = from;
            int greaterStart = to;
            while (cursor < greaterStart) {
                float currentValue = srcVecs[indices[cursor] * STRIDE + splitDim];
                if (currentValue < pivotVal) {
                    int swap = indices[lessEnd]; indices[lessEnd] = indices[cursor]; indices[cursor] = swap;
                    lessEnd++; cursor++;
                } else if (currentValue > pivotVal) {
                    greaterStart--;
                    int swap = indices[cursor]; indices[cursor] = indices[greaterStart]; indices[greaterStart] = swap;
                } else {
                    cursor++;
                }
            }

            if (targetPos < lessEnd) {
                to = lessEnd;
            } else if (targetPos >= greaterStart) {
                from = greaterStart;
            } else {
                return;
            }
        }
    }
}
