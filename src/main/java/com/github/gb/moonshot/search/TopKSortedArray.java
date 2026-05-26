package com.github.gb.moonshot.search;

/**
 * Fixed-capacity k=5 candidate set kept in ascending raw-i16-distance order. Insertions use a backward
 * shift loop; at k=5 this outperforms a binary heap whose sift-down branches mispredict more than
 * a tight counter-bound loop. See {@code TopKMicroBench} for the A/B that confirmed the loop form.
 */
final class TopKSortedArray {

    static final int MAX_K = 5;

    private final int[] ids = new int[MAX_K];
    private final int[] sums = new int[MAX_K];
    private int size;

    void clear() {
        size = 0;
    }

    int size() {
        return size;
    }

    float peekDist() {
        return sums[size - 1] * KdTree.INV_SCALE_SQ;
    }

    int peekSum() {
        return sums[size - 1];
    }

    boolean contains(int id) {
        return switch (size) {
            case 5 -> ids[4] == id || ids[3] == id || ids[2] == id || ids[1] == id || ids[0] == id;
            case 4 -> ids[3] == id || ids[2] == id || ids[1] == id || ids[0] == id;
            case 3 -> ids[2] == id || ids[1] == id || ids[0] == id;
            case 2 -> ids[1] == id || ids[0] == id;
            case 1 -> ids[0] == id;
            default -> false;
        };
    }

    void ensureCapacity(int k) {
        if (k > MAX_K) throw new IllegalArgumentException(
                "TopKSortedArray fixed at k=" + MAX_K + "; caller requested k=" + k);
    }

    void push(int id, int sum) {
        int[] localSums = sums;
        int[] localIds = ids;
        int pos = size;
        while (pos > 0 && localSums[pos - 1] > sum) {
            localSums[pos] = localSums[pos - 1];
            localIds[pos] = localIds[pos - 1];
            pos--;
        }
        localSums[pos] = sum;
        localIds[pos] = id;
        size++;
    }

    /**
     * Fully-unrolled cmov-friendly insert for k=5: each position becomes either the shifted-up
     * existing value or the new (id, sum) pair, decided by independent compares. C2 compiles
     * the ternary chain to CMOV instructions on x86 — fixed-cycle, branch-mispredict-free.
     * Microbench shows ~30 % faster per call vs the shift loop on JDK 25.
     */
    void replaceFarthest(int id, int sum) {
        int[] s = sums;
        int[] i = ids;
        int s0 = s[0], s1 = s[1], s2 = s[2], s3 = s[3];
        int i0 = i[0], i1 = i[1], i2 = i[2], i3 = i[3];
        boolean lt0 = sum < s0;
        boolean lt1 = sum < s1;
        boolean lt2 = sum < s2;
        boolean lt3 = sum < s3;
        s[0] = lt0 ? sum : s0;
        i[0] = lt0 ? id  : i0;
        s[1] = lt1 ? (lt0 ? s0 : sum) : s1;
        i[1] = lt1 ? (lt0 ? i0 : id)  : i1;
        s[2] = lt2 ? (lt1 ? s1 : sum) : s2;
        i[2] = lt2 ? (lt1 ? i1 : id)  : i2;
        s[3] = lt3 ? (lt2 ? s2 : sum) : s3;
        i[3] = lt3 ? (lt2 ? i2 : id)  : i3;
        s[4] = lt3 ? s3  : sum;
        i[4] = lt3 ? i3  : id;
    }

    int countFrauds(byte[] fraud) {
        // k is capped at 5; switch avoids loop control overhead on the hot path.
        return switch (size) {
            case 5 -> fraud[ids[0]] + fraud[ids[1]] + fraud[ids[2]] + fraud[ids[3]] + fraud[ids[4]];
            case 4 -> fraud[ids[0]] + fraud[ids[1]] + fraud[ids[2]] + fraud[ids[3]];
            case 3 -> fraud[ids[0]] + fraud[ids[1]] + fraud[ids[2]];
            case 2 -> fraud[ids[0]] + fraud[ids[1]];
            case 1 -> fraud[ids[0]];
            default -> 0;
        };
    }

    private static final long NAV_BYTE_OFFSET = (long) KdTree.LANE_NAV * 2L;
    private static final long POINT_STRIDE_BYTES = (long) KdTree.STRIDE * 2L;

    int countFraudsFromMmap(long ptsBaseAddr) {
        // k is capped at 5; switch keeps access branchless inside each case.
        return switch (size) {
            case 5 -> unpackFraudAtAddr(ptsBaseAddr + (long) ids[0] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[1] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[2] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[3] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[4] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET);
            case 4 -> unpackFraudAtAddr(ptsBaseAddr + (long) ids[0] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[1] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[2] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[3] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET);
            case 3 -> unpackFraudAtAddr(ptsBaseAddr + (long) ids[0] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[1] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[2] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET);
            case 2 -> unpackFraudAtAddr(ptsBaseAddr + (long) ids[0] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET)
                    + unpackFraudAtAddr(ptsBaseAddr + (long) ids[1] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET);
            case 1 -> unpackFraudAtAddr(ptsBaseAddr + (long) ids[0] * POINT_STRIDE_BYTES + NAV_BYTE_OFFSET);
            default -> 0;
        };
    }

    private static int unpackFraudAtAddr(long navAddr) {
        return KdTreeLayout.unpackFraud(KdTreeUnsafe.UNSAFE.getInt(navAddr));
    }

    int drainAscending(int[] out) {
        int n = size;
        System.arraycopy(ids, 0, out, 0, n);
        size = 0;
        return n;
    }

    int copyIds(int[] out) {
        System.arraycopy(ids, 0, out, 0, size);
        return size;
    }

    int copySums(int[] out) {
        System.arraycopy(sums, 0, out, 0, size);
        return size;
    }
}
