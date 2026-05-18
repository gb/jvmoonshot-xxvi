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
        if (size > 0 && ids[0] == id) return true;
        if (size > 1 && ids[1] == id) return true;
        if (size > 2 && ids[2] == id) return true;
        if (size > 3 && ids[3] == id) return true;
        if (size > 4 && ids[4] == id) return true;
        return false;
    }

    void ensureCapacity(int k) {
        if (k > MAX_K) throw new IllegalArgumentException(
                "TopKSortedArray fixed at k=" + MAX_K + "; caller requested k=" + k);
    }

    void push(int id, int sum) {
        int pos = size;
        while (pos > 0 && sums[pos - 1] > sum) {
            sums[pos] = sums[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        sums[pos] = sum;
        ids[pos] = id;
        size++;
    }

    void replaceFarthest(int id, int sum) {
        int pos = MAX_K - 1;
        while (pos > 0 && sums[pos - 1] > sum) {
            sums[pos] = sums[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        sums[pos] = sum;
        ids[pos] = id;
    }

    int countFrauds(byte[] fraud) {
        int count = 0;
        for (int i = 0; i < size; i++) count += fraud[ids[i]];
        return count;
    }

    /**
     * Mmap-mode fraud count: reads the fraud flag from lane {@link KdTree#LANE_FRAUD} of each
     * result node's pts block, eliminating the separate on-heap {@code fraud[]} array.
     * Each read lands in the same 40-byte cache line that was fetched during the search.
     */
    int countFraudsFromMmap(long ptsBaseAddr, int stride) {
        int count = 0;
        long fraudByteOffset = (long) KdTree.LANE_FRAUD * 2L; // short offset → byte offset
        for (int i = 0; i < size; i++) {
            count += KdTreeUnsafe.UNSAFE.getShort(
                    ptsBaseAddr + (long) ids[i] * ((long) stride * 2L) + fraudByteOffset) & 1;
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
