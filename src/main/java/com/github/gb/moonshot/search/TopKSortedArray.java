package com.github.gb.moonshot.search;

/**
 * Fixed-capacity k=5 candidate set kept in ascending distance order. Insertions use a backward
 * shift loop; at k=5 this outperforms a binary heap whose sift-down branches mispredict more than
 * a tight counter-bound loop. See {@code TopKMicroBench} for the A/B that confirmed the loop form.
 */
final class TopKSortedArray {

    static final int MAX_K = 5;

    private final int[] ids = new int[MAX_K];
    private final float[] dists = new float[MAX_K];
    private int size;

    void clear() {
        size = 0;
    }

    int size() {
        return size;
    }

    float peekDist() {
        return dists[size - 1];
    }

    boolean contains(int id) {
        for (int i = 0; i < size; i++) if (ids[i] == id) return true;
        return false;
    }

    void ensureCapacity(int k) {
        if (k > MAX_K) throw new IllegalArgumentException(
                "TopKSortedArray fixed at k=" + MAX_K + "; caller requested k=" + k);
    }

    void push(int id, float dist) {
        int pos = size;
        while (pos > 0 && dists[pos - 1] > dist) {
            dists[pos] = dists[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        dists[pos] = dist;
        ids[pos] = id;
        size++;
    }

    void replaceFarthest(int id, float dist) {
        int pos = MAX_K - 1;
        while (pos > 0 && dists[pos - 1] > dist) {
            dists[pos] = dists[pos - 1];
            ids[pos] = ids[pos - 1];
            pos--;
        }
        dists[pos] = dist;
        ids[pos] = id;
    }

    int countFrauds(byte[] fraud) {
        int count = 0;
        for (int i = 0; i < size; i++) count += fraud[ids[i]];
        return count;
    }

    int drainAscending(int[] out) {
        int n = size;
        System.arraycopy(ids, 0, out, 0, n);
        size = 0;
        return n;
    }
}
