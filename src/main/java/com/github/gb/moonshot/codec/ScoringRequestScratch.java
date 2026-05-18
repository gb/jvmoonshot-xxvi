package com.github.gb.moonshot.codec;

/**
 * Mutable, reusable parse result. Recycled across requests on the rinha-io thread for zero allocation
 * per parse. String-shaped fields are stored as byte-range views into the source body (offset + length);
 * the body byte array is owned by the NIO read buffer and never referenced across calls.
 * <p>
 * Fixed-length values (timestamps, mcc) store only an offset; variable-length values store offset + length.
 * Not thread-safe single-threaded by design. Callers must {@link #clear} before each new parse.
 */
public final class ScoringRequestScratch {

    /**
     * ISO-8601 {@code yyyy-MM-ddTHH:mm:ssZ} fixed length.
     */
    public static final int TIMESTAMP_LEN = 20;

    public static final int MCC_LEN = 4;

    /**
     * Sentinel for "field absent" / "last_transaction was null".
     */
    public static final int ABSENT = -1;

    public double amount;
    public int installments;
    public double customerAvgAmount;
    public int customerTxCount24h;
    public double merchantAvgAmount;
    public double kmFromHome;
    public boolean isOnline;
    public boolean cardPresent;
    public double lastKmFromCurrent;

    public int requestedAtOffset;

    /**
     * Offset of {@code last_transaction.timestamp}, or {@link #ABSENT} when {@code last_transaction == null}.
     */
    public int lastTimestampOffset;

    public int mccOffset;

    public int merchantIdOffset;
    public int merchantIdLength;
    public int merchantIdCode;
    public boolean merchantIdCodeValid;

    public int knownMerchantsCount;
    public int[] knownMerchantsOffsets = new int[16];
    public int[] knownMerchantsLengths = new int[16];
    public long knownMerchantMask;
    public boolean knownMerchantMaskValid;

    /**
     * Reset variable-cardinality state. Numeric/boolean fields are overwritten unconditionally on each parse.
     */
    public void clear() {
        knownMerchantsCount = 0;
        knownMerchantMask = 0L;
        knownMerchantMaskValid = true;
        merchantIdCode = -1;
        merchantIdCodeValid = false;
        lastTimestampOffset = ABSENT;
        requestedAtOffset = ABSENT;
    }

    /**
     * Grow {@code knownMerchants*} arrays on demand. Typical contest payload has 1-5 known merchants, so
     * the initial capacity (16) is never exceeded in steady state.
     */
    public void ensureKnownMerchantsCapacity(int n) {
        int cap = knownMerchantsOffsets.length;
        if (cap < n) {
            int newCap = Math.max(n, cap * 2);
            int[] newOffs = new int[newCap];
            int[] newLens = new int[newCap];
            System.arraycopy(knownMerchantsOffsets, 0, newOffs, 0, knownMerchantsCount);
            System.arraycopy(knownMerchantsLengths, 0, newLens, 0, knownMerchantsCount);
            knownMerchantsOffsets = newOffs;
            knownMerchantsLengths = newLens;
        }
    }
}
