package com.github.gb.moonshot.search;

import sun.misc.Unsafe;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;

/**
 * Raw-{@link Unsafe} read path for {@link KdTree}'s packed-nav reads. Replaces
 * {@code MemorySegment.get(INT_LE, off)} on the per-visit nav reads the Panama VarHandle
 * guard chain (checkSegment/checkEnclosingLayout/isAlignedForElement/checkBounds) was a
 * material chunk of descent CPU at ~2 nav reads x several thousand visits per query.
 *
 * <p>CAUTION: safety invariant. The pts mmap region is read-only, never moves once
 * {@link KdTreeIO#loadMmap} returns, and lives for the JVM's life. Tree indices arriving
 * here are always in {@code [0, n)} by construction the packed-nav encoding uses -1 as
 * the absent-child sentinel and every descent entry point checks
 * {@code if (treeIdx < 0) return} before any nav read. Within those bounds the address
 * {@code ptsBaseAddr + treeIdx * STRIDE * 4} stays inside the mapped region.
 *
 * <p>Heap mode: {@code ptsBaseAddr} is 0 and the heap-mode {@code pts != null} branch in
 * {@link KdTree#leftAndDimAt}/{@link KdTree#rightAt} short-circuits before any Unsafe call.
 */
public final class KdTreeUnsafe {

    /** Package-visible so descent inlines {@code UNSAFE.getInt} via {@code getstatic}. */
    static final Unsafe UNSAFE;
    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * Native base address of the pts mmap segment, captured once at tree load. Zero in heap
     * mode. Package-visible so descent reads it via {@code getstatic} (no method call).
     */
    static long ptsBaseAddr;

    static void bindPtsSegment(MemorySegment ptsSeg) {
        ptsBaseAddr = (ptsSeg != null) ? ptsSeg.address() : 0L;
    }

    private KdTreeUnsafe() {}
}
