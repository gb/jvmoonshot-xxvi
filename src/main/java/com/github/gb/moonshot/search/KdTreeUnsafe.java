package com.github.gb.moonshot.search;

import sun.misc.Unsafe;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;

/**
 * Raw Unsafe read for {@link KdTree}'s packed-nav reads from mmap'd pts.
 * leftAndDim is packed as two consecutive i16 shorts at pts byte-offset
 * {@code (treeIdx*STRIDE + LANE_LEFT_DIM)*2}. Reading a LE int32 at that offset
 * reconstructs leftAndDim directly (same byte order).
 */
public final class KdTreeUnsafe {

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
     * Native base address of the pts mmap segment; 0 in heap mode.
     */
    static long ptsBaseAddr;

    static void bindPtsSegment(MemorySegment seg) {
        ptsBaseAddr = (seg != null) ? seg.address() : 0L;
    }

    private KdTreeUnsafe() {
    }
}
