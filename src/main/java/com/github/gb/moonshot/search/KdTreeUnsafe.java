package com.github.gb.moonshot.search;

import sun.misc.Unsafe;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;

/**
 * Raw Unsafe read for {@link KdTree}'s mmap'd pts. The packed nav word is stored as two
 * consecutive i16 shorts at byte-offset {@code (treeIdx*STRIDE + LANE_NAV)*2}; reading
 * a LE int32 at that offset reconstructs it directly.
 *
 * <p>Software prefetch ({@code PREFETCHT0}) was investigated via {@code jdk.internal.misc.Unsafe.prefetchRead0}
 * but that method was removed in JDK 25. JNI/Panama downcall alternatives are not C2-intrinsified,
 * so their call overhead (~20-50 ns) exceeds the prefetch benefit. Prefetch sites remain in
 * {@link com.github.gb.moonshot.bench.PrefetchBench} as documentation.
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
