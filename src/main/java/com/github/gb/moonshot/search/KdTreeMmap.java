package com.github.gb.moonshot.search;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.MappedByteBuffer;

/**
 * Mmap-region helpers: linux madvise downcall + page-warm loop. Boot-time only never invoked
 * on the request hot path. Kept out of {@link KdTree} so the FFI boilerplate doesn't clutter
 * the search/distance code.
 */
public final class KdTreeMmap {

    public static final int MADV_RANDOM = 1;
    public static final int MADV_HUGEPAGE = 14;
    public static final int LINUX_PAGE_BYTES = 4096;

    private static final MethodHandle MADVISE;
    public static final String MADVISE_DIAG;

    static {
        MethodHandle madviseHandle = null;
        String diag;
        try {
            Linker linker = Linker.nativeLinker();
            SymbolLookup lookup = linker.defaultLookup();
            var symbol = lookup.find("madvise");
            if (symbol.isEmpty()) {
                diag = "symbol 'madvise' not found in defaultLookup";
            } else {
                madviseHandle = linker.downcallHandle(symbol.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
                diag = "ok";
            }
        } catch (Throwable t) {
            diag = "throw during lookup: " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
        MADVISE = madviseHandle;
        MADVISE_DIAG = diag;
    }

    private KdTreeMmap() {
    }

    /**
     * madvise an mmap'd MemorySegment. madvise requires page-aligned start; regions start at
     * file offsets that may not be page-aligned, so we skip the leading partial page and advise
     * the page-aligned subrange. Returns 0 on success, -1 on failure (errno not captured),
     * Integer.MIN_VALUE if FFM lookup failed or the segment is null.
     */
    public static int madvise(MemorySegment segment, int advice) {
        if (MADVISE == null || segment == null) return Integer.MIN_VALUE;
        try {
            long startAddr = segment.address();
            long byteSize = segment.byteSize();
            long misalign = startAddr & (LINUX_PAGE_BYTES - 1);
            long bytesToPageBoundary = (misalign == 0) ? 0 : (LINUX_PAGE_BYTES - misalign);
            long alignedLen = (byteSize - bytesToPageBoundary) & ~((long) LINUX_PAGE_BYTES - 1);
            if (alignedLen <= 0) return -1;
            MemorySegment aligned = segment.asSlice(bytesToPageBoundary, alignedLen);
            return (int) MADVISE.invoke(aligned, alignedLen, advice);
        } catch (Throwable t) {
            System.out.println("[kdtree] madvise invoke threw: " + t);
            return -1;
        }
    }

    /**
     * Touch one byte per 4 KB page across the entire region. Returns a DCE-defeat XOR.
     */
    public static int touchPages(MappedByteBuffer region) {
        int end = region.limit();
        int sink = 0;
        for (int i = 0; i < end; i += LINUX_PAGE_BYTES) sink ^= region.get(i);
        return sink;
    }
}
