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
    /** Linux 5.14+: synchronously populate page cache by read-faulting the entire range. */
    public static final int MADV_POPULATE_READ = 22;
    /** Linux 6.1+: synchronously promote a range to transparent hugepages. */
    public static final int MADV_COLLAPSE = 25;
    public static final int LINUX_PAGE_BYTES = 4096;

    private static final MethodHandle MADVISE;
    public static final String MADVISE_DIAG;
    private static final MethodHandle MLOCK;
    public static final String MLOCK_DIAG;

    static {
        MethodHandle madviseHandle = null;
        MethodHandle mlockHandle = null;
        String diag;
        String mlockDiag;
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

            var mlockSymbol = lookup.find("mlock");
            if (mlockSymbol.isEmpty()) {
                mlockDiag = "symbol 'mlock' not found in defaultLookup";
            } else {
                mlockHandle = linker.downcallHandle(mlockSymbol.get(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT,
                                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
                mlockDiag = "ok";
            }
        } catch (Throwable t) {
            diag = "throw during lookup: " + t.getClass().getSimpleName() + ": " + t.getMessage();
            mlockDiag = "throw during lookup: " + t.getClass().getSimpleName() + ": " + t.getMessage();
        }
        MADVISE = madviseHandle;
        MADVISE_DIAG = diag;
        MLOCK = mlockHandle;
        MLOCK_DIAG = mlockDiag;
    }

    private KdTreeMmap() {
    }

    /**
     * mlock the leading {@code bytes} of an mmap'd MemorySegment so the kernel cannot evict
     * those pages under memory pressure. Page-aligns the start; returns 0 on success, -1 on
     * failure, Integer.MIN_VALUE if FFM lookup failed or the segment is null.
     */
    public static int mlock(MemorySegment segment, long bytes) {
        if (MLOCK == null || segment == null) return Integer.MIN_VALUE;
        try {
            long startAddr = segment.address();
            long byteSize = segment.byteSize();
            long lockLen = Math.min(byteSize, bytes);
            long misalign = startAddr & (LINUX_PAGE_BYTES - 1);
            long bytesToPageBoundary = (misalign == 0) ? 0 : (LINUX_PAGE_BYTES - misalign);
            long alignedLen = (lockLen - bytesToPageBoundary) & ~((long) LINUX_PAGE_BYTES - 1);
            if (alignedLen <= 0) return -1;
            MemorySegment aligned = segment.asSlice(bytesToPageBoundary, alignedLen);
            return (int) MLOCK.invoke(aligned, alignedLen);
        } catch (Throwable t) {
            System.out.println("[kdtree] mlock invoke threw: " + t);
            return -1;
        }
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

    /**
     * Synchronously promote the mmap range to transparent hugepages (MADV_COLLAPSE) and
     * populate the page cache (MADV_POPULATE_READ). One syscall per advice, both kernel-side
     * loops, much faster than a Java per-page touch loop on HDD-backed page cache because
     * the kernel batches reads.
     *
     * <p>Returns {@code true} if {@code MADV_POPULATE_READ} succeeded (region is guaranteed
     * resident in page cache after this call). Returns {@code false} if the syscall is
     * unavailable on this kernel (Linux &lt; 5.14) or fails for any reason — caller should
     * fall back to {@link #touchPages}.
     *
     * <p>{@code MADV_COLLAPSE} is best-effort: failure is logged but does not abort the
     * sequence. On kernels &lt; 6.1 or under memory pressure the kernel may decline; the
     * existing {@code MADV_HUGEPAGE} advice from {@link KdTree#applyMmapHints} still
     * influences future page allocations.
     *
     * <p>Boot-log lines from this method are the primary diagnostic for contest-box behaviour:
     * if {@code MADV_POPULATE_READ ok} doesn't appear, the legacy touchPages path ran instead,
     * which on HDD-backed storage is significantly slower.
     */
    public static boolean prewarmSync(MemorySegment segment) {
        if (segment == null) return false;
        // POPULATE_READ: synchronously faults every 4 KB page into the page cache via kernel-
        // side readahead. On HDD-backed storage this is materially faster than a Java per-page
        // touch loop because the kernel batches reads. On warm page cache it's near-instant.
        int populate = madvise(segment, MADV_POPULATE_READ);
        if (populate < 0 || populate == Integer.MIN_VALUE) {
            System.out.println("[kdtree] MADV_POPULATE_READ unavailable (kernel < 5.14 or rc="
                    + populate + "); falling back to per-page touchPages loop");
            return false;
        }
        System.out.println("[kdtree] MADV_POPULATE_READ ok (page cache prewarmed via single syscall)");
        // COLLAPSE: best-effort, expected to return rc=-1 on file-backed regular-file mmap
        // (Linux 6.x: MADV_COLLAPSE primarily targets anonymous memory and shmem; ext4 file-
        // backed THP support is not yet generally available). Advisory MADV_HUGEPAGE from
        // KdTree.applyMmapHints still drives opportunistic THP allocation by the kernel.
        // Keep the call so a future kernel/FS combo that gains file-backed THP starts
        // benefitting without a code change. Quiet log on the expected-failure path.
        int collapse = madvise(segment, MADV_COLLAPSE);
        if (collapse == 0) {
            System.out.println("[kdtree] MADV_COLLAPSE ok (range synchronously promoted to THP)");
        }
        return true;
    }

    /**
     * Map and madvise the AOT cache file (/app/app.aot) with MADV_DONTNEED to drop its pages
     * from the page cache after CDS startup. Reclaims ~30-50 MB the JVM no longer needs.
     */
    public static void releaseAotCachePages() {
        String pathStr = "/app/app.aot";
        java.nio.file.Path path = java.nio.file.Path.of(pathStr);
        if (!java.nio.file.Files.exists(path)) {
            return;
        }
        try (var channel = java.nio.channels.FileChannel.open(path, java.nio.file.StandardOpenOption.READ)) {
            long size = channel.size();
            if (size <= 0) return;
            java.nio.MappedByteBuffer buffer = channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, size);
            MemorySegment segment = MemorySegment.ofBuffer(buffer);
            int MADV_DONTNEED = 4;
            int rc = madvise(segment, MADV_DONTNEED);
            System.out.println("[boot] released AOT cache page cache: rc=" + rc + " (size=" + size + " bytes)");
        } catch (Throwable t) {
            System.out.println("[boot] failed to release AOT page cache: " + t.getMessage());
        }
    }
}
