package com.github.gb.moonshot.search;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static com.github.gb.moonshot.search.KdTree.DIMS;
import static com.github.gb.moonshot.search.KdTree.STRIDE;

/**
 * On-disk format read/write for {@link KdTree}. Magic {@value #MAGIC}; bumped on each layout
 * change so an old artifact fails fast at {@link #loadMmap} instead of producing silent
 * garbage. Layout: pts in variance-descending lane order (see {@link KdTree#DIM_PERMUTATION}),
 * native little-endian (avoids per-load BSWAP). origId + topBbox follow pts so
 * {@link #loadMmap} can page-cache-back them off-heap.
 *
 * <p>Header: 8-byte magic + 4 little-endian ints (n, dims, stride, rootIdx). Bulk regions use
 * 8 MB chunked transfers.
 */
public final class KdTreeIO {

    static final String MAGIC = "RKDTR008";

    private static final int HEADER_BYTES = 8 + 4 * 4;
    private static final int IO_CHUNK_BYTES = 8 * 1024 * 1024;

    private KdTreeIO() {}

    public static void save(KdTree tree, Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file,
                 StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                 StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            header.put(MAGIC.getBytes());
            header.putInt(tree.n);
            header.putInt(DIMS);
            header.putInt(STRIDE);
            header.putInt(0); // root index always 0
            header.flip();
            writeFully(channel, header);

            writeFloats(channel, tree.pts, tree.n * STRIDE);
            writeInts(channel, tree.origId, tree.n);
            writeBytes(channel, tree.fraud);

            ByteBuffer meta = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            meta.putInt(tree.topNodeCount);
            meta.flip();
            writeFully(channel, meta);
            writeFloats(channel, tree.topBbox, tree.topNodeCount * DIMS * 2);
            writeInts(channel, tree.topSlot, tree.n);
        }
    }

    public static KdTree load(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            int nodeCount = readAndCheckHeader(channel);

            float[] pts = new float[nodeCount * STRIDE];
            readFloats(channel, pts, nodeCount * STRIDE);
            int[] origId = new int[nodeCount];
            readInts(channel, origId, nodeCount);
            byte[] fraud = new byte[nodeCount];
            readBytes(channel, fraud);

            ByteBuffer meta = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, meta);
            meta.flip();
            int topNodeCount = meta.getInt();
            float[] topBbox = new float[topNodeCount * DIMS * 2];
            readFloats(channel, topBbox, topNodeCount * DIMS * 2);
            int[] topSlot = new int[nodeCount];
            readInts(channel, topSlot, nodeCount);

            return new KdTree(nodeCount, pts, null, null, origId, null, fraud, topSlot, topBbox, null, topNodeCount);
        }
    }

    /**
     * Hybrid mmap loader for the production path. Three regions stay file-backed (page-cache):
     * pts (includes packed-nav fields at lanes 14-15), origId, and topBbox. The cgroup heap
     * can't hold any one of them, let alone all three. The small fraud byte[n] stays on heap
     * because it's read every query and the heap cost is trivial.
     */
    public static KdTree loadMmap(Path file) throws IOException {
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            int nodeCount = readAndCheckHeader(channel);

            long ptsOff = channel.position();
            long ptsLen = (long) nodeCount * STRIDE * 4;
            MappedByteBuffer ptsBuf = channel.map(FileChannel.MapMode.READ_ONLY, ptsOff, ptsLen);
            ptsBuf.order(ByteOrder.LITTLE_ENDIAN);
            FloatBuffer ptsFloats = ptsBuf.asFloatBuffer();
            channel.position(ptsOff + ptsLen);

            // origId stays mmap'd see field-level rationale.
            long origIdOff = channel.position();
            long origIdLen = (long) nodeCount * 4L;
            MappedByteBuffer origIdBuf = channel.map(FileChannel.MapMode.READ_ONLY, origIdOff, origIdLen);
            origIdBuf.order(ByteOrder.LITTLE_ENDIAN);
            channel.position(origIdOff + origIdLen);

            byte[] fraud = new byte[nodeCount];
            readBytes(channel, fraud);

            ByteBuffer meta = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            readFully(channel, meta);
            meta.flip();
            int topNodeCount = meta.getInt();

            // Mmap topBbox instead of heap-loading it would OOM the cgroup heap. Page-cache
            // backing is fine; the contest box has plenty of RAM outside the container budget.
            long topBboxOff = channel.position();
            long topBboxLen = (long) topNodeCount * DIMS * 2 * 4;
            MappedByteBuffer topBboxBuf = channel.map(FileChannel.MapMode.READ_ONLY, topBboxOff, topBboxLen);
            topBboxBuf.order(ByteOrder.LITTLE_ENDIAN);
            channel.position(topBboxOff + topBboxLen);

            int[] topSlot = new int[nodeCount];
            readInts(channel, topSlot, nodeCount);

            return new KdTree(nodeCount, null, ptsBuf, ptsFloats, null, origIdBuf, fraud, topSlot, null, topBboxBuf, topNodeCount);
        }
    }

    private static int readAndCheckHeader(FileChannel channel) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(channel, header);
        header.flip();
        byte[] magic = new byte[8];
        header.get(magic);
        if (!MAGIC.equals(new String(magic))) throw new IOException("bad magic: " + new String(magic));
        int nodeCount = header.getInt();
        int dims = header.getInt();
        int stride = header.getInt();
        int rootIdx = header.getInt();
        if (dims != DIMS || stride != STRIDE || rootIdx != 0) {
            throw new IOException("unexpected dims/stride/root: " + dims + "/" + stride + "/" + rootIdx);
        }
        return nodeCount;
    }

    private static void writeFully(FileChannel channel, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) channel.write(buf);
    }

    private static void readFully(FileChannel channel, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            if (channel.read(buf) < 0) throw new IOException("unexpected EOF");
        }
    }

    private static void writeBytes(FileChannel channel, byte[] arr) throws IOException {
        int off = 0;
        while (off < arr.length) {
            int len = Math.min(IO_CHUNK_BYTES, arr.length - off);
            writeFully(channel, ByteBuffer.wrap(arr, off, len));
            off += len;
        }
    }

    private static void readBytes(FileChannel channel, byte[] arr) throws IOException {
        int off = 0;
        while (off < arr.length) {
            int len = Math.min(IO_CHUNK_BYTES, arr.length - off);
            readFully(channel, ByteBuffer.wrap(arr, off, len));
            off += len;
        }
    }

    /** Copies {@code len} elements between a typed source/destination and the byte buffer at chunk boundary. */
    @FunctionalInterface
    private interface ChunkCopier {
        void copy(ByteBuffer buf, int srcOrDstOff, int len);
    }

    private static void writeChunked(FileChannel channel, int total, int bytesPerElement, ChunkCopier intoBuf)
            throws IOException {
        int elementsPerChunk = IO_CHUNK_BYTES / bytesPerElement;
        ByteBuffer buf = ByteBuffer.allocate(elementsPerChunk * bytesPerElement).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < total) {
            int len = Math.min(elementsPerChunk, total - off);
            buf.clear();
            intoBuf.copy(buf, off, len);
            buf.position(0).limit(len * bytesPerElement);
            writeFully(channel, buf);
            off += len;
        }
    }

    private static void readChunked(FileChannel channel, int total, int bytesPerElement, ChunkCopier fromBuf)
            throws IOException {
        int elementsPerChunk = IO_CHUNK_BYTES / bytesPerElement;
        ByteBuffer buf = ByteBuffer.allocate(elementsPerChunk * bytesPerElement).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < total) {
            int len = Math.min(elementsPerChunk, total - off);
            buf.clear();
            buf.limit(len * bytesPerElement);
            readFully(channel, buf);
            buf.flip();
            fromBuf.copy(buf, off, len);
            off += len;
        }
    }

    private static void writeInts(FileChannel channel, int[] arr, int total) throws IOException {
        writeChunked(channel, total, 4, (buf, off, len) -> buf.asIntBuffer().put(arr, off, len));
    }

    private static void readInts(FileChannel channel, int[] arr, int total) throws IOException {
        readChunked(channel, total, 4, (buf, off, len) -> buf.asIntBuffer().get(arr, off, len));
    }

    private static void writeFloats(FileChannel channel, float[] arr, int total) throws IOException {
        writeChunked(channel, total, 4, (buf, off, len) -> buf.asFloatBuffer().put(arr, off, len));
    }

    private static void readFloats(FileChannel channel, float[] arr, int total) throws IOException {
        readChunked(channel, total, 4, (buf, off, len) -> buf.asFloatBuffer().get(arr, off, len));
    }
}
