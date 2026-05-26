package com.github.gb.moonshot.search;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

import static com.github.gb.moonshot.search.KdTree.*;

/**
 * Persist and reload {@link KdTree}. Magic {@value #MAGIC}; bumped on layout changes.
 * Layout: short[] pts (n*STRIDE), int[] origId (n), byte[] fraud (n),
 * int topNodeCount, short[] topBbox (topNodeCount*STRIDE_BBOX), int[] topSlot (n).
 * All multi-byte values little-endian.
 * Note: right child, split dim, left-present flag, and fraud flag are packed into
 * pts[LANE_NAV..LANE_NAV+1]; there is no separate right[] section in the file.
 */
public final class KdTreeIO {

    static final String MAGIC = "RKDTS025"; // RKDT + S(short) + 025: nav adds hasBbox + childrenHaveBbox bits

    private static final int HEADER_BYTES = 64;
    private static final int IO_CHUNK = 8 * 1024 * 1024;

    private KdTreeIO() {
    }

    public static void save(KdTree tree, Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer hdr = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            hdr.put(MAGIC.getBytes());
            hdr.putInt(tree.n);
            hdr.putInt(DIMS);
            hdr.putInt(STRIDE);
            hdr.putInt(0); // root always 0
            hdr.position(HEADER_BYTES);
            hdr.flip();
            writeFully(ch, hdr);

            writeShorts(ch, tree.pts, tree.n * STRIDE);
            writeInts(ch, tree.origId, tree.n);
            writeBytes(ch, tree.fraud);

            ByteBuffer meta = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            meta.putInt(tree.topNodeCount);
            meta.flip();
            writeFully(ch, meta);

            writeShorts(ch, tree.topBbox, tree.topNodeCount * STRIDE_BBOX);
            writeInts(ch, tree.topSlot, tree.n);
        }
    }

    public static KdTree load(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            int n = readAndCheckHeader(ch);

            short[] pts = new short[n * STRIDE];
            readShorts(ch, pts, n * STRIDE);
            int[] origId = new int[n];
            readInts(ch, origId, n);
            byte[] fraud = new byte[n];
            readBytes(ch, fraud);

            ByteBuffer meta = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, meta);
            meta.flip();
            int topNodeCount = meta.getInt();

            short[] topBbox = new short[topNodeCount * STRIDE_BBOX];
            readShorts(ch, topBbox, topNodeCount * STRIDE_BBOX);
            int[] topSlot = new int[n];
            readInts(ch, topSlot, n);

            return new KdTree(n, pts, origId, fraud, topSlot, topBbox, topNodeCount);
        }
    }

    /**
     * Production loader: mmap pts (n*STRIDE*2 bytes) off the JVM heap to stay within -Xmx65m.
     * origId and fraud are skipped — fraud flag lives in the packed nav word,
     * read directly from the mmap in {@link TopKSortedArray#countFraudsFromMmap}.
     * Heap: topSlot+topBbox ≈ 28 MB for n=3M; pts is off-heap via mmap.
     */
    public static KdTree loadMmap(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            int n = readAndCheckHeader(ch);

            long ptsOff = ch.position();
            long ptsLen = (long) n * STRIDE * 2L;
            MappedByteBuffer ptsBuf = ch.map(FileChannel.MapMode.READ_ONLY, ptsOff, ptsLen);
            ptsBuf.order(ByteOrder.LITTLE_ENDIAN);
            ch.position(ptsOff + ptsLen);

            // Skip origId (not on hot path).
            ch.position(ch.position() + (long) n * 4L);
            // Skip separate fraud section — fraud is now in the packed nav word.
            ch.position(ch.position() + (long) n);

            ByteBuffer meta = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, meta);
            meta.flip();
            int topNodeCount = meta.getInt();

            short[] topBbox = new short[topNodeCount * STRIDE_BBOX];
            readShorts(ch, topBbox, topNodeCount * STRIDE_BBOX);
            int[] topSlot = new int[n];
            readInts(ch, topSlot, n);

            return new KdTree(n, ptsBuf, null, null, topSlot, topBbox, topNodeCount);
        }
    }

    private static int readAndCheckHeader(FileChannel ch) throws IOException {
        ByteBuffer hdr = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        readFully(ch, hdr);
        hdr.flip();
        byte[] magic = new byte[8];
        hdr.get(magic);
        String magicStr = new String(magic);
        if (!MAGIC.equals(magicStr)) throw new IOException("bad magic: " + magicStr);
        int n = hdr.getInt();
        int dims = hdr.getInt();
        int stride = hdr.getInt();
        int root = hdr.getInt();
        if (dims != DIMS || stride != STRIDE || root != 0)
            throw new IOException("unexpected dims/stride/root: " + dims + "/" + stride + "/" + root);
        return n;
    }

    private static void writeFully(FileChannel ch, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) ch.write(buf);
    }

    private static void readFully(FileChannel ch, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) {
            if (ch.read(buf) < 0) throw new IOException("unexpected EOF");
        }
    }

    private static void writeBytes(FileChannel ch, byte[] arr) throws IOException {
        int off = 0;
        while (off < arr.length) {
            int len = Math.min(IO_CHUNK, arr.length - off);
            writeFully(ch, ByteBuffer.wrap(arr, off, len));
            off += len;
        }
    }

    private static void readBytes(FileChannel ch, byte[] arr) throws IOException {
        int off = 0;
        while (off < arr.length) {
            int len = Math.min(IO_CHUNK, arr.length - off);
            readFully(ch, ByteBuffer.wrap(arr, off, len));
            off += len;
        }
    }

    private static void writeShorts(FileChannel ch, short[] arr, int total) throws IOException {
        int elemsPerChunk = IO_CHUNK / 2;
        ByteBuffer buf = ByteBuffer.allocate(Math.min(total, elemsPerChunk) * 2).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < total) {
            int len = Math.min(elemsPerChunk, total - off);
            buf.clear();
            buf.asShortBuffer().put(arr, off, len);
            buf.limit(len * 2);
            writeFully(ch, buf);
            off += len;
        }
    }

    private static void readShorts(FileChannel ch, short[] arr, int total) throws IOException {
        int elemsPerChunk = IO_CHUNK / 2;
        ByteBuffer buf = ByteBuffer.allocate(Math.min(total, elemsPerChunk) * 2).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < total) {
            int len = Math.min(elemsPerChunk, total - off);
            buf.clear();
            buf.limit(len * 2);
            readFully(ch, buf);
            buf.flip();
            buf.asShortBuffer().get(arr, off, len);
            off += len;
        }
    }

    private static void writeInts(FileChannel ch, int[] arr, int total) throws IOException {
        int elemsPerChunk = IO_CHUNK / 4;
        ByteBuffer buf = ByteBuffer.allocate(Math.min(total, elemsPerChunk) * 4).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < total) {
            int len = Math.min(elemsPerChunk, total - off);
            buf.clear();
            buf.asIntBuffer().put(arr, off, len);
            buf.limit(len * 4);
            writeFully(ch, buf);
            off += len;
        }
    }

    private static void readInts(FileChannel ch, int[] arr, int total) throws IOException {
        int elemsPerChunk = IO_CHUNK / 4;
        ByteBuffer buf = ByteBuffer.allocate(Math.min(total, elemsPerChunk) * 4).order(ByteOrder.LITTLE_ENDIAN);
        int off = 0;
        while (off < total) {
            int len = Math.min(elemsPerChunk, total - off);
            buf.clear();
            buf.limit(len * 4);
            readFully(ch, buf);
            buf.flip();
            buf.asIntBuffer().get(arr, off, len);
            off += len;
        }
    }
}
