package com.github.gb.moonshot.bench;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Read/write for the {@code RKPB0001} payload-bank file format. Counterpart of
 * {@link BankIO} but holds raw JSON payload bytes (length-prefixed) rather
 * than pre-vectorized float arrays.
 *
 * <p>Produced by {@link BuildPayloadBank}, consumed by {@link P99Bench} in
 * {@code --mode warm-e2e}.
 *
 * <p>Format (binary, big-endian):
 * <pre>
 *   8 bytes            ASCII magic "RKPB0001"
 *   4 bytes            int32 count
 *   for each payload:
 *     4 bytes          int32 length (bytes)
 *     length bytes     raw JSON payload body
 * </pre>
 */
public final class PayloadBankIO {

    public static final String MAGIC = "RKPB0001";

    private static final int HEADER_BYTES      = 12;
    private static final int LENGTH_BYTES      = 4;
    private static final int MAGIC_BYTES       = 8;
    /** 1 MiB sanity ceiling per payload anything beyond that is a corrupted stream. */
    private static final int MAX_PAYLOAD_BYTES = 1 << 20;
    /** Sanity ceiling on the {@code count} field; anything above this on read is treated as corruption. */
    private static final int MAX_COUNT         = 100_000_000;

    private PayloadBankIO() {}

    public static long write(Path path, byte[][] payloads) throws IOException {
        try (FileChannel ch = FileChannel.open(path,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            writeHeader(ch, payloads.length);
            writePayloads(ch, payloads);
            return ch.size();
        }
    }

    private static void writeHeader(FileChannel ch, int count) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.BIG_ENDIAN);
        header.put(MAGIC.getBytes(StandardCharsets.US_ASCII));
        header.putInt(count);
        header.flip();
        BankFileFormat.drain(ch, header);
    }

    private static void writePayloads(FileChannel ch, byte[][] payloads) throws IOException {
        ByteBuffer lengthBuf = ByteBuffer.allocate(LENGTH_BYTES).order(ByteOrder.BIG_ENDIAN);
        for (byte[] payload : payloads) {
            lengthBuf.clear();
            lengthBuf.putInt(payload.length);
            lengthBuf.flip();
            BankFileFormat.drain(ch, lengthBuf);
            BankFileFormat.drain(ch, ByteBuffer.wrap(payload));
        }
    }

    public static byte[][] read(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            int count = readHeader(ch);
            return readPayloads(ch, count);
        }
    }

    private static int readHeader(FileChannel ch) throws IOException {
        ByteBuffer head = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.BIG_ENDIAN);
        BankFileFormat.fillFully(ch, head, "header");
        head.flip();
        BankFileFormat.requireMagic(head, MAGIC, MAGIC_BYTES, "BuildPayloadBank");
        int count = head.getInt();
        if (count < 0 || count > MAX_COUNT) {
            throw new IOException("implausible payload count " + count + " (max=" + MAX_COUNT + ")");
        }
        return count;
    }

    private static byte[][] readPayloads(FileChannel ch, int count) throws IOException {
        byte[][] payloads = new byte[count][];
        ByteBuffer lengthBuf = ByteBuffer.allocate(LENGTH_BYTES).order(ByteOrder.BIG_ENDIAN);
        for (int i = 0; i < count; i++) {
            payloads[i] = readOnePayload(ch, lengthBuf, i);
        }
        return payloads;
    }

    private static byte[] readOnePayload(FileChannel ch, ByteBuffer lengthBuf, int index) throws IOException {
        lengthBuf.clear();
        BankFileFormat.fillFully(ch, lengthBuf, "length at i=" + index);
        lengthBuf.flip();
        int size = lengthBuf.getInt();
        if (size < 0 || size > MAX_PAYLOAD_BYTES) {
            throw new IOException("implausible payload size " + size + " at i=" + index);
        }
        ByteBuffer body = ByteBuffer.allocate(size);
        BankFileFormat.fillFully(ch, body, "body at i=" + index);
        return body.array();
    }
}
