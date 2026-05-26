package com.github.gb.moonshot.codec;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * Pre-encoded full HTTP/1.1 responses for {@code /fraud-score}. fraud_score is one of
 * {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and approved is determined by {@code fraud_score < 0.6}, so six
 * possible responses are pre-framed once at class load. Hot path takes a per-connection
 * {@link ByteBuffer#duplicate()} of the shared immutable direct buffer and hands it to
 * {@link java.nio.channels.GatheringByteChannel#write(ByteBuffer[], int, int)} — no per-response memcpy.
 */
public final class ResponseEncoder {

    /** Stable response indices; also used as slots in the per-connection duplicate cache. */
    public static final int RESP_FRAUD_0 = 0;
    public static final int RESP_FRAUD_1 = 1;
    public static final int RESP_FRAUD_2 = 2;
    public static final int RESP_FRAUD_3 = 3;
    public static final int RESP_FRAUD_4 = 4;
    public static final int RESP_FRAUD_5 = 5;
    public static final int RESP_READY = 6;
    public static final int RESP_NOT_READY = 7;
    public static final int RESP_NOT_FOUND = 8;
    public static final int RESP_BAD_REQUEST_CLOSE = 9;
    public static final int RESP_PAYLOAD_TOO_LARGE = 10;
    public static final int RESP_COUNT = 11;

    /** Shared, never-mutated direct buffers — callers duplicate() before writing. */
    private static final ByteBuffer[] SHARED = new ByteBuffer[RESP_COUNT];
    /** Heap copies retained for tests / benches that need byte[] inspection. Not used on the hot path. */
    private static final byte[][] HEAP_COPIES = new byte[RESP_COUNT][];
    /** Pre-computed lengths so per-conn duplicates can reset limit without re-reading. */
    private static final int[] LENGTHS = new int[RESP_COUNT];

    static {
        // Fraud-score table — six approved/score pairs.
        for (int fraudCount = 0; fraudCount <= 5; fraudCount++) {
            double score = fraudCount / 5.0;
            boolean approved = score < 0.6;
            String body = "{\"approved\":" + approved + ",\"fraud_score\":" + formatScore(score) + "}";
            byte[] bodyBytes = body.getBytes(StandardCharsets.UTF_8);
            String head = "HTTP/1.1 200 OK\r\n"
                    + "Content-Length: " + bodyBytes.length + "\r\n"
                    + "\r\n";
            byte[] headBytes = head.getBytes(StandardCharsets.UTF_8);
            byte[] full = new byte[headBytes.length + bodyBytes.length];
            System.arraycopy(headBytes, 0, full, 0, headBytes.length);
            System.arraycopy(bodyBytes, 0, full, headBytes.length, bodyBytes.length);
            install(fraudCount, full);
        }

        install(RESP_READY,
                "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK".getBytes(StandardCharsets.UTF_8));
        install(RESP_NOT_READY,
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.UTF_8));
        install(RESP_NOT_FOUND,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.UTF_8));
        install(RESP_BAD_REQUEST_CLOSE,
                "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".getBytes(StandardCharsets.UTF_8));
        install(RESP_PAYLOAD_TOO_LARGE,
                "HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.UTF_8));
    }

    private static void install(int idx, byte[] bytes) {
        // allocateDirect maps an anonymous page that is COW-zero until first touch.
        // Touch each page (only 1 page since <4 KB) immediately so the first channel.write
        // doesn't fault on the hot IO thread.
        ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length);
        buf.put(bytes).flip();
        buf.get(0); // touch first byte
        SHARED[idx] = buf.asReadOnlyBuffer(); // freezes content; duplicates inherit readonly view
        HEAP_COPIES[idx] = bytes;
        LENGTHS[idx] = bytes.length;
    }

    private ResponseEncoder() {
    }

    /**
     * Return a fresh per-write view of response {@code idx}. The returned ByteBuffer aliases
     * the shared direct memory; position/limit are owned by the caller. ~40 B young-gen alloc
     * per call — orders of magnitude cheaper than the memcpy it replaces.
     */
    public static ByteBuffer duplicateFor(int idx) {
        return SHARED[idx].duplicate();
    }

    public static int responseLength(int idx) {
        return LENGTHS[idx];
    }

    /** Map a fraud-neighbor count (0..5) to its response index. */
    public static int fraudCountToIndex(int fraudCount) {
        return fraudCount; // RESP_FRAUD_0..RESP_FRAUD_5 are 0..5 by construction.
    }

    /** Test/bench-only byte[] view of a pre-framed response. Do NOT use on the hot path. */
    public static byte[] fullResponseForFraudCount(int fraudCount) {
        return HEAP_COPIES[fraudCount];
    }

    /** Test/bench-only byte[] view of an arbitrary response by index. */
    public static byte[] responseBytes(int idx) {
        return HEAP_COPIES[idx];
    }

    private static String formatScore(double score) {
        if (score == 0.0) return "0.0";
        if (score == 0.2) return "0.2";
        if (score == 0.4) return "0.4";
        if (score == 0.6) return "0.6";
        if (score == 0.8) return "0.8";
        return "1.0";
    }
}
