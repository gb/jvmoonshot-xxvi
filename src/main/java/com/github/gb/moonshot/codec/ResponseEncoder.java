package com.github.gb.moonshot.codec;

import java.nio.charset.StandardCharsets;

/**
 * Pre-encoded full HTTP/1.1 responses for {@code /fraud-score}. fraud_score is one of
 * {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} and approved is determined by {@code fraud_score < 0.6} six possible
 * responses pre-framed once at class-load. Hot path is a single buffer copy.
 */
public final class ResponseEncoder {

    private static final byte[][] FULL_RESPONSES = new byte[6][];
    private static final byte[] READY_RESPONSE =
        ("HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nOK")
            .getBytes(StandardCharsets.UTF_8);
    private static final byte[] NOT_FOUND_RESPONSE =
        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.UTF_8);
    private static final byte[] NOT_READY_RESPONSE =
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.UTF_8);
    private static final byte[] BAD_REQUEST_CLOSE =
        "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            .getBytes(StandardCharsets.UTF_8);
    private static final byte[] PAYLOAD_TOO_LARGE_KEEPALIVE =
        "HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.UTF_8);

    static {
        for (int fraudCount = 0; fraudCount <= 5; fraudCount++) {
            double score = fraudCount / 5.0;
            boolean approved = score < 0.6;
            String body = "{\"approved\":" + approved + ",\"fraud_score\":" + formatScore(score) + "}";
            byte[] bodyBytes = body.getBytes(StandardCharsets.UTF_8);
            String head = "HTTP/1.1 200 OK\r\n"
                + "Content-Type: application/json\r\n"
                + "Content-Length: " + bodyBytes.length + "\r\n"
                + "\r\n";
            byte[] headBytes = head.getBytes(StandardCharsets.UTF_8);
            byte[] full = new byte[headBytes.length + bodyBytes.length];
            System.arraycopy(headBytes, 0, full, 0, headBytes.length);
            System.arraycopy(bodyBytes, 0, full, headBytes.length, bodyBytes.length);
            FULL_RESPONSES[fraudCount] = full;
        }
    }

    private ResponseEncoder() {}

    public static byte[] fullResponseForFraudCount(int fraudCount) {
        return FULL_RESPONSES[fraudCount];
    }

    public static byte[] readyResponse() {
        return READY_RESPONSE;
    }

    /** 503 returned for {@code GET /ready} until the runtime warmup pump finishes. */
    public static byte[] notReadyResponse() {
        return NOT_READY_RESPONSE;
    }

    public static byte[] notFoundResponse() {
        return NOT_FOUND_RESPONSE;
    }

    /** 400 with {@code Connection: close} issued on malformed frames or wedged headers. */
    public static byte[] badRequestClose() {
        return BAD_REQUEST_CLOSE;
    }

    /** 413 with keep-alive issued when the request body exceeds the read buffer. */
    public static byte[] payloadTooLargeKeepAlive() {
        return PAYLOAD_TOO_LARGE_KEEPALIVE;
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
