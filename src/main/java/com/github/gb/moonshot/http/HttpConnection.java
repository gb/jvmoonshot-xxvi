package com.github.gb.moonshot.http;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Per-connection state + single-pass HTTP/1.1 header parser. Recycled across pipelined keep-alive requests on the
 * same socket.
 * <p>
 * Fields are package-private intentionally: {@link NioHttpServer} reads {@code readBuf.array()}, {@code bodyStart},
 * {@code bodyLen} directly on the per-request hot path so the JIT can inline access without going through accessors.
 * Do not add getters.
 */
final class HttpConnection {

    // Int codes (not enum) so drainRequests's four-arm dispatch stays a tableswitch the JIT can fold cleanly.
    static final int NEED_MORE = 0;
    static final int READY = 1;
    static final int MALFORMED = 2;
    static final int TOO_LARGE = 3;

    static final int READ_BUF_SIZE = 4096;
    static final int WRITE_BUF_SIZE = 4096;

    // Lowercase: callers OR each input byte with 0x20, idempotent on '-', ':', and lowercase ASCII.
    private static final byte[] CONTENT_LENGTH_LC = "content-length:".getBytes(StandardCharsets.US_ASCII);

    // SWAR scan for '\r' (0x0d): JIT intrinsifies byteArrayViewVarHandle::get(long) into a single unaligned MOV on
    // x86-64; the hasValue pattern detects any matching byte across 8 lanes in parallel.
    private static final VarHandle LONG_LE =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final long CR_BROADCAST = 0x0d0d0d0d0d0d0d0dL;
    private static final long LOW_BITS = 0x0101010101010101L;
    private static final long HIGH_BITS = 0x8080808080808080L;

    final ByteBuffer readBuf = ByteBuffer.allocate(READ_BUF_SIZE);
    // Direct write buffer avoids the JDK's per-write heap-to-native staging copy on SocketChannel.write().
    // Responses are tiny but p99 is hypersensitive to extra memcpy/JNI staging on the single IO thread.
    final ByteBuffer writeBuf = ByteBuffer.allocateDirect(WRITE_BUF_SIZE);

    int routeId = Router.ROUTE_NOT_FOUND;
    int bodyStart = -1;
    int bodyLen = -1;
    boolean closeAfterWrite = false;
    int bytesToDrain = 0;

    boolean isDraining() {
        return bytesToDrain > 0;
    }

    int tryParse() {
        if (bodyStart < 0) {
            byte[] data = readBuf.array();
            int n = readBuf.position();
            if (n < 4) return NEED_MORE;

            byte method = data[0];
            routeId = method == 'P' ? Router.ROUTE_FRAUD_SCORE
                    : method == 'G' ? Router.ROUTE_READY
                    : Router.ROUTE_NOT_FOUND;

            int contentLen = 0;

            int cr = findCrLf(data, 0, n);
            if (cr < 0) return NEED_MORE;
            int i = cr + 2;

            while (true) {
                if (i + 1 >= n) return NEED_MORE;
                if (data[i] == '\r' && data[i + 1] == '\n') {
                    i += 2;
                    break;
                }

                // Gate the 15-byte case-fold compare on a single-byte match so every non-C line rejects after one
                // byte. Unrolled matcher avoids repeated prefix-array loads.
                if ((data[i] | 0x20) == 'c' && hasContentLengthHeader(data, i, n)) {
                    int v = i + CONTENT_LENGTH_LC.length;
                    while (v < n && (data[v] == ' ' || data[v] == '\t')) v++;
                    while (v < n && data[v] >= '0' && data[v] <= '9') {
                        contentLen = contentLen * 10 + (data[v] - '0');
                        v++;
                    }
                }

                int next = findCrLf(data, i, n);
                if (next < 0) return NEED_MORE;
                i = next + 2;
            }

            bodyStart = i;
            bodyLen = contentLen;

            int requestEnd = bodyStart + bodyLen;
            // Body won't fit would loop NEED_MORE forever. Caller replies 413, drains, keeps alive.
            if (requestEnd > readBuf.capacity()) return TOO_LARGE;
        }
        int have = readBuf.position() - bodyStart;
        if (have < bodyLen) return NEED_MORE;
        return READY;
    }

    // Returns offset of '\r' such that data[cr+1] == '\n' within [from, limit), or -1 if no CRLF pair exists. A lone
    // '\r' (no '\n' follow-on) is skipped and the scan resumes one byte past it.
    private static int findCrLf(byte[] data, int from, int limit) {
        int i = from;
        while (i < limit) {
            int cr = indexOfCr(data, i, limit);
            if (cr < 0 || cr + 1 >= limit) return -1;
            if (data[cr + 1] == '\n') return cr;
            i = cr + 1;
        }
        return -1;
    }

    private static int indexOfCr(byte[] data, int from, int limit) {
        int i = from;
        int end = limit - 7;
        while (i <= end) {
            long w = (long) LONG_LE.get(data, i);
            long x = w ^ CR_BROADCAST;
            long m = (x - LOW_BITS) & ~x & HIGH_BITS;
            if (m != 0) {
                return i + (Long.numberOfTrailingZeros(m) >>> 3);
            }
            i += 8;
        }
        while (i < limit) {
            if (data[i] == '\r') return i;
            i++;
        }
        return -1;
    }

    void enterDrainMode() {
        int alreadyReceived = readBuf.position() - bodyStart;
        bytesToDrain = bodyLen - alreadyReceived;
        routeId = Router.ROUTE_NOT_FOUND;
        bodyStart = -1;
        bodyLen = -1;
        readBuf.clear();
    }

    void advanceAfterRequest() {
        int requestEnd = bodyStart + bodyLen;
        int extra = readBuf.position() - requestEnd;
        if (extra > 0) {
            System.arraycopy(readBuf.array(), requestEnd, readBuf.array(), 0, extra);
        }
        readBuf.position(extra);
        routeId = Router.ROUTE_NOT_FOUND;
        bodyStart = -1;
        bodyLen = -1;
    }

    // Straight-line compares (no loop) keep the parser branch-hot and avoid repeated prefix-array loads. The | 0x20
    // fold is idempotent on '-', ':', and lowercase ASCII.
    private static boolean hasContentLengthHeader(byte[] data, int from, int n) {
        return n - from >= CONTENT_LENGTH_LC.length
                && (data[from] | 0x20) == 'c'
                && (data[from + 1] | 0x20) == 'o'
                && (data[from + 2] | 0x20) == 'n'
                && (data[from + 3] | 0x20) == 't'
                && (data[from + 4] | 0x20) == 'e'
                && (data[from + 5] | 0x20) == 'n'
                && (data[from + 6] | 0x20) == 't'
                && data[from + 7] == '-'
                && (data[from + 8] | 0x20) == 'l'
                && (data[from + 9] | 0x20) == 'e'
                && (data[from + 10] | 0x20) == 'n'
                && (data[from + 11] | 0x20) == 'g'
                && (data[from + 12] | 0x20) == 't'
                && (data[from + 13] | 0x20) == 'h'
                && data[from + 14] == ':';
    }
}
