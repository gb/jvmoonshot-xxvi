package com.github.gb.moonshot.http;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;

/**
 * AOT training hooks for the HTTP layer, called once at boot from {@link com.github.gb.moonshot.WarmupDriver} under
 * {@code AOT_TRAINING=1}. Lives apart from {@link NioHttpServer} so the runtime server class only contains
 * steady-state code paths.
 */
public final class NioAotTraining {

    private static final byte[] WARMUP_RESP =
            "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] WARMUP_REQ =
            "GET /ready HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"
                    .getBytes(StandardCharsets.US_ASCII);

    private static final byte[] CONTENT_LENGTH_LC =
            "content-length:".getBytes(StandardCharsets.US_ASCII);

    private NioAotTraining() {
    }

    /**
     * Pumps {@code iters} HTTP requests through the bound NIO server via real client sockets, spread across
     * {@code connections} sequential channels. Drives the full accept → select → handleRead → drainRequests →
     * flushWriteBuf chain the server-side hot path that {@link #trainAotHotPath} can't reach (that one bypasses
     * the SocketChannel/Selector layer and calls {@code HttpConnection.tryParse} directly).
     * <p>
     * Gets the JDK-internal NIO methods ({@code sun.nio.ch.Util$BufferCache}, {@code SocketChannelImpl},
     * {@code ReentrantLock$NonfairSync}, {@code Unsafe::copyMemory}) past C2's tier-4 threshold so they land in the
     * AOT cache instead of tier-4 compiling under k6 load.
     */
    public static int trainLiveListener(SocketAddress addr, byte[][] requestFrames, int iters,
                                        long deadlineNanos, int connections) {
        if (iters <= 0 || requestFrames.length == 0) return 0;
        if (connections <= 0) connections = 1;
        boolean unix = addr instanceof UnixDomainSocketAddress;
        StandardProtocolFamily family = unix ? StandardProtocolFamily.UNIX : StandardProtocolFamily.INET;

        int perConn = iters / connections;
        int remainder = iters - perConn * connections;
        int done = 0;

        for (int c = 0; c < connections && System.nanoTime() < deadlineNanos; c++) {
            int connIters = perConn + (c < remainder ? 1 : 0);
            if (connIters <= 0) continue;
            try (SocketChannel client = SocketChannel.open(family)) {
                client.configureBlocking(true);
                client.connect(addr);
                ByteBuffer response = ByteBuffer.allocate(512);
                int connDone = 0;
                while (connDone < connIters && System.nanoTime() < deadlineNanos) {
                    byte[] frame = requestFrames[(done + connDone) % requestFrames.length];
                    ByteBuffer request = ByteBuffer.wrap(frame);
                    while (request.hasRemaining()) client.write(request);
                    if (!readOneHttpResponse(client, response)) break;
                    connDone++;
                }
                done += connDone;
            } catch (Exception e) {
                System.err.println("[live-warmup] connection " + c + " failed after "
                        + done + "/" + iters + " iters: " + e.getClass().getSimpleName()
                        + ": " + e.getMessage());
            }
        }
        return done;
    }

    /**
     * Pumps real HTTP requests through the FD-passing entrypoint used by lapada:
     * one persistent Unix control connection (simulating lapada's persistent stream)
     * → {@link FdReceiver}'s recvLoop → {@link NioHttpServer}'s selector.
     * Exercises FdPassing.receive, FdConnHandler.wrapFd, NioHttpServer.injectChannel,
     * and the full NIO read/write path for FD-passed connections.
     */
    public static int trainFdPassing(String fdSocketPath, byte[][] requestFrames, int iters,
                                     long deadlineNanos, int connections) {
        if (fdSocketPath == null || fdSocketPath.isEmpty() || iters <= 0 || requestFrames.length == 0) return 0;
        if (connections <= 0) connections = 1;

        int perConn = iters / connections;
        int remainder = iters - perConn * connections;
        int done = 0;
        UnixDomainSocketAddress fdAddr = UnixDomainSocketAddress.of(fdSocketPath);

        try (ServerSocketChannel tcpServer = ServerSocketChannel.open(StandardProtocolFamily.INET)) {
            tcpServer.bind(new InetSocketAddress("127.0.0.1", 0), connections);
            tcpServer.configureBlocking(true);
            InetSocketAddress tcpAddr = (InetSocketAddress) tcpServer.getLocalAddress();

            // One persistent control connection — mirrors lapada's new architecture.
            try (SocketChannel ctrl = SocketChannel.open(StandardProtocolFamily.UNIX)) {
                ctrl.configureBlocking(true);
                ctrl.connect(fdAddr);
                int ctrlFd = FdReceiver.extractRawFd(ctrl);

                for (int c = 0; c < connections && System.nanoTime() < deadlineNanos; c++) {
                    int connIters = perConn + (c < remainder ? 1 : 0);
                    if (connIters <= 0) continue;
                    try (SocketChannel client = SocketChannel.open(StandardProtocolFamily.INET)) {
                        client.configureBlocking(true);
                        client.connect(tcpAddr);

                        SocketChannel accepted = tcpServer.accept();
                        try {
                            int acceptedFd = FdReceiver.extractRawFd(accepted);
                            if (!FdPassing.send(ctrlFd, acceptedFd)) {
                                throw new IOException("sendmsg(SCM_RIGHTS) returned short/error");
                            }
                        } finally {
                            accepted.close();
                        }

                        ByteBuffer response = ByteBuffer.allocate(512);
                        int connDone = 0;
                        while (connDone < connIters && System.nanoTime() < deadlineNanos) {
                            byte[] frame = requestFrames[(done + connDone) % requestFrames.length];
                            ByteBuffer request = ByteBuffer.wrap(frame);
                            while (request.hasRemaining()) client.write(request);
                            if (!readOneHttpResponse(client, response)) break;
                            connDone++;
                        }
                        done += connDone;
                    } catch (Exception e) {
                        System.err.println("[fd-warmup] connection " + c + " failed after "
                                + done + "/" + iters + " iters: " + e.getClass().getSimpleName()
                                + ": " + e.getMessage());
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("[fd-warmup] setup failed: " + e.getClass().getSimpleName()
                    + ": " + e.getMessage());
        }
        return done;
    }

    private static boolean readOneHttpResponse(SocketChannel client, ByteBuffer response) throws IOException {
        response.clear();
        int bodyStart = -1;
        int contentLen = -1;
        while (true) {
            int n = client.read(response);
            if (n < 0) return false;
            if (n == 0) continue;

            byte[] data = response.array();
            int pos = response.position();
            if (bodyStart < 0) {
                for (int i = 0; i + 3 < pos; i++) {
                    if (data[i] == '\r' && data[i + 1] == '\n' && data[i + 2] == '\r' && data[i + 3] == '\n') {
                        bodyStart = i + 4;
                        break;
                    }
                }
                if (bodyStart >= 0) {
                    contentLen = parseContentLength(data, bodyStart);
                    if (contentLen < 0) return false;
                }
            }
            if (bodyStart >= 0 && pos >= bodyStart + contentLen) return true;
            if (!response.hasRemaining()) return false;
        }
    }

    private static int parseContentLength(byte[] data, int headerEnd) {
        for (int i = 0; i < headerEnd - CONTENT_LENGTH_LC.length; i++) {
            if (startsWithIgnoreCaseAscii(data, i, CONTENT_LENGTH_LC)) {
                int v = i + CONTENT_LENGTH_LC.length;
                while (v < headerEnd && (data[v] == ' ' || data[v] == '\t')) v++;
                int len = 0;
                boolean sawDigit = false;
                while (v < headerEnd && data[v] >= '0' && data[v] <= '9') {
                    len = len * 10 + (data[v] - '0');
                    v++;
                    sawDigit = true;
                }
                return sawDigit ? len : -1;
            }
        }
        return -1;
    }

    private static boolean startsWithIgnoreCaseAscii(byte[] data, int from, byte[] prefixLc) {
        for (int i = 0; i < prefixLc.length; i++) {
            if ((data[from + i] | 0x20) != prefixLc[i]) return false;
        }
        return true;
    }

    /**
     * Exercises the tryParse → route → writeBuf cycle so the JIT-compiled forms land in the AOT cache instead of
     * compiling under live k6 traffic. Returns a DCE-defeat sink.
     */
    public static long trainAotHotPath(Router router, byte[][] requestFrames, int iters) {
        HttpConnection conn = new HttpConnection();
        long sink = 0;
        int frameCount = requestFrames.length;
        for (int i = 0; i < iters; i++) {
            byte[] frame = requestFrames[i % frameCount];
            conn.readBuf.clear();
            conn.readBuf.put(frame);
            int result = conn.tryParse();
            if (result != HttpConnection.READY) continue;

            byte[] body = conn.readBuf.array();
            byte[] response = conn.routeId == Router.ROUTE_FRAUD_SCORE
                    ? router.fraudScoreResponse(body, conn.bodyStart, conn.bodyLen)
                    : router.route(conn.routeId, body, conn.bodyStart, conn.bodyLen);
            sink ^= response[0];

            conn.writeBuf.clear();
            conn.writeBuf.put(response);
            conn.writeBuf.flip();
            while (conn.writeBuf.hasRemaining()) sink ^= conn.writeBuf.get();
            conn.writeBuf.clear();
            conn.advanceAfterRequest();
        }
        return sink;
    }

    /**
     * Loopback echo to compile {@code SocketChannel.read/write} and the heap-buffer {@code IOUtil} dance. Blocking
     * only selector cost is dominated by epoll_wait, which the JIT can't compile.
     */
    public static int trainAotSocketIo(int iters, long deadlineNanos) {
        try (ServerSocketChannel ssc = ServerSocketChannel.open()) {
            ssc.bind(new InetSocketAddress("127.0.0.1", 0), 1);
            ssc.configureBlocking(true);
            InetSocketAddress addr = (InetSocketAddress) ssc.getLocalAddress();

            Thread server = startEchoServer(ssc, iters);
            try {
                return runClientLoop(addr, iters, deadlineNanos);
            } finally {
                // Bounded join server is a daemon; safe to leak if echo blocks past 500 ms.
                try {
                    server.join(500);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            return 0;
        }
    }

    private static Thread startEchoServer(ServerSocketChannel ssc, int iters) {
        Thread server = new Thread(() -> {
            try (SocketChannel sc = ssc.accept()) {
                sc.configureBlocking(true);
                ByteBuffer in = ByteBuffer.allocate(HttpConnection.READ_BUF_SIZE);
                ByteBuffer out = ByteBuffer.wrap(WARMUP_RESP);
                for (int j = 0; j < iters; j++) {
                    in.clear();
                    int n = sc.read(in);
                    if (n < 0) return;
                    out.rewind();
                    while (out.hasRemaining()) sc.write(out);
                }
            } catch (IOException ignored) {
                // Client closed during warmup expected.
            }
        }, "aot-warmup-srv");
        server.setDaemon(true);
        server.start();
        return server;
    }

    private static int runClientLoop(InetSocketAddress addr, int iters, long deadlineNanos) throws IOException {
        try (SocketChannel client = SocketChannel.open(addr)) {
            client.configureBlocking(true);
            ByteBuffer req = ByteBuffer.wrap(WARMUP_REQ);
            final int RESP_LEN = WARMUP_RESP.length;
            ByteBuffer resp = ByteBuffer.allocate(RESP_LEN);
            int i = 0;
            while (i < iters && System.nanoTime() < deadlineNanos) {
                req.rewind();
                while (req.hasRemaining()) client.write(req);
                resp.clear();
                int totalRead = 0;
                while (totalRead < RESP_LEN) {
                    int n = client.read(resp);
                    if (n < 0) break;
                    totalRead += n;
                }
                i++;
            }
            return i;
        }
    }
}
