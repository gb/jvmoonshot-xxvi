package com.github.gb.moonshot.http;

import com.github.gb.moonshot.instrumentation.StageTimer;
import com.github.gb.moonshot.codec.ResponseEncoder;

import java.io.IOException;
import java.net.SocketAddress;
import java.net.StandardProtocolFamily;
import java.net.StandardSocketOptions;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermission;
import java.util.EnumSet;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Single-thread NIO event loop. Zero-alloc steady-state: each connection owns its read/write buffers and reuses them
 * across pipelined keep-alive requests.
 *
 * Contest workload: only {@code POST /fraud-score} (and {@code GET /ready} once at boot). Everything else is treated
 * as a malformed peer.
 */
public final class NioHttpServer {

    private static final byte[] HTTP_400_CLOSE = ResponseEncoder.badRequestClose();
    private static final byte[] HTTP_413_KEEPALIVE = ResponseEncoder.payloadTooLargeKeepAlive();

    private static final int LISTEN_BACKLOG = 1024;
    // Largest real response frame is ~110 B; 256 is the headroom guard so drainRequests bails before a pipelined
    // burst overflows writeBuf.
    private static final int MAX_RESPONSE_BYTES = 256;

    /** @return the IO loop thread (used by allocation profilers via {@code ThreadMXBean}). */
    public Thread start(SocketAddress addr, Router router) throws IOException {
        boolean unix = addr instanceof UnixDomainSocketAddress;
        StandardProtocolFamily family = unix ? StandardProtocolFamily.UNIX : StandardProtocolFamily.INET;
        ServerSocketChannel server = ServerSocketChannel.open(family);
        if (!unix) {
            // UDS has no TIME_WAIT; JDK throws if SO_REUSEADDR is set on it.
            server.setOption(StandardSocketOptions.SO_REUSEADDR, true);
        }
        server.bind(addr, LISTEN_BACKLOG);
        if (unix) {
            // JDK creates the UDS inode 0755; HAProxy connects as non-root and needs write perm.
            Set<PosixFilePermission> rw666 = EnumSet.of(
                PosixFilePermission.OWNER_READ,  PosixFilePermission.OWNER_WRITE,
                PosixFilePermission.GROUP_READ,  PosixFilePermission.GROUP_WRITE,
                PosixFilePermission.OTHERS_READ, PosixFilePermission.OTHERS_WRITE
            );
            try {
                Files.setPosixFilePermissions(((UnixDomainSocketAddress) addr).getPath(), rw666);
            } catch (UnsupportedOperationException ignore) {
                // Windows local dev.
            }
        }
        server.configureBlocking(false);

        Selector selector = Selector.open();
        server.register(selector, SelectionKey.OP_ACCEPT);

        Thread t = new Thread(() -> loop(selector, server, router, !unix), "rinha-io");
        t.setDaemon(false);
        t.start();
        return t;
    }

    private static void loop(Selector selector, ServerSocketChannel server, Router router, boolean tcpOptions) {
        // Allocated once; select(Consumer) auto-removes keys after callback returns.
        Consumer<SelectionKey> handle = key -> {
            try {
                if (!key.isValid()) return;
                if (key.isAcceptable()) handleAccept(server, selector, tcpOptions);
                else if (key.isReadable()) handleRead(key, router);
                else if (key.isWritable()) handleWrite(key);
            } catch (Throwable t) {
                // Catch Throwable so a per-key error closes only this key, not the whole loop.
                t.printStackTrace();
                close(key);
            }
        };
        while (!Thread.interrupted()) {
            try {
                selector.select(handle);
            } catch (Throwable t) {
                // One bad select() iteration must not terminate the loop; loop-fatal conditions surface via
                // Thread.interrupted.
                t.printStackTrace();
            }
        }
    }

    private static void handleAccept(ServerSocketChannel server, Selector selector, boolean tcpOptions) {
        // Catch inside the loop: a propagated accept() failure would land on the server-channel key and close the
        // listener.
        while (true) {
            SocketChannel channel = null;
            try {
                channel = server.accept();
                if (channel == null) return;
                channel.configureBlocking(false);
                if (tcpOptions) {
                    // UDS rejects these with UnsupportedOperationException and doesn't need them.
                    channel.setOption(StandardSocketOptions.TCP_NODELAY, true);
                    channel.setOption(StandardSocketOptions.SO_KEEPALIVE, true);
                }
                channel.register(selector, SelectionKey.OP_READ, new HttpConnection());
            } catch (Throwable t) {
                t.printStackTrace();
                if (channel != null) {
                    try { channel.close(); } catch (IOException ignored) {}
                }
                return;
            }
        }
    }

    private static void handleRead(SelectionKey key, Router router) throws IOException {
        SocketChannel channel = (SocketChannel) key.channel();
        HttpConnection state = (HttpConnection) key.attachment();

        if (state.isDraining()) {
            drainOversizedBody(key, channel, state);
            return;
        }

        int bytesRead = channel.read(state.readBuf);
        if (bytesRead < 0) { close(key); return; }
        if (bytesRead == 0) return;

        if (drainRequests(state, router)) {
            flushWriteBuf(key, channel, state);
        }
    }

    // Cap readBuf.limit so a drain read can't swallow bytes from the next request.
    private static void drainOversizedBody(SelectionKey key, SocketChannel channel, HttpConnection state) throws IOException {
        int cap = Math.min(HttpConnection.READ_BUF_SIZE, state.bytesToDrain);
        state.readBuf.limit(cap);
        int n = channel.read(state.readBuf);
        if (n < 0) { close(key); return; }
        state.bytesToDrain -= n;
        state.readBuf.clear();
    }

    private static boolean drainRequests(HttpConnection state, Router router) {
        boolean queued = false;
        while (true) {
            // tryParse is idempotent once bodyStart >= 0, so bailing here and resuming on the next OP_READ event
            // re-enters the same request.
            if (state.writeBuf.remaining() < MAX_RESPONSE_BYTES) {
                return true;
            }
            int parseResult = state.tryParse();
            if (parseResult == HttpConnection.NEED_MORE) {
                // Wedged headers (no \r\n\r\n in 4 KB): force-close, else the level-triggered selector spins
                // forever on OP_READ.
                if (state.bodyStart < 0 && !state.readBuf.hasRemaining()) {
                    state.writeBuf.put(HTTP_400_CLOSE);
                    state.closeAfterWrite = true;
                    return true;
                }
                return queued;
            }
            if (parseResult == HttpConnection.MALFORMED) {
                state.writeBuf.put(HTTP_400_CLOSE);
                state.closeAfterWrite = true;
                return true;
            }
            if (parseResult == HttpConnection.TOO_LARGE) {
                state.writeBuf.put(HTTP_413_KEEPALIVE);
                state.enterDrainMode();
                return true;
            }
            if (StageTimer.ENABLED) StageTimer.t0 = System.nanoTime();
            byte[] response = router.route(
                state.routeId, state.readBuf.array(), state.bodyStart, state.bodyLen);
            if (StageTimer.ENABLED) StageTimer.mark(3, System.nanoTime());
            state.writeBuf.put(response);
            if (StageTimer.ENABLED) {
                StageTimer.mark(4, System.nanoTime());
                StageTimer.complete();
            }
            state.advanceAfterRequest();
            queued = true;
        }
    }

    private static void flushWriteBuf(SelectionKey key, SocketChannel channel, HttpConnection state) throws IOException {
        state.writeBuf.flip();
        // Try one immediate write saves a select() round-trip when the kernel send buffer has space.
        channel.write(state.writeBuf);
        if (state.writeBuf.hasRemaining()) {
            key.interestOps(SelectionKey.OP_WRITE);
        } else {
            state.writeBuf.clear();
            if (state.closeAfterWrite) {
                close(key);
            }
        }
    }

    private static void handleWrite(SelectionKey key) throws IOException {
        SocketChannel channel = (SocketChannel) key.channel();
        HttpConnection state = (HttpConnection) key.attachment();

        channel.write(state.writeBuf);
        if (state.writeBuf.hasRemaining()) return;

        state.writeBuf.clear();
        if (state.closeAfterWrite) {
            close(key);
            return;
        }
        key.interestOps(SelectionKey.OP_READ);
    }

    private static void close(SelectionKey key) {
        try { key.channel().close(); } catch (IOException ignored) {}
        key.cancel();
    }
}
