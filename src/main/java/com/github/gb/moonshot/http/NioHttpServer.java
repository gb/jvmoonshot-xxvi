package com.github.gb.moonshot.http;

import com.github.gb.moonshot.codec.ResponseEncoder;
import com.github.gb.moonshot.instrumentation.StageTimer;

import java.io.IOException;
import java.net.SocketAddress;
import java.net.StandardProtocolFamily;
import java.net.StandardSocketOptions;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermission;
import java.util.EnumSet;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Single-thread NIO event loop. Zero-alloc-per-request steady-state: each connection owns its read buffer and
 * gathering-write pipeline, and pre-baked shared direct response buffers are duplicated (~40 B young-gen) per
 * response.
 * <p>
 * Contest workload: only {@code POST /fraud-score} (and {@code GET /ready} once at boot). Everything else is treated
 * as a malformed peer.
 */
public final class NioHttpServer {

    private static final int LISTEN_BACKLOG = 1024;

    /** Error logging is off by default: synchronized stderr writes are catastrophic during short p99 runs. */
    private static final boolean LOG_IO_ERRORS = "1".equals(System.getenv("LOG_IO_ERRORS"));

    /** FD-passed channels queued by {@link FdReceiver} for registration on the next select() iteration. */
    private final ConcurrentLinkedQueue<SocketChannel> pendingChannels = new ConcurrentLinkedQueue<>();
    private final AtomicBoolean pendingWakeupArmed = new AtomicBoolean();
    private volatile Selector selectorRef;

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
                    PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE,
                    PosixFilePermission.GROUP_READ, PosixFilePermission.GROUP_WRITE,
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
        this.selectorRef = selector; // visible to injectChannel() after Thread.start() happens-before
        server.register(selector, SelectionKey.OP_ACCEPT);

        Thread t = new Thread(() -> loop(selector, server, router, !unix, pendingChannels, pendingWakeupArmed), "rinha-io");
        t.setDaemon(false);
        // Best effort only; some container runtimes ignore Java priorities, but when honored this keeps the
        // selector/response thread ahead of warmup leftovers and JVM housekeeping during the contest window.
        t.setPriority(Thread.MAX_PRIORITY);
        t.start();
        return t;
    }

    /**
     * Register a non-blocking {@link SocketChannel} received via FD passing into this server's selector.
     * Thread-safe: can be called from any thread (e.g. {@link FdReceiver}'s recv loop thread).
     * The channel must already be configured non-blocking by the caller.
     */
    public void injectChannel(SocketChannel ch) {
        pendingChannels.add(ch);
        Selector sel = selectorRef;
        if (sel != null && pendingWakeupArmed.compareAndSet(false, true)) sel.wakeup();
    }

    private static void loop(Selector selector, ServerSocketChannel server, Router router, boolean tcpOptions,
                             ConcurrentLinkedQueue<SocketChannel> pending, AtomicBoolean pendingWakeupArmed) {
        // Allocated once; select(Consumer) auto-removes keys after callback returns.
        Consumer<SelectionKey> handle = key -> {
            try {
                if (!key.isValid()) return;
                if (key.isAcceptable()) handleAccept(server, selector, tcpOptions);
                else if (key.isReadable()) handleRead(key, router);
                else if (key.isWritable()) handleWrite(key);
            } catch (Throwable t) {
                // Catch Throwable so a per-key error closes only this key, not the whole loop.
                logIoError(t);
                close(key);
            }
        };
        while (!Thread.interrupted()) {
            try {
                // Drain FD-passed channels before blocking. ConcurrentLinkedQueue.poll() is
                // wait-free; draining here (on the selector thread) avoids any locking on Selector.
                SocketChannel inj;
                boolean drainedInjected = false;
                while ((inj = pending.poll()) != null) {
                    drainedInjected = true;
                    try {
                        inj.register(selector, SelectionKey.OP_READ, new HttpConnection());
                    } catch (Throwable t) {
                        logIoError(t);
                        try { inj.close(); } catch (IOException ignored) {}
                    }
                }
                if (drainedInjected) {
                    pendingWakeupArmed.set(false);
                }
                // Non-blocking pass first: under keep-alive load the next request bytes are already
                // in the socket buffer by the time we finish flushing the response. selectNow() catches
                // them without an epoll_wait syscall (~10-50µs saved per round-trip). Only block when
                // truly idle (selectNow found nothing AND no FD-passed channels queued).
                if (selector.selectNow(handle) == 0 && pending.isEmpty()) {
                    selector.select(handle);
                }
            } catch (Throwable t) {
                // One bad select() iteration must not terminate the loop; loop-fatal conditions surface via
                // Thread.interrupted.
                logIoError(t);
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
                logIoError(t);
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
        if (bytesRead < 0) {
            close(key);
            return;
        }
        if (bytesRead == 0) return;

        if (drainRequests(state, router)) {
            flushPipeline(key, channel, state);
        }
    }

    // Cap readBuf.limit so a drain read can't swallow bytes from the next request.
    private static void drainOversizedBody(SelectionKey key, SocketChannel channel, HttpConnection state) throws IOException {
        int cap = Math.min(HttpConnection.READ_BUF_SIZE, state.bytesToDrain);
        state.readBuf.limit(cap);
        int n = channel.read(state.readBuf);
        if (n < 0) {
            close(key);
            return;
        }
        state.bytesToDrain -= n;
        state.readBuf.clear();
    }

    private static boolean drainRequests(HttpConnection state, Router router) {
        boolean queued = false;
        final boolean timed = StageTimer.ENABLED;
        while (true) {
            // Force a flush before queueing if the per-conn pipeline is full. The cap bounds
            // per-conn ByteBuffer[] memory and ensures gathering write fits in one syscall under
            // realistic SNDBUF sizes.
            if (state.pipelineCount >= HttpConnection.MAX_PIPELINE_SLOTS) {
                return true;
            }
            // tryParse is idempotent once bodyStart >= 0, so bailing here and resuming on the next OP_READ event
            // re-enters the same request.
            int parseResult = state.tryParse();
            if (parseResult == HttpConnection.NEED_MORE) {
                // Wedged headers (no \r\n\r\n in 4 KB): force-close, else the level-triggered selector spins
                // forever on OP_READ.
                if (state.bodyStart < 0 && !state.readBuf.hasRemaining()) {
                    queueResponse(state, ResponseEncoder.RESP_BAD_REQUEST_CLOSE);
                    state.closeAfterWrite = true;
                    return true;
                }
                return queued;
            }
            if (parseResult == HttpConnection.MALFORMED) {
                queueResponse(state, ResponseEncoder.RESP_BAD_REQUEST_CLOSE);
                state.closeAfterWrite = true;
                return true;
            }
            if (parseResult == HttpConnection.TOO_LARGE) {
                queueResponse(state, ResponseEncoder.RESP_PAYLOAD_TOO_LARGE);
                state.enterDrainMode();
                return true;
            }
            if (timed) StageTimer.t0 = System.nanoTime();
            int respIdx = state.routeId == Router.ROUTE_FRAUD_SCORE
                    ? router.fraudScoreResponseIndex(state.readBuf.array(), state.bodyStart, state.bodyLen)
                    : router.routeResponseIndex(state.routeId, state.readBuf.array(), state.bodyStart, state.bodyLen);
            if (timed) StageTimer.mark(3, System.nanoTime());
            queueResponse(state, respIdx);
            if (timed) {
                StageTimer.mark(4, System.nanoTime());
                StageTimer.complete();
            }
            state.advanceAfterRequest();
            queued = true;
        }
    }

    private static void queueResponse(HttpConnection state, int respIdx) {
        // Per-write duplicate aliases shared direct memory but owns its own position/limit so the
        // gathering write can advance it independently. ~40 B young-gen alloc per response.
        state.pipeline[state.pipelineCount++] = ResponseEncoder.duplicateFor(respIdx);
    }

    private static void flushPipeline(SelectionKey key, SocketChannel channel, HttpConnection state) throws IOException {
        // Single writev for all queued responses. Each ByteBuffer's position is advanced by the
        // bytes consumed from it; fully-drained buffers reach position==limit and contribute nothing
        // on a subsequent call, partially-drained buffers resume from where they left off.
        channel.write(state.pipeline, 0, state.pipelineCount);
        if (pipelineHasRemaining(state)) {
            key.interestOps(SelectionKey.OP_WRITE);
        } else {
            resetPipeline(state);
            if (state.closeAfterWrite) {
                close(key);
            }
        }
    }

    private static void handleWrite(SelectionKey key) throws IOException {
        SocketChannel channel = (SocketChannel) key.channel();
        HttpConnection state = (HttpConnection) key.attachment();

        channel.write(state.pipeline, 0, state.pipelineCount);
        if (pipelineHasRemaining(state)) return;

        resetPipeline(state);
        if (state.closeAfterWrite) {
            close(key);
            return;
        }
        key.interestOps(SelectionKey.OP_READ);
    }

    private static boolean pipelineHasRemaining(HttpConnection state) {
        for (int i = 0; i < state.pipelineCount; i++) {
            if (state.pipeline[i].hasRemaining()) return true;
        }
        return false;
    }

    private static void resetPipeline(HttpConnection state) {
        // Null out slot references so the per-write duplicates become unreachable and get
        // collected on the next young-gen pass. Avoids retaining ByteBuffer wrappers across
        // long-lived keep-alive connections.
        ByteBuffer[] arr = state.pipeline;
        int n = state.pipelineCount;
        for (int i = 0; i < n; i++) arr[i] = null;
        state.pipelineCount = 0;
    }

    private static void logIoError(Throwable t) {
        if (LOG_IO_ERRORS) t.printStackTrace();
    }

    private static void close(SelectionKey key) {
        try {
            key.channel().close();
        } catch (IOException ignored) {
        }
        key.cancel();
    }
}
