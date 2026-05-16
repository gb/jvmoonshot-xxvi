package com.github.gb.moonshot.http;

import com.github.gb.moonshot.codec.ResponseEncoder;

import java.io.FileDescriptor;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.nio.channels.Channels;
import java.nio.channels.SocketChannel;
import java.nio.channels.spi.SelectorProvider;

/**
 * Blocking HTTP/1.1 keep-alive handler for a connection received via FD passing.
 *
 * <p>Takes a raw POSIX file descriptor (a TCP socket passed by lapada via
 * {@code SCM_RIGHTS}), wraps it as a {@link SocketChannel} via reflection on
 * {@code sun.nio.ch.SocketChannelImpl}, and runs a parse→route→respond loop
 * equivalent to {@link NioHttpServer}'s per-connection state machine.
 *
 * <p>SocketChannel (not FileInputStream) is used because the VT spike showed
 * that {@code FileInputStream.read()} on a socket fd pins the carrier thread
 * on macOS Java 25 (and may on other platforms too). SocketChannel.read()
 * is the documented VT-safe I/O path: it registers the fd with the JVM's
 * internal poller (epoll on Linux) and properly unmounts the virtual thread
 * while waiting for data.
 *
 * <p>The {@link #SOCKET_CHANNEL_CTOR} reflection is cached at class load,
 * so the per-connection cost is one constructor invocation and one FileDescriptor
 * field set — negligible at 450 connections/second.
 *
 * <p>Designed to run on a Java 21+ virtual thread: blocking reads unmount the
 * virtual thread without consuming a platform thread.  At 100 concurrent
 * connections the virtual thread overhead is negligible (&lt;10 MB total).
 *
 * <p><b>Known limitation — no read timeout.</b>  {@link SocketChannel} in
 * blocking mode does not expose {@code SO_TIMEOUT}.  A misbehaving client that
 * stalls mid-request parks the virtual thread indefinitely.  Acceptable for
 * the contest workload (k6 VUs always complete).
 */
public final class FdConnHandler {

    /**
     * {@code sun.nio.ch.SocketChannelImpl(SelectorProvider, FileDescriptor, InetSocketAddress)}
     * — the only public-facing way to wrap a raw socket fd in a SocketChannel without
     * going through the OS accept() path.  Cached once at class load.
     */
    private static final Constructor<?> SOCKET_CHANNEL_CTOR;
    private static final Field FD_FIELD;

    static {
        try {
            Class<?> implClass = Class.forName("sun.nio.ch.SocketChannelImpl");
            SOCKET_CHANNEL_CTOR = implClass.getDeclaredConstructor(
                    SelectorProvider.class, FileDescriptor.class, java.net.InetSocketAddress.class);
            SOCKET_CHANNEL_CTOR.setAccessible(true);

            FD_FIELD = FileDescriptor.class.getDeclaredField("fd");
            FD_FIELD.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * Handle one TCP connection lifetime for a file descriptor received via FD passing.
     * Blocks until the client closes or a parse error is detected.
     * Designed to be called from a virtual thread.
     *
     * @param rawFd  the raw int fd of the accepted TCP socket
     * @param router the request router (shared, thread-safe because FraudScoreHandler is)
     */
    public static void handle(int rawFd, Router router) {
        FileDescriptor jfd = new FileDescriptor();
        try {
            FD_FIELD.set(jfd, rawFd);
        } catch (IllegalAccessException e) {
            return;
        }

        SocketChannel channel;
        try {
            // Create a SocketChannel wrapping the received fd.  Blocking mode ensures
            // channel.read() blocks for data (unmounting the VT via the JVM poller)
            // rather than returning 0 immediately.
            channel = (SocketChannel) SOCKET_CHANNEL_CTOR.newInstance(
                    SelectorProvider.provider(), jfd, null);
            channel.configureBlocking(true);
        } catch (Exception e) {
            return;
        }

        try (SocketChannel ch = channel;
             InputStream in = Channels.newInputStream(ch);
             OutputStream out = Channels.newOutputStream(ch)) {
            runLoop(in, out, router);
        } catch (IOException ignored) {
            // Connection closed or reset — normal end of connection.
        }
    }

    private static void runLoop(InputStream in, OutputStream out, Router router)
            throws IOException {
        HttpConnection state = new HttpConnection();
        byte[] raw = state.readBuf.array();

        while (true) {
            // Fill readBuf until tryParse() has a complete request or reports an error.
            while (true) {
                int pos = state.readBuf.position();
                int cap = state.readBuf.capacity();
                if (pos == cap) {
                    // 4 KB buffer exhausted without finding \r\n\r\n: headers are malformed
                    // or abnormally large.  Send 400 and close — do NOT try to drain because
                    // bodyStart is -1 so enterDrainMode() would set bytesToDrain to a large
                    // negative value and the drain loop would not execute, leaving unread
                    // data in the TCP stream that would corrupt the next request.
                    out.write(ResponseEncoder.badRequestClose());
                    return;
                }
                int n = in.read(raw, pos, cap - pos);
                if (n < 0) return; // client closed
                state.readBuf.position(pos + n);

                int result = state.tryParse();
                if (result == HttpConnection.READY) break;
                if (result == HttpConnection.MALFORMED) {
                    out.write(ResponseEncoder.badRequestClose());
                    return; // close
                }
                if (result == HttpConnection.TOO_LARGE) {
                    out.write(ResponseEncoder.payloadTooLargeKeepAlive());
                    state.enterDrainMode();
                    while (state.isDraining()) {
                        int nd = in.read(raw, 0, Math.min(cap, state.bytesToDrain));
                        if (nd < 0) return;
                        state.bytesToDrain -= nd;
                    }
                    state.readBuf.clear();
                    break; // next request
                }
                // NEED_MORE → keep reading
            }

            if (state.bodyStart < 0) continue; // came from drain-reset, no request to route

            byte[] response = router.route(state.routeId, raw, state.bodyStart, state.bodyLen);
            // write() copies directly into the kernel TCP send buffer.
            out.write(response);
            state.advanceAfterRequest();
        }
    }

    private FdConnHandler() {
    }
}
