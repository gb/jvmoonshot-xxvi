package com.github.gb.moonshot.http;

import com.github.gb.moonshot.codec.ResponseEncoder;

import java.io.FileDescriptor;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.VarHandle;
import java.net.ProtocolFamily;
import java.net.StandardProtocolFamily;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.nio.channels.spi.SelectorProvider;

/**
 * Raw-fd wrapping helper for client TCP sockets received from lapada via
 * {@code SCM_RIGHTS}. Runtime traffic uses {@link #wrapFd(int)} so
 * {@link FdReceiver} can inject each socket into {@link NioHttpServer}'s
 * selector; the blocking {@link #handle(int, Router)} loop is kept only for
 * AOT training and diagnostics.
 *
 * <p>The internal constructor and {@link FileDescriptor#fd} setter are cached as
 * method-handle/var-handle call sites at class load, so the per-connection runtime
 * avoids reflective invocation while still skipping the OS accept() path.
 */
public final class FdConnHandler {

    /**
     * {@code sun.nio.ch.SocketChannelImpl(SelectorProvider, ProtocolFamily, FileDescriptor, SocketAddress)}
     * — the only public-facing way to wrap a raw socket fd in a SocketChannel without
     * going through the OS accept() path.  Cached once at class load.
     */
    private static final MethodHandle SOCKET_CHANNEL_CTOR;
    private static final VarHandle FD_FIELD;

    static {
        try {
            Class<?> implClass = Class.forName("sun.nio.ch.SocketChannelImpl");
            // JDK 25 changed the signature: added ProtocolFamily, generalized InetSocketAddress→SocketAddress.
            MethodHandles.Lookup implLookup = MethodHandles.privateLookupIn(implClass, MethodHandles.lookup());
            SOCKET_CHANNEL_CTOR = implLookup.findConstructor(implClass, MethodType.methodType(
                    void.class, SelectorProvider.class, ProtocolFamily.class, FileDescriptor.class,
                    java.net.SocketAddress.class));

            FD_FIELD = MethodHandles.privateLookupIn(FileDescriptor.class, MethodHandles.lookup())
                    .findVarHandle(FileDescriptor.class, "fd", int.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    /**
     * Wrap a raw POSIX TCP socket fd in a {@link SocketChannel}.
     * The caller is responsible for setting the blocking mode and registering or handling the channel.
     */
    static SocketChannel wrapFd(int rawFd) throws Throwable {
        FileDescriptor jfd = new FileDescriptor();
        FD_FIELD.set(jfd, rawFd);
        return (SocketChannel) SOCKET_CHANNEL_CTOR.invoke(
                SelectorProvider.provider(), StandardProtocolFamily.INET, jfd, null);
    }

    /**
     * Handle one TCP connection lifetime for a file descriptor received via FD passing.
     * Blocks until the client closes or a parse error is detected.
     * Designed to be called from a virtual thread (AOT training path only in production;
     * the NIO selector path is used at runtime).
     *
     * @param rawFd  the raw int fd of the accepted TCP socket
     * @param router the request router (shared, thread-safe because FraudScoreHandler is)
     */
    public static void handle(int rawFd, Router router) {
        SocketChannel channel;
        try {
            channel = wrapFd(rawFd);
            channel.configureBlocking(true);
        } catch (Throwable e) {
            return;
        }

        try (SocketChannel ch = channel) {
            runLoop(ch, router);
        } catch (IOException ignored) {
            // Connection closed or reset — normal end of connection.
        }
    }

    private static void runLoop(SocketChannel channel, Router router) throws IOException {
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
                    writeResponse(channel, ResponseEncoder.RESP_BAD_REQUEST_CLOSE);
                    return;
                }
                int n = channel.read(state.readBuf);
                if (n < 0) return; // client closed
                if (n == 0) continue;

                int result = state.tryParse();
                if (result == HttpConnection.READY) break;
                if (result == HttpConnection.MALFORMED) {
                    writeResponse(channel, ResponseEncoder.RESP_BAD_REQUEST_CLOSE);
                    return; // close
                }
                if (result == HttpConnection.TOO_LARGE) {
                    writeResponse(channel, ResponseEncoder.RESP_PAYLOAD_TOO_LARGE);
                    state.enterDrainMode();
                    while (state.isDraining()) {
                        state.readBuf.limit(Math.min(cap, state.bytesToDrain));
                        int nd = channel.read(state.readBuf);
                        if (nd < 0) return;
                        if (nd == 0) continue;
                        state.bytesToDrain -= nd;
                        state.readBuf.clear();
                    }
                    state.readBuf.clear();
                    break; // next request
                }
                // NEED_MORE → keep reading
            }

            if (state.bodyStart < 0) continue; // came from drain-reset, no request to route

            int respIdx = router.routeResponseIndex(state.routeId, raw, state.bodyStart, state.bodyLen);
            writeResponse(channel, respIdx);
            state.advanceAfterRequest();
        }
    }

    private static void writeResponse(SocketChannel channel, int respIdx) throws IOException {
        ByteBuffer out = ResponseEncoder.duplicateFor(respIdx);
        while (out.hasRemaining()) {
            channel.write(out);
        }
    }

    private FdConnHandler() {
    }
}
