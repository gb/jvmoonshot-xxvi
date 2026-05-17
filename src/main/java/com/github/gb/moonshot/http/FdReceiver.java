package com.github.gb.moonshot.http;

import java.io.IOException;
import java.net.StandardProtocolFamily;
import java.net.StandardSocketOptions;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.attribute.PosixFilePermission;
import java.util.EnumSet;
import java.util.Set;

/**
 * Listens on a Unix control socket for lapada FD-passing connections.
 *
 * <p>Protocol: lapada connects once and holds a persistent control connection.
 * For each client TCP socket it accepts, lapada sends the raw fd via
 * {@code sendmsg(SCM_RIGHTS)} over the persistent connection.
 *
 * <p>On the Java side, {@link #acceptLoop} accepts one control connection at a time.
 * A dedicated platform thread runs {@link #recvLoop}, calling {@link FdPassing#receive}
 * in a tight loop. Each received fd is wrapped as a non-blocking {@link SocketChannel}
 * and injected into {@link NioHttpServer}'s selector via {@link NioHttpServer#injectChannel},
 * so FD-passed connections share the same single-thread NIO event loop as regular
 * UDS connections — no virtual thread per connection.
 *
 * <p>When the persistent control connection drops (lapada restart or sendmsg failure),
 * {@code recvLoop} exits and {@code acceptLoop} waits for lapada to reconnect.
 */
public final class FdReceiver {

    /**
     * {@code sun.nio.ch.SelChImpl.getFDVal()} cached at class-load. Called once per
     * accepted control connection to extract the raw OS fd for {@link FdPassing#receive}.
     */
    private static final java.lang.reflect.Method GET_FD_VAL;

    static {
        try {
            Class<?> sel = Class.forName("sun.nio.ch.SelChImpl");
            GET_FD_VAL = sel.getDeclaredMethod("getFDVal");
            GET_FD_VAL.setAccessible(true);
        } catch (Exception e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private final String socketPath;
    private final NioHttpServer nioServer;

    public FdReceiver(String socketPath, NioHttpServer nioServer) {
        this.socketPath = socketPath;
        this.nioServer = nioServer;
    }

    /**
     * Binds the control socket and starts the accept loop on a daemon thread.
     */
    public void start() throws IOException {
        UnixDomainSocketAddress addr = UnixDomainSocketAddress.of(socketPath);
        Files.deleteIfExists(addr.getPath());

        ServerSocketChannel server = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
        server.bind(addr, 1024);

        Set<PosixFilePermission> rw666 = EnumSet.of(
                PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE,
                PosixFilePermission.GROUP_READ, PosixFilePermission.GROUP_WRITE,
                PosixFilePermission.OTHERS_READ, PosixFilePermission.OTHERS_WRITE
        );
        try {
            Files.setPosixFilePermissions(addr.getPath(), rw666);
        } catch (UnsupportedOperationException ignore) {
        }

        Thread t = new Thread(() -> acceptLoop(server), "rinha-fd-receiver");
        t.setDaemon(false);
        t.start();
    }

    private void acceptLoop(ServerSocketChannel server) {
        while (!Thread.interrupted()) {
            try {
                SocketChannel control = server.accept();
                if (control == null) continue;
                // One platform thread per persistent control connection (lapada reconnects rarely).
                SocketChannel capturedControl = control;
                Thread.ofPlatform().daemon(true).name("rinha-fd-recv").start(() -> recvLoop(capturedControl));
            } catch (Throwable e) {
                if (!Thread.interrupted()) e.printStackTrace();
            }
        }
    }

    /**
     * Receive FDs in a tight loop from a persistent lapada control connection.
     * Each received fd is wrapped as a non-blocking SocketChannel and injected into the NIO selector.
     * Returns when the connection drops (recvmsg returns -1).
     */
    private void recvLoop(SocketChannel control) {
        int controlFd = extractRawFd(control);
        while (true) {
            int rawFd = FdPassing.receive(controlFd);
            if (rawFd < 0) break;
            SocketChannel ch = null;
            try {
                ch = FdConnHandler.wrapFd(rawFd);
                ch.configureBlocking(false);
                ch.setOption(StandardSocketOptions.TCP_NODELAY, true);
                nioServer.injectChannel(ch);
            } catch (Throwable t) {
                if (ch != null) {
                    try { ch.close(); } catch (IOException ignored) {}
                }
            }
        }
        try { control.close(); } catch (IOException ignored) {}
        // acceptLoop will accept the next persistent control connection from lapada.
    }

    /**
     * Extract the raw OS file descriptor from a {@link SocketChannel} via the
     * {@code sun.nio.ch.SelChImpl} internal interface.
     */
    static int extractRawFd(SocketChannel channel) {
        try {
            return (int) GET_FD_VAL.invoke(channel);
        } catch (Exception e) {
            throw new RuntimeException("Cannot extract raw fd from SocketChannel", e);
        }
    }
}
