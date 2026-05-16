package com.github.gb.moonshot.http;

import java.io.IOException;
import java.net.StandardProtocolFamily;
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
 * <p>Protocol: lapada connects → sends one client TCP socket FD via
 * {@code sendmsg(SCM_RIGHTS)} → closes control connection.  For each received FD,
 * a virtual thread is started that handles the full HTTP/1.1 keep-alive lifetime
 * via {@link FdConnHandler}.
 *
 * <p>Unlike {@link NioHttpServer}, this uses one virtual thread per connection rather
 * than a single-thread NIO event loop.  For the contest's ~100 concurrent connections,
 * virtual-thread overhead is negligible while the code is significantly simpler.
 */
public final class FdReceiver {

    /**
     * {@code sun.nio.ch.SelChImpl.getFDVal()} cached at class-load.  Called once per
     * accepted control connection (~450/s) to extract the raw OS fd for {@link FdPassing#receive}.
     * Caching avoids per-call {@link Class#forName} + {@link Class#getDeclaredMethod} lookups.
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
    private final Router router;

    public FdReceiver(String socketPath, Router router) {
        this.socketPath = socketPath;
        this.router = router;
    }

    /**
     * Binds the control socket and starts the accept loop on a daemon thread.
     */
    public void start() throws IOException {
        UnixDomainSocketAddress addr = UnixDomainSocketAddress.of(socketPath);
        Files.deleteIfExists(addr.getPath());

        ServerSocketChannel server = ServerSocketChannel.open(StandardProtocolFamily.UNIX);
        server.bind(addr, 1024);

        // HAProxy must be able to connect as non-root.
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
                // Each lapada control connection may carry one FD (connect-per-client model).
                // Start a virtual thread to receive the FD and handle the connection.
                Thread.startVirtualThread(() -> receiveAndHandle(control));
            } catch (IOException e) {
                if (!Thread.interrupted()) e.printStackTrace();
            }
        }
    }

    /**
     * Receive one FD from the lapada control connection, then handle the TCP connection.
     * The control channel is closed after the FD is received (lapada uses connect-per-client).
     */
    private void receiveAndHandle(SocketChannel control) {
        int rawFd;
        try {
            // Extract the raw socket FD of the control channel for recvmsg().
            int controlFd = extractRawFd(control);
            rawFd = FdPassing.receive(controlFd);
        } finally {
            try {
                control.close();
            } catch (IOException ignored) {
            }
        }

        if (rawFd < 0) return; // nothing received

        // The virtual thread we're already on handles the full connection lifetime.
        FdConnHandler.handle(rawFd, router);
    }

    /**
     * Extract the raw OS file descriptor from a {@link SocketChannel} via the
     * {@code sun.nio.ch.SelChImpl} internal interface exposed on NIO channels.
     * Equivalent to {@code ((sun.nio.ch.SelChImpl) channel).getFDVal()}.
     * The Method object is cached in {@link #GET_FD_VAL} to avoid per-call reflection overhead.
     */
    private static int extractRawFd(SocketChannel channel) {
        try {
            return (int) GET_FD_VAL.invoke(channel);
        } catch (Exception e) {
            throw new RuntimeException("Cannot extract raw fd from SocketChannel", e);
        }
    }
}
