package com.github.gb.moonshot;

import com.github.gb.moonshot.http.FdReceiver;
import com.github.gb.moonshot.http.NioAotTraining;
import com.github.gb.moonshot.http.NioHttpServer;
import com.github.gb.moonshot.http.Router;
import com.github.gb.moonshot.codec.ScoringRequestParser;
import com.github.gb.moonshot.search.KdTree;
import com.github.gb.moonshot.search.KdTreeIO;
import com.github.gb.moonshot.search.VectorIndex;
import com.github.gb.moonshot.vector.Vectorizer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.UnixDomainSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

/**
 * MoonShot: A highly optimized bad idea.
 * Listen address: {@code LISTEN_SOCKET} (Unix domain) takes priority over {@code PORT} (default 9999).
 *
 * <p>KdTree tuning — epsilon-relaxation, soft cap, BBF depth, prime fan-out — is governed by static
 * fields on {@link com.github.gb.moonshot.search.KdTreeTuning}, initialized from env vars
 * ({@code KDTREE_RELAX_EPSILON}, etc.) at class load time.
 *
 * <p>{@code AOT_TRAINING=1}: runs the same boot but exits after pumping traffic so the Leyden
 * recorder can dump its cache.
 */
public final class MoonShot {

    public static void main(String[] args) throws Exception {
        VectorIndex index = loadIndex();
        ScoringRequestParser parser = new ScoringRequestParser();
        Vectorizer vectorizer = new Vectorizer();
        Router router = new Router(new FraudScoreHandler(parser, vectorizer, index));

        new WarmupDriver(index, parser, vectorizer, router).run();
        SocketAddress addr = listenAddress();
        NioHttpServer nioServer = new NioHttpServer();
        nioServer.start(addr, router);

        preinitAotClasses();
        runLiveListenerPump(addr);
        String fdSocket = startFdReceiver(nioServer);

        if (aotTrainingMode()) {
            runFdPassingPump(fdSocket);
            log("AOT_TRAINING=1 exiting after live-listener and FD-passing pumps");
            System.exit(0);
        }

        router.markReady();
        log("listening on " + addr);
    }

    private static void preinitAotClasses() {
        // Static initializers (SocketChannelImpl + FileDescriptor reflection) must run during
        // AOT training to appear in the Leyden compilation profile.
        try {
            Class.forName("com.github.gb.moonshot.http.FdPassing");
            Class.forName("com.github.gb.moonshot.http.FdConnHandler");
            Class.forName("com.github.gb.moonshot.http.FdReceiver");
        } catch (ClassNotFoundException | ExceptionInInitializerError ignored) {
            // Non-fatal on non-Linux: FD passing is Linux-only.
        }
    }

    private static String startFdReceiver(NioHttpServer nioServer) throws IOException {
        // Starts before markReady so the receiver is listening when lapada gets the ready signal.
        String fdSocket = env("FD_SOCKET", "");
        if (!fdSocket.isEmpty()) {
            new FdReceiver(fdSocket, nioServer).start();
            log("FD receiver listening on " + fdSocket);
        }
        return fdSocket;
    }

    /**
     * 15 000 sequential cycles clear C2's tier-4 threshold for every per-request method; 100 channels matches the
     * contest k6's 100-VU concurrency so accept-side state is exercised at the right cardinality.
     */
    private static void runLiveListenerPump(SocketAddress addr) {
        int iters = Integer.parseInt(env("LIVE_WARMUP_ITERS", "15000"));
        int connections = Integer.parseInt(env("LIVE_WARMUP_CONNECTIONS", "100"));
        long deadlineNanos = System.nanoTime() + 10_000L * 1_000_000L;
        byte[][] bodies = WarmupDriver.loadBodies();
        byte[][] frames = Arrays.stream(bodies).map(WarmupDriver::buildHttpFrame).toArray(byte[][]::new);

        long t0 = System.nanoTime();
        int done = NioAotTraining.trainLiveListener(addr, frames, iters, deadlineNanos, connections);
        log("live-listener pump: " + done + " requests in " + millisSince(t0) + " ms");
    }

    private static void runFdPassingPump(String fdSocket) {
        if (fdSocket.isEmpty()) return;
        int iters = Integer.parseInt(env("FD_WARMUP_ITERS", env("LIVE_WARMUP_ITERS", "15000")));
        int connections = Integer.parseInt(env("FD_WARMUP_CONNECTIONS", env("LIVE_WARMUP_CONNECTIONS", "100")));
        long deadlineNanos = System.nanoTime() + 10_000L * 1_000_000L;
        byte[][] bodies = WarmupDriver.loadBodies();
        byte[][] frames = Arrays.stream(bodies).map(WarmupDriver::buildHttpFrame).toArray(byte[][]::new);

        long t0 = System.nanoTime();
        int done = NioAotTraining.trainFdPassing(fdSocket, frames, iters, deadlineNanos, connections);
        log("FD-passing pump: " + done + " requests in " + millisSince(t0) + " ms");
    }

    private static SocketAddress listenAddress() throws Exception {
        String socketPath = env("LISTEN_SOCKET", "");
        if (!socketPath.isEmpty()) {
            Path path = Path.of(socketPath);
            Files.deleteIfExists(path);
            return UnixDomainSocketAddress.of(path);
        }
        int port = Integer.parseInt(env("PORT", "9999"));
        return new InetSocketAddress(port);
    }

    private static VectorIndex loadIndex() throws Exception {
        String kind = env("INDEX_KIND", "kdtree-i16");
        return switch (kind) {
            case "kdtree-i16" -> loadKdTreeIndex(Path.of(env("KDTREE_GRAPH", "data/kdtree-i16.bin")));
            default -> throw new IllegalArgumentException("INDEX_KIND must be kdtree-i16, got: " + kind);
        };
    }

    private static KdTree loadKdTreeIndex(Path graphPath) throws Exception {
        log("loading KdTree (mmap pts) from " + graphPath);
        long t0 = System.nanoTime();
        KdTree index = KdTreeIO.loadMmap(graphPath);
        log("KdTree loaded in " + millisSince(t0) + " ms, n=" + index.size());
        index.applyMmapHints();
        long prewarmStart = System.nanoTime();
        index.prewarm();
        log("KdTree prewarmed in " + millisSince(prewarmStart) + " ms");
        return index;
    }

    private static boolean aotTrainingMode() {
        return "1".equals(env("AOT_TRAINING", ""));
    }

    private static String env(String key, String fallback) {
        return Objects.requireNonNullElse(System.getenv(key), fallback);
    }

    private static long millisSince(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000;
    }

    private static void log(String msg) {
        System.out.println("[boot] " + msg);
    }
}
