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

    /**
     * Hard cap on total startup wall-clock (boot + index load + WarmupDriver + listener pump +
     * FD-passing pump + markReady + writeReadyFile). Defaults to 55 s so we land comfortably
     * inside the docker healthcheck's 60 s {@code start_period} — once start_period elapses the
     * fallback {@code interval=60s} kicks in and a healthy API can sit unobserved for up to a
     * full minute before lapada is allowed to start.
     */
    private static final long DEFAULT_STARTUP_BUDGET_MS = 55_000L;

    public static void main(String[] args) throws Exception {
        long startNanos = System.nanoTime();
        long startupBudgetMs = Long.parseLong(env("MAX_STARTUP_MS", Long.toString(DEFAULT_STARTUP_BUDGET_MS)));
        long startupDeadlineNanos = startNanos + startupBudgetMs * 1_000_000L;

        clearReadyFile();
        VectorIndex index = loadIndex();
        ScoringRequestParser parser = new ScoringRequestParser();
        Vectorizer vectorizer = new Vectorizer();
        Router router = new Router(new FraudScoreHandler(parser, vectorizer, index));

        byte[][] warmupBodies = WarmupDriver.loadBodies();
        new WarmupDriver(index, parser, vectorizer, router).run(warmupBodies);
        byte[][] warmupFrames = buildHttpFrames(warmupBodies);
        SocketAddress addr = listenAddress();
        NioHttpServer nioServer = new NioHttpServer();
        nioServer.start(addr, router);

        preinitAotClasses();
        runLiveListenerPump(addr, warmupFrames, startupDeadlineNanos);
        String fdSocket = startFdReceiver(nioServer);

        if (aotTrainingMode()) {
            runFdPassingPump(fdSocket, warmupFrames, "FD-passing pump",
                    Long.parseLong(env("FD_WARMUP_BUDGET_MS", "10000")),
                    Integer.parseInt(env("FD_WARMUP_ITERS", env("LIVE_WARMUP_ITERS", "15000"))),
                    Integer.parseInt(env("FD_WARMUP_CONNECTIONS", env("LIVE_WARMUP_CONNECTIONS", "100"))),
                    startupDeadlineNanos);
            log("AOT_TRAINING=1 exiting after live-listener and FD-passing pumps");
            System.exit(0);
        }

        // Always exercise the FD-passing accept path in production startup before flipping ready,
        // so the first k6 connections don't catch FdReceiver / injectChannel / pending-drain on a
        // cold path. With the AOT cache these methods are already pre-compiled; this confirms
        // they're loaded into the code cache and shakes out the SCM_RIGHTS plumbing end-to-end.
        // Shorter budget than AOT training — we only need a smoke pass, not full tier-4 promotion.
        runFdPassingPump(fdSocket, warmupFrames, "FD-passing prod smoke",
                Long.parseLong(env("FD_PROD_WARMUP_BUDGET_MS", "5000")),
                Integer.parseInt(env("FD_PROD_WARMUP_ITERS", "5000")),
                Integer.parseInt(env("FD_PROD_WARMUP_CONNECTIONS", "50")),
                startupDeadlineNanos);

        router.markReady();
        writeReadyFile();
        com.github.gb.moonshot.search.KdTreeMmap.releaseAotCachePages();
        long startupElapsedMs = (System.nanoTime() - startNanos) / 1_000_000L;
        log("listening on " + addr + " (startup " + startupElapsedMs + " ms, budget " + startupBudgetMs + " ms)");
        if (startupElapsedMs > startupBudgetMs) {
            log("WARN: startup exceeded " + startupBudgetMs + " ms budget by "
                    + (startupElapsedMs - startupBudgetMs)
                    + " ms — docker healthcheck may have slipped past start_period");
        }
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
     * The phase deadline is the lesser of the per-phase budget (default 10 s) and the overall
     * startup deadline, so a slow earlier phase shortens this one rather than overflowing.
     */
    private static void runLiveListenerPump(SocketAddress addr, byte[][] frames, long startupDeadlineNanos) {
        int iters = Integer.parseInt(env("LIVE_WARMUP_ITERS", "15000"));
        int connections = Integer.parseInt(env("LIVE_WARMUP_CONNECTIONS", "100"));
        long phaseBudgetMs = Long.parseLong(env("LIVE_WARMUP_BUDGET_MS", "10000"));
        long phaseDeadlineNanos = Math.min(startupDeadlineNanos,
                System.nanoTime() + phaseBudgetMs * 1_000_000L);

        long t0 = System.nanoTime();
        int done = NioAotTraining.trainLiveListener(addr, frames, iters, phaseDeadlineNanos, connections);
        log("live-listener pump: " + done + "/" + iters + " requests in " + millisSince(t0) + " ms");
    }

    /**
     * Shared FD-passing pump used by both the AOT-record stage (full iters, full budget) and the
     * production startup smoke (smaller iters, shorter budget). The phase deadline is the lesser
     * of {@code phaseBudgetMs} and {@code startupDeadlineNanos} so it can't push total startup
     * past the overall budget.
     */
    private static void runFdPassingPump(String fdSocket, byte[][] frames, String label,
                                         long phaseBudgetMs, int iters, int connections,
                                         long startupDeadlineNanos) {
        if (fdSocket.isEmpty()) return;
        long phaseDeadlineNanos = Math.min(startupDeadlineNanos,
                System.nanoTime() + phaseBudgetMs * 1_000_000L);

        long t0 = System.nanoTime();
        int done = NioAotTraining.trainFdPassing(fdSocket, frames, iters, phaseDeadlineNanos, connections);
        log(label + ": " + done + "/" + iters + " requests in " + millisSince(t0) + " ms");
    }

    private static byte[][] buildHttpFrames(byte[][] bodies) {
        byte[][] frames = new byte[bodies.length][];
        for (int i = 0; i < bodies.length; i++) {
            frames[i] = WarmupDriver.buildHttpFrame(bodies[i]);
        }
        return frames;
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
            case "kdtree-i16" -> loadKdTreeIndex(Path.of(env("KDTREE_GRAPH", "data/kdtree.bin")));
            case "stub" -> {
                log("INDEX_KIND=stub: no dataset load, fixed fraud count=2 (HTTP transport bench mode)");
                yield new com.github.gb.moonshot.search.StubVectorIndex();
            }
            default -> throw new IllegalArgumentException("INDEX_KIND must be kdtree-i16|stub, got: " + kind);
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

    private static void clearReadyFile() throws IOException {
        String readyFile = env("READY_FILE", "");
        if (!readyFile.isEmpty()) {
            Files.deleteIfExists(Path.of(readyFile));
        }
    }

    private static void writeReadyFile() throws IOException {
        String readyFile = env("READY_FILE", "");
        if (!readyFile.isEmpty()) {
            Files.writeString(Path.of(readyFile), "ready");
        }
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
