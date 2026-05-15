package com.github.gb.moonshot;

import com.github.gb.moonshot.http.NioAotTraining;
import com.github.gb.moonshot.http.NioHttpServer;
import com.github.gb.moonshot.http.Router;
import com.github.gb.moonshot.codec.ScoringRequestParser;
import com.github.gb.moonshot.search.IvfFlatIndex;
import com.github.gb.moonshot.search.KdTree;
import com.github.gb.moonshot.search.KdTreeIO;
import com.github.gb.moonshot.search.VectorIndex;
import com.github.gb.moonshot.vector.Vectorizer;

import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.net.UnixDomainSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * MoonShot: A highly optimized bad idea.
 *
 * Loads the configured vector index, builds the request pipeline, warms the JIT, and starts the HTTP server.
 * Listen address and index backend are selected via environment variables.
 *
 * AOT_TRAINING=1 runs the same boot but exits after pumping traffic, so the Leyden recorder can dump its cache.
 */
public final class MoonShot {

    public static void main(String[] args) throws Exception {
        VectorIndex index = loadIndex();
        ScoringRequestParser parser = new ScoringRequestParser();
        Vectorizer vectorizer = new Vectorizer();
        Router router = new Router(new FraudScoreHandler(parser, vectorizer, index));

        runWarmup(index, parser, vectorizer, router);
        SocketAddress addr = listenAddress();
        new NioHttpServer().start(addr, router);
        runLiveListenerPump(addr);

        if (aotTrainingMode()) {
            log("AOT_TRAINING=1 exiting after live-listener pump");
            System.exit(0);
        }

        router.markReady();
        log("listening on " + addr);
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
        byte[][] frames = new byte[bodies.length][];

        for (int i = 0; i < bodies.length; i++) {
            frames[i] = WarmupDriver.buildHttpFrame(bodies[i]);
        }

        long t0 = System.nanoTime();
        int done = NioAotTraining.trainLiveListener(addr, frames, iters, deadlineNanos, connections);
        log("live-listener pump: " + done + " requests in " + millisSince(t0) + " ms");
    }

    private static SocketAddress listenAddress() throws Exception {
        String socketPath = System.getenv("LISTEN_SOCKET");
        if (socketPath != null && !socketPath.isEmpty()) {
            Path path = Path.of(socketPath);
            // Stale socket from a prior run blocks bind() with AddressInUse; container restart re-uses the volume.
            Files.deleteIfExists(path);
            return UnixDomainSocketAddress.of(path);
        }

        int port = Integer.parseInt(env("PORT", "9999"));
        return new InetSocketAddress(port);
    }

    private static VectorIndex loadIndex() throws Exception {
        String kind = env("INDEX_KIND", "kdtree");
        return switch (kind) {
            case "kdtree" -> loadKdTreeIndex(Path.of(env("KDTREE_GRAPH", "data/kdtree.bin")));
            case "ivf" -> loadIvfIndex(Path.of(env("IVF_GRAPH", "data/ivf.bin")));
            default -> throw new IllegalArgumentException("INDEX_KIND must be kdtree|ivf, got: " + kind);
        };
    }

    private static KdTree loadKdTreeIndex(Path graphPath) throws Exception {
        log("loading KD-tree from " + graphPath);
        long loadStarted = System.nanoTime();
        KdTree index = KdTreeIO.loadMmap(graphPath);
        log("KD-tree loaded in " + millisSince(loadStarted) + " ms");

        index.applyMmapHints();
        long prewarmStarted = System.nanoTime();
        index.prewarm();
        log("prewarmed mmap pages in " + millisSince(prewarmStarted) + " ms");
        return index;
    }

    private static IvfFlatIndex loadIvfIndex(Path graphPath) throws Exception {
        int nprobe = Integer.parseInt(env("IVF_NPROBE", "16"));
        log("loading IVF from " + graphPath + " (nprobe=" + nprobe + ")");

        long loadStarted = System.nanoTime();
        IvfFlatIndex index = IvfFlatIndex.load(graphPath, nprobe);
        log("IVF loaded in " + millisSince(loadStarted) + " ms "
            + index.size() + " vectors across " + index.nlist() + " cells");

        return index;
    }

    private static void runWarmup(VectorIndex index, ScoringRequestParser parser, Vectorizer vectorizer, Router router) {
        if (index instanceof KdTree kd) {
            new WarmupDriver(kd, parser, vectorizer, router).run();
        } else if (index instanceof IvfFlatIndex ivf) {
            warmIvf(ivf);
        }
    }

    /**
     * IVF's hot path is structurally branch-free (linear cell sweep), so a simple random-vector loop suffices no
     * need for the input-distribution care that {@link WarmupDriver} takes for KdTree.
     */
    private static void warmIvf(IvfFlatIndex index) {
        log("IVF warmup: 15000 random queries");
        long deadlineNanos = System.nanoTime() + 30_000L * 1_000_000L;
        float[] query = new float[com.github.gb.moonshot.Dataset.STRIDE];
        long state = 0xC0DECAFE12345678L;
        int sink = 0;
        long t0 = System.nanoTime();
        int iter = 0;
        while (iter < 15_000 && System.nanoTime() < deadlineNanos) {
            for (int d = 0; d < com.github.gb.moonshot.Dataset.DIMS; d++) {
                state = state ^ (state << 13);
                state = state ^ (state >>> 7);
                state = state ^ (state << 17);
                query[d] = (state & 0xFFFF) * (1.0f / 65535.0f);
            }
            sink ^= index.countFraudsInTopK(query, 5);
            iter++;
        }
        warmIvfSink = sink; // defeat DCE
        log("IVF warmup: " + iter + " queries in " + millisSince(t0) + " ms");
    }

    @SuppressWarnings("unused")
    private static volatile int warmIvfSink;

    private static boolean aotTrainingMode() {
        return "1".equals(System.getenv("AOT_TRAINING"));
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
