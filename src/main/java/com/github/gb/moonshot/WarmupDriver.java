package com.github.gb.moonshot;

import com.github.gb.moonshot.bench.PayloadBankIO;
import com.github.gb.moonshot.http.NioAotTraining;
import com.github.gb.moonshot.http.Router;
import com.github.gb.moonshot.codec.ScoringRequestParser;
import com.github.gb.moonshot.codec.ScoringRequestScratch;
import com.github.gb.moonshot.search.KdTree;
import com.github.gb.moonshot.search.VectorIndex;
import com.github.gb.moonshot.vector.Vectorizer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Pre-flight JIT + page-cache warmup. Runs after KD-tree load + mmap prewarm, before the HTTP server starts
 * listening, so {@code /ready} only flips to 200 once the hot path is C2-compiled.
 *
 * CAUTION: affects production p99. Refactors that introduce indirection (lambda dispatch, captured arrays,
 * extra helpers) change which methods get compiled or how — keep inner loops monolithic and direct.
 */
public final class WarmupDriver {

    /** Clears tiered C2's once-per-query threshold (descend, prime, countFraudsInTopK) ~10-15K invocations. */
    private static final int SEARCH_ITERS = 15_000;

    /** 20K pushes parseInto (226 B) and vectorizeInto (273 B) past tier-4; 5K only gets parseInto there. */
    private static final int PARSER_ITERS = 20_000;

    /** Router dispatch + ResponseEncoder lookup; search/parser already warm from dedicated loops. */
    private static final int ROUTE_ITERS = 200;

    /** Pushes tryParse (362 B), Router.route (56 B), countFraudNeighbors (108 B) past tier 4. */
    private static final int HTTP_PARSE_ITERS = 20_000;

    /** Socket I/O loopback to compile SocketChannel.read/write paths. */
    private static final int HTTP_SOCKET_ITERS = 1_000;

    /** Leaves ~25 s headroom in the contest engine's 60 s window (20 retries × 3000 ms). */
    private static final long WALL_CLOCK_BUDGET_MS = 35_000;

    /**
     * Every Nth search swaps the sampled-perturbed query for pure-random so the epsilon-relaxation
     * code path (active above RELAX_SOFT_CAP visits) sees realistic high-visit queries during JIT
     * warm-up. Period=16 → 6.25% far queries (up from 1.56% at period=64); far queries reach
     * visit counts well above SOFT_CAP=1500 while near queries stay below it, so both branches
     * of relaxScale get meaningful profile coverage before production traffic starts.
     */
    private static final int FAR_QUERY_PERIOD = 16;

    private static final int  TEMPLATE_COUNT  = 16;
    private static final long SEARCH_RNG_SEED = 0xC0DECAFE12345678L;

    private final VectorIndex index;
    private final ScoringRequestParser parser;
    private final Vectorizer vectorizer;
    private final Router router;

    public WarmupDriver(VectorIndex index, ScoringRequestParser parser, Vectorizer vectorizer, Router router) {
        this.index = Objects.requireNonNull(index, "index");
        this.parser = Objects.requireNonNull(parser, "parser");
        this.vectorizer = Objects.requireNonNull(vectorizer, "vectorizer");
        this.router = Objects.requireNonNull(router, "router");
    }

    public void run() {
        long deadlineNanos = System.nanoTime() + WALL_CLOCK_BUDGET_MS * 1_000_000L;
        log("starting warmup (budget " + WALL_CLOCK_BUDGET_MS + " ms)");

        byte[][] templateBodies = loadBodies();

        long t0 = System.nanoTime();
        int searchDone = warmSearch(deadlineNanos);
        long t1 = System.nanoTime();
        log("search done in " + ms(t1 - t0) + " ms (" + searchDone + " iters)");
        int parserDone = warmParser(deadlineNanos, templateBodies);
        long t2 = System.nanoTime();
        log("parser done in " + ms(t2 - t1) + " ms (" + parserDone + " iters)");
        int routeDone = warmFullRoute(deadlineNanos, templateBodies);
        long t3 = System.nanoTime();
        log("route done in " + ms(t3 - t2) + " ms (" + routeDone + " iters)");
        int httpParseDone = warmHttpParse(deadlineNanos, templateBodies);
        long t4 = System.nanoTime();
        log("http-parse done in " + ms(t4 - t3) + " ms (" + httpParseDone + " iters)");
        int httpSocketDone = warmHttpSocketIo(deadlineNanos);
        long t5 = System.nanoTime();
        log("http-io done in " + ms(t5 - t4) + " ms (" + httpSocketDone + " iters)");

        log(String.format(
            "warmup done: search=%d in %d ms, parser=%d in %d ms, route=%d in %d ms, "
            + "http-parse=%d in %d ms, http-io=%d in %d ms (total %d ms)",
            searchDone, ms(t1 - t0),
            parserDone, ms(t2 - t1),
            routeDone, ms(t3 - t2),
            httpParseDone, ms(t4 - t3),
            httpSocketDone, ms(t5 - t4),
            ms(t5 - t0)
        ));
    }

    private int warmHttpParse(long deadlineNanos, byte[][] templateBodies) {
        byte[][] frames = new byte[templateBodies.length][];
        for (int i = 0; i < templateBodies.length; i++) {
            frames[i] = buildHttpFrame(templateBodies[i]);
        }
        int iter = 0;
        while (iter < HTTP_PARSE_ITERS && System.nanoTime() < deadlineNanos) {
            int batch = Math.min(100, HTTP_PARSE_ITERS - iter);
            sink ^= NioAotTraining.trainAotHotPath(router, frames, batch);
            iter += batch;
        }
        return iter;
    }

    private int warmHttpSocketIo(long deadlineNanos) {
        return NioAotTraining.trainAotSocketIo(HTTP_SOCKET_ITERS, deadlineNanos);
    }

    public static byte[] buildHttpFrame(byte[] body) {
        String head = "POST /fraud-score HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
            + body.length + "\r\n\r\n";
        byte[] headBytes = head.getBytes(StandardCharsets.UTF_8);
        byte[] full = new byte[headBytes.length + body.length];
        System.arraycopy(headBytes, 0, full, 0, headBytes.length);
        System.arraycopy(body, 0, full, headBytes.length, body.length);
        return full;
    }

    private int warmSearch(long deadlineNanos) {
        int n = index.size();
        long state = SEARCH_RNG_SEED;
        float[] query = new float[KdTree.STRIDE];
        int iter = 0;

        while (iter < SEARCH_ITERS && System.nanoTime() < deadlineNanos) {
            if ((iter & (FAR_QUERY_PERIOD - 1)) == 0) {
                for (int d = 0; d < KdTree.DIMS; d++) {
                    state = xorshift64(state);
                    query[d] = (state & 0xFFFF) * (1.0f / 65535.0f);
                }
            } else {
                state = xorshift64(state);
                int treeIdx = (int) ((state >>> 1) % n);
                index.copyNodeVector(treeIdx, query);
                for (int d = 0; d < KdTree.DIMS; d++) {
                    state = xorshift64(state);
                    int signedByte = (int) (state & 0xFF) - 128;
                    query[d] += signedByte * 0.0002f;
                }
            }
            sink ^= index.countFraudsInTop5(query);
            iter++;
        }
        return iter;
    }

    private int warmParser(long deadlineNanos, byte[][] bodies) {
        ScoringRequestScratch scratch = new ScoringRequestScratch();
        float[] vector = new float[com.github.gb.moonshot.Dataset.STRIDE];
        int iter = 0;
        while (iter < PARSER_ITERS && System.nanoTime() < deadlineNanos) {
            byte[] body = bodies[iter % bodies.length];
            parser.parseInto(body, 0, body.length, scratch);
            vectorizer.vectorizeInto(body, scratch, vector);
            sink ^= Float.floatToRawIntBits(vector[0]);
            iter++;
        }
        return iter;
    }

    private int warmFullRoute(long deadlineNanos, byte[][] bodies) {
        int iter = 0;
        while (iter < ROUTE_ITERS && System.nanoTime() < deadlineNanos) {
            byte[] body = bodies[iter % bodies.length];
            byte[] response = router.route(Router.ROUTE_FRAUD_SCORE, body, 0, body.length);
            sink ^= response[0];
            iter++;
        }
        return iter;
    }

    /**
     * Loads the warmup payload bank from {@code /data/payloads.bin} (or the
     * path in {@code WARMUP_PAYLOADS} env var). Falls back to the in-class
     * 16-template synthesizer if the bank is missing or unreadable, so
     * dev / bench invocations without the bank still work.
     */
    public static byte[][] loadBodies() {
        String envPath = System.getenv("WARMUP_PAYLOADS");
        Path path = Path.of((envPath == null || envPath.isEmpty()) ? "/data/payloads.bin" : envPath);
        if (Files.isReadable(path)) {
            try {
                byte[][] bank = PayloadBankIO.read(path);
                log("loaded " + bank.length + " payloads from " + path);
                return bank;
            } catch (IOException e) {
                log("WARN: failed to read " + path + " (" + e.getMessage() + "); falling back to templates");
            }
        } else {
            log("WARN: " + path + " not present; falling back to 16 in-class templates");
        }
        return buildTemplateBodies();
    }

    public static byte[][] buildTemplateBodies() {
        String[] mccs = { "5912", "5411", "5732", "4111", "7011" };
        int[] installments = { 1, 3, 12, 6 };
        int[] knownMerchantsCounts = { 0, 1, 3, 5 };
        boolean[] flags = { false, true };

        byte[][] out = new byte[TEMPLATE_COUNT][];
        int idx = 0;
        for (int t = 0; t < TEMPLATE_COUNT; t++) {
            String mcc = mccs[t % mccs.length];
            int inst = installments[t % installments.length];
            int kmc = knownMerchantsCounts[t % knownMerchantsCounts.length];
            boolean online = flags[t % 2];
            boolean cardPresent = flags[(t / 2) % 2];
            boolean hasLastTx = (t % 2) == 0;
            double amount = 100.0 + t * 47.5;
            double customerAvg = 200.0 + t * 31.0;
            double merchantAvg = 150.0 + t * 23.0;
            double kmFromHome = 1.0 + t * 4.7;
            double kmFromCurrent = 0.5 + t * 3.3;
            int tx24h = 1 + (t * 7) % 50;
            int hour = (t * 5) % 24;

            StringBuilder km = new StringBuilder("[");
            for (int k = 0; k < kmc; k++) {
                if (k > 0) km.append(',');
                km.append("\"MERC-").append(String.format("%03d", k)).append('"');
            }
            km.append(']');

            String body = String.format(
                "{\"id\":\"tx-warmup-%04d\","
                + "\"transaction\":{\"amount\":%.2f,\"installments\":%d,\"requested_at\":\"2026-03-11T%02d:23:35Z\"},"
                + "\"customer\":{\"avg_amount\":%.2f,\"tx_count_24h\":%d,\"known_merchants\":%s},"
                + "\"merchant\":{\"id\":\"MERC-001\",\"mcc\":\"%s\",\"avg_amount\":%.2f},"
                + "\"terminal\":{\"is_online\":%b,\"card_present\":%b,\"km_from_home\":%.4f},"
                + "\"last_transaction\":%s}",
                t, amount, inst, hour,
                customerAvg, tx24h, km.toString(),
                mcc, merchantAvg,
                online, cardPresent, kmFromHome,
                hasLastTx
                    ? String.format("{\"timestamp\":\"2026-03-11T%02d:58:35Z\",\"km_from_current\":%.4f}", (hour + 23) % 24, kmFromCurrent)
                    : "null"
            );
            out[idx++] = body.getBytes(StandardCharsets.UTF_8);
        }
        return out;
    }

    /** DCE-defeat sink — warmup loops XOR result lanes here so the JIT can't elide calls. */
    @SuppressWarnings("unused")
    private static volatile int sink;

    private static long xorshift64(long x) {
        x ^= x << 13;
        x ^= x >>> 7;
        x ^= x << 17;
        return x;
    }

    private static long ms(long nanos) { return nanos / 1_000_000L; }

    private static void log(String msg) { System.out.println("[warmup] " + msg); }
}
