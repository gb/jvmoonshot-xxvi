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
import java.util.Locale;
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

    /** Pushes tryParse, Router.fraudScoreResponse, and countFraudNeighborsFast past tier 4. */
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
     * of relaxScaleQ16 get meaningful profile coverage before production traffic starts.
     */
    private static final int FAR_QUERY_PERIOD = 16;

    private static final int  TEMPLATE_COUNT  = 16;
    private static final long SEARCH_RNG_SEED = 0xC0DECAFE12345678L;
    private static final byte[] HTTP_FRAME_PREFIX =
            "POST /fraud-score HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                    .getBytes(StandardCharsets.US_ASCII);
    private static final byte[] HTTP_FRAME_SUFFIX = "\r\n\r\n".getBytes(StandardCharsets.US_ASCII);

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
        run(loadBodies());
    }

    public void run(byte[][] templateBodies) {
        long deadlineNanos = System.nanoTime() + WALL_CLOCK_BUDGET_MS * 1_000_000L;
        log("starting warmup (budget " + WALL_CLOCK_BUDGET_MS + " ms)");

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

        log("warmup done: search=" + searchDone + " in " + ms(t1 - t0)
                + " ms, parser=" + parserDone + " in " + ms(t2 - t1)
                + " ms, route=" + routeDone + " in " + ms(t3 - t2)
                + " ms, http-parse=" + httpParseDone + " in " + ms(t4 - t3)
                + " ms, http-io=" + httpSocketDone + " in " + ms(t5 - t4)
                + " ms (total " + ms(t5 - t0) + " ms)");
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
        int digits = decimalDigits(body.length);
        byte[] full = new byte[HTTP_FRAME_PREFIX.length + digits + HTTP_FRAME_SUFFIX.length + body.length];
        int p = 0;
        System.arraycopy(HTTP_FRAME_PREFIX, 0, full, p, HTTP_FRAME_PREFIX.length);
        p += HTTP_FRAME_PREFIX.length;
        writeDecimal(full, p, body.length, digits);
        p += digits;
        System.arraycopy(HTTP_FRAME_SUFFIX, 0, full, p, HTTP_FRAME_SUFFIX.length);
        p += HTTP_FRAME_SUFFIX.length;
        System.arraycopy(body, 0, full, p, body.length);
        return full;
    }

    private static int decimalDigits(int v) {
        if (v >= 100_000_000) return v >= 1_000_000_000 ? 10 : 9;
        if (v >= 1_000_000) return v >= 10_000_000 ? 8 : 7;
        if (v >= 10_000) return v >= 100_000 ? 6 : 5;
        if (v >= 1000) return 4;
        if (v >= 100) return 3;
        if (v >= 10) return 2;
        return 1;
    }

    private static void writeDecimal(byte[] out, int offset, int value, int digits) {
        int p = offset + digits;
        int v = value;
        do {
            out[--p] = (byte) ('0' + (v % 10));
            v /= 10;
        } while (p > offset);
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
            int respIdx = router.fraudScoreResponseIndex(body, 0, body.length);
            sink ^= respIdx;
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
        String[] requestedAt = {
            "2026-03-01T09:23:35Z", "2026-07-18T16:12:04Z", "2027-02-09T11:44:19Z", "2027-10-27T19:51:02Z",
            "2028-01-14T02:08:43Z", "2028-06-03T05:37:10Z", "2028-12-22T00:19:58Z", "2029-04-11T06:45:31Z",
            "2029-09-28T03:02:17Z", "2030-01-06T14:26:50Z", "2030-05-19T20:05:13Z", "2030-12-28T18:39:44Z",
            "2026-11-07T10:17:25Z", "2027-05-16T22:48:07Z", "2029-02-24T13:59:36Z", "2030-08-12T04:31:21Z"
        };
        String[] lastAt = {
            null, "2023-04-21T08:11:02Z", "2026-12-29T15:39:44Z", "2022-10-05T13:27:19Z",
            "2019-03-08T01:41:03Z", null, "2020-12-17T20:22:50Z", "2024-06-03T05:18:12Z",
            "2016-08-30T23:59:58Z", "2028-11-19T11:02:45Z", null, "2026-01-14T17:09:32Z",
            "2021-07-01T07:26:54Z", "2025-05-16T21:48:07Z", "2028-02-22T13:59:36Z", "2027-09-12T04:31:21Z"
        };
        double[] amount = {
            41.12, 288.45, 476.90, 123.34,
            7340.10, 9505.97, 2218.55, 6682.32,
            4890.71, 510.40, 1840.25, 2990.99,
            395.00, 430.50, 2760.80, 2450.65
        };
        int[] installments = { 1, 2, 3, 1, 8, 10, 6, 12, 7, 3, 5, 7, 3, 4, 6, 5 };
        double[] customerAvg = {
            82.24, 410.12, 953.80, 232.10,
            123.55, 88.91, 245.40, 174.20,
            298.73, 340.00, 208.65, 455.35,
            700.20, 112.00, 280.50, 430.15
        };
        int[] tx24h = { 1, 3, 5, 2, 8, 19, 12, 15, 10, 4, 9, 11, 5, 6, 18, 7 };
        int[] knownStart = { 1, 4, 7, 10, 2, 5, 8, 11, 14, 3, 6, 9, 12, 15, 17, 1 };
        int[] knownCount = { 2, 4, 5, 3, 2, 5, 4, 3, 5, 2, 4, 5, 3, 4, 2, 5 };
        String[] merchantId = {
            "MERC-001", "MERC-006", "MERC-011", "MERC-012",
            "MERC-050", "MERC-088", "MERC-063", "MERC-097",
            "MERC-074", "MERC-031", "MERC-047", "MERC-059",
            "MERC-014", "MERC-038", "MERC-091", "MERC-055"
        };
        String[] mccs = {
            "5411", "5812", "5912", "5311",
            "7995", "7801", "7802", "7995",
            "7801", "5944", "4511", "5999",
            "5411", "7802", "7995", "5944"
        };
        double[] merchantAvg = {
            55.10, 302.78, 488.45, 140.00,
            31.70, 74.25, 93.80, 45.55,
            88.40, 120.10, 211.35, 285.70,
            430.80, 150.00, 62.95, 240.20
        };
        boolean[] online = {
            false, true, false, false,
            true, true, true, false,
            true, false, true, false,
            false, true, true, true
        };
        boolean[] cardPresent = {
            true, false, true, true,
            false, false, false, true,
            false, true, false, true,
            true, false, false, false
        };
        double[] kmFromHome = {
            2.7, 18.4, 47.6, 6.1,
            920.3, 403.7, 281.4, 765.2,
            612.8, 55.1, 288.9, 397.5,
            22.0, 84.6, 948.4, 331.6
        };
        double[] kmFromCurrent = {
            0.0, 8.2, 12.6, 19.7,
            811.3, 452.9, 278.4, 923.6,
            554.1, 31.5, 118.2, 260.4,
            5.5, 74.0, 875.2, 210.9
        };

        byte[][] out = new byte[TEMPLATE_COUNT][];
        int idx = 0;
        for (int t = 0; t < TEMPLATE_COUNT; t++) {
            String knownMerchants = knownMerchantsJson(knownStart[t], knownCount[t]);

            String body = String.format(Locale.ROOT,
                "{\"id\":\"tx-warmup-%04d\","
                + "\"transaction\":{\"amount\":%.2f,\"installments\":%d,\"requested_at\":\"%s\"},"
                + "\"customer\":{\"avg_amount\":%.2f,\"tx_count_24h\":%d,\"known_merchants\":%s},"
                + "\"merchant\":{\"id\":\"%s\",\"mcc\":\"%s\",\"avg_amount\":%.2f},"
                + "\"terminal\":{\"is_online\":%b,\"card_present\":%b,\"km_from_home\":%.4f},"
                + "\"last_transaction\":%s}",
                t, amount[t], installments[t], requestedAt[t],
                customerAvg[t], tx24h[t], knownMerchants,
                merchantId[t], mccs[t], merchantAvg[t],
                online[t], cardPresent[t], kmFromHome[t],
                lastAt[t] == null
                    ? "null"
                    : String.format(Locale.ROOT,
                        "{\"timestamp\":\"%s\",\"km_from_current\":%.4f}",
                        lastAt[t], kmFromCurrent[t])
            );
            out[idx++] = body.getBytes(StandardCharsets.UTF_8);
        }
        return out;
    }

    private static String knownMerchantsJson(int start, int count) {
        StringBuilder km = new StringBuilder("[");
        for (int k = 0; k < count; k++) {
            if (k > 0) km.append(',');
            km.append("\"MERC-").append(String.format(Locale.ROOT, "%03d", ((start + k - 1) % 19) + 1)).append('"');
        }
        km.append(']');
        return km.toString();
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
