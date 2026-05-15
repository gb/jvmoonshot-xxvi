package com.github.gb.moonshot.instrumentation;

import java.util.Arrays;

/**
 * Per-request stage timing collector. Single-threaded by design only the rinha-io NIO loop calls
 * {@link #mark(int, long)} / {@link #complete()}, so the {@code idx}/{@code BUF}/{@code VISITS} fields need no
 * synchronization. Allocates one {@code long[N*STAGES]} at startup; no per-request allocation.
 *
 * Enable with {@code STAGE_TIMING=1}. When disabled, {@link #mark} / {@link #complete} short-circuit on the first
 * {@code if (!ENABLED) return;} HotSpot eliminates the disabled path once {@code ENABLED} stabilises.
 *
 * Stages (nanoseconds relative to {@code t0}, the request-bytes-ready instant): 0 jsonParse, 1 vectorize, 2 search,
 * 3 encode, 4 write.
 *
 * Buffers fill once, dump to stdout, then reset. With N=100_000 and a 250K-request bench you get two dumps; the
 * second is the steady-state picture (the first includes JIT warmup).
 *
 * Not thread-safe all calls must come from the rinha-io NIO loop thread; callers outside that thread will see
 * torn or lost samples.
 */
public final class StageTimer {

    public static final boolean ENABLED;
    static {
        String value = System.getenv("STAGE_TIMING");
        ENABLED = value != null && !value.isEmpty() && !"0".equals(value) && !"false".equalsIgnoreCase(value);
    }

    public static final int STAGES = 5;
    public static final String[] NAMES = { "jsonParse", "vectorize", "search", "encode", "write" };

    public static final int STAGE_JSON_PARSE = 0;
    public static final int STAGE_VECTORIZE  = 1;
    public static final int STAGE_SEARCH     = 2;
    public static final int STAGE_ENCODE     = 3;
    public static final int STAGE_WRITE      = 4;

    /**
     * Requests per dump. Default 100K (≈ one dump per two contest tests); {@code STAGE_DUMP_EVERY} overrides to
     * fire dumps within a single test when comparing stage breakdowns between fast and slow runs.
     */
    private static final int N;
    static {
        String dumpEvery = System.getenv("STAGE_DUMP_EVERY");
        int parsed = 100_000;
        if (dumpEvery != null && !dumpEvery.isEmpty()) {
            try { parsed = Integer.parseInt(dumpEvery); }
            catch (NumberFormatException ignored) { }
        }
        N = Math.max(1, parsed);
    }
    private static final long[] BUF = new long[N * STAGES];
    private static final int[]  VISITS = new int[N];
    private static int idx;
    private static int dumpCount;

    /** Set by {@code NioHttpServer} at the start of each request. Public for hot-path inlining. */
    public static long t0;

    private static final double[] PERCENTILES = { 0.50, 0.90, 0.95, 0.99, 0.999 };
    private static final String   PERCENTILE_HEADER_FMT = "%-10s  %8s  %8s  %8s  %8s  %8s  %8s%n";
    private static final String   PERCENTILE_DOUBLE_FMT = "%-10s  %8.1f  %8.1f  %8.1f  %8.1f  %8.1f  %8.1f%n";
    private static final String   PERCENTILE_INT_FMT    = "%-10s  %8d  %8d  %8d  %8d  %8d  %8d%n";
    private static final double   NANOS_PER_MICRO       = 1000.0;
    private static final double   FAST_FRACTION         = 0.50;
    private static final double   SLOW_FRACTION         = 0.99;
    private static final double   ALGORITHMIC_RATIO_HINT = 4.0;

    public static void mark(int stage, long now) {
        if (!ENABLED) return;
        if (idx >= N) return;
        BUF[idx * STAGES + stage] = now - t0;
    }

    public static void recordVisits(int n) {
        if (!ENABLED) return;
        if (idx >= N) return;
        VISITS[idx] = n;
    }

    public static void complete() {
        if (!ENABLED) return;
        if (idx < N) {
            idx++;
            if (idx == N) {
                dumpCount++;
                dump();
                idx = 0;
            }
        }
    }

    private static void dump() {
        System.out.println();
        System.out.println("=== STAGE TIMINGS dump #" + dumpCount
            + " (" + N + " requests, μs cumulative from request start) ===");
        System.out.printf(PERCENTILE_HEADER_FMT, "stage", "p50", "p90", "p95", "p99", "p99.9", "max");

        long[] tmp = new long[N];
        printStagePercentiles(tmp);
        printVisitPercentiles();
        printLatencyVisitCorrelation();
    }

    private static void printStagePercentiles(long[] tmp) {
        for (int stage = 0; stage < STAGES; stage++) {
            for (int i = 0; i < N; i++) tmp[i] = BUF[i * STAGES + stage];
            Arrays.sort(tmp);
            System.out.printf(PERCENTILE_DOUBLE_FMT,
                NAMES[stage],
                quantileMicros(tmp, PERCENTILES[0]),
                quantileMicros(tmp, PERCENTILES[1]),
                quantileMicros(tmp, PERCENTILES[2]),
                quantileMicros(tmp, PERCENTILES[3]),
                quantileMicros(tmp, PERCENTILES[4]),
                tmp[N - 1] / NANOS_PER_MICRO);
        }
    }

    private static void printVisitPercentiles() {
        int[] visitsSorted = VISITS.clone();
        Arrays.sort(visitsSorted);
        System.out.printf(PERCENTILE_INT_FMT,
            "visits",
            quantileInt(visitsSorted, PERCENTILES[0]),
            quantileInt(visitsSorted, PERCENTILES[1]),
            quantileInt(visitsSorted, PERCENTILES[2]),
            quantileInt(visitsSorted, PERCENTILES[3]),
            quantileInt(visitsSorted, PERCENTILES[4]),
            visitsSorted[N - 1]);
    }

    /**
     * Ranks requests by per-request search-stage duration and compares the visits distribution of the bottom 50%
     * (fast) against the top 1% (slow). Ratio ≈ 4× suggests algorithmic cost; ratio ≈ 1× suggests memory-bound.
     */
    private static void printLatencyVisitCorrelation() {
        long[] searchDurations = new long[N];
        for (int i = 0; i < N; i++) {
            searchDurations[i] = BUF[i * STAGES + STAGE_SEARCH] - BUF[i * STAGES + STAGE_VECTORIZE];
        }
        int[] orderByDurationAsc = sortIndicesByLongAsc(searchDurations);

        int fastEnd   = (int) (N * FAST_FRACTION);
        int slowStart = (int) (N * SLOW_FRACTION);
        int slowEnd   = N;

        VisitsBucket fast = collectVisitsAt(orderByDurationAsc, 0, fastEnd);
        VisitsBucket slow = collectVisitsAt(orderByDurationAsc, slowStart, slowEnd);

        System.out.println();
        System.out.println("--- visits correlated with search-stage latency ---");
        System.out.printf("bottom 50%% by latency: visits mean=%.1f  median=%d  (n=%d)%n",
            fast.mean, fast.median, fast.count);
        System.out.printf("top    1%% by latency: visits mean=%.1f  median=%d  (n=%d)%n",
            slow.mean, slow.median, slow.count);
        System.out.printf("ratio (slow/fast):     mean=%.2fx  median=%.2fx%n",
            slow.mean / fast.mean, slow.median / (double) fast.median);
        System.out.println("(ratio ≈ " + ALGORITHMIC_RATIO_HINT + "x → algorithmic; ratio ≈ 1x → memory-bound)");
        System.out.println();
    }

    private static VisitsBucket collectVisitsAt(int[] orderByDurationAsc, int start, int endExclusive) {
        int count = endExclusive - start;
        int[] visits = new int[count];
        long sum = 0;
        for (int i = 0; i < count; i++) {
            int requestIdx = orderByDurationAsc[start + i];
            visits[i] = VISITS[requestIdx];
            sum += visits[i];
        }
        Arrays.sort(visits);
        double mean = sum / (double) count;
        int    median = visits[count / 2];
        return new VisitsBucket(mean, median, count);
    }

    /**
     * Returns indices into {@code keys} sorted by ascending key, using primitive-long packing rather than
     * {@code Integer[]} + comparator to avoid 100K boxed integers per dump. Requires non-negative keys.
     */
    private static int[] sortIndicesByLongAsc(long[] keys) {
        int n = keys.length;
        long[] packed = new long[n];
        for (int i = 0; i < n; i++) {
            packed[i] = (keys[i] << 32) | (i & 0xFFFFFFFFL);
        }
        Arrays.sort(packed);
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = (int) (packed[i] & 0xFFFFFFFFL);
        return order;
    }

    private static double quantileMicros(long[] sorted, double p) {
        return sorted[(int) (sorted.length * p)] / NANOS_PER_MICRO;
    }

    private static int quantileInt(int[] sorted, double p) {
        return sorted[(int) (sorted.length * p)];
    }

    private record VisitsBucket(double mean, int median, int count) {}

    private StageTimer() {}
}
