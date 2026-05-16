package com.github.gb.moonshot.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.Month;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.Optional;
import java.util.Random;

/**
 * Builds {@code data/payloads.bin} the deterministic raw-JSON payload bank
 * for {@link P99Bench} {@code --mode warm-e2e} / {@code warm-parse} /
 * {@code warm-vectorize}.
 *
 * <h2>Source priority</h2>
 *
 * <ol>
 *   <li>{@code --source sample path/to/example-payloads.json} (or env
 *       {@code PAYLOADS_JSON}) sample with replacement from real contest
 *       payloads. Exact distribution, limited diversity
 *       (~50 distinct payloads in the official fixture, cycled up to N).
 *   <li>Default: synthetic, seeded. Matches the contest data-generator's
 *       observable shape without copying its fraud/legit profile logic (that
 *       would let the bench exploit label structure, which doesn't exist at
 *       runtime anyway).
 * </ol>
 *
 * <h2>Synthetic distribution choices</h2>
 *
 * Parameters derived from {@code example-payloads.json} (50 samples) +
 * {@code rinha-de-backend-2026/data-generator/main.c}. The goal is to exercise
 * the same parser / vectorizer branches at realistic frequency, NOT to
 * reproduce the contest fraud/legit label structure.
 *
 * <h2>Why not copy the LEGIT/FRAUD/BORDERLINE profiles</h2>
 *
 * The contest generator uses three behavioral profiles correlated with
 * label. At bench time we don't know the label, so sampling from a union of
 * the three profiles (which is what this class does) is strictly more
 * conservative: it exercises the same value-range of parser inputs without
 * letting bench code accidentally exploit label structure.
 *
 * <h2>Usage</h2>
 *
 * <pre>
 *   java -cp target/jvmoonshot-xxvi.jar com.github.gb.moonshot.bench.BuildPayloadBank \
 *        [--source sample PAYLOADS_JSON]
 *        [--out data/payloads.bin] [--n 10000] [--seed 42]
 * </pre>
 */
public final class BuildPayloadBank {

    private static final String LOG_PREFIX = "[build-payloads]";

    private static final int    MERCHANT_POOL              = 100;
    /** The 10 MCCs listed in mcc_risk.json hit the fast path ~90% of the time. */
    private static final String[] KNOWN_MCCS = {
        "5411","5812","5912","5944","7801","7802","7995","4511","5311","5999"
    };

    // ---- Synthetic-distribution constants (see Javadoc summary below) ----
    private static final long   ID_BASE                    = 1_000_000_000L;
    private static final double AMOUNT_LO                  = 10.0;
    private static final double AMOUNT_HI                  = 10_000.0;
    private static final int    INSTALLMENTS_MAX           = 12;
    private static final double CUST_AVG_LO                = 50.0;
    private static final double CUST_AVG_HI                = 1_000.0;
    private static final int    TX_24H_HI_EXCLUSIVE        = 21;
    private static final int    KNOWN_MERCHANTS_MIN        = 2;
    private static final int    KNOWN_MERCHANTS_RANGE      = 5;   // gives [2..6] inclusive
    private static final double MERCHANT_KNOWN_PROBABILITY = 0.5;
    private static final double UNKNOWN_MCC_PROBABILITY    = 0.10;
    private static final int    UNKNOWN_MCC_MIN            = 1000;
    private static final int    UNKNOWN_MCC_RANGE          = 9000;
    private static final double MERCH_AVG_LO               = 20.0;
    private static final double MERCH_AVG_HI               = 500.0;
    private static final double ONLINE_PROBABILITY         = 0.40;
    private static final double CARD_PRESENT_GIVEN_ONLINE  = 0.10;
    private static final double CARD_PRESENT_GIVEN_OFFLINE = 0.90;
    private static final double KM_RANGE                   = 1000.0;
    private static final double NULL_LAST_TX_PROBABILITY   = 0.20;
    private static final int    LAST_TX_MAX_MINUTES_BACK   = 720;
    private static final int    INITIAL_JSON_CAPACITY      = 512;

    // ---- Timestamp window (yyyy-MM-dd uniformly across March 2026) ----
    private static final int   TS_YEAR  = 2026;
    private static final Month TS_MONTH = Month.MARCH;
    private static final DateTimeFormatter ISO_FORMATTER =
        DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH:mm:ss'Z'");

    public static void main(String[] args) throws IOException {
        String source     = CliArgs.string(args, "--source", "");
        String fixturePath = CliArgs.string(args, "--source-path",
            System.getenv().getOrDefault("PAYLOADS_JSON",
                "../rinha-de-backend-2026/resources/example-payloads.json"));
        Path   outPath    = Path.of(CliArgs.string(args, "--out", "data/payloads.bin"));
        int    count      = CliArgs.intVal(args, "--n", 10_000);
        long   seed       = CliArgs.longVal(args, "--seed", 42L);
        if (count <= 0) {
            throw new IllegalArgumentException("--n must be positive, got " + count);
        }

        BuildKdTree.ensureParentDirectory(outPath);

        byte[][] bank = chooseSource(source, fixturePath, count, seed);

        long bytes = PayloadBankIO.write(outPath, bank);
        log("wrote " + bank.length + " payloads → " + outPath + " (" + bytes + " bytes)");
        log("bank stats: " + describe(bank));
    }

    private static byte[][] chooseSource(String source, String fixturePath, int count, long seed) throws IOException {
        if ("sample".equals(source) && Files.exists(Path.of(fixturePath))) {
            log("source: real contest payloads (" + fixturePath + ") sampled with replacement");
            return sampleFromContest(Path.of(fixturePath), count, seed);
        }
        if ("sample".equals(source)) {
            log("source: contest fixture requested but not found at " + fixturePath + " using synthetic");
        } else {
            log("source: synthetic (pass --source sample to use contest payloads)");
        }
        return synthesize(count, seed);
    }

    // =========================================================================
    // Contest-fixture sampling (deterministic with replacement)
    // =========================================================================

    private static byte[][] sampleFromContest(Path fixturePath, int count, long seed) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode array = mapper.readTree(fixturePath.toFile());
        int sourceCount = array.size();
        if (sourceCount == 0) throw new IllegalStateException("empty fixture: " + fixturePath);

        byte[][] sourceBodies = new byte[sourceCount][];
        for (int i = 0; i < sourceCount; i++) sourceBodies[i] = mapper.writeValueAsBytes(array.get(i));

        Random rng = new Random(seed);
        byte[][] bank = new byte[count][];
        for (int i = 0; i < count; i++) bank[i] = sourceBodies[rng.nextInt(sourceCount)];
        return bank;
    }

    // =========================================================================
    // Synthetic generator
    // =========================================================================

    private static byte[][] synthesize(int count, long seed) {
        Random rng = new Random(seed);
        byte[][] bank = new byte[count][];
        for (int i = 0; i < count; i++) bank[i] = synthOne(rng);
        return bank;
    }

    private static byte[] synthOne(Random rng) {
        SyntheticPayload payload = drawPayload(rng);
        return renderJson(payload).getBytes(StandardCharsets.UTF_8);
    }

    private static SyntheticPayload drawPayload(Random rng) {
        long idNum = ID_BASE + ((long) rng.nextInt(Integer.MAX_VALUE));
        double amount       = round2(logUniform(rng, AMOUNT_LO, AMOUNT_HI));
        int    installments = 1 + rng.nextInt(INSTALLMENTS_MAX);
        LocalDateTime requestedAt = randomTimestamp(rng);
        double customerAvg  = round2(logUniform(rng, CUST_AVG_LO, CUST_AVG_HI));
        int    tx24h        = rng.nextInt(TX_24H_HI_EXCLUSIVE);
        int    knownCount   = KNOWN_MERCHANTS_MIN + rng.nextInt(KNOWN_MERCHANTS_RANGE);
        String[] knownMerchants = new String[knownCount];
        for (int i = 0; i < knownCount; i++) {
            knownMerchants[i] = formatMerchantId(rng.nextInt(MERCHANT_POOL) + 1);
        }
        String merchantId = (rng.nextDouble() < MERCHANT_KNOWN_PROBABILITY)
            ? knownMerchants[rng.nextInt(knownCount)]
            : formatMerchantId(rng.nextInt(MERCHANT_POOL) + 1);
        String mcc = (rng.nextDouble() < UNKNOWN_MCC_PROBABILITY)
            ? String.format("%04d", UNKNOWN_MCC_MIN + rng.nextInt(UNKNOWN_MCC_RANGE))
            : KNOWN_MCCS[rng.nextInt(KNOWN_MCCS.length)];
        double merchantAvg  = round2(logUniform(rng, MERCH_AVG_LO, MERCH_AVG_HI));
        boolean online      = rng.nextDouble() < ONLINE_PROBABILITY;
        double cardPresentP = online ? CARD_PRESENT_GIVEN_ONLINE : CARD_PRESENT_GIVEN_OFFLINE;
        boolean cardPresent = rng.nextDouble() < cardPresentP;
        double kmFromHome   = rng.nextDouble() * KM_RANGE;

        Optional<LastTransaction> lastTx;
        if (rng.nextDouble() < NULL_LAST_TX_PROBABILITY) {
            lastTx = Optional.empty();
        } else {
            int minsBack = 1 + rng.nextInt(LAST_TX_MAX_MINUTES_BACK);
            lastTx = Optional.of(new LastTransaction(
                requestedAt.minusMinutes(minsBack),
                rng.nextDouble() * KM_RANGE));
        }

        return new SyntheticPayload(
            idNum, amount, installments, requestedAt,
            customerAvg, tx24h, knownMerchants,
            merchantId, mcc, merchantAvg,
            online, cardPresent, kmFromHome,
            lastTx
        );
    }

    private static String renderJson(SyntheticPayload p) {
        StringBuilder b = new StringBuilder(INITIAL_JSON_CAPACITY);
        b.append("{\"id\":\"tx-").append(p.idNum).append("\",");
        b.append("\"transaction\":{");
        b.append("\"amount\":").append(p.amount);
        b.append(",\"installments\":").append(p.installments);
        b.append(",\"requested_at\":\"").append(formatTs(p.requestedAt)).append("\"},");
        b.append("\"customer\":{");
        b.append("\"avg_amount\":").append(p.customerAvg);
        b.append(",\"tx_count_24h\":").append(p.tx24h);
        b.append(",\"known_merchants\":[");
        for (int i = 0; i < p.knownMerchants.length; i++) {
            if (i > 0) b.append(',');
            b.append('"').append(p.knownMerchants[i]).append('"');
        }
        b.append("]},");
        b.append("\"merchant\":{");
        b.append("\"id\":\"").append(p.merchantId).append("\",");
        b.append("\"mcc\":\"").append(p.mcc).append("\",");
        b.append("\"avg_amount\":").append(p.merchantAvg).append("},");
        b.append("\"terminal\":{");
        b.append("\"is_online\":").append(p.online);
        b.append(",\"card_present\":").append(p.cardPresent);
        b.append(",\"km_from_home\":").append(p.kmFromHome).append("},");
        if (p.lastTx.isEmpty()) {
            b.append("\"last_transaction\":null");
        } else {
            LastTransaction last = p.lastTx.get();
            b.append("\"last_transaction\":{");
            b.append("\"timestamp\":\"").append(formatTs(last.timestamp())).append("\",");
            b.append("\"km_from_current\":").append(last.kmFromCurrent()).append('}');
        }
        b.append('}');
        return b.toString();
    }

    private static double logUniform(Random rng, double lo, double hi) {
        return lo * Math.pow(hi / lo, rng.nextDouble());
    }

    private static String formatMerchantId(int n) {
        return String.format("MERC-%03d", n);
    }

    /**
     * Uniformly random instant within the entire 31-day window of March 2026.
     * Note: the prior implementation was capped at day 28; this one covers
     * the full month.
     */
    private static LocalDateTime randomTimestamp(Random rng) {
        int daysInMonth = TS_MONTH.length(false);
        int day  = 1 + rng.nextInt(daysInMonth);
        int hour = rng.nextInt(24);
        int min  = rng.nextInt(60);
        int sec  = rng.nextInt(60);
        return LocalDateTime.of(LocalDate.of(TS_YEAR, TS_MONTH, day),
                                LocalTime.of(hour, min, sec));
    }

    private static String formatTs(LocalDateTime ts) {
        return ts.atOffset(ZoneOffset.UTC).format(ISO_FORMATTER);
    }

    private static double round2(double x) {
        return Math.round(x * 100.0) / 100.0;
    }

    // =========================================================================
    // Reporting
    // =========================================================================

    private static String describe(byte[][] bank) {
        long totalBytes = 0;
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (byte[] payload : bank) {
            totalBytes += payload.length;
            if (payload.length < min) min = payload.length;
            if (payload.length > max) max = payload.length;
        }
        return String.format("size min=%d max=%d mean=%.0f  total=%dKB",
            min, max, totalBytes / (double) bank.length, totalBytes / 1024);
    }

    private static void log(String message) {
        System.out.println(LOG_PREFIX + " " + message);
    }

    private BuildPayloadBank() {}

    /** Plain data carrier for one synthetic payload keeps {@link #renderJson} pure. */
    private record SyntheticPayload(
        long idNum,
        double amount,
        int installments,
        LocalDateTime requestedAt,
        double customerAvg,
        int tx24h,
        String[] knownMerchants,
        String merchantId,
        String mcc,
        double merchantAvg,
        boolean online,
        boolean cardPresent,
        double kmFromHome,
        Optional<LastTransaction> lastTx
    ) {}

    /** Companion record for {@link SyntheticPayload}'s optional last-transaction segment. */
    private record LastTransaction(LocalDateTime timestamp, double kmFromCurrent) {}
}
