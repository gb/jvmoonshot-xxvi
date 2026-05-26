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
 *       {@code PAYLOADS_JSON}) sample with replacement from fixture payloads.
 *       Exact distribution, limited diversity
 *       (~50 distinct payloads, cycled up to N).
 *   <li>Default: synthetic, seeded. Matches the fixture's observable JSON
 *       shape, randomized date range, and observable profile mix.
 * </ol>
 *
 * <h2>Synthetic distribution choices</h2>
 *
 * Parameters keep the synthetic bank close to the production traffic shape so
 * warmup reaches the same parser and vectorizer branches without embedding
 * labels in the generated bank.
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

    private enum Profile {
        LEGIT, FRAUD, BORDERLINE
    }

    private static final int    MERCHANT_POOL              = 100;
    private static final double FRAUD_RATIO_PAYLOADS       = 0.47;
    private static final double BORDERLINE_SHARE_OF_FRAUD  = 0.10;
    private static final String[] LEGIT_MCCS = {"5411", "5812", "5912", "5311"};
    private static final String[] FRAUD_MCCS = {"7995", "7801", "7802"};
    /** Ordered to match resources/mcc_risk.json. */
    private static final String[] ALL_MCCS = {
        "5411", "5812", "5912", "5944", "7801", "7802", "7995", "4511", "5311", "5999"
    };

    // ---- Synthetic-distribution constants (see Javadoc summary below) ----
    private static final long   ID_BASE                    = 1_000_000_000L;
    private static final double NULL_LAST_TX_PROBABILITY   = 0.20;
    private static final long   RANDOMIZED_LAST_TX_MAX_SECONDS_BACK = 10L * 365 * 24 * 3600;
    private static final int    INITIAL_JSON_CAPACITY      = 512;

    private static final int RANDOM_DATE_START_YEAR = 2026;
    private static final int RANDOM_DATE_END_YEAR_EXCLUSIVE = 2031;
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
            log("source: fixture payloads (" + fixturePath + ") sampled with replacement");
            return sampleFromFixture(Path.of(fixturePath), count, seed);
        }
        if ("sample".equals(source)) {
            log("source: fixture requested but not found at " + fixturePath + " using synthetic");
        } else {
            log("source: synthetic (pass --source sample to use fixture payloads)");
        }
        return synthesize(count, seed);
    }

    // =========================================================================
    // Fixture sampling (deterministic with replacement)
    // =========================================================================

    private static byte[][] sampleFromFixture(Path fixturePath, int count, long seed) throws IOException {
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
        Profile profile = pickProfile(rng);
        long idNum = ID_BASE + ((long) rng.nextInt(Integer.MAX_VALUE));
        double amount = round2(switch (profile) {
            case LEGIT -> uniform(rng, 10, 500);
            case FRAUD -> uniform(rng, 2000, 10000);
            case BORDERLINE -> uniform(rng, 400, 3000);
        });
        int installments = switch (profile) {
            case LEGIT -> rngInt(rng, 1, 4);
            case FRAUD -> rngInt(rng, 6, 13);
            case BORDERLINE -> rngInt(rng, 3, 8);
        };
        LocalDateTime requestedAt = randomTimestamp(rng, profile);
        double customerAvg = round2(switch (profile) {
            case LEGIT -> uniform(rng, amount / 0.5, amount * 2.0);
            case FRAUD -> uniform(rng, 50, 300);
            case BORDERLINE -> uniform(rng, 100, 500);
        });
        int tx24h = switch (profile) {
            case LEGIT -> rngInt(rng, 1, 6);
            case FRAUD -> rngInt(rng, 8, 21);
            case BORDERLINE -> rngInt(rng, 4, 12);
        };
        int knownCount = rngInt(rng, 2, 6);
        String[] knownMerchants = new String[knownCount];
        for (int i = 0; i < knownCount; i++) {
            knownMerchants[i] = formatMerchantId(rngInt(rng, 1, 20));
        }
        String merchantId = switch (profile) {
            case LEGIT -> knownMerchants[rng.nextInt(knownCount)];
            case FRAUD -> formatMerchantId(rngInt(rng, 50, MERCHANT_POOL));
            case BORDERLINE -> rng.nextDouble() < 0.5
                    ? knownMerchants[rng.nextInt(knownCount)]
                    : formatMerchantId(rngInt(rng, 30, 60));
        };
        String mcc = switch (profile) {
            case LEGIT -> LEGIT_MCCS[rng.nextInt(LEGIT_MCCS.length)];
            case FRAUD -> FRAUD_MCCS[rng.nextInt(FRAUD_MCCS.length)];
            case BORDERLINE -> ALL_MCCS[rng.nextInt(ALL_MCCS.length)];
        };
        double merchantAvg = round2(switch (profile) {
            case LEGIT -> uniform(rng, 30, 500);
            case FRAUD -> uniform(rng, 20, 100);
            case BORDERLINE -> uniform(rng, 50, 300);
        });
        boolean online = rng.nextDouble() < switch (profile) {
            case LEGIT -> 0.3;
            case FRAUD -> 0.8;
            case BORDERLINE -> 0.5;
        };
        boolean cardPresent = online ? false : rng.nextDouble() < 0.9;
        double kmFromHome = switch (profile) {
            case LEGIT -> uniform(rng, 0, 50);
            case FRAUD -> uniform(rng, 200, 1000);
            case BORDERLINE -> uniform(rng, 30, 400);
        };

        Optional<LastTransaction> lastTx;
        if (rng.nextDouble() < NULL_LAST_TX_PROBABILITY) {
            lastTx = Optional.empty();
        } else {
            long secondsBack = 60 + rng.nextLong(RANDOMIZED_LAST_TX_MAX_SECONDS_BACK);
            lastTx = Optional.of(new LastTransaction(
                requestedAt.minusSeconds(secondsBack),
                switch (profile) {
                    case LEGIT -> uniform(rng, 0, 20);
                    case FRAUD -> uniform(rng, 200, 1000);
                    case BORDERLINE -> uniform(rng, 20, 300);
                }));
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

    private static Profile pickProfile(Random rng) {
        double borderline = FRAUD_RATIO_PAYLOADS * BORDERLINE_SHARE_OF_FRAUD;
        double v = rng.nextDouble();
        if (v < 1.0 - FRAUD_RATIO_PAYLOADS) return Profile.LEGIT;
        if (v < 1.0 - borderline) return Profile.FRAUD;
        return Profile.BORDERLINE;
    }

    private static int rngInt(Random rng, int loInclusive, int hiExclusive) {
        return loInclusive + rng.nextInt(hiExclusive - loInclusive);
    }

    private static double uniform(Random rng, double lo, double hi) {
        return lo + rng.nextDouble() * (hi - lo);
    }

    private static String formatMerchantId(int n) {
        return String.format("MERC-%03d", n);
    }

    /**
     * Randomized payload date window: requested_at in 2026-03..2030-12, with
     * days limited to 1..28 so every month is valid. The hour band remains
     * profile-specific.
     */
    private static LocalDateTime randomTimestamp(Random rng, Profile profile) {
        int year = rngInt(rng, RANDOM_DATE_START_YEAR, RANDOM_DATE_END_YEAR_EXCLUSIVE);
        int month = year == 2026 ? rngInt(rng, 3, 13) : rngInt(rng, 1, 13);
        int day = rngInt(rng, 1, 29);
        int hour = switch (profile) {
            case LEGIT -> rngInt(rng, 8, 21);
            case FRAUD -> rngInt(rng, 0, 7);
            case BORDERLINE -> rngInt(rng, 6, 23);
        };
        int min  = rng.nextInt(60);
        int sec  = rng.nextInt(60);
        return LocalDateTime.of(LocalDate.of(year, month, day), LocalTime.of(hour, min, sec));
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
