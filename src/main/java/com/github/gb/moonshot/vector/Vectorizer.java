package com.github.gb.moonshot.vector;

import com.github.gb.moonshot.Dataset;
import com.github.gb.moonshot.codec.ScoringRequestScratch;
import com.github.gb.moonshot.codec.ScoringRequest;

import static com.github.gb.moonshot.vector.NormalizationConstants.*;

/**
 * ScoringRequest to 14-dim normalized vector.
 *
 * CAUTION: indexes 5/6 receive a -1 sentinel when last_transaction is null the ONLY case where the vector contains
 * a value outside [0,1]. Reference vectors follow the same convention, so KNN groups "no history" rows together
 * naturally; do NOT filter or replace -1.
 *
 * Contest timestamps are fixed UTC ISO-8601 strings, so hour/day/epoch-minute extraction is done with direct
 * character arithmetic rather than java.time.
 */
public final class Vectorizer {

    private static final double MAX_HOUR_OF_DAY = 23.0;
    private static final double MAX_DAY_OF_WEEK = 6.0;
    private static final double INV_MAX_HOUR_OF_DAY = 1.0 / MAX_HOUR_OF_DAY;
    private static final double INV_MAX_DAY_OF_WEEK = 1.0 / MAX_DAY_OF_WEEK;
    private static final long MINUTES_PER_DAY = 1440L;
    private static final long MINUTES_PER_HOUR = 60L;
    /** Days from year 0 (proleptic Gregorian) to 1970-01-01 (Unix epoch). */
    private static final long EPOCH_DAYS_FROM_YEAR_ZERO = 719528L;
    /** Multiply by this to scale to 4-decimal grid; cheaper than div by 10000.0. */
    private static final double INV_TEN_THOUSAND = 1.0 / 10000.0;

    /**
     * Writes normalized floats into {@code outVector[0..14)}. Caller guarantees {@code outVector.length >=
     * Dataset.STRIDE} (16); lanes 14-15 are intentionally not written and must already be 0 so the SIMD distance
     * kernel sees them as zero contributions.
     */
    public void vectorizeInto(byte[] body, ScoringRequestScratch s, float[] outVector) {
        outVector[0] = clampRound(s.amount * INV_MAX_AMOUNT);
        outVector[1] = clampRound(s.installments * INV_MAX_INSTALLMENTS);
        // (amount/avg)/RATIO collapses to amount * INV_RATIO / avg = 1 div + 1 mul (vs 2 div). Guard zero-avg to
        // avoid NaN flowing downstream.
        double avgAmount = s.customerAvgAmount;
        outVector[2] = clampRound(avgAmount > 0
            ? (s.amount * INV_AMOUNT_VS_AVG_RATIO) / avgAmount
            : 0.0);

        // Single-pass timestamp parse: extract year/month/day/hour/min once, derive epochDay once, then compute
        // BOTH the epoch minute (lanes 5/6 delta against lastTimestamp) AND the day-of-week + hour fields.
        int reqOff   = s.requestedAtOffset;
        int reqYear  = fourDigits(body, reqOff);
        int reqMonth = twoDigits(body, reqOff + 5);
        int reqDay   = twoDigits(body, reqOff + 8);
        int reqHour  = twoDigits(body, reqOff + 11);
        int reqMin   = twoDigits(body, reqOff + 14);
        long reqEpochDay = toEpochDay(reqYear, reqMonth, reqDay);
        long requestedMinute = reqEpochDay * MINUTES_PER_DAY
                             + reqHour * MINUTES_PER_HOUR
                             + reqMin;
        // % vs Math.floorMod: epochDay is always positive in the valid range, so sign-correct floorMod isn't needed.
        int dayOfWeek = (int) ((reqEpochDay + 3L) % 7L);

        outVector[3] = round4(reqHour * INV_MAX_HOUR_OF_DAY);
        outVector[4] = round4(dayOfWeek * INV_MAX_DAY_OF_WEEK);

        if (s.lastTimestampOffset < 0) {
            outVector[5] = -1f;
            outVector[6] = -1f;
        } else {
            long lastMinute = parseEpochMinuteUtc(body, s.lastTimestampOffset);
            outVector[5] = clampRound((requestedMinute - lastMinute) * INV_MAX_MINUTES);
            outVector[6] = clampRound(s.lastKmFromCurrent * INV_MAX_KM);
        }

        outVector[7]  = clampRound(s.kmFromHome * INV_MAX_KM);
        outVector[8]  = clampRound(s.customerTxCount24h * INV_MAX_TX_COUNT_24H);
        outVector[9]  = s.isOnline ? 1f : 0f;
        outVector[10] = s.cardPresent ? 1f : 0f;
        outVector[11] = isUnknownMerchant(body, s) ? 1f : 0f;
        outVector[12] = round4(MccRiskTable.riskFor(body, s.mccOffset));
        outVector[13] = clampRound(s.merchantAvgAmount * INV_MAX_MERCHANT_AVG_AMOUNT);
    }

    /** Cold-path overload used by warmup/build/bench tools where allocation isn't on the hot path. */
    public float[] vectorize(ScoringRequest request) {
        float[] vector = new float[Dataset.STRIDE];

        vector[0] = clampRound(request.amount() * INV_MAX_AMOUNT);
        vector[1] = clampRound(request.installments() * INV_MAX_INSTALLMENTS);
        double avgAmount = request.customerAvgAmount();
        vector[2] = clampRound(avgAmount > 0
            ? (request.amount() * INV_AMOUNT_VS_AVG_RATIO) / avgAmount
            : 0.0);

        String requestedAt = request.requestedAt();
        int reqYear  = fourDigits(requestedAt, 0);
        int reqMonth = twoDigits(requestedAt, 5);
        int reqDay   = twoDigits(requestedAt, 8);
        int reqHour  = twoDigits(requestedAt, 11);
        int reqMin   = twoDigits(requestedAt, 14);
        long reqEpochDay = toEpochDay(reqYear, reqMonth, reqDay);
        long requestedMinute = reqEpochDay * MINUTES_PER_DAY
                             + reqHour * MINUTES_PER_HOUR
                             + reqMin;
        int dayOfWeek = (int) ((reqEpochDay + 3L) % 7L);

        vector[3] = round4(reqHour * INV_MAX_HOUR_OF_DAY);
        vector[4] = round4(dayOfWeek * INV_MAX_DAY_OF_WEEK);

        if (request.lastTimestamp() == null) {
            vector[5] = -1f;
            vector[6] = -1f;
        } else {
            long lastMinute = parseEpochMinuteUtc(request.lastTimestamp());
            vector[5] = clampRound((requestedMinute - lastMinute) * INV_MAX_MINUTES);
            vector[6] = clampRound(request.lastKmFromCurrent() * INV_MAX_KM);
        }

        vector[7]  = clampRound(request.kmFromHome() * INV_MAX_KM);
        vector[8]  = clampRound(request.customerTxCount24h() * INV_MAX_TX_COUNT_24H);
        vector[9]  = request.isOnline() ? 1f : 0f;
        vector[10] = request.cardPresent() ? 1f : 0f;
        vector[11] = isUnknown(request.merchantId(), request.knownMerchants()) ? 1f : 0f;
        vector[12] = round4(MccRiskTable.riskFor(request.mcc()));
        vector[13] = clampRound(request.merchantAvgAmount() * INV_MAX_MERCHANT_AVG_AMOUNT);

        return vector;
    }

    /** Parses contest timestamp shape {@code yyyy-MM-ddTHH:mm:ssZ} into epoch minutes. */
    private static long parseEpochMinuteUtc(byte[] body, int offset) {
        int year  = fourDigits(body, offset);
        int month = twoDigits(body, offset + 5);
        int day   = twoDigits(body, offset + 8);
        long epochDay = toEpochDay(year, month, day);
        return epochDay * MINUTES_PER_DAY
            + twoDigits(body, offset + 11) * MINUTES_PER_HOUR
            + twoDigits(body, offset + 14);
    }

    private static int fourDigits(byte[] body, int offset) {
        return (body[offset]     - '0') * 1000
             + (body[offset + 1] - '0') * 100
             + (body[offset + 2] - '0') * 10
             + (body[offset + 3] - '0');
    }

    private static int twoDigits(byte[] body, int offset) {
        return (body[offset] - '0') * 10 + (body[offset + 1] - '0');
    }

    private static boolean isUnknownMerchant(byte[] body, ScoringRequestScratch s) {
        int merchOff = s.merchantIdOffset;
        int merchLen = s.merchantIdLength;
        for (int i = 0; i < s.knownMerchantsCount; i++) {
            if (s.knownMerchantsLengths[i] != merchLen) continue;
            int knownOff = s.knownMerchantsOffsets[i];
            if (bytesEqual(body, knownOff, merchOff, merchLen)) return false;
        }
        return true;
    }

    private static boolean bytesEqual(byte[] body, int a, int b, int len) {
        for (int i = 0; i < len; i++) {
            if (body[a + i] != body[b + i]) return false;
        }
        return true;
    }

    private static long parseEpochMinuteUtc(String timestamp) {
        int year  = fourDigits(timestamp, 0);
        int month = twoDigits(timestamp, 5);
        int day   = twoDigits(timestamp, 8);
        long epochDay = toEpochDay(year, month, day);
        return epochDay * MINUTES_PER_DAY + twoDigits(timestamp, 11) * MINUTES_PER_HOUR + twoDigits(timestamp, 14);
    }

    private static int fourDigits(String timestamp, int offset) {
        return (timestamp.charAt(offset) - '0') * 1000
            + (timestamp.charAt(offset + 1) - '0') * 100
            + (timestamp.charAt(offset + 2) - '0') * 10
            + (timestamp.charAt(offset + 3) - '0');
    }

    private static int twoDigits(String timestamp, int offset) {
        return (timestamp.charAt(offset) - '0') * 10 + (timestamp.charAt(offset + 1) - '0');
    }

    private static boolean isUnknown(String merchantId, String[] knownMerchants) {
        for (String known : knownMerchants) {
            if (known.equals(merchantId)) return false;
        }
        return true;
    }

    private static long toEpochDay(int year, int month, int day) {
        long y = year;
        long m = month;
        long total = 365L * y;
        if (y >= 0) {
            total += (y + 3L) / 4L - (y + 99L) / 100L + (y + 399L) / 400L;
        } else {
            total -= y / -4L - y / -100L + y / -400L;
        }
        total += (367L * m - 362L) / 12L;
        total += day - 1L;
        if (m > 2) {
            total--;
            if (!isLeapYear(year)) total--;
        }
        return total - EPOCH_DAYS_FROM_YEAR_ZERO;
    }

    private static boolean isLeapYear(int year) {
        return ((year & 3) == 0) && (year % 100 != 0 || year % 400 == 0);
    }

    /**
     * Clamp to [0, 1] AND round to 4 decimals both required for KNN to match contest ground truth. The contest
     * data-generator rounds references at save time; an unrounded query produces tied-distance neighbors that
     * resolve in different order at the rank-K boundary, flipping the verdict on edge cases.
     *
     * Math.min/max compile to branchless MAXSD/MINSD on AVX2. NaN guard is upstream (callers check zero-divisors
     * before invoking) so no NaN reaches here.
     */
    private static float clampRound(double value) {
        return round4(Math.min(1.0, Math.max(0.0, value)));
    }

    /**
     * Round non-negative double in [0,1] to 4 decimal places. Cheaper than {@code Math.round}: the long cast skips
     * IEEE 754 corner-case logic, and {@code * INV_TEN_THOUSAND} avoids a runtime divide.
     */
    private static float round4(double value) {
        return (float) ((long) (value * 10000.0 + 0.5) * INV_TEN_THOUSAND);
    }
}
