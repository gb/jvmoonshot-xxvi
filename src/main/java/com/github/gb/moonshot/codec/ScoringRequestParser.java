package com.github.gb.moonshot.codec;

/**
 * Byte-level, single-pass, schema-driven JSON parser for the fixed /fraud-score request shape.
 * <p>
 * Hot path {@link #parseInto} writes byte-range views into a caller-owned {@link ScoringRequestScratch} zero
 * allocation per call. Keys are dispatched by byte comparison; string-shaped values are stored as
 * offset + length into the source body. Cold path {@link #parse} materializes a {@link ScoringRequest}.
 * <p>
 * Not thread-safe: the instance owns a single mutable Cursor reused across calls. Exactly one rinha-io
 * thread calls into the parser in production.
 */
public final class ScoringRequestParser {

    private static final String[] EMPTY_STRINGS = new String[0];

    private static final int K_TOP_UNKNOWN = 0;
    private static final int K_TOP_ID = 1;
    private static final int K_TOP_TRANSACTION = 2;
    private static final int K_TOP_CUSTOMER = 3;
    private static final int K_TOP_MERCHANT = 4;
    private static final int K_TOP_TERMINAL = 5;
    private static final int K_TOP_LAST_TRANSACTION = 6;

    private static final int K_TX_UNKNOWN = 0;
    private static final int K_TX_AMOUNT = 1;
    private static final int K_TX_INSTALLMENTS = 2;
    private static final int K_TX_REQUESTED_AT = 3;

    private static final int K_CUST_UNKNOWN = 0;
    private static final int K_CUST_AVG_AMOUNT = 1;
    private static final int K_CUST_TX_COUNT_24H = 2;
    private static final int K_CUST_KNOWN_MERCHANTS = 3;

    private static final int K_MERCH_UNKNOWN = 0;
    private static final int K_MERCH_ID = 1;
    private static final int K_MERCH_MCC = 2;
    private static final int K_MERCH_AVG_AMOUNT = 3;

    private static final int K_TERM_UNKNOWN = 0;
    private static final int K_TERM_IS_ONLINE = 1;
    private static final int K_TERM_CARD_PRESENT = 2;
    private static final int K_TERM_KM_FROM_HOME = 3;

    private static final int K_LAST_UNKNOWN = 0;
    private static final int K_LAST_TIMESTAMP = 1;
    private static final int K_LAST_KM_FROM_CURRENT = 2;

    private final Cursor cursor = new Cursor();
    private final ScoringRequestScratch coldScratch = new ScoringRequestScratch();

    private static ScoringRequest materializePayload(byte[] body, ScoringRequestScratch s) {
        String requestedAt = s.requestedAtOffset >= 0
                ? new String(body, s.requestedAtOffset, ScoringRequestScratch.TIMESTAMP_LEN)
                : null;
        String lastTimestamp = s.lastTimestampOffset >= 0
                ? new String(body, s.lastTimestampOffset, ScoringRequestScratch.TIMESTAMP_LEN)
                : null;
        String mcc = new String(body, s.mccOffset, ScoringRequestScratch.MCC_LEN);
        String merchantId = new String(body, s.merchantIdOffset, s.merchantIdLength);

        String[] knownMerchants;
        if (s.knownMerchantsCount == 0) {
            knownMerchants = EMPTY_STRINGS;
        } else {
            knownMerchants = new String[s.knownMerchantsCount];
            for (int i = 0; i < s.knownMerchantsCount; i++) {
                knownMerchants[i] = new String(body, s.knownMerchantsOffsets[i], s.knownMerchantsLengths[i]);
            }
        }

        return new ScoringRequest(
                null, // id skipped on hot path; no consumer reads it
                s.amount, s.installments, requestedAt,
                s.customerAvgAmount, s.customerTxCount24h, knownMerchants,
                merchantId, mcc, s.merchantAvgAmount,
                s.isOnline, s.cardPresent, s.kmFromHome,
                lastTimestamp, s.lastKmFromCurrent
        );
    }

    /**
     * Hot-path parse: writes parsed fields into {@code scratch} as primitive values + byte-range views.
     * Variable-cardinality fields are reset by {@link ScoringRequestScratch#clear()}.
     */
    public void parseInto(byte[] body, int offset, int length, ScoringRequestScratch scratch) {
        scratch.clear();
        cursor.reset(body, offset, offset + length);

        cursor.skipWs();
        cursor.expect('{');
        while (cursor.skipWsAndPeek() != '}') {
            int key = cursor.readTopKey();
            cursor.skipWs();
            cursor.expect(':');
            cursor.skipWs();

            switch (key) {
                case K_TOP_ID -> cursor.skipString();
                case K_TOP_TRANSACTION -> parseTransaction(scratch);
                case K_TOP_CUSTOMER -> parseCustomer(scratch);
                case K_TOP_MERCHANT -> parseMerchant(scratch);
                case K_TOP_TERMINAL -> parseTerminal(scratch);
                case K_TOP_LAST_TRANSACTION -> parseLastTransaction(scratch);
                default -> cursor.skipValue();
            }
            cursor.skipWs();
            if (cursor.peek() == ',') cursor.consume();
        }
        cursor.expect('}');
    }

    public ScoringRequest parse(byte[] body, int offset, int length) {
        parseInto(body, offset, length, coldScratch);
        return materializePayload(body, coldScratch);
    }

    public ScoringRequest parse(byte[] body) {
        return parse(body, 0, body.length);
    }

    private void parseTransaction(ScoringRequestScratch scratch) {
        cursor.expect('{');
        while (cursor.skipWsAndPeek() != '}') {
            int fieldKey = cursor.readTransactionKey();
            cursor.skipWs();
            cursor.expect(':');
            cursor.skipWs();
            switch (fieldKey) {
                case K_TX_AMOUNT -> scratch.amount = cursor.readNumber();
                case K_TX_INSTALLMENTS -> scratch.installments = cursor.readInt();
                case K_TX_REQUESTED_AT ->
                        scratch.requestedAtOffset = cursor.captureFixedString(ScoringRequestScratch.TIMESTAMP_LEN);
                default -> cursor.skipValue();
            }
            cursor.skipWs();
            if (cursor.peek() == ',') cursor.consume();
        }
        cursor.expect('}');
    }

    private void parseCustomer(ScoringRequestScratch scratch) {
        cursor.expect('{');
        while (cursor.skipWsAndPeek() != '}') {
            int fieldKey = cursor.readCustomerKey();
            cursor.skipWs();
            cursor.expect(':');
            cursor.skipWs();
            switch (fieldKey) {
                case K_CUST_AVG_AMOUNT -> scratch.customerAvgAmount = cursor.readNumber();
                case K_CUST_TX_COUNT_24H -> scratch.customerTxCount24h = cursor.readInt();
                case K_CUST_KNOWN_MERCHANTS -> cursor.captureKnownMerchants(scratch);
                default -> cursor.skipValue();
            }
            cursor.skipWs();
            if (cursor.peek() == ',') cursor.consume();
        }
        cursor.expect('}');
    }

    private void parseMerchant(ScoringRequestScratch scratch) {
        cursor.expect('{');
        while (cursor.skipWsAndPeek() != '}') {
            int fieldKey = cursor.readMerchantKey();
            cursor.skipWs();
            cursor.expect(':');
            cursor.skipWs();
            switch (fieldKey) {
                case K_MERCH_ID -> {
                    long range = cursor.captureStringRange();
                    scratch.merchantIdOffset = (int) (range >>> 32);
                    scratch.merchantIdLength = (int) range;
                    int code = cursor.decodeMerchantCode(scratch.merchantIdOffset, scratch.merchantIdLength);
                    scratch.merchantIdCode = code;
                    scratch.merchantIdCodeValid = code >= 0;
                }
                case K_MERCH_MCC -> scratch.mccOffset = cursor.captureFixedString(ScoringRequestScratch.MCC_LEN);
                case K_MERCH_AVG_AMOUNT -> scratch.merchantAvgAmount = cursor.readNumber();
                default -> cursor.skipValue();
            }
            cursor.skipWs();
            if (cursor.peek() == ',') cursor.consume();
        }
        cursor.expect('}');
    }

    private void parseTerminal(ScoringRequestScratch scratch) {
        cursor.expect('{');
        while (cursor.skipWsAndPeek() != '}') {
            int fieldKey = cursor.readTerminalKey();
            cursor.skipWs();
            cursor.expect(':');
            cursor.skipWs();
            switch (fieldKey) {
                case K_TERM_IS_ONLINE -> scratch.isOnline = cursor.readBool();
                case K_TERM_CARD_PRESENT -> scratch.cardPresent = cursor.readBool();
                case K_TERM_KM_FROM_HOME -> scratch.kmFromHome = cursor.readNumber();
                default -> cursor.skipValue();
            }
            cursor.skipWs();
            if (cursor.peek() == ',') cursor.consume();
        }
        cursor.expect('}');
    }

    private void parseLastTransaction(ScoringRequestScratch scratch) {
        cursor.skipWs();
        if (cursor.peek() == 'n') {
            cursor.expectLiteral("null");
            return; // lastTimestampOffset stays ABSENT from clear()
        }
        cursor.expect('{');
        while (cursor.skipWsAndPeek() != '}') {
            int fieldKey = cursor.readLastTxKey();
            cursor.skipWs();
            cursor.expect(':');
            cursor.skipWs();
            switch (fieldKey) {
                case K_LAST_TIMESTAMP ->
                        scratch.lastTimestampOffset = cursor.captureFixedString(ScoringRequestScratch.TIMESTAMP_LEN);
                case K_LAST_KM_FROM_CURRENT -> scratch.lastKmFromCurrent = cursor.readNumber();
                default -> cursor.skipValue();
            }
            cursor.skipWs();
            if (cursor.peek() == ',') cursor.consume();
        }
        cursor.expect('}');
    }

    /**
     * Linear byte-array scanner reset on every {@link #parseInto} call so it allocates exactly once at
     * parser construction. Helpers either consume bytes or capture an (offset, length) view into the
     * source body no String allocation in the hot path.
     */
    private static final class Cursor {
        // Compile-time literals: JVM stores the nearest-representable IEEE 754 value same accuracy
        // as Math.pow but ~30x faster. NEG_POW10[k] = 10^-k, POW10[k] = 10^k.
        private static final int MAX_POW10_INDEX = 18;
        private static final double[] NEG_POW10 = {
                1.0,
                1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
                1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
                1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18
        };
        private static final double[] POW10 = {
                1.0,
                1e1, 1e2, 1e3, 1e4, 1e5, 1e6,
                1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
                1e13, 1e14, 1e15, 1e16, 1e17, 1e18
        };
        byte[] bytes;
        int end;
        int pos;

        void reset(byte[] bytes, int offset, int end) {
            this.bytes = bytes;
            this.pos = offset;
            this.end = end;
        }

        byte peek() {
            return bytes[pos];
        }

        byte consume() {
            return bytes[pos++];
        }

        byte skipWsAndPeek() {
            skipWs();
            return bytes[pos];
        }

        void skipWs() {
            while (pos < end) {
                byte b = bytes[pos];
                if (b == ' ' || b == '\n' || b == '\r' || b == '\t') pos++;
                else return;
            }
        }

        void expect(char c) {
            if (bytes[pos] != (byte) c) {
                throw new IllegalStateException("expected '" + c + "' at " + pos + " got '" + (char) bytes[pos] + "'");
            }
            pos++;
        }

        void expectLiteral(String literal) {
            for (int i = 0; i < literal.length(); i++) {
                if (bytes[pos + i] != (byte) literal.charAt(i)) {
                    throw new IllegalStateException("expected literal '" + literal + "' at " + pos);
                }
            }
            pos += literal.length();
        }

        void skipString() {
            expect('"');
            while (bytes[pos] != '"') {
                if (bytes[pos] == '\\') pos++;
                pos++;
            }
            pos++;
        }

        /**
         * Capture a quoted string value as a packed {@code (start << 32) | length}.
         */
        long captureStringRange() {
            expect('"');
            int start = pos;
            while (bytes[pos] != '"') {
                if (bytes[pos] == '\\') pos++;
                pos++;
            }
            int len = pos - start;
            pos++;
            return ((long) start << 32) | (len & 0xFFFFFFFFL);
        }

        /**
         * Capture a quoted string of schema-fixed length (mcc=4, timestamp=20). Advances blindly past
         * body bytes plus closing quote; malformed input surfaces at the next expect().
         */
        int captureFixedString(int expectedLen) {
            expect('"');
            int start = pos;
            pos += expectedLen + 1;
            return start;
        }

        void captureKnownMerchants(ScoringRequestScratch scratch) {
            expect('[');
            skipWs();
            if (bytes[pos] == ']') {
                pos++;
                return;
            }
            while (true) {
                skipWs();
                int count = scratch.knownMerchantsCount;
                scratch.ensureKnownMerchantsCapacity(count + 1);
                long range = captureStringRange();
                scratch.knownMerchantsOffsets[count] = (int) (range >>> 32);
                scratch.knownMerchantsLengths[count] = (int) range;
                if (scratch.knownMerchantMaskValid) {
                    int code = decodeMerchantCode(scratch.knownMerchantsOffsets[count],
                            scratch.knownMerchantsLengths[count]);
                    if (code >= 0 && code < Long.SIZE) {
                        scratch.knownMerchantMask |= 1L << code;
                    } else {
                        scratch.knownMerchantMaskValid = false;
                    }
                }
                scratch.knownMerchantsCount = count + 1;
                skipWs();
                if (bytes[pos] == ',') {
                    pos++;
                    continue;
                }
                expect(']');
                return;
            }
        }

        int readTopKey() {
            long range = captureStringRange();
            int start = (int) (range >>> 32);
            int len = (int) range;
            switch (len) {
                case 2: // "id"
                    return (bytes[start] == 'i' && bytes[start + 1] == 'd') ? K_TOP_ID : K_TOP_UNKNOWN;
                case 8: { // "customer" | "merchant" | "terminal"
                    byte first = bytes[start];
                    if (first == 'c') return K_TOP_CUSTOMER;
                    if (first == 'm') return K_TOP_MERCHANT;
                    if (first == 't') return K_TOP_TERMINAL;
                    return K_TOP_UNKNOWN;
                }
                case 11: // "transaction"
                    return (bytes[start] == 't') ? K_TOP_TRANSACTION : K_TOP_UNKNOWN;
                case 16: // "last_transaction"
                    return (bytes[start] == 'l') ? K_TOP_LAST_TRANSACTION : K_TOP_UNKNOWN;
                default:
                    return K_TOP_UNKNOWN;
            }
        }

        int readTransactionKey() {
            long range = captureStringRange();
            int start = (int) (range >>> 32);
            int len = (int) range;
            switch (len) {
                case 6:  // "amount"
                    return (bytes[start] == 'a') ? K_TX_AMOUNT : K_TX_UNKNOWN;
                case 12: // "installments" | "requested_at"
                    byte first = bytes[start];
                    if (first == 'i') return K_TX_INSTALLMENTS;
                    if (first == 'r') return K_TX_REQUESTED_AT;
                    return K_TX_UNKNOWN;
                default:
                    return K_TX_UNKNOWN;
            }
        }

        int readCustomerKey() {
            long range = captureStringRange();
            int start = (int) (range >>> 32);
            int len = (int) range;
            switch (len) {
                case 10: // "avg_amount"
                    return (bytes[start] == 'a') ? K_CUST_AVG_AMOUNT : K_CUST_UNKNOWN;
                case 12: // "tx_count_24h"
                    return (bytes[start] == 't') ? K_CUST_TX_COUNT_24H : K_CUST_UNKNOWN;
                case 15: // "known_merchants"
                    return (bytes[start] == 'k') ? K_CUST_KNOWN_MERCHANTS : K_CUST_UNKNOWN;
                default:
                    return K_CUST_UNKNOWN;
            }
        }

        int readMerchantKey() {
            long range = captureStringRange();
            int start = (int) (range >>> 32);
            int len = (int) range;
            switch (len) {
                case 2:  // "id"
                    return (bytes[start] == 'i') ? K_MERCH_ID : K_MERCH_UNKNOWN;
                case 3:  // "mcc"
                    return (bytes[start] == 'm') ? K_MERCH_MCC : K_MERCH_UNKNOWN;
                case 10: // "avg_amount"
                    return (bytes[start] == 'a') ? K_MERCH_AVG_AMOUNT : K_MERCH_UNKNOWN;
                default:
                    return K_MERCH_UNKNOWN;
            }
        }

        int readTerminalKey() {
            long range = captureStringRange();
            int start = (int) (range >>> 32);
            int len = (int) range;
            switch (len) {
                case 9:  // "is_online"
                    return (bytes[start] == 'i') ? K_TERM_IS_ONLINE : K_TERM_UNKNOWN;
                case 12: // "card_present" | "km_from_home"
                    byte first = bytes[start];
                    if (first == 'c') return K_TERM_CARD_PRESENT;
                    if (first == 'k') return K_TERM_KM_FROM_HOME;
                    return K_TERM_UNKNOWN;
                default:
                    return K_TERM_UNKNOWN;
            }
        }

        int readLastTxKey() {
            long range = captureStringRange();
            int start = (int) (range >>> 32);
            int len = (int) range;
            switch (len) {
                case 9:  // "timestamp"
                    return (bytes[start] == 't') ? K_LAST_TIMESTAMP : K_LAST_UNKNOWN;
                case 15: // "km_from_current"
                    return (bytes[start] == 'k') ? K_LAST_KM_FROM_CURRENT : K_LAST_UNKNOWN;
                default:
                    return K_LAST_UNKNOWN;
            }
        }

        double readNumber() {
            int p = pos;
            boolean negative = false;
            if (bytes[p] == '-') {
                negative = true;
                p++;
            } else if (bytes[p] == '+') {
                p++;
            }

            long intPart = 0;
            while (p < end) {
                byte b = bytes[p];
                if (b < '0' || b > '9') break;
                intPart = intPart * 10 + (b - '0');
                p++;
            }

            double value = intPart;
            if (p < end && bytes[p] == '.') {
                p++;
                long fracDigits = 0;
                int fracCount = 0;
                while (p < end) {
                    byte b = bytes[p];
                    if (b < '0' || b > '9') break;
                    fracDigits = fracDigits * 10 + (b - '0');
                    fracCount++;
                    p++;
                }
                if (fracCount > 0) {
                    int idx = fracCount <= MAX_POW10_INDEX ? fracCount : MAX_POW10_INDEX;
                    value += fracDigits * NEG_POW10[idx];
                }
            }

            if (p < end && (bytes[p] == 'e' || bytes[p] == 'E')) {
                p++;
                boolean exponentNegative = false;
                if (bytes[p] == '-') {
                    exponentNegative = true;
                    p++;
                } else if (bytes[p] == '+') {
                    p++;
                }
                int exp = 0;
                while (p < end) {
                    byte b = bytes[p];
                    if (b < '0' || b > '9') break;
                    exp = exp * 10 + (b - '0');
                    p++;
                }
                if (exp <= MAX_POW10_INDEX) {
                    value *= exponentNegative ? NEG_POW10[exp] : POW10[exp];
                } else {
                    value *= Math.pow(10.0d, exponentNegative ? -exp : exp);
                }
            }

            pos = p;
            return negative ? -value : value;
        }

        int readInt() {
            int p = pos;
            int value = 0;
            while (p < end) {
                byte b = bytes[p];
                if (b < '0' || b > '9') break;
                value = value * 10 + (b - '0');
                p++;
            }
            pos = p;
            return value;
        }

        boolean readBool() {
            if (bytes[pos] == 't') {
                expectLiteral("true");
                return true;
            }
            if (bytes[pos] == 'f') {
                expectLiteral("false");
                return false;
            }
            throw new IllegalStateException("expected boolean at " + pos);
        }

        void skipValue() {
            skipWs();
            byte b = bytes[pos];
            switch (b) {
                case '"' -> skipString();
                case '{' -> skipObject();
                case '[' -> skipArray();
                case 't' -> expectLiteral("true");
                case 'f' -> expectLiteral("false");
                case 'n' -> expectLiteral("null");
                default -> readNumber();
            }
        }

        private void skipObject() {
            expect('{');
            skipWs();
            if (bytes[pos] == '}') {
                pos++;
                return;
            }
            while (true) {
                skipWs();
                skipString();
                skipWs();
                expect(':');
                skipValue();
                skipWs();
                if (bytes[pos] == ',') {
                    pos++;
                    continue;
                }
                expect('}');
                return;
            }
        }

        private void skipArray() {
            expect('[');
            skipWs();
            if (bytes[pos] == ']') {
                pos++;
                return;
            }
            while (true) {
                skipValue();
                skipWs();
                if (bytes[pos] == ',') {
                    pos++;
                    continue;
                }
                expect(']');
                return;
            }
        }

        private int decodeMerchantCode(int start, int len) {
            if (len != 8
                    || bytes[start] != 'M'
                    || bytes[start + 1] != 'E'
                    || bytes[start + 2] != 'R'
                    || bytes[start + 3] != 'C'
                    || bytes[start + 4] != '-') {
                return -1;
            }
            byte a = bytes[start + 5];
            byte b = bytes[start + 6];
            byte c = bytes[start + 7];
            if (a < '0' || a > '9' || b < '0' || b > '9' || c < '0' || c > '9') return -1;
            return (a - '0') * 100 + (b - '0') * 10 + (c - '0');
        }
    }
}
