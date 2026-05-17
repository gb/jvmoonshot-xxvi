package com.github.gb.moonshot.http;

import com.github.gb.moonshot.FraudScoreHandler;
import com.github.gb.moonshot.codec.ResponseEncoder;

/**
 * Top-level request dispatcher. Maps the route id parsed by {@link com.github.gb.moonshot.http.HttpConnection} to a pre-framed
 * HTTP response for the fraud-score route, the response is selected from a 6-entry table indexed by the
 * fraud-neighbor count returned by {@link FraudScoreHandler}. The {@code /ready} route is gated on
 * {@link #markReady()} so the contest health check only flips green once the runtime warmup has stabilised JIT
 * speculation.
 */
public final class Router {

    public static final int ROUTE_NOT_FOUND = 0;
    public static final int ROUTE_READY = 1;
    public static final int ROUTE_FRAUD_SCORE = 2;

    private static final int MAX_FRAUD_COUNT = 5;

    private static final byte[][] FRAUD_RESPONSES = new byte[MAX_FRAUD_COUNT + 1][];
    private static final byte[] READY_RESPONSE = ResponseEncoder.readyResponse();
    private static final byte[] NOT_READY_RESPONSE = ResponseEncoder.notReadyResponse();
    private static final byte[] NOT_FOUND_RESPONSE = ResponseEncoder.notFoundResponse();

    static {
        for (int fraudCount = 0; fraudCount <= MAX_FRAUD_COUNT; fraudCount++) {
            FRAUD_RESPONSES[fraudCount] = ResponseEncoder.fullResponseForFraudCount(fraudCount);
        }
    }

    private final FraudScoreHandler fraudHandler;

    /**
     * Stays false until {@link #markReady()} fires set after the runtime live-listener pump has serviced enough
     * live traffic to stabilise JIT speculation (notably the ThreadLocal single-receiver speculation that deopts
     * when the carrier-thread variant first appears).
     */
    private volatile boolean ready;

    public Router(FraudScoreHandler fraudHandler) {
        this.fraudHandler = fraudHandler;
    }

    public void markReady() {
        ready = true;
    }

    /** Fast contest path: caller already proved {@code ROUTE_FRAUD_SCORE}, so skip switch + ready volatile read. */
    public byte[] fraudScoreResponse(byte[] body, int bodyOffset, int bodyLength) {
        return FRAUD_RESPONSES[fraudHandler.countFraudNeighborsFast(body, bodyOffset, bodyLength)];
    }

    public byte[] route(int routeId, byte[] body, int bodyOffset, int bodyLength) {
        return switch (routeId) {
            case ROUTE_FRAUD_SCORE -> FRAUD_RESPONSES[fraudHandler.countFraudNeighbors(body, bodyOffset, bodyLength)];
            case ROUTE_READY -> ready ? READY_RESPONSE : NOT_READY_RESPONSE;
            default -> NOT_FOUND_RESPONSE;
        };
    }
}
