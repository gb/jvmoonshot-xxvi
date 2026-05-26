package com.github.gb.moonshot.http;

import com.github.gb.moonshot.FraudScoreHandler;
import com.github.gb.moonshot.codec.ResponseEncoder;

/**
 * Top-level request dispatcher. Returns a {@link ResponseEncoder} response index for the parsed
 * routeId. The fraud-score route picks among six pre-framed responses (one per fraud-neighbor
 * count). The {@code /ready} route is gated on {@link #markReady()} so the contest health check
 * only flips green once the runtime warmup has stabilised JIT speculation.
 */
public final class Router {

    public static final int ROUTE_NOT_FOUND = 0;
    public static final int ROUTE_READY = 1;
    public static final int ROUTE_FRAUD_SCORE = 2;

    private final FraudScoreHandler fraudHandler;

    /**
     * Stays false until {@link #markReady()} fires — set after the runtime live-listener pump has
     * serviced enough live traffic to stabilise JIT speculation (notably the ThreadLocal
     * single-receiver speculation that deopts when the carrier-thread variant first appears).
     */
    private volatile boolean ready;

    public Router(FraudScoreHandler fraudHandler) {
        this.fraudHandler = fraudHandler;
    }

    public void markReady() {
        ready = true;
    }

    /** Fast contest path: caller already proved {@code ROUTE_FRAUD_SCORE}, so skip switch + ready volatile read. */
    public int fraudScoreResponseIndex(byte[] body, int bodyOffset, int bodyLength) {
        return ResponseEncoder.fraudCountToIndex(
                fraudHandler.countFraudNeighborsFast(body, bodyOffset, bodyLength));
    }

    public int routeResponseIndex(int routeId, byte[] body, int bodyOffset, int bodyLength) {
        return switch (routeId) {
            case ROUTE_FRAUD_SCORE -> ResponseEncoder.fraudCountToIndex(
                    fraudHandler.countFraudNeighbors(body, bodyOffset, bodyLength));
            case ROUTE_READY -> ready ? ResponseEncoder.RESP_READY : ResponseEncoder.RESP_NOT_READY;
            default -> ResponseEncoder.RESP_NOT_FOUND;
        };
    }
}
