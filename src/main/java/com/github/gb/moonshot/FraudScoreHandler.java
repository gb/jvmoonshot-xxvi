package com.github.gb.moonshot;

import com.github.gb.moonshot.codec.ScoringRequestParser;
import com.github.gb.moonshot.codec.ScoringRequestScratch;
import com.github.gb.moonshot.instrumentation.StageTimer;
import com.github.gb.moonshot.search.KdTreeProbes;
import com.github.gb.moonshot.search.VectorIndex;
import com.github.gb.moonshot.vector.Vectorizer;

/**
 * {@code body} is sliced by {@code (offset, length)} from the NIO read buffer — nothing is copied between the
 * wire and the parser.
 *
 * <p><b>Thread-safety</b>: {@code scratch} and {@code queryVector} live in a {@link ThreadLocal} so the same
 * instance can be called from the {@code rinha-io} NIO thread (NioHttpServer) and from ForkJoinPool carrier
 * threads (FdConnHandler virtual threads) without per-call allocation.
 */
public final class FraudScoreHandler {

    /**
     * Scoring weights are FP=1, FN=3, Err=5;
     * returning 3 → score 0.6 → DENY costs at most 1 FP, far cheaper than letting the request surface as 5xx.
     */
    private static final int FAIL_SAFE_FRAUD_COUNT = 3;

    /** {@code printStackTrace()} locks on {@code System.err}; disabled by default to avoid contention under error load. */
    private static final boolean LOG_ERRORS = "1".equals(System.getenv("LOG_ERRORS"));

    private static final class Buffers {
        final ScoringRequestScratch scratch = new ScoringRequestScratch();
        /** Lanes 14-15 are stride padding; the vectorizer never writes them, so they stay 0. */
        final float[] queryVector = new float[Dataset.STRIDE];
    }

    private static final ThreadLocal<Buffers> BUFFERS_THREAD_LOCAL = ThreadLocal.withInitial(Buffers::new);

    private final ScoringRequestParser parser;
    private final Vectorizer vectorizer;
    private final VectorIndex vectorIndex;

    public FraudScoreHandler(ScoringRequestParser parser, Vectorizer vectorizer, VectorIndex vectorIndex) {
        this.parser = parser;
        this.vectorizer = vectorizer;
        this.vectorIndex = vectorIndex;
    }

    public int countFraudNeighbors(byte[] body, int offset, int length) {
        try {
            Buffers buffers = BUFFERS_THREAD_LOCAL.get();
            parser.parseInto(body, offset, length, buffers.scratch);
            if (StageTimer.ENABLED) StageTimer.mark(0, System.nanoTime());

            vectorizer.vectorizeInto(body, buffers.scratch, buffers.queryVector);
            if (StageTimer.ENABLED) StageTimer.mark(1, System.nanoTime());

            int fraudCount = vectorIndex.countFraudsInTop5(buffers.queryVector);
            if (StageTimer.ENABLED) {
                StageTimer.mark(2, System.nanoTime());
                StageTimer.recordVisits(KdTreeProbes.lastQueryNodesVisited());
            }
            return fraudCount;
        } catch (Throwable error) {
            if (LOG_ERRORS) error.printStackTrace();
            return FAIL_SAFE_FRAUD_COUNT;
        }
    }
}
