package com.github.gb.moonshot;

import com.github.gb.moonshot.codec.ScoringRequestParser;
import com.github.gb.moonshot.codec.ScoringRequestScratch;
import com.github.gb.moonshot.instrumentation.StageTimer;
import com.github.gb.moonshot.search.KdTreeProbes;
import com.github.gb.moonshot.search.VectorIndex;
import com.github.gb.moonshot.vector.Vectorizer;

/**
 * Turns a raw HTTP request body into a fraud-neighbor count: parses the JSON payload, vectorizes it, and asks the
 * vector index how many of the top-5 nearest neighbors are flagged as fraud (k=5 is fixed in
 * {@link com.github.gb.moonshot.search.VectorIndex#countFraudsInTop5}). Returns
 * {@link #FAIL_SAFE_FRAUD_COUNT} on any exception so the request loop never surfaces a 5xx.
 *
 * The body is passed as an {@code (offset, length)} slice into the NIO read buffer so nothing is copied between the
 * wire and the parser. The hot path uses zero-alloc {@code parseInto} + {@code vectorizeInto} with reusable scratch
 * buffers.
 *
 * Single-threaded by design, the rinha-io NIO loop is the only caller, so the reusable scratch and vector buffers
 * are instance fields, not thread-locals.
 */
public final class FraudScoreHandler {

    /**
     * Scoring weights are FP=1, FN=3, Err=5; returning 3 → score 0.6 → DENY costs at most 1 FP, far cheaper than
     * letting the request surface as 5xx.
     */
    private static final int FAIL_SAFE_FRAUD_COUNT = 3;

    /** Off by default under failure load, {@code printStackTrace()} synchronizes on {@code System.err}. */
    private static final boolean LOG_ERRORS = "1".equals(System.getenv("LOG_ERRORS"));

    private final ScoringRequestParser parser;
    private final Vectorizer vectorizer;
    private final VectorIndex vectorIndex;

    private final ScoringRequestScratch scratch = new ScoringRequestScratch();

    /** Lanes 14-15 are stride padding; the vectorizer never writes them, so they stay 0. */
    private final float[] queryVector = new float[Dataset.STRIDE];

    public FraudScoreHandler(ScoringRequestParser parser, Vectorizer vectorizer, VectorIndex vectorIndex) {
        this.parser = parser;
        this.vectorizer = vectorizer;
        this.vectorIndex = vectorIndex;
    }

    /**
     * Contest hot path: valid {@code POST /fraud-score} payloads only. Avoids wrapping the parse/vector/search chain
     * in a broad exception region so C2 can optimize the straight-line p99 path as aggressively as possible.
     */
    public int countFraudNeighborsFast(byte[] body, int offset, int length) {
        parser.parseInto(body, offset, length, scratch);
        if (StageTimer.ENABLED) StageTimer.mark(0, System.nanoTime());

        vectorizer.vectorizeInto(body, scratch, queryVector);
        if (StageTimer.ENABLED) StageTimer.mark(1, System.nanoTime());

        int fraudCount = vectorIndex.countFraudsInTop5(queryVector);
        if (StageTimer.ENABLED) {
            StageTimer.mark(2, System.nanoTime());
            StageTimer.recordVisits(KdTreeProbes.lastQueryNodesVisited());
        }
        return fraudCount;
    }

    public int countFraudNeighbors(byte[] body, int offset, int length) {
        try {
            return countFraudNeighborsFast(body, offset, length);
        } catch (Throwable error) {
            if (LOG_ERRORS) error.printStackTrace();
            return FAIL_SAFE_FRAUD_COUNT;
        }
    }
}
