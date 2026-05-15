package com.github.gb.moonshot.vector;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;

/**
 * MCC to risk weight. Hardcoded from {@code resources/mcc_risk.json}; default {@value #DEFAULT_RISK} for unlisted MCCs.
 * Hot path parses 4 ASCII digits directly into a {@code float[10000]} LUT no String, no Map probe, no autoboxing.
 */
public final class MccRiskTable {

    public static final double DEFAULT_RISK = 0.5;

    private static final Map<String, Double> RISK_BY_MCC = Map.ofEntries(
        Map.entry("5411", 0.15),
        Map.entry("5812", 0.30),
        Map.entry("5912", 0.20),
        Map.entry("5944", 0.45),
        Map.entry("7801", 0.80),
        Map.entry("7802", 0.75),
        Map.entry("7995", 0.85),
        Map.entry("4511", 0.35),
        Map.entry("5311", 0.25),
        Map.entry("5999", 0.50)
    );

    /** All possible 4-digit MCCs (0000-9999). Unlisted entries hold {@link #DEFAULT_RISK}. */
    private static final float[] RISK_LUT;
    static {
        float[] lut = new float[10000];
        Arrays.fill(lut, (float) DEFAULT_RISK);
        for (Map.Entry<String, Double> e : RISK_BY_MCC.entrySet()) {
            int idx = Integer.parseInt(e.getKey());
            lut[idx] = e.getValue().floatValue();
        }
        RISK_LUT = lut;
    }

    private MccRiskTable() {}

    public static double riskFor(String mcc) {
        return Objects.requireNonNullElse(RISK_BY_MCC.get(mcc), DEFAULT_RISK);
    }

    /** Caller guarantees {@code body[offset..offset+4]} is 4 ASCII digits (contest schema enforces this on merchant.mcc). */
    public static float riskFor(byte[] body, int offset) {
        int idx = (body[offset]     - '0') * 1000
                + (body[offset + 1] - '0') * 100
                + (body[offset + 2] - '0') * 10
                + (body[offset + 3] - '0');
        return RISK_LUT[idx];
    }
}
