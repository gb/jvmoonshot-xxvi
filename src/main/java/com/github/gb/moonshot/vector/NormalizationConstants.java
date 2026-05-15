package com.github.gb.moonshot.vector;

/** From resources/normalization.json. Constants don't change between runs. */
public final class NormalizationConstants {

    public static final double MAX_AMOUNT = 10_000;
    public static final double MAX_INSTALLMENTS = 12;
    public static final double AMOUNT_VS_AVG_RATIO = 10;
    public static final double MAX_MINUTES = 1_440;
    public static final double MAX_KM = 1_000;
    public static final double MAX_TX_COUNT_24H = 20;
    public static final double MAX_MERCHANT_AVG_AMOUNT = 10_000;

    // Reciprocals
    public static final double INV_MAX_AMOUNT = 1.0 / MAX_AMOUNT;
    public static final double INV_MAX_INSTALLMENTS = 1.0 / MAX_INSTALLMENTS;
    public static final double INV_AMOUNT_VS_AVG_RATIO = 1.0 / AMOUNT_VS_AVG_RATIO;
    public static final double INV_MAX_MINUTES = 1.0 / MAX_MINUTES;
    public static final double INV_MAX_KM = 1.0 / MAX_KM;
    public static final double INV_MAX_TX_COUNT_24H = 1.0 / MAX_TX_COUNT_24H;
    public static final double INV_MAX_MERCHANT_AVG_AMOUNT = 1.0 / MAX_MERCHANT_AVG_AMOUNT;

    private NormalizationConstants() {}
}
