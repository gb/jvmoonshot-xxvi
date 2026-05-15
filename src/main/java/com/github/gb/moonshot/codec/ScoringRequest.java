package com.github.gb.moonshot.codec;

/** Flat record of all fields needed from a /fraud-score request. {@code lastTimestamp == null} encodes "no last_transaction". */
public record ScoringRequest(
    String id,
    double amount,
    int installments,
    String requestedAt,
    double customerAvgAmount,
    int customerTxCount24h,
    String[] knownMerchants,
    String merchantId,
    String mcc,
    double merchantAvgAmount,
    boolean isOnline,
    boolean cardPresent,
    double kmFromHome,
    String lastTimestamp,
    double lastKmFromCurrent
) {}
