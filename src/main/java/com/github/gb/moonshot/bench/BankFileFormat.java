package com.github.gb.moonshot.bench;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;

/**
 * Low-level helpers shared by {@link BankIO} and {@link PayloadBankIO} —
 * draining a buffer to a channel, filling a buffer fully from a channel,
 * and validating an ASCII magic header. Hides the {@code while
 * (hasRemaining)} loop that both file formats need but neither should
 * own.
 */
final class BankFileFormat {

    private BankFileFormat() {}

    /** Writes every remaining byte of {@code buf} to {@code ch}. */
    static void drain(FileChannel ch, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) ch.write(buf);
    }

    /**
     * Reads from {@code ch} into {@code buf} until the buffer has no
     * remaining capacity. Throws on EOF.
     */
    static void fillFully(FileChannel ch, ByteBuffer buf, String what) throws IOException {
        while (buf.hasRemaining()) {
            if (ch.read(buf) < 0) throw new IOException("truncated " + what);
        }
    }

    /**
     * Reads {@code magicBytes} bytes from the front of {@code header} and
     * verifies they match {@code expectedMagic}. Throws an
     * {@link IOException} with a hint pointing at {@code rebuildHint} (e.g.
     * "BuildQueryBank") on mismatch.
     */
    static void requireMagic(ByteBuffer header, String expectedMagic, int magicBytes, String rebuildHint)
            throws IOException {
        byte[] magicBuf = new byte[magicBytes];
        header.get(magicBuf);
        String magic = new String(magicBuf, StandardCharsets.US_ASCII);
        if (!expectedMagic.equals(magic)) {
            throw new IOException("bad magic: " + magic + " (expected " + expectedMagic
                + "). Rebuild via " + rebuildHint + ".");
        }
    }
}
