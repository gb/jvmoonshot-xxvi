package com.github.gb.moonshot.bench;

/**
 * Shared argv parsing + timing helpers for the {@code com.github.gb.moonshot.bench} and
 * {@code com.github.gb.moonshot.search} CLI tools. All callers use the same conventions:
 *
 * <ul>
 *   <li>Flag with value: {@code --name value} (consumes two argv slots).
 *   <li>Boolean flag: {@code --name} (no value).
 *   <li>Missing flag → returns the supplied default.
 * </ul>
 *
 * Extracted to keep individual CLI mains free of duplicated argv-parsing
 * boilerplate; previously the same five helpers were copy-pasted across
 * eight tools.
 */
public final class CliArgs {

    private CliArgs() {}

    /** Returns the argument that follows {@code name}, or {@code defaultValue} if absent. */
    public static String string(String[] args, String name, String defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) return args[i + 1];
        }
        return defaultValue;
    }

    /**
     * Returns {@code Integer.parseInt} of the value following {@code name}, or
     * {@code defaultValue} when the flag is absent. Throws
     * {@link IllegalArgumentException} (with the offending flag name) when the
     * value is present but not a valid integer.
     */
    public static int intVal(String[] args, String name, int defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                try {
                    return Integer.parseInt(args[i + 1]);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "invalid integer for " + name + ": " + args[i + 1], e);
                }
            }
        }
        return defaultValue;
    }

    /** Same contract as {@link #intVal}, for {@code long}. */
    public static long longVal(String[] args, String name, long defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                try {
                    return Long.parseLong(args[i + 1]);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "invalid long for " + name + ": " + args[i + 1], e);
                }
            }
        }
        return defaultValue;
    }

    /** Same contract as {@link #intVal}, for {@code float}. */
    public static float floatVal(String[] args, String name, float defaultValue) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                try {
                    return Float.parseFloat(args[i + 1]);
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(
                        "invalid float for " + name + ": " + args[i + 1], e);
                }
            }
        }
        return defaultValue;
    }

    /** Returns {@code true} if the boolean flag {@code name} is present in {@code args}. */
    public static boolean flag(String[] args, String name) {
        for (String s : args) {
            if (s.equals(name)) return true;
        }
        return false;
    }

    /** Milliseconds elapsed since the supplied {@code System.nanoTime()} stamp. */
    public static long elapsedMs(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000;
    }
}
