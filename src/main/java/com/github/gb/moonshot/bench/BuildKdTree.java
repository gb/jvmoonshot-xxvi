package com.github.gb.moonshot.bench;

import com.github.gb.moonshot.Dataset;
import com.github.gb.moonshot.search.KdTree;
import com.github.gb.moonshot.search.KdTreeBuilder;
import com.github.gb.moonshot.search.KdTreeIO;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Offline build: load dataset, construct i16-quantized KD-tree, save to disk.
 *
 *   java -cp target/jvmoonshot-xxvi.jar com.github.gb.moonshot.bench.BuildKdTree \
 *        [--in  path/to/references.json.gz] \
 *        [--out path/to/kdtree.bin]
 */
public final class BuildKdTree {

    private static final String LOG_PREFIX = "[kdtree-i16]";

    public static void main(String[] args) throws IOException {
        String referencesPath = CliArgs.string(args, "--in", System.getenv().getOrDefault(
            "REFERENCES_PATH",
            "../rinha-de-backend-2026/resources/references.json.gz"));
        String outPathArg = CliArgs.string(args, "--out", "data/kdtree.bin");
        int n = CliArgs.intVal(args, "--n", 0);

        log("loading dataset from " + referencesPath + (n > 0 ? " (first " + n + ")" : ""));
        long t0 = System.nanoTime();
        Dataset dataset = (n > 0) ? Dataset.loadFirst(Path.of(referencesPath), n)
                                  : Dataset.load(Path.of(referencesPath));
        log("loaded " + dataset.size() + " vectors in " + CliArgs.elapsedMs(t0) + " ms");

        log("building KdTree (scale=" + KdTree.SCALE + ", sliding-midpoint)");
        t0 = System.nanoTime();
        KdTree tree = KdTreeBuilder.build(dataset);
        log("built " + tree.size() + " nodes in " + CliArgs.elapsedMs(t0) + " ms");

        Path outPath = Path.of(outPathArg);
        Files.createDirectories(outPath.getParent() == null ? Path.of(".") : outPath.getParent());
        log("saving to " + outPath);
        t0 = System.nanoTime();
        KdTreeIO.save(tree, outPath);
        log("saved in " + CliArgs.elapsedMs(t0) + " ms");
    }

    static void ensureParentDirectory(Path path) throws IOException {
        Path parent = path.getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    private static void log(String msg) { System.out.println(LOG_PREFIX + " " + msg); }
    private BuildKdTree() {}
}
