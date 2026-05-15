package com.github.gb.moonshot.bench;

import com.github.gb.moonshot.Dataset;
import com.github.gb.moonshot.search.KdTree;
import com.github.gb.moonshot.search.KdTreeBuilder;
import com.github.gb.moonshot.search.KdTreeIO;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Offline build: load int8 dataset from gzipped JSON references, construct
 * balanced KD-tree, save to disk. One-shot tool, ~10-30 sec total.
 *
 *   java -cp target/jvmoonshot-xxvi.jar com.github.gb.moonshot.bench.BuildKdTree \
 *        [--in C:/.../references.json.gz] \
 *        [--out C:/.../kdtree.bin]
 */
public final class BuildKdTree {

    private static final String LOG_PREFIX = "[kdtree]";

    public static void main(String[] args) throws IOException {
        String referencesPath = CliArgs.string(args, "--in", System.getenv().getOrDefault(
            "REFERENCES_PATH",
            "../rinha-de-backend-2026/resources/references.json.gz"));
        String outPathArg = CliArgs.string(args, "--out", "data/kdtree.bin");
        int    n           = CliArgs.intVal(args, "--n", 0);  // 0 = full dataset

        Dataset dataset = loadDataset(referencesPath, n);
        KdTree tree = buildTree(dataset);
        saveTree(tree, Path.of(outPathArg));
    }

    private static Dataset loadDataset(String path, int n) throws IOException {
        log("loading float dataset from " + path + (n > 0 ? " (first " + n + ")" : ""));
        long startNanos = System.nanoTime();
        Dataset dataset = (n > 0) ? Dataset.loadFirst(Path.of(path), n) : Dataset.load(Path.of(path));
        log("loaded " + dataset.size() + " float vectors in " + CliArgs.elapsedMs(startNanos) + " ms");
        return dataset;
    }

    private static KdTree buildTree(Dataset dataset) {
        log("building KD-tree (sliding-midpoint split, 33% imbalance cap)");
        long startNanos = System.nanoTime();
        KdTree tree = KdTreeBuilder.build(dataset);
        log("built " + tree.size() + " nodes in " + CliArgs.elapsedMs(startNanos) + " ms");
        return tree;
    }

    private static void saveTree(KdTree tree, Path outPath) throws IOException {
        log("saving to " + outPath);
        long startNanos = System.nanoTime();
        ensureParentDirectory(outPath);
        KdTreeIO.save(tree, outPath);
        log("saved in " + CliArgs.elapsedMs(startNanos) + " ms");
    }

    static void ensureParentDirectory(Path outPath) throws IOException {
        Path parent = outPath.getParent();
        if (parent != null) Files.createDirectories(parent);
    }

    private static void log(String message) {
        System.out.println(LOG_PREFIX + " " + message);
    }

    private BuildKdTree() {}
}
