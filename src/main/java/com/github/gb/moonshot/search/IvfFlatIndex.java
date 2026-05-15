package com.github.gb.moonshot.search;

import com.github.gb.moonshot.Dataset;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Random;

/**
 * IVF-Flat ANN: approximate at cell-selection (nprobe of nlist), exact SIMD distance within
 * chosen cells. CELL-MAJOR vector layout is load-bearing: the per-query sweep is one streaming
 * pass per cell so the HW prefetcher saturates and pages survive cgroup pressure. An indirect
 * layout (cellIds[j] -> vectors[pid]) regresses badly under mmap.
 */
public final class IvfFlatIndex implements VectorIndex {

    public static final String MAGIC = "RIVF0003";
    public static final int DIMS = Dataset.DIMS;
    public static final int STRIDE = Dataset.STRIDE;

    private static final int MAX_K = 8;
    private static final int HEADER_BYTES = 8 + 4 * 4;

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;
    private static final ValueLayout.OfInt INT_LE =
        ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfByte BYTE = ValueLayout.JAVA_BYTE;

    private final int nlist;
    private final int n;
    private final int nprobe;

    private final float[] centroids;
    private final int[]   cellOffsets;
    /**
     * Per-cell radius. Enables the triangle-inequality cell prune:
     * (sqrt(d(q,c)) - r(c))^2 > topCutoff skips the whole cell. Lossless.
     */
    private final float[] cellRadius;

    private final MappedByteBuffer vectorsBuf;
    private final MemorySegment    vectorsSeg;
    private final MappedByteBuffer fraudBuf;
    private final MemorySegment    fraudSeg;
    private final MappedByteBuffer cellIdsBuf;
    private final MemorySegment    cellIdsSeg;

    private final long bytesMmaped;

    private final float[] probeDistBuf;
    private final int[]   probeCellBuf;
    private final float[] topDistBuf = new float[MAX_K];
    private final int[]   topPosBuf  = new int[MAX_K];

    private IvfFlatIndex(int nprobe, int nlist, int n,
                         float[] centroids, int[] cellOffsets, float[] cellRadius,
                         MappedByteBuffer vectorsBuf,
                         MappedByteBuffer fraudBuf,
                         MappedByteBuffer cellIdsBuf,
                         long bytesMmaped) {
        if (STRIDE != 16) {
            throw new IllegalStateException("IvfFlatIndex assumes STRIDE=16, got " + STRIDE);
        }
        if (SPECIES.length() != 8) {
            throw new IllegalStateException("IvfFlatIndex assumes 8-lane SPECIES_256");
        }
        if (nprobe < 1 || nprobe > nlist) {
            throw new IllegalArgumentException("nprobe must be in [1, nlist=" + nlist + "], got " + nprobe);
        }
        this.nprobe = nprobe;
        this.nlist = nlist;
        this.n = n;
        this.centroids = centroids;
        this.cellOffsets = cellOffsets;
        this.cellRadius = cellRadius;
        this.vectorsBuf = vectorsBuf;
        this.vectorsSeg = MemorySegment.ofBuffer(vectorsBuf);
        this.fraudBuf = fraudBuf;
        this.fraudSeg = MemorySegment.ofBuffer(fraudBuf);
        this.cellIdsBuf = cellIdsBuf;
        this.cellIdsSeg = MemorySegment.ofBuffer(cellIdsBuf);
        this.bytesMmaped = bytesMmaped;
        this.probeDistBuf = new float[nprobe];
        this.probeCellBuf = new int[nprobe];
    }

    @Override
    public int size() { return n; }

    public int nlist()  { return nlist; }
    public int nprobe() { return nprobe; }
    public long bytesMmaped() { return bytesMmaped; }

    public int[] cellSizes() {
        int[] sizes = new int[nlist];
        for (int c = 0; c < nlist; c++) sizes[c] = cellOffsets[c + 1] - cellOffsets[c];
        return sizes;
    }

    @Override
    public int countFraudsInTopK(float[] query, int k) {
        if (k > MAX_K) throw new IllegalArgumentException("k=" + k + " exceeds MAX_K=" + MAX_K);

        final FloatVector q0 = FloatVector.fromArray(SPECIES, query, 0);
        final FloatVector q1 = FloatVector.fromArray(SPECIES, query, 8);

        final int np = this.nprobe;
        final float[] probeDist = this.probeDistBuf;
        final int[]   probeCell = this.probeCellBuf;
        Arrays.fill(probeDist, Float.POSITIVE_INFINITY);
        Arrays.fill(probeCell, -1);
        float probeCutoff = Float.POSITIVE_INFINITY;

        final float[] cs = this.centroids;
        final int nl = this.nlist;
        for (int c = 0; c < nl; c++) {
            int coff = c << 4;
            FloatVector v0 = FloatVector.fromArray(SPECIES, cs, coff);
            FloatVector v1 = FloatVector.fromArray(SPECIES, cs, coff + 8);
            FloatVector d0 = v0.sub(q0);
            FloatVector d1 = v1.sub(q1);
            float dist = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
            if (dist < probeCutoff) {
                int pos = np - 1;
                while (pos > 0 && probeDist[pos - 1] > dist) {
                    probeDist[pos] = probeDist[pos - 1];
                    probeCell[pos] = probeCell[pos - 1];
                    pos--;
                }
                probeDist[pos] = dist;
                probeCell[pos] = c;
                probeCutoff = probeDist[np - 1];
            }
        }

        // Closest cell first so topCutoff tightens fastest, enabling triangle-inequality
        // skips on later cells.
        final float[] topDist = this.topDistBuf;
        final int[]   topPos  = this.topPosBuf;
        for (int i = 0; i < k; i++) { topDist[i] = Float.POSITIVE_INFINITY; topPos[i] = -1; }
        float topCutoff = Float.POSITIVE_INFINITY;

        final int[] off = this.cellOffsets;
        final float[] cellR = this.cellRadius;
        final MemorySegment vecS = this.vectorsSeg;
        final int kMinus1 = k - 1;
        final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

        for (int p = 0; p < np; p++) {
            int c = probeCell[p];
            float centroidD2 = probeDist[p];
            float diff = (float) Math.sqrt(centroidD2) - cellR[c];
            if (diff > 0f && diff * diff >= topCutoff) continue;

            int lo = off[c];
            int hi = off[c + 1];
            for (int j = lo; j < hi; j++) {
                long vbytes = (long) j * STRIDE * 4L;
                FloatVector v0 = FloatVector.fromMemorySegment(SPECIES, vecS, vbytes, LE);
                FloatVector v1 = FloatVector.fromMemorySegment(SPECIES, vecS, vbytes + 32L, LE);
                FloatVector d0 = v0.sub(q0);
                FloatVector d1 = v1.sub(q1);
                float dist = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
                if (dist < topCutoff) {
                    int pos = kMinus1;
                    while (pos > 0 && topDist[pos - 1] > dist) {
                        topDist[pos] = topDist[pos - 1];
                        topPos[pos] = topPos[pos - 1];
                        pos--;
                    }
                    topDist[pos] = dist;
                    topPos[pos] = j;
                    topCutoff = topDist[kMinus1];
                }
            }
        }

        int frauds = 0;
        final MemorySegment frS = this.fraudSeg;
        for (int i = 0; i < k; i++) {
            int pos = topPos[i];
            if (pos >= 0 && frS.get(BYTE, (long) pos) != 0) frauds++;
        }
        return frauds;
    }

    /**
     * Profile-mode mirror of countFraudsInTopK. Diagnostic only never on the request hot path.
     * out[0]=centroid-scan ns, out[1]=candidate-scan ns, out[2]=fraud-lookup ns,
     * out[3]=cells visited, out[4]=candidates scanned, out[5]=sum of nprobe cell sizes.
     */
    public int countFraudsInTopKTimed(float[] query, int k, long[] out) {
        if (k > MAX_K) throw new IllegalArgumentException("k=" + k + " exceeds MAX_K=" + MAX_K);

        long t0 = System.nanoTime();

        final FloatVector q0 = FloatVector.fromArray(SPECIES, query, 0);
        final FloatVector q1 = FloatVector.fromArray(SPECIES, query, 8);

        final int np = this.nprobe;
        final float[] probeDist = this.probeDistBuf;
        final int[]   probeCell = this.probeCellBuf;
        Arrays.fill(probeDist, Float.POSITIVE_INFINITY);
        Arrays.fill(probeCell, -1);
        float probeCutoff = Float.POSITIVE_INFINITY;

        final float[] cs = this.centroids;
        final int nl = this.nlist;
        for (int c = 0; c < nl; c++) {
            int coff = c << 4;
            FloatVector v0 = FloatVector.fromArray(SPECIES, cs, coff);
            FloatVector v1 = FloatVector.fromArray(SPECIES, cs, coff + 8);
            FloatVector d0 = v0.sub(q0);
            FloatVector d1 = v1.sub(q1);
            float dist = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
            if (dist < probeCutoff) {
                int pos = np - 1;
                while (pos > 0 && probeDist[pos - 1] > dist) {
                    probeDist[pos] = probeDist[pos - 1];
                    probeCell[pos] = probeCell[pos - 1];
                    pos--;
                }
                probeDist[pos] = dist;
                probeCell[pos] = c;
                probeCutoff = probeDist[np - 1];
            }
        }

        long t1 = System.nanoTime();

        final float[] topDist = this.topDistBuf;
        final int[]   topPos  = this.topPosBuf;
        for (int i = 0; i < k; i++) { topDist[i] = Float.POSITIVE_INFINITY; topPos[i] = -1; }
        float topCutoff = Float.POSITIVE_INFINITY;

        final int[] off = this.cellOffsets;
        final MemorySegment vecS = this.vectorsSeg;
        final int kMinus1 = k - 1;
        final ByteOrder LE = ByteOrder.LITTLE_ENDIAN;

        long candidatesScanned = 0;
        long totalProbeCellSize = 0;
        int cellsVisited = 0;
        final float[] cellR = this.cellRadius;
        for (int p = 0; p < np; p++) {
            int c = probeCell[p];
            int lo = off[c];
            int hi = off[c + 1];
            totalProbeCellSize += (hi - lo);
            float centroidD2 = probeDist[p];
            float diffR = (float) Math.sqrt(centroidD2) - cellR[c];
            if (diffR > 0f && diffR * diffR >= topCutoff) continue; // mirror hot-path prune
            cellsVisited++;
            for (int j = lo; j < hi; j++) {
                long vbytes = (long) j * STRIDE * 4L;
                FloatVector v0 = FloatVector.fromMemorySegment(SPECIES, vecS, vbytes, LE);
                FloatVector v1 = FloatVector.fromMemorySegment(SPECIES, vecS, vbytes + 32L, LE);
                FloatVector d0 = v0.sub(q0);
                FloatVector d1 = v1.sub(q1);
                float dist = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
                candidatesScanned++;
                if (dist < topCutoff) {
                    int pos = kMinus1;
                    while (pos > 0 && topDist[pos - 1] > dist) {
                        topDist[pos] = topDist[pos - 1];
                        topPos[pos] = topPos[pos - 1];
                        pos--;
                    }
                    topDist[pos] = dist;
                    topPos[pos] = j;
                    topCutoff = topDist[kMinus1];
                }
            }
        }

        long t2 = System.nanoTime();

        int frauds = 0;
        final MemorySegment frS = this.fraudSeg;
        for (int i = 0; i < k; i++) {
            int pos = topPos[i];
            if (pos >= 0 && frS.get(BYTE, (long) pos) != 0) frauds++;
        }

        long t3 = System.nanoTime();

        out[0] = t1 - t0;
        out[1] = t2 - t1;
        out[2] = t3 - t2;
        out[3] = cellsVisited;
        out[4] = candidatesScanned;
        out[5] = totalProbeCellSize;
        return frauds;
    }

    /** Diagnostic / recall path touches the cold cellIds mmap region. NOT for the hot path. */
    public int[] topK(float[] query, int k) {
        if (k > MAX_K) throw new IllegalArgumentException("k=" + k + " exceeds MAX_K=" + MAX_K);
        countFraudsInTopK(query, k); // populates topPosBuf
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            int pos = topPosBuf[i];
            out[i] = (pos >= 0) ? cellIdsSeg.get(INT_LE, (long) pos * 4L) : -1;
        }
        return out;
    }

    @Override
    public void prewarm() {
        // cellIds intentionally NOT prewarmed diagnostic-only, pages should stay cold.
        int sink = 0;
        sink ^= touchPages(vectorsBuf);
        sink ^= touchPages(fraudBuf);
        prewarmSink = sink; // defeat DCE
    }

    @SuppressWarnings("unused")
    private static volatile int prewarmSink;

    private static int touchPages(MappedByteBuffer buf) {
        int len = buf.limit();
        int sink = 0;
        for (int i = 0; i < len; i += 4096) sink ^= buf.get(i);
        return sink;
    }

    public static BuildReport buildAndSave(Dataset dataset, Path outPath, int nlist,
                                           int trainerSize, int iters, long seed) throws IOException {
        final int n = dataset.size();
        final float[] vectors = dataset.vectors();
        final boolean[] fraudLabels = dataset.fraudLabels();

        if (nlist <= 0 || nlist > n) {
            throw new IllegalArgumentException("nlist=" + nlist + " out of (0, n=" + n + "]");
        }
        if (trainerSize > n) trainerSize = n;
        if (trainerSize < nlist) trainerSize = Math.min(n, nlist * 8);

        Random rng = new Random(seed);
        long buildStart = System.nanoTime();

        int initSize = Math.min(n, Math.max(nlist * 8, 50_000));
        int[] initSubsample = sampleIndices(initSize, n, rng);
        float[] centroids = kmeansPlusPlusInit(vectors, initSubsample, nlist, rng);

        int[] trainSubsample = sampleIndices(trainerSize, n, rng);
        float[] sums   = new float[nlist * STRIDE];
        int[]   counts = new int[nlist];
        for (int iter = 0; iter < iters; iter++) {
            Arrays.fill(sums, 0f);
            Arrays.fill(counts, 0);
            for (int idx : trainSubsample) {
                int voff = idx * STRIDE;
                int c = nearestCentroidSimd(vectors, voff, centroids, nlist);
                int coff = c * STRIDE;
                for (int d = 0; d < DIMS; d++) sums[coff + d] += vectors[voff + d];
                counts[c]++;
            }
            for (int c = 0; c < nlist; c++) {
                int coff = c * STRIDE;
                if (counts[c] > 0) {
                    float inv = 1f / counts[c];
                    for (int d = 0; d < DIMS; d++) centroids[coff + d] = sums[coff + d] * inv;
                } else {
                    int idx = trainSubsample[rng.nextInt(trainSubsample.length)];
                    int voff = idx * STRIDE;
                    for (int d = 0; d < DIMS; d++) centroids[coff + d] = vectors[voff + d];
                }
            }
        }

        int[] assignment = new int[n];
        int[] cellCounts = new int[nlist];
        for (int i = 0; i < n; i++) {
            int c = nearestCentroidSimd(vectors, i * STRIDE, centroids, nlist);
            assignment[i] = c;
            cellCounts[c]++;
        }

        int[] cellOffsets = new int[nlist + 1];
        for (int c = 0; c < nlist; c++) cellOffsets[c + 1] = cellOffsets[c] + cellCounts[c];
        int[] cursors = cellOffsets.clone();

        float[] cellVectors = new float[n * STRIDE];
        byte[]  cellFraud   = new byte[n];
        int[]   cellIds     = new int[n];
        for (int i = 0; i < n; i++) {
            int c = assignment[i];
            int dstSlot = cursors[c]++;
            int dstOff = dstSlot * STRIDE;
            int srcOff = i * STRIDE;
            System.arraycopy(vectors, srcOff, cellVectors, dstOff, STRIDE);
            cellFraud[dstSlot] = fraudLabels[i] ? (byte) 1 : (byte) 0;
            cellIds[dstSlot]   = i;
        }

        int empty = 0, maxCell = 0;
        int[] sizesSorted = cellCounts.clone();
        Arrays.sort(sizesSorted);
        for (int s : cellCounts) {
            if (s == 0) empty++;
            if (s > maxCell) maxCell = s;
        }
        BuildReport report = new BuildReport();
        report.n = n;
        report.nlist = nlist;
        report.iters = iters;
        report.trainerSize = trainerSize;
        report.emptyCells = empty;
        report.minCellSize = sizesSorted[0];
        report.p50CellSize = sizesSorted[(int) (nlist * 0.50)];
        report.p99CellSize = sizesSorted[Math.min(nlist - 1, (int) (nlist * 0.99))];
        report.maxCellSize = maxCell;
        report.meanCellSize = (double) n / nlist;
        report.buildNanos = System.nanoTime() - buildStart;

        // Write order must match load() read order.
        try (FileChannel ch = FileChannel.open(outPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.WRITE,
                StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            header.put(MAGIC.getBytes(StandardCharsets.US_ASCII));
            header.putInt(nlist);
            header.putInt(n);
            header.putInt(DIMS);
            header.putInt(STRIDE);
            header.flip();
            drainFully(ch, header);

            writeFloats(ch, centroids, centroids.length);
            writeInts(ch, cellOffsets, cellOffsets.length);
            writeFloats(ch, cellVectors, n * STRIDE);
            writeBytes(ch, cellFraud);
            writeInts(ch, cellIds, n);
        }

        return report;
    }

    public static final class BuildReport {
        public int n;
        public int nlist;
        public int iters;
        public int trainerSize;
        public int emptyCells;
        public int minCellSize;
        public int p50CellSize;
        public int p99CellSize;
        public int maxCellSize;
        public double meanCellSize;
        public long buildNanos;

        @Override public String toString() {
            return String.format(
                "BuildReport{n=%d nlist=%d iters=%d train=%d empty=%d "
                + "cellSize min=%d p50=%d p99=%d max=%d mean=%.1f buildMs=%d}",
                n, nlist, iters, trainerSize, emptyCells,
                minCellSize, p50CellSize, p99CellSize, maxCellSize, meanCellSize,
                buildNanos / 1_000_000);
        }
    }

    private static int nearestCentroidSimd(float[] vectors, int voff, float[] centroids, int nlist) {
        FloatVector q0 = FloatVector.fromArray(SPECIES, vectors, voff);
        FloatVector q1 = FloatVector.fromArray(SPECIES, vectors, voff + 8);
        int best = 0;
        float bestDist = Float.POSITIVE_INFINITY;
        for (int c = 0; c < nlist; c++) {
            int coff = c << 4;
            FloatVector v0 = FloatVector.fromArray(SPECIES, centroids, coff);
            FloatVector v1 = FloatVector.fromArray(SPECIES, centroids, coff + 8);
            FloatVector d0 = v0.sub(q0);
            FloatVector d1 = v1.sub(q1);
            float dist = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
            if (dist < bestDist) {
                bestDist = dist;
                best = c;
            }
        }
        return best;
    }

    private static float[] kmeansPlusPlusInit(float[] vectors, int[] subsample, int nlist, Random rng) {
        float[] centroids = new float[nlist * STRIDE];
        int m = subsample.length;
        float[] minDist = new float[m];
        Arrays.fill(minDist, Float.POSITIVE_INFINITY);

        int first = subsample[rng.nextInt(m)];
        System.arraycopy(vectors, first * STRIDE, centroids, 0, DIMS);

        for (int chosen = 1; chosen < nlist; chosen++) {
            int prev = chosen - 1;
            int pcoff = prev * STRIDE;
            double sumDist = 0;
            for (int s = 0; s < m; s++) {
                int voff = subsample[s] * STRIDE;
                float d2 = 0f;
                for (int d = 0; d < DIMS; d++) {
                    float diff = vectors[voff + d] - centroids[pcoff + d];
                    d2 += diff * diff;
                }
                if (d2 < minDist[s]) minDist[s] = d2;
                sumDist += minDist[s];
            }
            double target = rng.nextDouble() * sumDist;
            double acc = 0;
            int pick = subsample[m - 1];
            for (int s = 0; s < m; s++) {
                acc += minDist[s];
                if (acc >= target) { pick = subsample[s]; break; }
            }
            int dst = chosen * STRIDE;
            System.arraycopy(vectors, pick * STRIDE, centroids, dst, DIMS);
        }
        return centroids;
    }

    private static int[] sampleIndices(int count, int n, Random rng) {
        if (count >= n) {
            int[] all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            shuffle(all, rng);
            return all;
        }
        java.util.HashSet<Integer> picked = new java.util.HashSet<>(count * 2);
        for (int i = n - count; i < n; i++) {
            int t = rng.nextInt(i + 1);
            if (!picked.add(t)) picked.add(i);
        }
        int[] out = new int[picked.size()];
        int j = 0;
        for (int v : picked) out[j++] = v;
        return out;
    }

    private static void shuffle(int[] arr, Random rng) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
        }
    }

    public static IvfFlatIndex load(Path path, int nprobe) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            fillFully(ch, header, "header");
            header.flip();

            byte[] magic = new byte[8];
            header.get(magic);
            String got = new String(magic, StandardCharsets.US_ASCII);
            if (!MAGIC.equals(got)) throw new IOException("bad magic: expected " + MAGIC + " got " + got);

            int nlist  = header.getInt();
            int n      = header.getInt();
            int dims   = header.getInt();
            int stride = header.getInt();
            if (dims != DIMS || stride != STRIDE) {
                throw new IOException("dim/stride mismatch: file dims=" + dims + " stride=" + stride
                    + " (expected " + DIMS + "/" + STRIDE + ")");
            }

            float[] centroids = new float[nlist * STRIDE];
            readFloats(ch, centroids, centroids.length);

            int[] cellOffsets = new int[nlist + 1];
            readInts(ch, cellOffsets, cellOffsets.length);
            if (cellOffsets[nlist] != n) {
                throw new IOException("cellOffsets[nlist]=" + cellOffsets[nlist] + " != n=" + n);
            }

            long vecOff = ch.position();
            long vecLen = (long) n * STRIDE * 4L;
            MappedByteBuffer vectorsBuf = ch.map(FileChannel.MapMode.READ_ONLY, vecOff, vecLen);
            vectorsBuf.order(ByteOrder.LITTLE_ENDIAN);
            ch.position(vecOff + vecLen);

            long fraudOff = ch.position();
            long fraudLen = (long) n;
            MappedByteBuffer fraudBuf = ch.map(FileChannel.MapMode.READ_ONLY, fraudOff, fraudLen);
            ch.position(fraudOff + fraudLen);

            long idsOff = ch.position();
            long idsLen = (long) n * 4L;
            MappedByteBuffer cellIdsBuf = ch.map(FileChannel.MapMode.READ_ONLY, idsOff, idsLen);
            cellIdsBuf.order(ByteOrder.LITTLE_ENDIAN);
            ch.position(idsOff + idsLen);

            // One streaming SIMD pass to compute max-radius per cell; nlist floats on heap.
            MemorySegment vSeg = MemorySegment.ofBuffer(vectorsBuf);
            float[] cellRadius = new float[nlist];
            for (int c = 0; c < nlist; c++) {
                int lo = cellOffsets[c];
                int hi = cellOffsets[c + 1];
                int coff = c << 4;
                FloatVector c0 = FloatVector.fromArray(SPECIES, centroids, coff);
                FloatVector c1 = FloatVector.fromArray(SPECIES, centroids, coff + 8);
                float maxD2 = 0f;
                for (int j = lo; j < hi; j++) {
                    long vbytes = (long) j * STRIDE * 4L;
                    FloatVector v0 = FloatVector.fromMemorySegment(SPECIES, vSeg, vbytes, ByteOrder.LITTLE_ENDIAN);
                    FloatVector v1 = FloatVector.fromMemorySegment(SPECIES, vSeg, vbytes + 32L, ByteOrder.LITTLE_ENDIAN);
                    FloatVector d0 = v0.sub(c0);
                    FloatVector d1 = v1.sub(c1);
                    float d2 = d0.mul(d0).add(d1.mul(d1)).reduceLanes(VectorOperators.ADD);
                    if (d2 > maxD2) maxD2 = d2;
                }
                cellRadius[c] = (float) Math.sqrt(maxD2);
            }

            long bytesMmaped = vecLen + fraudLen + idsLen;
            return new IvfFlatIndex(nprobe, nlist, n, centroids, cellOffsets, cellRadius,
                vectorsBuf, fraudBuf, cellIdsBuf, bytesMmaped);
        }
    }

    private static final int IO_CHUNK_FLOATS = 64 * 1024;
    private static final int IO_CHUNK_INTS   = 64 * 1024;
    private static final int IO_CHUNK_BYTES  = 256 * 1024;

    private static void writeFloats(FileChannel ch, float[] data, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(IO_CHUNK_FLOATS * 4).order(ByteOrder.LITTLE_ENDIAN);
        int written = 0;
        while (written < count) {
            int chunk = Math.min(IO_CHUNK_FLOATS, count - written);
            buf.clear();
            for (int i = 0; i < chunk; i++) buf.putFloat(data[written + i]);
            buf.flip();
            drainFully(ch, buf);
            written += chunk;
        }
    }

    private static void writeInts(FileChannel ch, int[] data, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(IO_CHUNK_INTS * 4).order(ByteOrder.LITTLE_ENDIAN);
        int written = 0;
        while (written < count) {
            int chunk = Math.min(IO_CHUNK_INTS, count - written);
            buf.clear();
            for (int i = 0; i < chunk; i++) buf.putInt(data[written + i]);
            buf.flip();
            drainFully(ch, buf);
            written += chunk;
        }
    }

    private static void writeBytes(FileChannel ch, byte[] data) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(IO_CHUNK_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        int written = 0;
        while (written < data.length) {
            int chunk = Math.min(IO_CHUNK_BYTES, data.length - written);
            buf.clear();
            buf.put(data, written, chunk);
            buf.flip();
            drainFully(ch, buf);
            written += chunk;
        }
    }

    private static void readFloats(FileChannel ch, float[] out, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(IO_CHUNK_FLOATS * 4).order(ByteOrder.LITTLE_ENDIAN);
        int read = 0;
        while (read < count) {
            int rc = ch.read(buf);
            boolean eof = rc < 0;
            buf.flip();
            while (buf.remaining() >= Float.BYTES && read < count) out[read++] = buf.getFloat();
            if (read >= count) {
                if (buf.hasRemaining()) ch.position(ch.position() - buf.remaining());
                return;
            }
            if (eof) throw new IOException("truncated float stream at " + read + "/" + count);
            buf.compact();
        }
    }

    private static void readInts(FileChannel ch, int[] out, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(IO_CHUNK_INTS * 4).order(ByteOrder.LITTLE_ENDIAN);
        int read = 0;
        while (read < count) {
            int rc = ch.read(buf);
            boolean eof = rc < 0;
            buf.flip();
            while (buf.remaining() >= Integer.BYTES && read < count) out[read++] = buf.getInt();
            if (read >= count) {
                if (buf.hasRemaining()) ch.position(ch.position() - buf.remaining());
                return;
            }
            if (eof) throw new IOException("truncated int stream at " + read + "/" + count);
            buf.compact();
        }
    }

    private static void drainFully(FileChannel ch, ByteBuffer buf) throws IOException {
        while (buf.hasRemaining()) ch.write(buf);
    }

    private static void fillFully(FileChannel ch, ByteBuffer buf, String what) throws IOException {
        while (buf.hasRemaining()) {
            int rc = ch.read(buf);
            if (rc < 0) throw new IOException("truncated " + what + ": " + buf.remaining() + " bytes left");
        }
    }
}
