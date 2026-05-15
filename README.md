# jvmoonshot-xxvi

My solution for the [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## What is it about

Rinha de Backend is the Brazilian dev community's annual party around a backend challenge, a friendly competition where you build a backend under tight constraints. This year's theme is **fraud detection via vector search**: for every card transaction the API receives, you turn it into a 14-dimensional vector, find its 5 nearest neighbors in a reference set of 3 million labeled vectors, and answer `approved` or `denied` in milliseconds. The constraints this year: 1 CPU and 350 MB of memory total, split across at least one load balancer and two API instances.

## Why the JVM

The smart money is on a small native binary in C, Rust, or Zig. I'm writing it on the JVM anyway. It's the stack I know best, and I want to see how far careful engineering (off-heap memory, SIMD, a hand-rolled HNSW index, a tiny HTTP loop) can push a runtime that nobody picks for this kind of problem. It probably won't win, and that's fine. The point is to have fun, learn a lot, and see if the JVM still has teeth when you treat it like a serious tool.
