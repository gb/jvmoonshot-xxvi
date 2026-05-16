# Build stage: rust:alpine is musl-native, so `cargo build` produces a fully
# static binary without any cross-toolchain setup. Result is ~2-3 MB.
FROM rust:1.83-alpine AS build
RUN apk add --no-cache musl-dev

WORKDIR /lapada

# target-cpu=haswell: contest box is 2014 Mac Mini (i5-4xxxU). Lets rustc
# emit AVX2/BMI2 in any vectorizable byte ops (memcpy, slice copies, intrinsic
# bytes). Free perf since we're shipping a Haswell-only binary.
ENV RUSTFLAGS="-C target-cpu=haswell"

# Layer-cache the dependency build so source-only edits skip the slow tokio
# compile. Trick: build with a stub main so cargo resolves+compiles deps,
# then drop the stub artifacts before copying real sources. Cargo.lock is
# committed for reproducibility (host build vs in-image build see same deps).
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
    && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src target/release/lapada target/release/lapada.d \
              target/release/deps/lapada-*

COPY src ./src
RUN cargo build --release && strip target/release/lapada

# Runtime stage: scratch + a single static binary. No shell, no libc, no PID 1
# init. tokio handles SIGTERM via the default async-task drop on exit.
FROM scratch
COPY --from=build /lapada/target/release/lapada /lapada
ENTRYPOINT ["/lapada"]
