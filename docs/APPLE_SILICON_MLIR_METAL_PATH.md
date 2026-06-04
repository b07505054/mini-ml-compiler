# Apple Silicon MLIR-to-Metal Path

## Goal

Build a measured compiler/runtime path where an MLIR RMSNorm operation lowers
to typed HIR, consumes Apple Silicon benchmark evidence, and selects a real
Metal kernel only for shape buckets where it beats the CPU fallback.

## Measured Metal Kernel

`metal/rmsnorm.metal` implements FP32 RMSNorm as one threadgroup per token. Each
thread accumulates squared values, the threadgroup performs a reduction, and
the normalized output is written with the learned weight.

`apps/benchmark_metal_rmsnorm.mm`:

- Compiles and dispatches the real Metal kernel.
- Compares against a CPU RMSNorm reference.
- Checks max absolute difference with an explicit `1e-4` threshold.
- Sweeps tokens `1, 16, 128` and hidden sizes `768, 1024, 4096, 8192`.
- Records Metal and CPU p50/p95 latency, speedup, and effective bandwidth.
- Emits `trace/metal_rmsnorm_benchmark.json` for compiler cost-table ingestion.

Build and run:

```bash
cmake -S . -B build-metal
cmake --build build-metal --target benchmark_metal_rmsnorm
./build-metal/benchmark_metal_rmsnorm
ctest --test-dir build-metal -R metal_rmsnorm_correctness --output-on-failure
```

## Apple M5 Result

All twelve FP32 shapes pass numeric correctness. The measured crossover is
shape-dependent:

- Small token counts remain on CPU because Metal command-buffer overhead
  dominates.
- `16x4096:f32` and larger representative workloads select Metal.
- `128x8192:f32` demonstrates the strongest measured Metal advantage.

This crossover is the intended compiler decision signal. The compiler should
not select Metal because it is available; it should select Metal because the
profile table proves it wins for the requested shape bucket.
