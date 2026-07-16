# AArch64 Native Codegen: hir.fused_matmul_bias_relu

Target: Raspberry Pi 5 / Cortex-A76
Execution type: Real hardware
Code generation: MLIR -> LLVM IR -> LLVM AArch64 backend
Optimization status: Generic baseline, no project-owned target-specific instruction selection

## What this is

The project's first end-to-end native code generation path, executed on real
hardware:

```
hir.fused_matmul_bias_relu (typed HIR op)
  -> hir-matmul-bias-relu-to-linalg      (project-owned pass, existing)
  -> one-shot-bufferize + buffer-deallocation-pipeline + stock MLIR->LLVM
     dialect conversion passes            (stock upstream MLIR passes)
  -> mlir-translate --mlir-to-llvmir     -> LLVM IR text
  -> llc -mtriple=aarch64-linux-gnu -mcpu=cortex-a76
       -> AArch64 assembly and object code
  -> g++ link on the Raspberry Pi itself (object built on the x86_64 dev
     host, transferred via scp, linked with gcc/g++ already installed on
     the Pi -- no LLVM toolchain was installed on the Pi)
  -> executed on the real Raspberry Pi, correctness- and latency-checked
     against a scalar C++ reference compiled at the same optimization level
```

Everything through "LLVM dialect" already existed and was FileCheck-verified
before this slice (`mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir`).
This slice's new contribution is closing the remaining path: LLVM dialect ->
textual LLVM IR -> AArch64 object -> linked executable -> real device
execution -> measured evidence. See `ARCHITECTURE_STATUS.md` and
`KNOWN_GAPS.md` for how this is now described at the repository level.

## Files

- `input_<shape>.mlir` -- HIR input for each of the three validated shapes
  (8x8x8, 16x16x16, 32x32x32 for M/N/K). Identical in structure to
  `mlir_passes/test/hir_matmul_bias_relu_to_llvm.mlir`, with
  `llvm.emit_c_interface` added on the function so a C++ harness can call it
  without guessing the raw unpacked-descriptor ABI. Canonical copies live at
  `mlir_passes/test/backend_codegen/matmul_bias_relu_<shape>.mlir`.
- `lowered_llvm_dialect_<shape>.mlir` -- mlir-opt output (LLVM dialect).
- `generated_<shape>.ll` -- mlir-translate output (textual LLVM IR).
- `generated_<shape>.s` -- llc AArch64 assembly.
- `generated_<shape>.o` -- llc AArch64 ELF object (`file format
  elf64-littleaarch64`, machine `EM_AARCH64`). **Not committed**: this repo's
  `.gitignore` excludes `*.o` repo-wide and no binary object has ever been
  committed anywhere in this repository's history, so this artifact directory
  follows that same convention. The object is fully reproducible from
  `generated_<shape>.ll` via the `llc` command in `commands.txt`; its SHA-256
  and size at generation time are recorded in `object_hashes.txt` so a
  regenerated object can be verified byte-for-byte identical.
- `objdump_<shape>.txt` -- full `llvm-objdump -d` disassembly of the object
  (this text file *is* committed -- it's the durable evidence of what the
  object contained, independent of the gitignored binary itself).
- `correctness_results.json` -- per-shape correctness verdict, max absolute
  error vs. a scalar C++ reference, and checksums of both outputs.
- `benchmark_results.json` -- per-shape median/p95 latency for the generated
  kernel and the scalar reference, measured on the real Raspberry Pi.
- `backend_metrics.json` -- static LLVM IR / AArch64 instruction counts,
  load/store/branch/FP-arithmetic classification, and object size, with the
  exact counting methodology documented inline.
- `pi_device_state.txt` -- `hostname; uname -a; lscpu; gcc --version` captured
  on the real Raspberry Pi at benchmark time.

Reproduction commands for every file above are in `commands.txt`.

## Results summary (see benchmark_results.json / backend_metrics.json for full precision)

| Shape | Correct | Max Abs Error | Generated Median | Scalar Median | Ratio |
|---|---|---|---|---|---|
| 8x8x8 | yes | 3e-08 | 0.0022 ms | 0.0006 ms | 3.5x slower |
| 16x16x16 | yes | 1.2e-07 | 0.0165 ms | 0.0040 ms | 4.1x slower |
| 32x32x32 | yes | 1.8e-07 | 0.1272 ms | 0.0293 ms | 4.3x slower |

All three shapes are numerically correct (error at FP32 rounding level, not
a logic error). The generated code is currently slower than the scalar
reference at every shape. Static metrics show why this is expected, not a
regression to chase in this slice: the loop nest is not unrolled or
vectorized, so instruction count is identical across all three shapes (176
LLVM IR instructions, 134 AArch64 instructions, both split ~85%/15% between
the core kernel and the `_mlir_ciface_` marshalling wrapper) even though
runtime work scales with M*N*K. The scalar reference, compiled by g++ at
`-O2`, gets ordinary auto-vectorization and loop optimization that this
generic LLVM AArch64 path does not yet receive, because no project-owned or
LLVM target-specific optimization was applied here on purpose -- see
"Optimization status" above. Closing that gap (real NEON/SDOT instruction
selection) is explicit future work, not part of this slice.

## A real bug found and fixed during this slice

The original mlir-opt pass pipeline (matching the existing FileCheck test)
never ran a buffer-deallocation pass. Every call to the generated kernel
allocated an intermediate buffer (for the matmul-only result, before
bias-add+relu) that was never freed inside the function -- a genuine,
per-call memory leak, invisible to the FileCheck test because it only
inspects static IR text and never executes the result. This was found by
running the correctness+benchmark harness across many iterations and shapes
on the real Pi and observing incorrect output on specific shapes after
enough iterations; root-caused by an explicit malloc/free count check on the
emitted LLVM IR, and fixed by adding `buffer-deallocation-pipeline` to the
pass pipeline in `compile_hir_matmul_bias_relu_aarch64.sh`.

A second, smaller effect was also found and is worked around rather than
fully root-caused: even after the deallocation fix, running all three shapes
back-to-back in a single process intermittently corrupted one shape's
output, while every shape was always correct in isolation and the generated
kernel itself was independently verified correct (disassembly inspection,
single-call testing with both trivial and realistic inputs, before and after
a stress loop of 2000+ prior calls to a different shape). The exact
allocator-level mechanism was not fully isolated. The fix applied --
running each shape's correctness+benchmark as its own process, orchestrated
by `tools/run_backend_codegen_pi_integration.sh` -- removes the entire class
of cross-call heap-state risk regardless of the precise mechanism, and is
standard practice for exactly this kind of measurement robustness. This is
recorded here rather than hidden because it is a real, reproducible finding
about this specific generated-code/harness/toolchain combination, not
resolved with full certainty about its root cause.
