# MLIR Compiler Passes

This directory contains real MLIR C++ pass infrastructure, separate from the
project's custom toy graph IR.

The plugin currently contains MLIR compiler passes for canonicalization,
fusion detection, lowering, and verification:

- `hir` dialect: defines typed runtime-facing fused ops such as
  `hir.fused_matmul_bias_relu`, `hir.fused_rmsnorm`, `hir.quantize`,
  `hir.dequantize`, `hir.qmatmul`, and `hir.fused_qmatmul_bias_relu`.

- `hir-canonicalize`: rewrites small tensor-level canonical forms before
  optimization.
- `matmul-bias-relu-fusion`: detects a tensor-level MatMul + Bias Add + ReLU
  fusion candidate.
- `rmsnorm-kernel-selection`: marks `llm.rmsnorm` ops for runtime-aware HIR
  lowering.
- `hir-fusion-lowering`: lowers annotated fusion candidates into typed HIR
  dialect ops.
- `hir-verify-fused-ops`: verifies invariants on emitted HIR fused ops.

The canonicalization pass performs real IR rewrites with MLIR
`OpRewritePattern`, `PatternRewriter`, `RewritePatternSet`, and the greedy
rewrite driver:

```text
linalg.map addf(x, 0.0)
  -> x

linalg.map maximumf(linalg.map maximumf(x, 0.0), 0.0)
  -> linalg.map maximumf(x, 0.0)
```

The fusion pass then detects:

```text
linalg.matmul
  -> linalg.map arith.addf
  -> linalg.map arith.maximumf with zero
```

The MatMul fusion pass is legality-driven rather than purely detect-and-annotate.
Before it marks a candidate, it checks:

```text
matmul result has one use
bias-add result has one use
bias input is legal for the fused output shape
lhs/rhs/result are ranked rank-2 tensors
dtype is supported by the target path
M/N/K are static multiples of the target tile shape
```

Legal candidates are annotated and later lowered to a typed HIR op. Illegal
patterns remain unfused so the original IR can fall back conservatively.

The target model attached during lowering is intentionally small but real:

```text
target.model = "sparsecore_like_v1"
tile_m/tile_n/tile_k = 16/16/32
target.sram_kb = 256
target.vector_bytes = 128
target.alignment = 128
target.sparse_layout = "dense_or_2_4"
target.memory_hierarchy = "global_sram_register"
```

The HIR verifier checks these attributes when present, so invalid target
metadata is rejected before artifact export.

The runtime JSON bridge also turns target metadata into a dispatch descriptor.
For fused MatMul and qmatmul HIR ops, it parses M/N/K/dtype from the emitted
MLIR type, enumerates candidate tiles, rejects tiles that violate shape,
SRAM, or vector-alignment constraints, and records the selected tile plus
rejected candidates in the execution plan.

RMSNorm lowering uses MLIR dialect-conversion infrastructure:

```text
ConversionTarget marks "llm.rmsnorm" illegal
TypeConverter provides the HIR result-type conversion hook
ConversionPattern rewrites "llm.rmsnorm" -> hir.fused_rmsnorm
hir.fused_rmsnorm::verify checks op invariants
hir-verify-fused-ops provides a pipeline-level verification stage
```

The HIR dialect also includes an INT8 quantization path for mobile accelerator
compiler work:

```text
hir.quantize
hir.qmatmul
hir.fused_qmatmul_bias_relu
hir.dequantize
```

The quantized ops verify INT8 dtype metadata, scale, zero point, per-channel
activation quantization metadata, and mobile accelerator layout constraints:

```text
input_layout = "NHWC"
weight_layout = "blocked_kc"
alignment = 128
K dimension multiple of 32
INT8 output channel dimension multiple of 32
```

These checks model Qualcomm-style DSP/NPU constraints where layout, tile shape,
and memory alignment affect whether a lowered kernel is legal.

Profile-guided quantized lowering is intentionally conditional. A MatMul chain
only lowers to `hir.fused_qmatmul_bias_relu` when imported profile metadata marks
the INT8 path as valid and faster:

```text
f32 linalg.matmul + bias + relu
  -> profile.quantized_path = "faster"
  -> hir.fused_qmatmul_bias_relu
  -> int8_qmatmul_bias_relu
```

The committed qmatmul benchmark profile records the INT8 path as faster for the
`128x128x128:i8` shape bucket, so the generated qmatmul execution plan selects
`int8_qmatmul_bias_relu` from the profile-calibrated cost table.

The test `canonicalization_enables_fusion.mlir` demonstrates why the passes run
in that order: the input graph contains an identity `add(x, 0.0)` between
MatMul and BiasAdd. `hir-canonicalize` removes the identity map first, then
`matmul-bias-relu-fusion` can see and annotate the cleaned fusion chain.

Negative FileCheck tests cover missing ReLU, multi-use MatMul results, dynamic
target shapes, invalid HIR bias broadcast metadata, and invalid target tile
metadata. These tests are meant to prove the pass avoids wrong-code rewrites,
not only that it finds the happy path.

## Requirements

Tested target: LLVM/MLIR built from `/Users/allen/Developer/llvm-build`.

Use the same LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck`.
Mixing a Homebrew `mlir-opt` with a locally built plugin can cause plugin
registration failures.

Required CMake packages:

```text
/Users/allen/Developer/llvm-build/lib/cmake/mlir
/Users/allen/Developer/llvm-build/lib/cmake/llvm
```

## Build

From the `ml-graph-compiler-runtime` repo root:

```bash
cmake -S mlir_passes -B build-mlir \
  -DMLIR_DIR=/Users/allen/Developer/llvm-build/lib/cmake/mlir \
  -DLLVM_DIR=/Users/allen/Developer/llvm-build/lib/cmake/llvm

cmake --build build-mlir
```

On macOS this builds:

```text
build-mlir/HIRMatMulBiasReluFusionPass.dylib
```

## Run The Fusion Pipeline

This plugin is registered as an MLIR pass pipeline. Use the same `mlir-opt`
binary from the LLVM/MLIR build used by CMake.

```bash
/Users/allen/Developer/llvm-build/bin/mlir-opt \
  --load-dialect-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)'
```

Expected output includes:

```mlir
linalg.matmul {fusion.candidate = "matmul_bias_relu"}
```

## FileCheck

```bash
/Users/allen/Developer/llvm-build/bin/mlir-opt \
  --load-dialect-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)' \
  | /Users/allen/Developer/llvm-build/bin/FileCheck mlir_passes/test/matmul_bias_relu.mlir
```

## Runtime Bridge

The fusion pipeline can emit runtime-facing artifacts:

```bash
tools/run_mlir_fusion_pipeline.sh
```

Outputs:

```text
trace/mlir_fused_graph.mlir
trace/mlir_lowered_graph.json
trace/mlir_execution_plan.json
trace/hir_runtime_benchmark_report.json
trace/hir_runtime_benchmark_report.md
```

This connects the real MLIR pass output to the existing C++ heterogeneous
runtime planning story by first lowering an annotated `linalg.matmul` chain into
the typed MLIR op `hir.fused_matmul_bias_relu`, then exporting that HIR op to
runtime-facing JSON.

The default bridge pipeline runs:

```text
hir-canonicalize
  -> matmul-bias-relu-fusion
  -> rmsnorm-kernel-selection
  -> hir-fusion-lowering
  -> hir-verify-fused-ops
```

Quick verification:

```bash
grep 'hir.fused_matmul_bias_relu' trace/mlir_fused_graph.mlir
grep 'FusedMatMulBiasReLU' trace/mlir_lowered_graph.json
grep 'dispatch_selected_kernel' trace/mlir_execution_plan.json
grep 'fused_matmul_add_relu' trace/mlir_execution_plan.json
```

The JSON bridge also attaches a lightweight cost model with estimated FLOPs,
memory traffic, and arithmetic intensity so the fused op can be consumed by
backend placement or scheduling heuristics.

The C++ `CostBasedPlanner` consumes the compiler `CostReport` instead of
discarding it. Planner candidates now estimate each op from FLOPs, bytes moved,
backend bandwidth/compute, launch overhead, and backend transfer cost, then
export per-op cost source and decision reason in
`trace/cv_cost_based_planner.json`.

The runtime-aware cost model is calibrated from benchmark artifacts into:

```text
cost_table[fusion_candidate][backend][shape_bucket][dtype]
```

`tools/build_profile_cost_table.py` builds
`trace/profile_calibrated_cost_table.json` from runtime profiles. The JSON
bridge consumes that table before selecting a runtime kernel, so the planner
uses measured shape-bucket evidence instead of assuming fused kernels always
win.

The pass also assigns a `fusion.group` and per-op `fusion.role` metadata to
the MatMul, bias-add, and ReLU operations, making the detected producer and
consumer chain explicit before runtime lowering.

The runtime bridge records the concrete selected C++ dispatch path:

```text
hir.fused_matmul_bias_relu
  -> OpType::FusedMatMulAddReLU
  -> selected kernel from runtime benchmark evidence
```

`run_mlir_fused_kernel_benchmark` verifies that the emitted execution plan names
that custom fused kernel path, dispatches the fused op through `OpRegistry`, and
checks fused-vs-unfused numerical correctness. The benchmark is used as a
compiler/runtime integration check; CPU speedup depends on shape and local
microarchitecture.

`tools/generate_hir_runtime_benchmark_report.py` ties the compiler and runtime
evidence together. It fails if the execution plans do not contain typed
`hir.fused_*` ops, profile-calibrated runtime kernel selections, benchmark
latencies, and correctness-passing evidence for both MatMul-Bias-ReLU and
RMSNorm.

`tools/generate_compiler_runtime_decision_report.py` is the compact
interview-facing artifact. It combines the CostReport-driven C++ planner,
profile-calibrated kernel choices, target dispatch descriptors, selected tile
shapes, SRAM usage, and fallback reasons into
`trace/compiler_runtime_decision_report.md`.

`tools/generate_rmsnorm_case_study.py` preserves the RMSNorm transformer-kernel
story as a standalone compiler/runtime case study. It checks that
`llm.rmsnorm` lowers to `hir.fused_rmsnorm`, imports the measured CUDA benchmark
from `heterogeneous-inference-runtime`, and records the selected
`fused_rmsnorm_cuda` path against the `torch_rmsnorm` fallback with correctness,
latency, bandwidth, and roofline metadata.

`tools/generate_attention_kv_bandwidth_model.py` adds a decode-attention
KV-cache bandwidth model. This is not a full FlashAttention implementation; it
is a planning artifact that explains why decode attention is KV-memory-bound
and why compiler-selected layout, block size, and runtime memory pressure matter
for paged KV-cache serving.

`docs/GPU_COMPILER_RUNTIME_CASE_STUDY.md` is the repo-level narrative that ties
the runtime evidence, MLIR lowering, profile-calibrated decision, roofline
analysis, and KV-cache serving extension into one interview-ready story.

## Resume Bullet

Added an MLIR C++ compiler pipeline with `RewritePattern` canonicalization,
typed HIR dialect ops, MatMul-Bias-ReLU fusion detection,
`ConversionTarget`/`TypeConverter`-aware RMSNorm lowering, op verification,
FileCheck coverage, runtime-facing lowering artifacts, and correctness/benchmark
reports for a heterogeneous C++ execution planner, including a measured CUDA
RMSNorm compiler/runtime case study and an attention decode KV-cache bandwidth
model.
