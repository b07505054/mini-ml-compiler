# MLIR Compiler Passes

This directory contains the real MLIR C++ pass infrastructure for the
**execution-planning ML compiler**. It is separate from the project's custom toy
graph IR in `src/`.

The compiler here reuses existing MLIR infrastructure for generic graph
semantics, lifts backend-relevant regions into a decision-oriented HIR, and
produces an explainable hardware-aware execution plan consumed by heterogeneous
runtimes.

## 16-Pass Hardware-Aware Serving Pipeline

The primary current compiler work is a complete 16-pass execution-planning
pipeline implemented in `lib/serving/`, with related HIR/quantization helpers
in `lib/`. Each pass annotates ops with structured MLIR attrs. No pass modifies
IR structure. No pass selects a winner until `PlanSelectionPass`. No pass
materializes boundary ops.

```text
HIR / Serving IR input
  -> ServingPhaseAnalysis            (serving.phase = prefill | decode)
  -> KVLayoutPlanningPass            (kv.layout, kv.block_size)
  -> ReplayEligibilityPass           (replay.eligible)
  -> ExecutionProviderPlanningPass   (execution_provider.backend)
  -> RepresentationPlanningPass      (representation.effective_dtype)
  -> LayoutPlanningPass              (layout.effective_layout)
  -> BoundaryPlanningPass            (boundary.cast_required, boundary.dequant_required)
  -> WeightClassificationPlanningPass (weight.classification)
  -> QuantizationStrategyPlanningPass (quant.strategy, quant.activation_dtype)
  -> KernelAvailabilityPlanningPass   (kernel.exists, kernel.lowering_status)
  -> LoweringDecisionPlanningPass     (lowering.decision)
  -> QuantizedBoundaryRefinementPass  (boundary.weight_dequant_required refined)
  -> AlternativeLoweringPlanningPass  (alternative.candidates)
  -> CandidateGenerationPass          (compiler.candidates, compiler.rejected_candidates)
  -> CandidateEvaluationPass          (compiler.evaluated_candidates, evaluation.*)
  -> PlanSelectionPass                (selected_plan.*, compiler.selected_candidates)
  -> Execution Plan JSON / Runtime Contract
```

Truth boundary discipline: every planning annotation carries a `truth_boundary`
field. Static penalty scores in `CandidateEvaluationPass` carry
`candidate_evaluation_static_penalty_not_measured_latency`. Declared profile
data carries `declared_profile`. No annotation presents a static relative
penalty as measured hardware latency.

Fallback is last resort, not first response. `backend_fallback` is only emitted
when direct kernel, algebraic decomposition, representation conversion, layout
conversion, and cast conversion paths are all unavailable or invalid.

FileCheck tests for the serving pipeline are in `test/serving/`, with related
HIR and quantization tests in `test/`. Run with `tools/run_mlir_pass_tests.sh`.

## Other MLIR Passes

The plugin also contains MLIR compiler passes for canonicalization,
fusion detection, lowering, verification, serving-plan generation, and CV
execution-plan generation:

- `hir` dialect: defines typed runtime-facing fused ops such as
  `hir.fused_matmul_bias_relu`, `hir.fused_rmsnorm`, `hir.quantize`,
  `hir.dequantize`, `hir.qmatmul`, and `hir.fused_qmatmul_bias_relu`.
- `cv` dialect: defines the current CV compiler IR, including seven CV ops and
  variadic `cv.detect_head` support for the raw CV graph milestone.

- `hir-canonicalize`: rewrites small tensor-level canonical forms before
  optimization.
- `matmul-bias-relu-fusion`: detects a tensor-level MatMul + Bias Add + ReLU
  fusion candidate.
- `rmsnorm-kernel-selection`: marks `llm.rmsnorm` ops for runtime-aware HIR
  lowering.
- `hir-fusion-lowering`: lowers annotated fusion candidates into typed HIR
  dialect ops.
- `hir-verify-fused-ops`: verifies invariants on emitted HIR fused ops.
- `cv-frontend-normalization`: normalizes raw CV frontend metadata.
- `cv-shape-inference`: attaches static shape metadata for supported CV ops.
- `cv-memory-planning`: attaches tensor lifetime and allocation metadata.
- `cv-execution-domain-planning`: assigns execution-domain metadata for the
  current CV plan export path.

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
M/N/K are static; exact tile multiples lower directly, while near-tile shapes
can lower through `pad_to_tile_with_crop` when compute/output overhead is
below the profitability threshold
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

Sparse layout is now a compiler decision rather than passive metadata for the
supported constant-weight case. When a MatMul requests:

```text
sparse.candidate = "2_4"
profile.sparse_2_4_path = "faster"
```

the fusion pass checks the RHS/weight tensor at compile time. Along the K
dimension, every group of 4 values for each output channel must contain at most
2 non-zero values. Legal constant weights emit `target.sparse_layout =
"structured_2_4"` plus group metadata; illegal or non-constant weights fall
back to the dense fused path and record the sparse fallback reason. This models
accelerator sparse-layout legality without claiming a real SparseCore backend.

The runtime JSON bridge also turns target metadata into a dispatch descriptor.
For fused MatMul and qmatmul HIR ops, it parses M/N/K/dtype from the emitted
MLIR type, enumerates candidate tiles, accepts either exact divisibility or
profitable pad/crop tiles, rejects candidates that violate padding overhead,
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

The HIR dialect includes an executable INT8 path for the validated fused
Raspberry Pi operator, as well as broader mobile-accelerator compiler work:

```text
hir.quantize
hir.load_quantized_weight
hir.qmatmul
hir.dequantize
hir.portable_cpu_int8_fused_matmul_bias_relu
```

The quantization compiler pipeline now has three small graph-level passes:

```text
quantization-planning
  -> hir-quant-propagate
  -> hir-quant-canonicalize
  -> fusion / lowering
  -> hir-int8-operator-selection
  -> verification / artifact export
```

Implemented behavior:

- `hir-quant-propagate` forms INT8-capable islands through
  conservative ops: INT8-candidate `linalg.matmul`, ReLU-shaped `linalg.map`,
  and tensor reshape/cast ops.
- Unsupported ops are left unannotated and break INT8 island continuity.
- `hir-quant-canonicalize` eliminates adjacent Q/DQ or DQ/Q pairs only when
  `scale`, `zero_point`, `quantized_dtype`, and `quantization.mode` match.
- `hir-int8-operator-selection` uses a deterministic capability table for
  `matmul_bias_relu`, `matmul`, `relu`, and `reshape`, gated by backend,
  layout, shape alignment, and optional profile metadata.
- Illegal layout or unsupported backend records a fallback selection reason
  instead of pretending INT8 dispatch is available.

Implemented for the fused Slice 3 path:

- deterministic calibration-derived activation and weight scales;
- compiler-owned packed INT8 weights;
- explicit Q/DQ and integer tensor rewrite;
- canonical ExecutionPlan execution through the selected Cortex-A76 kernel;
- identity and correctness validation on Raspberry Pi.

Not implemented generally: full-model calibration, graph-wide mixed precision,
a complete layout optimizer, or propagation through arbitrary arithmetic.

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
target shapes, padding overhead fallback, invalid HIR bias broadcast metadata,
and invalid target tile/padded metadata. These tests are meant to prove the
pass avoids wrong-code rewrites, not only that it finds the happy path.

## Requirements

Use the same LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck`.
Mixing tools from different LLVM builds can cause plugin ABI or registration
failures.

Required CMake packages:

```text
$(brew --prefix llvm)/lib/cmake/mlir
$(brew --prefix llvm)/lib/cmake/llvm
```

## Build

From the `ml-graph-compiler-runtime` repo root:

```bash
cmake -S mlir_passes -B build-mlir \
  -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir" \
  -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"

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
mlir-opt \
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
mlir-opt \
  --load-dialect-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  --load-pass-plugin=build-mlir/HIRMatMulBiasReluFusionPass.dylib \
  mlir_passes/test/matmul_bias_relu.mlir \
  --pass-pipeline='builtin.module(hir-canonicalize,matmul-bias-relu-fusion)' \
  | FileCheck mlir_passes/test/matmul_bias_relu.mlir
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

The CV compiler artifact path is separate from the HIR fusion bridge:

```text
CV dialect input
  -> cv-frontend-normalization
  -> cv-shape-inference
  -> cv-memory-planning
  -> cv-execution-domain-planning
  -> CVExecutionPlanBuilder
  -> CVExecutionPlanExporter
  -> emit-cv-execution-plan
  -> artifacts/apple_demo/cv_execution_plan.json
```

This path proves the registered CV dialect and Phase 1 CV analysis/export
pipeline. It does not yet claim ONNX import, dynamic shapes, backend kernel
mapping, or PocketChef visualization.

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
