# MLIR Compiler Pipeline Summary

## What Was Added

- Real MLIR C++ pass plugin infrastructure in `mlir_passes/`
- Real `hir` MLIR dialect plugin with typed fused ops:
  `hir.fused_matmul_bias_relu` and `hir.fused_rmsnorm`
- INT8 quantization-aware HIR ops:
  `hir.quantize`, `hir.dequantize`, `hir.qmatmul`, and
  `hir.fused_qmatmul_bias_relu`
- HIR-focused canonicalization pass for tensor-level cleanup
- Canonicalization implemented with MLIR `RewritePattern`,
  `PatternRewriter`, `RewritePatternSet`, and the greedy rewrite driver
- MatMul + Bias Add + ReLU fusion candidate detection
- Fusion group and role annotations across producer and consumer ops
- SparseCore-like HIR target model metadata for fused MatMul lowering:
  tile shape 16x16x32, 256 KB SRAM model, 128-byte vector/alignment
  constraints, dense-or-2:4 sparse layout metadata, and memory-hierarchy tags
- Legality-driven MatMul-Bias-ReLU rewrite guards for one-use producer results,
  one-use bias-add results, ranked/static tensor shapes, supported dtypes,
  legal bias shape, and target tile multiples before fusion annotation
- FileCheck tests for canonicalization, positive fusion, and negative fusion cases
- Negative FileCheck/verifier coverage for multi-use MatMul results, dynamic
  target shapes, invalid HIR bias broadcast metadata, and invalid target tile
  metadata
- A canonicalization-enables-fusion test where `add(x, 0.0)` cleanup exposes
  the MatMul -> BiasAdd -> ReLU chain to the fusion detector
- MLIR Affine loop tiling test
- MLIR Affine vectorization test
- Pipeline script for generating runtime-facing artifacts
- MLIR HIR lowering pass that rewrites fusion candidates to typed HIR ops:
  `hir.fused_matmul_bias_relu` and `hir.fused_rmsnorm`
- RMSNorm lowering implemented with MLIR `ConversionPattern`,
  `ConversionTarget`, and `TypeConverter`
- HIR op verifiers plus a fused-op verifier pass for HIR invariants before
  artifact export
- Positive parse/print FileCheck coverage for typed HIR dialect ops and a
  negative verifier diagnostic test for invalid fused-op metadata
- Positive and negative FileCheck coverage for INT8 quantization metadata,
  scale, zero point, and per-channel quantization constraints
- Layout-aware verifier constraints for mobile accelerator legality:
  NHWC activations, blocked-KC weights, 128-byte alignment, and channel/tile
  dimensions that are multiples of 32
- Profile-guided INT8 lowering decision:
  `f32 matmul+bias+relu -> hir.fused_qmatmul_bias_relu` only when imported
  benchmark metadata marks the quantized path valid and faster
- JSON bridge from HIR MLIR ops to lowered graph and execution plan artifacts
- Target-aware dispatch descriptor export for fused MatMul and qmatmul HIR ops,
  including shape, dtype, selected tile, SRAM usage, vector alignment, and
  rejected tile candidates
- Profile-calibrated cost table:
  `cost_table[fusion_candidate][backend][shape_bucket][dtype]`
- C++ `CostBasedPlanner` now consumes `CostReport` FLOPs, bytes moved, launch
  overhead, and backend placement metadata instead of ignoring the report.
  Planner candidates include per-op compute time, memory time, transfer cost,
  cost source, and decision reason.
- Runtime execution plan metadata that maps the lowered fused op to
  `OpType::FusedMatMulAddReLU` and the `fused_matmul_add_relu` custom kernel
- A C++ benchmark that checks the MLIR execution plan and dispatches the fused
  op through the runtime registry for correctness and latency comparison
- End-to-end HIR runtime benchmark report that verifies typed HIR ops emitted
  by compiler passes are connected to measured runtime kernels with correctness
  and latency evidence
- Runtime-aware kernel-selection metadata for fusion candidates. The compiler
  now records selected, candidate, and fallback kernels plus runtime benchmark
  evidence instead of assuming that every fused candidate should dispatch to a
  custom kernel.
- RMSNorm compiler annotation and lowering support:
  `llm.rmsnorm -> hir.fused_rmsnorm`, with kernel selection driven by
  `heterogeneous-inference-runtime` RMSNorm benchmark artifacts.
- RMSNorm compiler/runtime case study report that proves the lowered
  `hir.fused_rmsnorm` op selects `fused_rmsnorm_cuda` from measured CUDA
  benchmark evidence, compares against `torch_rmsnorm`, and records
  correctness, latency, bandwidth, and roofline metadata.
- Real Apple Silicon MLIR-to-Metal RMSNorm path: a measured Metal kernel and
  CPU reference produce shape-bucket profiles, the compiler selects Metal only
  when correctness passes and measured latency wins, and a runtime executor
  reads the emitted plan before dispatching the selected kernel.
- MLIR-native HIR RMSNorm codegen path:
  `hir.fused_rmsnorm -> linalg.generic/math.rsqrt -> one-shot bufferization
  -> LLVM dialect -> mlir-runner`, with correctness checked against a CPU
  reference value.
- StableHLO-compatible MatMul decomposition test:
  `dot_general/add/maximum` frontend semantics represented as equivalent
  `linalg.matmul/linalg.map` IR, then lowered into `hir.fused_matmul_bias_relu`.
- StableHLO-compatible RMSNorm decomposition import:
  row sum-of-squares reduction plus `math.rsqrt` normalization represented in
  Linalg/Arith/Math IR, then rewritten into `hir.fused_rmsnorm` with
  `frontend.source = "stablehlo_compatible_rmsnorm_decomposition"`.
- OpenXLA toolchain status check for `stablehlo-opt`, `torch-mlir-opt`, JAX,
  TensorFlow, Torch-MLIR, and StableHLO Python modules. Native StableHLO tests
  are skipped until those tools are installed.
- StableHLO textual subset importer for the two high-value patterns used in
  this portfolio: RMSNorm decomposition and `dot_general + add + maximum`.
  The subset keeps `stablehlo.*` op names at the frontend boundary, imports
  them into standard MLIR, lowers them to HIR, and executes RMSNorm through the
  LLVM dialect path.
- IREE comparison layer for the same StableHLO textual subset, compiling
  RMSNorm and MatMul-Bias-ReLU through IREE VM/HAL with LLVM CPU target when
  `iree-compile` is available.
- Torch-MLIR tiny transformer probe that exports a minimal RMSNorm/Linear/ReLU
  block when `torch` and `torch_mlir` are installed, otherwise producing a
  clean skip report.
- Decode-attention KV-cache bandwidth model for serving-time attention, showing
  why context-length growth makes KV reads the bottleneck and why layout/block
  choices belong in the compiler/runtime planning loop.

## Pipeline

```text
MLIR input
  -> HIRCanonicalizationPass
  -> MatMulBiasReluFusionPass
  -> RMSNormKernelSelectionPass
  -> HIRFusionLoweringPass
  -> HIRFusedOpVerifierPass
  -> trace/mlir_fused_graph.mlir
  -> tools/mlir_fusion_to_runtime_json.py
  -> trace/mlir_lowered_graph.json
  -> trace/mlir_execution_plan.json
  -> runtime-aware dispatch to selected kernel or fallback
```

The native codegen path now continues from HIR without going through JSON:

```text
hir.fused_rmsnorm
  -> HIRRMSNormToLinalgPass
  -> linalg.generic + math.rsqrt tensor IR
  -> one-shot bufferization
  -> LLVM dialect
  -> mlir-runner executable CPU function
  -> trace/hir_rmsnorm_execution_engine_report.{json,md}
```

The Google/OpenXLA-facing frontend proof is:

```text
stablehlo textual subset
  -> tools/import_stablehlo_subset.py
  -> linalg/arith/math
  -> stablehlo-compatible RMSNorm import or MatMul fusion
  -> hir.fused_rmsnorm / hir.fused_matmul_bias_relu
  -> HIR RMSNorm LLVM executable path
  -> trace/stablehlo_subset_pipeline_report.{json,md}
```

RMSNorm-specific compiler artifacts are also preserved as
`trace/rmsnorm_fused_graph.mlir`, `trace/rmsnorm_lowered_graph.json`, and
`trace/rmsnorm_execution_plan.json` so the transformer-kernel selection path can
be inspected without overwriting the default MatMul-Bias-ReLU pipeline outputs.

The standalone RMSNorm case study is preserved as
`trace/rmsnorm_compiler_runtime_case_study.json` and
`trace/rmsnorm_compiler_runtime_case_study.md`. The attention decode model is
preserved as `trace/attention_kv_bandwidth_model.json` and
`trace/attention_kv_bandwidth_model.md`.

For the interview-level narrative that connects runtime evidence, compiler
lowering, roofline analysis, and KV-cache serving pressure, see
`docs/GPU_COMPILER_RUNTIME_CASE_STUDY.md`.

The cross-cutting planner/kernel/tile decision report is preserved as
`trace/compiler_runtime_decision_report.json` and
`trace/compiler_runtime_decision_report.md`. It shows the CostReport-driven C++
planner choosing a mixed CPU/Metal plan over all-Metal due to transfer cost,
plus profile-calibrated kernel selection and target tile decisions for HIR
fused ops.

## Verification

```bash
cmake --build build-mlir
tools/run_mlir_pass_tests.sh
tools/run_mlir_fusion_pipeline.sh
tools/build_profile_cost_table.py
tools/generate_hir_runtime_benchmark_report.py
tools/generate_rmsnorm_case_study.py
tools/generate_attention_kv_bandwidth_model.py
tools/validate_cost_based_planner.py
tools/validate_dispatch_descriptors.py
tools/generate_compiler_runtime_decision_report.py
```

Expected signals:

```text
add(x, 0.0) canonicalized away
relu(relu(x)) canonicalized to relu(x)
canonicalization_enables_fusion.mlir gains fusion.candidate after cleanup
fusion.candidate = "matmul_bias_relu"
fusion.group = "matmul_bias_relu_0"
target.model = "sparsecore_like_v1"
target.tile_m/tile_n/tile_k = 16/16/32
hir.fused_matmul_bias_relu emitted by HIRFusionLoweringPass
FusedMatMulBiasReLU
dispatch_selected_kernel
runtime_kernel = fused_matmul_add_relu
runtime_op_type = FusedMatMulAddReLU
kernel_selection.selected_kernel
kernel_selection.fallback_kernel
kernel_selection.evidence
profile_calibrated_cost_table chooses the measured winner for each shape bucket
llm.rmsnorm annotated as fusion.candidate = "rmsnorm"
hir.fused_rmsnorm emitted with fused_rmsnorm_cuda candidate and torch_rmsnorm fallback
hir.fused_qmatmul_bias_relu emitted when profile.quantized_path = "faster"
trace/hir_runtime_benchmark_report.json status = passed
trace/rmsnorm_compiler_runtime_case_study.json status = passed
trace/attention_kv_bandwidth_model.json status = modeled
trace/cv_cost_based_planner.json format = cost_based_planner.v2
planner op_costs include FLOPs/bytes/launch/memory/transfer breakdowns
target.dispatch_descriptor.v1 selects legal SRAM/alignment-aware tiles
trace/compiler_runtime_decision_report.json status = passed
```

## Engineering Relevance

This demonstrates:

- MLIR C++ pass/plugin development
- MLIR dialect/op definition with TableGen and C++ op verifiers
- Quantization-aware lowering surface for INT8 mobile/accelerator kernels
- Target-aware lowering legality for accelerator-style tile, memory hierarchy,
  sparse layout, and vector alignment constraints
- Target-aware tile planning that enumerates legal/rejected tiles and emits a
  dispatch descriptor consumed by execution-plan artifacts
- Layout-aware verifier checks for memory alignment and tile/channel
  constraints relevant to Qualcomm-style DSP/NPU paths
- Profile-driven quantized lowering from f32 MatMul chains to INT8 HIR fused ops
  when benchmark evidence selects the quantized kernel
- Pattern-based canonicalization rewrites before fusion
- Linalg pattern analysis
- Fusion candidate detection
- Legality-driven rewrite discipline that prevents wrong-code fusion when
  producer use-count, bias shape, dynamic shape, dtype, or target tile
  constraints are not satisfied
- MLIR lowering from source/fusion dialect patterns into typed HIR dialect ops
- MLIR dialect-conversion infrastructure through `ConversionTarget`,
  `TypeConverter`, and conversion patterns
- Op verifier and verifier-pass invariant checks for emitted HIR fused ops
- False-positive prevention with negative tests
- Affine loop tiling
- Affine vectorization
- Runtime lowering bridge
- Cost-model metadata for backend scheduling
- Profile-calibrated cost table for runtime-aware kernel and backend decisions
- CostReport-driven C++ planning where FLOPs, bytes moved, launch overhead, and
  backend transfer cost change backend selection
- Custom fused-kernel dispatch path from MLIR lowering metadata to C++ runtime
- End-to-end benchmark evidence tying compiler-emitted HIR dialect ops to
  runtime kernel correctness and latency data
- Runtime-aware compiler behavior: kernel selection is based on benchmark
  evidence when available and conservatively falls back when evidence is absent
- Measured RMSNorm case-study evidence with correctness, latency, speedup,
  bandwidth, and memory-bound roofline metadata
- Attention decode bandwidth modeling for KV-cache layout, paged blocks, and
  runtime memory-pressure decisions
