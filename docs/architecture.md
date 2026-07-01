# Architecture

## Purpose

This repository is a prototype ML graph compiler/runtime used to demonstrate
compiler passes, graph lowering, memory planning, CV execution-plan generation,
and runtime-facing artifact export. It is not a production inference runtime.
The code mixes implemented C++ demo-runtime components, real MLIR pass plugin
infrastructure, Python artifact tooling, demos, benchmark harnesses, and
simulated planning outputs.

The project should be read as two separate tracks:

1. **MLIR compiler track**: the primary compiler architecture. It owns
   HIR/CV/Serving dialects, compiler passes, quantization planning, INT8 island
   propagation, Q/DQ canonicalization, fusion/lowering,
   capability-gated operator-selection metadata, verification, and
   execution-plan/artifact export.
2. **Legacy/local C++ runtime demo harness**: a custom graph runtime used for
   local demos, benchmark bridges, memory-planning experiments, kernel-registry
   examples, and backend-sandbox experiments. It is useful evidence for
   compiler/runtime contracts, but it is not the production distributed runtime
   story.

New runtime/deployment features should go to the sibling
`heterogeneous-inference-runtime` repository. That project is the intended home
for distributed scheduling, prefill/decode split planning, backend dispatch,
runtime simulation, cost modeling, and runtime artifact consumption.

Clean architecture boundary:

```text
Model / MLIR input
  ->
MLIR Compiler Track
  ->
Execution Plan / Artifact
  ->
heterogeneous-inference-runtime
  ->
Runtime scheduling / backend dispatch / validation
```

Honest claims:

- This compiler repo exports runtime-facing metadata and artifacts.
- Quantization is currently implemented at the compiler-pass
  metadata/legality level.
- The C++ runtime inside this repo is a local demo harness and benchmark
  bridge.

Not claimed:

- Full INT8 graph runtime execution.
- Production calibration.
- Complete ONNX import.
- Generic Metal backend execution for all quantized kernels.
- The C++ runtime harness and the Python runtime project are one production
  system.

Assumptions for this handoff:

- Existing uncommitted changes in source, trace, and artifact files predate this documentation pass and should be treated as user work.
- Documentation should describe the repository as observed, not as a verified production system.
- Metrics in generated JSON, reports, and README examples are estimated or demo-derived unless a benchmark script explicitly measures them on the local machine.

## Top-Level Modules

### Core Graph IR

Implemented in `include/ir/` and `src/ir/`.

- `Graph` owns vectors of `Tensor` and `Node`.
- `Tensor` stores name, shape, dtype, float data, lifetime metadata, memory offset, runtime pointer, and persistence flag.
- `Node` stores name, `OpType`, input tensor ids, and output tensor ids.
- `OpType` covers simple NN ops, fused matmul variants, attention variants, and CV ops.

This is a custom toy IR. It is separate from the MLIR plugin in `mlir_passes/`.

### Compiler Passes Over Toy IR

Implemented in `include/pass/` and `src/pass/`.

- `PassManager` runs pass objects sequentially.
- Canonicalization, algebraic simplification, dead-node elimination, fusion, cost reporting, and runtime cost merge passes operate over the custom `Graph`.
- These passes are educational/compiler-prototype passes, not a full optimizer stack.

### Analysis

Implemented in `include/analysis/` and `src/analysis/`.

- Shape inference and graph verification support the custom graph pipeline.
- Analysis is lightweight and tailored to the demo graph structures.

### Runtime

Implemented in `include/runtime/` and `src/runtime/`.

- `Executor` runs an `ExecutionPlan` against a selected backend or backend scheduler.
- `CPUBackend` dispatches through the default op registry and real CPU kernels.
- `MockGPUBackend` logs simulated GPU execution, then dispatches the same CPU kernels.
- `MetalBackend` currently logs Metal device discovery and node dispatch in the generic backend path. It does not execute generic graph kernels.
- `MemoryPlanner` computes tensor lifetimes and assigns reusable arena offsets.
- `ArenaAllocator` binds tensors to a contiguous runtime buffer.
- `ExecutionPlanBuilder` converts lowered graph ops into dependency-aware `ExecutionPlanV2` steps.
- Runtime scheduling includes static scheduling, backend/provider scheduling, async/parallel executors, thread pool support, KV cache demos, continuous batching, and LLM scheduler prototypes.

Implemented behavior: CPU dispatch, CPU kernels for supported ops, mock GPU dispatch via CPU kernels, memory offset assignment, dependency extraction for lowered plans, and LLM scheduler/KV cache simulations.

Simulated or partial behavior: generic Metal backend execution, GPU execution through `MockGPUBackend`, runtime replan decisions, many latency/throughput fields in JSON artifacts, and full serving runtime behavior.

Boundary: this runtime directory is a local demo harness and benchmark bridge.
It is useful for showing how compiler metadata can meet memory planning,
kernel registration, scheduling, and backend sandboxing, but it should not be
described as the production distributed runtime layer. Runtime artifact
consumption, backend dispatch, distributed scheduling, PD split, cost modeling,
and runtime simulation belong in `heterogeneous-inference-runtime`.

### Kernels

Implemented in `include/kernels/` and `src/kernels/`.

- CPU kernels include matmul, tiled matmul, prefetch matmul, threaded variants, add, ReLU, fused matmul-add-ReLU, attention-like kernels, and layer norm style kernels.
- SIMD/architecture-specific variants exist where guarded by compile-time checks.
- Metal shader files exist under `metal/`, and separate Metal demo/profiling apps exist, but the generic runtime backend is not a full Metal graph executor.

### Compiler Planning

Implemented in `include/compiler/` and `src/compiler/`.

- `CostBasedPlanner` evaluates candidate backend assignments using `CostReport` entries when available.
- Cost modeling combines estimated FLOPs, estimated bytes, backend bandwidth/compute constants, launch overhead, and transfer cost.
- Missing cost data falls back to static hardcoded latency estimates.
- Subgraph partitioning and MLIR emitter code support the demo compiler-runtime story.

All static planner model values are estimates unless populated from actual runtime observations.

### MLIR Pass Plugin

Implemented separately in `mlir_passes/`.

- This is real MLIR C++ plugin infrastructure.
- It defines a `hir` dialect with runtime-facing fused ops and quantization-related ops.
- It defines a registered `cv` dialect for the current CV compiler milestone.
- It includes passes for canonicalization, matmul-bias-ReLU fusion detection, RMSNorm kernel selection, HIR lowering, and verification.
- It includes quantization planning, conservative INT8 island propagation, safe
  Q/DQ canonicalization, and capability-gated INT8 operator-selection metadata.
- It includes CV compiler passes for frontend normalization, shape inference, memory planning, and execution-domain planning.
- FileCheck tests under `mlir_passes/test/` cover positive and negative pass behavior.

This module is independent of the custom toy `Graph` IR. The bridge to the runtime story is through generated MLIR/text and JSON artifacts.

### CV Compiler Path

Current implemented flow:

```text
CV Dialect
  -> CVFrontendNormalizationPass
  -> CVShapeInferencePass
  -> CVMemoryPlanningPass
  -> CVExecutionDomainPlanningPass
  -> CVExecutionPlanBuilder
  -> CVExecutionPlanExporter
  -> emit-cv-execution-plan
  -> artifacts/apple_demo/cv_execution_plan.json
```

The frontend source before the CV dialect is future ONNX import:

```text
ONNX (future)
  -> CV Dialect
```

Implemented:

- Registered `cv` dialect.
- Seven CV operations for the raw CV graph milestone.
- Variadic `cv.detect_head`.
- Registered parsing path in CV tool/test contexts.
- Four Phase 1 CV compiler passes.
- C++ execution-plan builder/exporter.
- CLI artifact emission through `emit-cv-execution-plan`.
- Checked-in `cv_execution_plan.json` artifact.

Future work:

- ONNX importer.
- More CV operators.
- Dynamic-shape handling.
- Backend/kernel mapping.
- PocketChef visualization.

### Python Tooling

Implemented in `tools/` and `src/ml_graph_compiler_runtime/`.

- Tools generate, validate, and visualize compiler/runtime artifacts.
- LLM serving tools analyze a tiny GPT-style MLIR-like input, emit JSON plans, validate artifacts, and copy integration bundles.
- Visualization tools produce PNGs from JSON traces.

The Python artifact path is a planning/demo artifact generator. It does not run a real LLM inference server.

### Apps and Demos

Implemented in `apps/`.

- Demo executables exercise graph construction, passes, runtime dispatch, scheduling, allocators, attention/KV cache behavior, backend planning, and benchmarks.
- CUDA and Metal demos are optional/platform-specific.

## Implemented vs Simulated Behavior

Implemented:

- C++17 library build through CMake.
- Custom `Graph`/`Tensor`/`Node` IR.
- CPU op registry and CPU kernels for supported operations.
- Pass manager and several custom graph passes.
- Memory lifetime analysis and arena offset assignment.
- Execution plan and lowered graph data structures with JSON export.
- Cost-based planner that consumes `CostReport` entries and estimated models.
- Real MLIR plugin structure, HIR dialect, pass registration, TableGen-generated op/pass declarations, and FileCheck tests.
- Registered CV dialect, Phase 1 CV pass pipeline, CV execution-plan builder/exporter, and CV execution-plan artifact emission.
- Python artifact generation and validation for LLM serving plans.

Simulated, partial, or demo-only:

- `MockGPUBackend` uses CPU kernels.
- Generic `MetalBackend` logs dispatch and device info but does not execute graph kernels.
- Runtime replanning uses simple overload observations and predefined fallback assignments.
- CV and LLM runtime timelines are trace artifacts rather than traces from a production scheduler.
- The CV execution plan is a compiler artifact; it is not yet evidence of ONNX import, dynamic-shape support, backend kernel dispatch, or PocketChef visualization.
- Candidate serving plan latency and throughput numbers in artifact generation are estimated demo values.
- OpenXLA/IREE/Torch-MLIR paths are conditional probes and may skip if tools are absent.
