# Data Flow

## Inputs

Primary inputs include:

- C++ demo graph construction in `apps/`.
- ONNX/model artifacts under `models/`.
- MLIR examples under `mlir/`.
- MLIR pass test inputs under `mlir_passes/test/`.
- JSON configs such as `configs/tiny_gpt_llm_config.json`.
- Existing trace/artifact JSON used by visualization and validation tools.

Assumption: many trace and artifact files are generated outputs and may be stale relative to current source unless regenerated.

## Custom Graph Runtime Flow

1. A demo app constructs a `Graph`.
2. Tensors are added to `graph.tensors`; nodes are added to `graph.nodes`.
3. Optional compiler passes run through `PassManager`.
4. Analysis/planning components infer shapes, costs, fusions, memory lifetimes, backend placement, or execution ordering.
5. Lowering creates `LoweredGraph` records where each `LoweredOp` contains:
   - `op_id`
   - `source_op_name`
   - `lowered_op_type`
   - `backend`
   - `inputs`
   - `outputs`
   - `memory_offset`
6. `ExecutionPlanBuilder` converts lowered ops to `ExecutionPlan` steps with dependencies and launch config.
7. `Executor` binds tensor runtime pointers through `ArenaAllocator`.
8. Runtime dispatch selects a backend.
9. The backend dispatches each node through the op registry or logs simulated behavior.
10. Optional profiling records per-node timings and can export trace JSON.

## MLIR Plugin Flow

The real MLIR plugin flow is separate from the custom C++ graph IR:

```text
MLIR input
  -> hir-canonicalize
  -> matmul-bias-relu-fusion
  -> rmsnorm-kernel-selection
  -> hir-fusion-lowering
  -> hir-verify-fused-ops
  -> HIR MLIR output
  -> Python/runtime JSON bridge
  -> lowered graph and execution plan artifacts
```

Important data structures:

- HIR dialect ops such as `hir.fused_matmul_bias_relu`, `hir.fused_rmsnorm`, `hir.qmatmul`, and `hir.fused_qmatmul_bias_relu`.
- Fusion and target metadata attributes carried in MLIR.
- Runtime JSON descriptors emitted by bridge tools.

Implemented: MLIR pass plugin and FileCheck-oriented tests.

Simulated or artifact-level: dispatch from MLIR output into the general heterogeneous runtime is represented through JSON plans and demo harnesses.

## CV Compiler Artifact Flow

The implemented CV compiler path is MLIR-first and separate from the older
custom C++ CV graph demo:

```text
ONNX (future)
  -> CV Dialect
  -> CVFrontendNormalizationPass
  -> CVShapeInferencePass
  -> CVMemoryPlanningPass
  -> CVExecutionDomainPlanningPass
  -> CVExecutionPlanBuilder
  -> CVExecutionPlanExporter
  -> emit-cv-execution-plan
  -> artifacts/apple_demo/cv_execution_plan.json
```

Current inputs:

- MLIR CV dialect examples/tests.
- Raw CV graph-shaped MLIR under `mlir/`.

Current outputs:

- Compiler annotations from frontend normalization, shape inference, memory
  planning, and execution-domain planning.
- A runtime-facing CV execution-plan JSON artifact.

Future inputs/outputs:

- ONNX import into the CV dialect.
- Additional CV operators.
- Dynamic-shape metadata.
- Backend/kernel mapping.
- PocketChef visualization of the exported plan.

## LLM Serving Artifact Flow

Documented pipeline:

```text
mlir/tiny_gpt_serving.mlir
  -> tools/analyze_llm_serving_mlir.py
  -> trace/llm_serving_compiler_analysis.json
  -> tools/emit_llm_artifacts_from_analysis.py
  -> artifacts/apple_demo/*.json
  -> tools/validate_llm_serving_artifacts.py
  -> trace/llm_artifact_validation_report.json
  -> integration_bundle/apple_demo_artifacts/*.json
```

Generated artifact structures include:

- `llm_graph_ir.json`: model metadata, operator list, request workload.
- `serving_execution_plan.json`: prefill/decode phases and runtime contract.
- `cv_execution_plan.json`: CV compiler artifact emitted from the registered
  CV dialect/pass/exporter path.
- `kv_cache_plan.json`: block size, block count, token capacity, dtype, allocation policy, prefix cache metadata.
- `memory_plan.json`: estimated prefill/decode memory, KV memory, temporary buffers, budget fit.
- `scheduling_plan.json`: queues, batch size, decode step size, dashboard signals.
- `candidate_execution_plans.json`: estimated backend plans and selection objective.
- `serving_framework_contract.json`: target serving policies and metrics.
- `memory_timeline.json`: planned memory events.
- `validation_manifest.json`: expected outputs.

These artifacts describe intended runtime behavior. They do not prove that a production serving runtime exists in this repo.

## Outputs

Common outputs:

- Console logs from demos and pass pipelines.
- JSON traces under `trace/`.
- JSON artifacts under `artifacts/apple_demo/`.
- Integration bundle JSON under `integration_bundle/apple_demo_artifacts/`.
- PNG visualizations at repo root and from visualization tools.
- Benchmark reports under `reports/` or `trace/`.

Output freshness is not guaranteed. Regenerate artifacts before relying on them for decisions.

## Metrics

Metric categories:

- Measured local benchmark timings: produced by benchmark executables or profiling scripts when run on the current machine.
- Runtime observed latency: used by `CostBasedPlanner` only when present in `CostReport`.
- Estimated static model metrics: FLOPs, bytes, launch cost, compute time, memory time, transfer cost, and planner latency estimates.
- Demo serving metrics: TTFT, TPOT, throughput, queue wait, KV pressure, and SLO fields named in contracts or generated plans.

Rules for interpreting metrics:

- Treat planner constants as estimated.
- Treat generated candidate serving latency/throughput as estimated.
- Do not compare benchmark numbers across machines unless environment, compiler flags, hardware, and workload are documented.
- Do not invent benchmark numbers in future docs or PRs.

## Important Data Structures

- `Graph`: vector-backed container for tensors and nodes.
- `Tensor`: float32 tensor plus lifetime/runtime allocation metadata.
- `Node`: op type plus input/output tensor ids.
- `ExecutionPlan`: ordered nodes for the basic executor.
- `LoweredGraph` and `LoweredOp`: runtime-facing lowered representation.
- `ExecutionPlan` and `ExecutionStep`: dependency-aware execution plan.
- `CostReport` and `CostReportEntry`: compiler/runtime cost metadata.
- `PlannerCandidate` and `PlannerOpCost`: candidate backend assignments and estimated/observed cost breakdowns.
- `LLMRequest`: serving request state for scheduler demos.
- KV cache structures: block-based allocation metadata for serving simulations.
