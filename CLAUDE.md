# Claude Code Handoff

## Repository Summary

This repository is a prototype **execution-planning ML compiler**. It reuses existing MLIR infrastructure for generic graph semantics, lifts backend-relevant regions into a decision-oriented HIR, and produces an explainable hardware-aware execution plan consumed by heterogeneous runtimes.

The primary compiler story lives in `mlir_passes/`. The custom C++ runtime in `src/` and `apps/` is a local demo harness and benchmark bridge — it is not the production distributed runtime layer. New runtime/deployment features belong in the sibling `heterogeneous-inference-runtime` project.

Be careful to distinguish implemented behavior from simulations:

- Implemented: HIR/CV/Serving MLIR dialects, 16-pass serving pipeline, custom C++ `Graph`/`Tensor`/`Node` IR, CPU kernels, op registry, pass manager, memory planner, execution plan structures, cost planner, Python artifact generation/validation.
- Simulated or partial: `MockGPUBackend`, generic `MetalBackend` graph execution, runtime replanning, many timeline artifacts, and serving latency/throughput values in generated plans.

High-value implemented compiler evidence:

- HIR dialect ops, verifiers, canonicalization, fusion, conversion, and lowering tests under `mlir_passes/test/`.
- 16-pass hardware-aware serving pipeline (serving phase → KV layout → replay eligibility → execution provider → representation → layout → boundary → weight classification → quantization strategy → kernel availability → lowering decision → quantized boundary refinement → alternative lowering → candidate generation → candidate evaluation → plan selection).
- StableHLO-compatible textual subset import for RMSNorm and MatMul-Bias-ReLU patterns.
- HIR RMSNorm executable CPU path via the MLIR execution engine.
- Apple Silicon MLIR-to-Metal RMSNorm path with a real Metal kernel, generated execution plan, and dispatch validation when the MLIR pipeline has produced the required trace.
- CPU software-prefetch MatMul-Bias-ReLU backend candidate and benchmark executable.

Assume checked-in traces and artifacts may be stale unless regenerated.

## Architecture

### Execution-Planning Compiler Pipeline

```text
ONNX / model graph
  ->
Frontend (import / normalize)
  ->
Shared capability profiles
  (HardwareCapability / BackendCapability / KernelLibraryCapability)
  ->
Static optimization
  (Decision Engine: 16-pass serving pipeline)
  ->
ExecutionPlan  [compiler deliverable]
```

Hardware Model and Decision Engine are **sequential modules** in one compiler
pipeline — not parallel tracks.

### Compiler Responsibility

The compiler owns:

- Import / normalize model graph into MLIR
- Lift backend-relevant regions into HIR
- Read shared hardware/backend/kernel capability profiles
- Run hardware-aware static optimization passes (the Decision Engine)
- Select per-op execution plans using static penalty scoring
- Export `execution_plan.json` — the compiler deliverable

The compiler produces the **theoretical best static solution** given declared
capability profiles. It has no visibility into runtime state.

### Compiler Does NOT

- Perform runtime scheduling or dynamic execution ordering
- Manage memory at runtime (no allocation, deallocation, or arena management)
- Execute kernels or dispatch CUDA/Metal/CoreML APIs
- Perform speculative decoding or KV cache management
- Make deployment policy decisions (batching, prefill/decode split, SLO targets)
- Claim measured latency or runtime speedup unless a benchmark produced them
- Materialize IR unless a pass explicitly states it does so

Those responsibilities belong entirely to `heterogeneous-inference-runtime`.

### 16-Pass Serving Pipeline

Each pass answers one planning question and writes structured attrs. No pass
selects a winner until `PlanSelectionPass`. No pass materializes IR.

| Pass | Question answered | truth_boundary |
|---|---|---|
| ServingPhaseAnalysis | Prefill or decode? | declared_profile |
| KVLayoutPlanning | KV cache layout and block size? | declared_profile |
| ReplayEligibilityPass | CUDA-graph replayable? | declared_profile |
| ExecutionProviderPlanning | Which backend/API? | declared_profile |
| RepresentationPlanning | What dtype/layout does the backend expect? | declared_profile |
| LayoutPlanning | Which specific tensor layout per op? | declared_profile |
| BoundaryPlanning | Which boundary ops (cast/dequant/layout_transform) required? | declared_profile |
| WeightClassificationPlanning | Weight tensors constant or runtime-variable? | declared_profile |
| QuantizationStrategyPlanning | What quantization strategy per op? | declared_profile |
| KernelAvailabilityPlanning | Does a kernel exist for (op, dtype, layout, quant)? | declared_profile |
| LoweringDecisionPlanning | Direct lower / fallback / unsupported? | declared_profile |
| QuantizedBoundaryRefinement | Refine weight dequant after lowering decisions. | declared_profile |
| AlternativeLoweringPlanning | Legal alternatives when exact kernel is missing? | alternative_lowering_static_not_materialized_not_cost_evaluated |
| CandidateGenerationPass | What executable candidates exist per op? | candidate_generation_static_constraints_not_cost_evaluated |
| CandidateEvaluation | Static relative penalty score per candidate? | candidate_evaluation_static_penalty_not_measured_latency |
| PlanSelection | Which candidate wins? | plan_selection_static_penalty_not_measured_runtime |

### Decision Flow

```text
Backend selected
  -> Representation chosen
  -> Layout planned
  -> Boundary needs identified
  -> Weights classified
  -> Quantization strategy chosen
  -> Kernel availability checked
  -> Lowering decision made
  -> Quantized boundaries refined
  -> Alternative lowering candidates generated (if exact kernel missing)
  -> Candidates evaluated with static penalty (no measured latency)
  -> Plan selected (lowest-penalty evaluated candidate wins)
  -> Execution plan exported
```

Fallback is **not** chosen immediately after a kernel miss. It is a last-resort
candidate considered only after direct kernel, algebraic decomposition,
representation conversion, and other alternative paths have been found unavailable
or invalid.

### Hardware Model Layers

Three distinct layers with different scopes and truth_boundary values:

- **HardwareCapability**: theoretical hardware support from public docs or declared device profiles. `truth_boundary = public_docs | declared_profile`.
- **BackendCapability**: what the backend/API/compiler actually exposes, which may be more restrictive than hardware. `truth_boundary = declared_profile`.
- **KernelLibraryCapability**: actual kernel availability for a specific (op, dtype, layout, quant_mode) tuple. `truth_boundary = declared_profile | measured_profile`.

No layer claims measured hardware performance unless a benchmark explicitly produced it.

### Materialization Status

Planning is implemented. IR materialization is intentionally deferred.

Planning means (implemented):
- annotate `selected_plan.*` per op
- annotate `quant.*`, `kernel.*`, `lowering.*`, `alternative.*`
- export execution plan JSON / runtime contract

Materialization would mean (deferred):
- inserting `hir.cast`
- inserting `hir.dequantize` / `hir.requantize`
- inserting `hir.layout_transform`
- replacing `gelu` with primitive ops
- rewriting graph structure

No pass in the current pipeline materializes IR. IR structure is read-only
until a future materialization pass explicitly states otherwise.

### Apple/CoreML Quantization Demo

The Apple/CoreML demo demonstrates:
- Apple/CoreML-style quantization lowering decisions (`declared_profile`)
- weight classification preventing incorrect `weight_only_int8` on runtime activations
- kernel availability vs backend capability separation
- per-op plan selection and export

The demo does NOT claim:
- real ANE kernel internals or layout proofs
- measured Core ML performance
- a CoreML profile as evidence of ANE internal execution
- fallback is selected immediately after a kernel miss

### Shared Capability Profiles

The compiler and runtime intentionally read from the same capability data. Both
repositories are currently centered on the same development machine. The three
shared layers:

- **HardwareCapability** — theoretical hardware support. `truth_boundary = public_docs | declared_profile`. Neither repo may claim measured performance here unless a benchmark produced it.
- **BackendCapability** — what the backend/API actually exposes, which may be more restrictive than hardware. `truth_boundary = declared_profile`.
- **KernelLibraryCapability** — actual kernel availability for (op, dtype, layout, quant_mode). `truth_boundary = declared_profile | measured_profile`.

The runtime may extend these with a fourth layer:

- **MeasuredSupport** — runtime-measured latency/throughput evidence collected during execution. `truth_boundary = measured_profile`. This layer belongs to the runtime only; the compiler never writes it.

Neither repository maintains separate hardware truths. All shared profile data
must be consistent across both.

### Truth Boundary

The compiler makes only static optimization claims:

- No measured latency values unless an explicit benchmark produced them.
- No runtime speedup claims.
- No deployment performance claims.
- No claims about dynamic execution quality (batching efficiency, SLO hit rate, etc.).

Every planning annotation carries a `truth_boundary` field. If no benchmark
exists, the claim is `declared_profile` or `public_docs` — not `measured_profile`.

## Environment Policy

- Use the repo's `.venv` for all Python tooling; do not rely on a system Python.
- Install optional Python dependencies (e.g. `jax[cpu]`, `torch_mlir`) into `.venv` only when a specific demo requires them.
- For MLIR work, use a single consistent LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck` (see MLIR Notes).
- Treat missing toolchain/environment dependencies as expected in a fresh checkout; report them clearly rather than working around them silently.

## Working Rules

- Do not modify source code unless explicitly asked.
- Preserve user changes in the working tree.
- Do not refactor opportunistically.
- Do not change tests unless the task explicitly calls for it.
- Do not invent benchmark numbers.
- If metrics are estimated, label them estimated.
- Explain assumptions explicitly.
- Explain changes after implementation.

## Coding Preferences

- Python 3.11.
- Prefer dataclasses.
- Avoid unnecessary classes.
- Keep functions under 100 lines when practical.
- Use type hints.
- Prefer simple modular design.
- Avoid over-engineering.
- Composition over inheritance.
- No giant classes.
- Write tests for non-trivial logic.
- Run tests after changes.

## C++ Notes

- The main project is C++17 and built with CMake.
- The core library target is `mlcompiler`.
- Platform-specific paths exist for Apple Metal and optional CUDA.
- The generic backend API is intentionally small and not sufficient for full device runtime semantics.
- `MockGPUBackend` dispatches CPU kernels.
- Generic `MetalBackend` logs dispatch/device information; do not describe it as full graph-kernel execution.

## MLIR Notes

- `mlir_passes/` is separate from the custom toy graph IR.
- It uses LLVM/MLIR CMake packages, TableGen, a `hir` dialect, pass registration, and FileCheck tests.
- Use the same LLVM/MLIR build for CMake, `mlir-opt`, and `FileCheck`.
- Environmental toolchain failures are common; report them clearly.

## Artifact Notes

- `trace/`, `artifacts/apple_demo/`, and `integration_bundle/apple_demo_artifacts/` contain generated JSON outputs.
- Treat candidate serving latency/throughput and planner values as estimated unless a current benchmark produced them.
- Prefer regenerating artifacts with the relevant tool script before relying on them.
- Keep schema/version changes explicit.

## Suggested Verification

`scripts/check.sh` is the canonical baseline validation entrypoint. It runs:

```bash
scripts/check.sh
```

It does not install any system dependencies (no `brew`/`apt-get` calls) — if a
required tool is missing, the relevant step fails and that failure should be
reported as-is, not worked around.

### Known baseline validation gap: `metal_rmsnorm_plan_dispatch`

On Apple builds, `bash scripts/check.sh` currently surfaces a known issue in
the `metal_rmsnorm_plan_dispatch` test (`apps/run_metal_rmsnorm_plan.mm`,
registered in `CMakeLists.txt`): it requires
`trace/metal_rmsnorm_execution_plan.json`, which is generated only by the MLIR
pipeline (`tools/run_metal_rmsnorm_compiler_pipeline.sh`), not by the baseline
CMake build. In a baseline checkout that file is absent, so this test
is reported as skipped rather than exercising the dispatch path.

- Do not fabricate `trace/metal_rmsnorm_execution_plan.json` to make this test
  pass — it is an MLIR-pipeline-generated artifact, not baseline CMake output.
- Do not describe the dispatch logic as verified unless the MLIR pipeline has
  produced `trace/metal_rmsnorm_execution_plan.json` and the dispatch test has
  run against that artifact.

For MLIR changes, when LLVM/MLIR tools are installed:

```bash
cmake -S mlir_passes -B build-mlir \
  -DMLIR_DIR="$(brew --prefix llvm)/lib/cmake/mlir" \
  -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"
cmake --build build-mlir
tools/run_mlir_pass_tests.sh
```

For LLM serving artifacts:

```bash
tools/run_llm_serving_artifact_pipeline.sh
```

If a verification command cannot run because dependencies are missing, state that directly.

## Common Commands

- `scripts/check.sh` — canonical CMake build + CTest baseline check (see Suggested Verification above).
- `tools/run_metal_rmsnorm_compiler_pipeline.sh` — produces `trace/metal_rmsnorm_execution_plan.json`, required by the `metal_rmsnorm_plan_dispatch` CTest target.
- `tools/run_metal_rmsnorm_end_to_end.sh` — runs the Apple Silicon MLIR-to-Metal RMSNorm path end to end when the toolchain is available.
- `python3 tools/validate_metal_rmsnorm_path.py` — validates Metal RMSNorm generated artifacts and dispatch evidence.
- `python3 tools/run_stablehlo_subset_pipeline.py` — exercises the StableHLO-compatible textual subset importer.
- `.venv/bin/python tools/run_jax_stablehlo_pipeline.py` — exports supported JAX functions to StableHLO and runs the local HIR/LLVM path when JAX is installed.
- `.venv/bin/python tools/run_torch_mlir_tiny_transformer_probe.py` — probes Torch-MLIR export when `torch_mlir` is installed.
- `.venv/bin/python tools/run_iree_stablehlo_subset_comparison.py` — runs the optional IREE comparison path.
- `build/benchmark_prefetch_matmul` — benchmarks the CPU software-prefetch MatMul-Bias-ReLU candidate after CMake build.
- `tools/run_mlir_pass_tests.sh` — runs the MLIR pass plugin FileCheck tests against `build-mlir`.
- `tools/run_llm_serving_artifact_pipeline.sh` — runs the full MLIR-to-LLM-artifacts pipeline end to end.
- `python3 tools/validate_compiler_artifacts.py` — validates generated `trace/cv_*.json` compiler/runtime artifacts.
- `python3 tools/validate_llm_serving_artifacts.py` — validates generated `artifacts/apple_demo` LLM serving artifacts.
- `tools/check_openxla_toolchain.py` — checks optional StableHLO tooling availability.

## Compiler Core Policy: Zero Python / Zero JSON

The compiler core must be C++/MLIR-first.

Python and JSON are allowed only at the edges:
- Python: legacy prototypes, validation tooling, debug scripts, regression comparison, runtime/demo helpers.
- JSON: debug dumps, validation artifacts, temporary runtime/demo interchange.

They must not be the source of truth for new compiler functionality.

New compiler-core work should be implemented in:
- C++
- MLIR passes
- TableGen where appropriate
- FileCheck / CTest tests

Runtime metadata must originate from:
MLIR attributes or C++ RuntimeMetadataContract.

If JSON is emitted, it is a serialization/debug format only.

Do not add new Python+JSON planner logic as the primary compiler implementation.

## Portfolio-Level Policy

When this repository is maintained inside the `systems-portfolio` wrapper, follow the root `CLAUDE.md` for shared documentation hierarchy, benchmark honesty, and Git authorship rules. Keep this file focused on repository-specific capabilities, truth boundaries, and validation commands.
