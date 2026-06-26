# Claude Code Handoff

## Repository Summary

This repository is a prototype ML graph compiler/runtime. It includes a custom C++ graph IR and runtime demos, CPU kernels, mock/partial heterogeneous backends, memory planning, cost-based backend planning, LLM serving artifact generation, and a separate real MLIR pass plugin for HIR fusion/lowering experiments.

Be careful to distinguish implemented behavior from simulations:

- Implemented: custom C++ `Graph`/`Tensor`/`Node` IR, CPU kernels, op registry, pass manager, memory planner, execution plan structures, cost planner, Python artifact generation/validation, and real MLIR plugin infrastructure.
- Simulated or partial: `MockGPUBackend`, generic `MetalBackend` graph execution, runtime replanning, many timeline artifacts, and serving latency/throughput values in generated plans.

High-value implemented compiler/runtime evidence now includes:

- HIR dialect ops, verifiers, canonicalization, fusion, conversion, and lowering tests under `mlir_passes/test/`.
- StableHLO-compatible textual subset import for RMSNorm and MatMul-Bias-ReLU patterns.
- JAX StableHLO export into the local HIR/LLVM pipeline when optional JAX dependencies are installed.
- Torch-MLIR tiny transformer probe and IREE comparison probe when optional dependencies are installed.
- HIR RMSNorm executable CPU path via the MLIR execution engine.
- Apple Silicon MLIR-to-Metal RMSNorm path with a real Metal kernel, generated execution plan, and dispatch validation when the MLIR pipeline has produced the required trace.
- CPU software-prefetch MatMul-Bias-ReLU backend candidate and benchmark executable.

Assume checked-in traces and artifacts may be stale unless regenerated.

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
