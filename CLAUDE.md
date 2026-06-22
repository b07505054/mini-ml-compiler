# Claude Code Handoff

## Repository Summary

This repository is a prototype ML graph compiler/runtime. It includes a custom C++ graph IR and runtime demos, CPU kernels, mock/partial heterogeneous backends, memory planning, cost-based backend planning, LLM serving artifact generation, and a separate real MLIR pass plugin for HIR fusion/lowering experiments.

Be careful to distinguish implemented behavior from simulations:

- Implemented: custom C++ `Graph`/`Tensor`/`Node` IR, CPU kernels, op registry, pass manager, memory planner, execution plan structures, cost planner, Python artifact generation/validation, and real MLIR plugin infrastructure.
- Simulated or partial: `MockGPUBackend`, generic `MetalBackend` graph execution, runtime replanning, many timeline artifacts, and serving latency/throughput values in generated plans.

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

For C++ changes, run the one-shot baseline check:

```bash
scripts/check.sh
```

This runs `cmake -S . -B build`, `cmake --build build`, and
`ctest --test-dir build --output-on-failure`, and prints whether
`trace/metal_rmsnorm_execution_plan.json` is present before running CTest.

Note on `metal_rmsnorm_plan_dispatch`: this test requires
`trace/metal_rmsnorm_execution_plan.json`, which is produced only by
`tools/run_metal_rmsnorm_compiler_pipeline.sh` (a separate MLIR pipeline
requiring `mlir-opt` and a built `mlir_passes` plugin). In a clean baseline
build without that artifact, CTest reports this test as **skipped**
(`SKIP_RETURN_CODE 77`), not failed or passed. Treat "Not Run (Skipped)" as
expected in that case — it does not mean the dispatch logic was verified. To
exercise the real check, build `mlir_passes` and run
`tools/run_metal_rmsnorm_compiler_pipeline.sh` first.

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
- `tools/run_metal_rmsnorm_compiler_pipeline.sh` — produces `trace/metal_rmsnorm_execution_plan.json`, required for the `metal_rmsnorm_plan_dispatch` CTest target to run instead of skip.
- `tools/run_mlir_pass_tests.sh` — runs the MLIR pass plugin FileCheck tests against `build-mlir`.
- `tools/run_llm_serving_artifact_pipeline.sh` — runs the full MLIR-to-LLM-artifacts pipeline end to end.
- `python3 tools/validate_compiler_artifacts.py` — validates generated `trace/cv_*.json` compiler/runtime artifacts.
- `python3 tools/validate_llm_serving_artifacts.py` — validates generated `artifacts/apple_demo` LLM serving artifacts.
- `tools/check_openxla_toolchain.py` — checks optional StableHLO tooling availability.

