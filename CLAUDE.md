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
- `tools/run_mlir_pass_tests.sh` — runs the MLIR pass plugin FileCheck tests against `build-mlir`.
- `tools/run_llm_serving_artifact_pipeline.sh` — runs the full MLIR-to-LLM-artifacts pipeline end to end.
- `python3 tools/validate_compiler_artifacts.py` — validates generated `trace/cv_*.json` compiler/runtime artifacts.
- `python3 tools/validate_llm_serving_artifacts.py` — validates generated `artifacts/apple_demo` LLM serving artifacts.
- `tools/check_openxla_toolchain.py` — checks optional StableHLO tooling availability.

## Documentation Hierarchy

Truth must flow in the following order:

Code
↓
Artifacts
↓
README.md
↓
CLAUDE.md
↓
docs/

Lower levels must never contradict higher levels.

Documentation must describe reality rather than invent behavior.

If uncertainty exists, trust code and generated artifacts.

Never exaggerate capabilities.

Never claim production behavior unless code and artifacts support it.

## README Contract

README.md exists to answer:

1. What is it?
2. Why is it interesting?
3. How do I run it?
4. What results does it produce?

README should emphasize user-facing understanding.

Avoid implementation details unless necessary.

Avoid maintenance instructions.

## CLAUDE.md Contract

CLAUDE.md exists to answer:

1. How do I maintain it?
2. What commands are canonical?
3. Which components are implemented?
4. Which components are simulated?
5. Which validation commands must pass?
6. What files should not be changed casually?

CLAUDE.md is intended for maintainers and future AI agents.

## docs/ Contract

docs/ exists to answer:

1. Why is it designed this way?
2. What tradeoffs were made?
3. What is measured versus modeled?
4. What assumptions exist?
5. What limitations remain?
6. What future work is possible?

docs/ explains architecture and rationale rather than usage.

## Documentation Principles

Code > Artifacts > README > CLAUDE.md > docs/

Never reverse this order.

Never infer unsupported features.

Never create claims unsupported by code or artifacts.

Prefer conservative wording.

Call synthetic benchmarks synthetic.

Call simulated systems simulated.

Distinguish measured behavior from modeled behavior.

## Git Authorship Policy

The user is the sole maintainer and owner of this repository.

AI agents may modify files as requested.

AI agents must not add AI authorship metadata.

Never add:

* Co-Authored-By entries
* Co-authored-by trailers
* Claude authorship metadata
* AI signatures
* Generated-by-AI footers
* any metadata that makes an AI system appear as a repository contributor

Commit policy:

* By default, do not run git commit.
* If the user explicitly asks in the current conversation to commit, an AI agent may run git add and git commit.
* Commits created by an AI agent must use the user's configured git author and committer identity.
* Commit messages must not mention AI authorship unless the user explicitly asks.
* Before committing, show git status and the staged diff summary when practical.

Push policy:

* By default, do not run git push.
* Only run git push if the user explicitly asks in the current conversation.
* Never force-push unless the user explicitly asks for a force push and the reason is explained.

History policy:

* Do not create branches, rewrite history, rebase, reset, or amend commits unless the user explicitly asks in the current conversation.
* Never rewrite public history without explicit user approval.

Ownership rule:

* The user remains the sole author/maintainer for portfolio presentation purposes.
* No AI system should appear as a repository contributor.
