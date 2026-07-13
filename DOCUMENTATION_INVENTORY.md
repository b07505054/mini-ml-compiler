# Documentation Inventory

Last verified: 2026-07-13\nSource host: GPU Linux /home/allen/Desktop/Project/ml-graph-compiler-runtime\nVerified compiler HEAD: e30c54cc477aab771525661d4dfc3c53419cd8a9 (master, ahead 1 of origin/master)\nVerified runtime HEAD: f4cc98bc93e1e8e5ecea32ffb0779b0a5c801097 (main, ahead 1 of origin/main)\nVerified capabilities HEAD: 84cf1d229788390f3b95254416636672fabe8d20 (main, origin-aligned)\nIVP source: Mac-only divergent checkout at 3f11a0422123e88eab7f90cff06d8ab7a7d48f24, ahead 1 / behind 2\nRaspberry Pi: execution/evidence target only; no canonical source repositories verified there\n

This inventory classifies relevant documentation by status. Generated benchmark reports and raw evidence are intentionally not rewritten.

## Current Canonical

- `ARCHITECTURE_CONSTITUTION.md`
- `CURRENT_STATE.md`
- `ARCHITECTURE_STATUS.md`
- `KNOWN_GAPS.md`
- `PROJECT_STATUS_2026.md`
- `ARCHITECTURE_PATHS.md`
- `DOCUMENTATION_INVENTORY.md`

## Current Supporting

- `README.md`
- `mlir_passes/README.md`
- `docs/EXECUTION_PLAN_SCHEMA.md`
- `docs/RUNTIME_KERNEL_CONTRACT.md`
- `docs/GENERIC_GRAPH_IR_SCHEMA.md`
- `docs/GENERIC_GRAPH_IR_TO_MLIR_LOWERING_CONTRACT.md`
- `docs/MLIR_COMPILER_PIPELINE_SUMMARY.md`
- `docs/QUANTIZATION_CODESIGN.md`
- `docs/VLLM_SERVING_CODESIGN.md`
- `DOC/result/P1B_RASPBERRY_PI_ARM_CPU_KERNEL_REPORT.md`
- `DOC/result/P1C_RASPBERRY_PI_MULTI_CANDIDATE_REPORT.md`
- `DOC/result/P1C1_LOW_REGRET_STATIC_DEFAULT_REVIEW.md`
- `DOC/result/P1D_RASPBERRY_PI_THREAD_DECOMPOSITION_REPORT.md`
- Triton result reports under `DOC/result/` and `trace/` as evidence reports.

## Historical / Superseded

- `DOC/result/RASPBERRY_PI5_P1A_TARGET_PROFILE.md` reflects the P1A target-profile phase and older commit heads.
- `docs/YOLOSEG_EXECUTION_PLAN_ENVIRONMENT_AND_RUNTIME_SCOPE_AUDIT.md` reflects a Mac-host audit before GPU Linux/P1D state.
- `docs/future_work.md` is historical planning guidance, not current canonical architecture.
- `docs/architecture.md`, `docs/data_flow.md`, `docs/design_decisions.md`, `docs/future_work.md`, `docs/technical_debt.md` are supporting/historical unless explicitly linked from a canonical doc.
- `integration_bundle/*` and Apple demo docs are path-specific historical/experimental evidence.

## Generated / Non-documentation Evidence

- `trace/**/*.md`
- `reports/**/*.md`
- `artifacts/**` generated summaries
- benchmark run reports under `trace/matmul_postop_*`

These are evidence/report artifacts. Preserve them as generated history; do not rewrite them to match current architecture prose.

## Incorrect/Stale Claims Corrected By Canonical Docs

- Mac-only “no CUDA/no AutoAWQ” statements are historical host-scoped claims, not project-wide current facts.
- Absence from `runtimeKernels[]` does not imply AWQ is non-executable.
- Absence of a Triton pass in `compile-for-target` does not imply no real Triton decision pipeline.
- P1D threshold policy is evidence-backed offline research, not shipped compiler behavior.
- Raspberry Pi is an execution/evidence target, not a canonical repository host.
- ExecuTorch comparison is not fully integrated.
