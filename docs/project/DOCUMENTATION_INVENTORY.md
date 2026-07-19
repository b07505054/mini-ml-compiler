# Documentation Inventory

Last verified: 2026-07-14.

## Current Canonical

- `README.md`
- `docs/project/ARCHITECTURE_CONSTITUTION.md`
- `docs/project/CURRENT_STATE.md`
- `docs/project/ARCHITECTURE_STATUS.md`
- `docs/project/KNOWN_GAPS.md`
- `docs/project/ARCHITECTURE_PATHS.md`
- `docs/project/PROJECT_STATUS_2026.md`
- `docs/project/PUBLICATION_STATUS.md`
- `docs/project/PROJECT_MATURITY.md`
- `docs/project/WHY_THIS_PROJECT.md`
- `publication_claims.json`
- `docs/QUANTIZATION_CODESIGN.md`
- `docs/EXECUTION_PLAN_SCHEMA.md`
- `docs/architecture.md`
- `mlir_passes/README.md`

## Current Supporting

- `DOC/result/P1D1_OFFLINE_CALIBRATED_THREAD_POLICY_REPORT.md`
- `DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md`
- `docs/VLLM_SERVING_CODESIGN.md` as supporting serving material, bounded by the canonical truth boundary.

## Runtime-Owned Evidence Documents

- `heterogeneous-inference-runtime/RUNTIME_CONTRACT.md`
- `heterogeneous-inference-runtime/docs/E1_EXECUTORCH_RASPBERRY_PI_BASELINE_BRINGUP.md`
- `heterogeneous-inference-runtime/docs/E2_EXECUTORCH_CONTROLLED_HEAD_TO_HEAD.md`
- `heterogeneous-inference-runtime/docs/E2_1_EXECUTORCH_CORRECTNESS_REPAIRED_COMPARISON.md`
- `heterogeneous-inference-runtime/docs/E3_LIVE_COMPILER_XNNPACK_COMPARISON.md`
- `heterogeneous-inference-runtime/docs/MEASURED_BASELINES.md`
- `heterogeneous-inference-runtime/docs/QUANTIZATION_ANALYSIS.md`
- `heterogeneous-inference-runtime/results/executorch_e*/**`
- External Slice 3E-3G workspaces supplied to the import tooling through
  explicit command-line paths; these local binary/raw-evidence workspaces are
  not repository content.

## Historical / Superseded

Older docs under `docs/`, `trace/`, `reports/`, `integration_bundle/`, and phase reports preserve their original phase evidence unless explicitly listed as canonical above. Do not rewrite raw evidence or historical measurements to match newer interpretation; add boundary notes instead.

Root-level slice diaries, finalization audits, changelogs, and stage-by-stage
writeups are intentionally not canonical. `DOC/result/` should contain only
final achievement/result reports for current main lines, not intermediate phase
logs or superseded analysis snapshots.

## Pre-Edit Contradiction Table

| Finding | Pre-S1 location | Resolution |
|---|---|---|
| P1D.1 described as not started | old compiler `README.md` | corrected: P1D.1 complete and integrated |
| Stale pre-E3 commit/status lines | compiler canonical docs | updated to S1 heads |
| E2.1 presented as narrow ExecuTorch-faster result | runtime E2.1 doc and older summaries | reclassified as implementation-stack comparison |
| E3 absent from canonical public narrative | compiler/portfolio docs | added E3 same-XNNPACK section |
| Quantization overclaim risk | compiler/runtime/portfolio docs | downgraded to AWQ artifact + vLLM materialization + serving evidence; no accuracy claim |
| Capability DB overclaim risk | compiler/capabilities/portfolio docs | clarified partial ownership and synchronization gap |
| Placeholder `ExecuTorch C++ ~= 5.7 ms` risk | S0 finding | forbidden as measured evidence; no current canonical use |
| Runtime-boundary ambiguity | compiler/runtime docs | restated: Runtime validates/executes, does not choose |
| Slice 3 quantization described as planning/metadata only | quantization, architecture, MLIR, and runtime docs | corrected for the fused-operator scope: calibration, packed weights, Q/DQ integer IR, kernel lowering, ExecutionPlan validation, and Pi execution are implemented |
| Historical ExecuTorch `1.49 ms` row mixed with unrelated workloads | runtime README | removed; controlled shape-specific Slice 3F/3G values now live in the existing ExecuTorch and measured-baseline sections |
