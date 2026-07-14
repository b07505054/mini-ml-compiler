# Documentation Inventory

Last verified: 2026-07-14.

## Current Canonical

- `README.md`
- `ARCHITECTURE_CONSTITUTION.md`
- `CURRENT_STATE.md`
- `ARCHITECTURE_STATUS.md`
- `KNOWN_GAPS.md`
- `ARCHITECTURE_PATHS.md`
- `PROJECT_STATUS_2026.md`
- `PUBLICATION_STATUS.md`
- `PROJECT_MATURITY.md`
- `WHY_THIS_PROJECT.md`
- `publication_claims.json`

## Current Supporting

- `DOC/result/P1B_RASPBERRY_PI_ARM_CPU_KERNEL_REPORT.md`
- `DOC/result/P1C_RASPBERRY_PI_MULTI_CANDIDATE_REPORT.md`
- `DOC/result/P1C1_LOW_REGRET_STATIC_DEFAULT_REVIEW.md`
- `DOC/result/P1D_RASPBERRY_PI_THREAD_DECOMPOSITION_REPORT.md`
- `DOC/result/P1D1_OFFLINE_CALIBRATED_THREAD_POLICY_REPORT.md`
- `DOC/result/A1_MINIMAL_UNIFIED_IMPLEMENTATION_CANDIDATE_FOUNDATION.md` through `DOC/result/A6_IR_ROOTED_TRITON_PROVIDER_SHADOW_INTEGRATION.md`
- `docs/QUANTIZATION_CODESIGN.md` and `docs/VLLM_SERVING_CODESIGN.md` as supporting/historical quantization material, bounded by S1 canonical docs.

## Runtime-Owned Evidence Documents

- `heterogeneous-inference-runtime/RUNTIME_CONTRACT.md`
- `heterogeneous-inference-runtime/docs/E1_EXECUTORCH_RASPBERRY_PI_BASELINE_BRINGUP.md`
- `heterogeneous-inference-runtime/docs/E2_EXECUTORCH_CONTROLLED_HEAD_TO_HEAD.md`
- `heterogeneous-inference-runtime/docs/E2_1_EXECUTORCH_CORRECTNESS_REPAIRED_COMPARISON.md`
- `heterogeneous-inference-runtime/docs/E3_LIVE_COMPILER_XNNPACK_COMPARISON.md`
- `heterogeneous-inference-runtime/results/executorch_e*/**`

## Historical / Superseded

Older docs under `docs/`, `trace/`, `reports/`, `integration_bundle/`, and phase reports preserve their original phase evidence unless explicitly listed as canonical above. Do not rewrite raw evidence or historical measurements to match newer interpretation; add boundary notes instead.

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
