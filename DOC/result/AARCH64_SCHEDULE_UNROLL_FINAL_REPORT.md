# AArch64 Schedule-Unroll Achievement Report

This report records the final achievement for the Raspberry Pi 5 / Cortex-A76
AArch64 schedule-unroll slice. It intentionally summarizes outcomes, evidence,
scope, and truth boundaries rather than preserving a step-by-step stage diary.

## Achievement

The compiler can generate a narrow FP32 fused MatMul + Bias + ReLU AArch64
microkernel path from HIR through stock MLIR and LLVM, then choose an opt-in
schedule-unroll variant using exact-domain Raspberry Pi evidence.

For the measured Raspberry Pi 5 / Cortex-A76 domain set, `schedule-unroll-k=4`
was the fastest candidate in all six tested shape/tile domains. The calibrated
selector chose the measured winner whenever compatible exact-domain evidence was
available, while incompatible or missing evidence fell back to conservative
manual behavior.

## Evidence Summary

| Evidence | Result |
|---|---|
| Target | Raspberry Pi 5, Cortex-A76, FP32 fused MatMul + Bias + ReLU |
| Pipeline | HIR -> Linalg -> Transform dialect tiling/vectorization/unroll -> LLVM dialect -> LLVM IR -> stock LLVM 21.1.8 AArch64 object |
| Candidate set | Tiles `{8x8x8, 8x8x4, 4x8x8}` and unroll factors `{1, 2, 4}` |
| Correctness | All measured candidates were bit-exact against the reference within the recorded validation scope |
| Measured winner | `schedule-unroll-k=4` in 6/6 tested domains |
| Static-only ranking | Correct in 2/6 domains, showing static backend evidence alone is insufficient |
| Important finding | Spill/reload counts are real backend-cost signals but not reliable standalone runtime predictors |
| Selection boundary | Calibrated mode is explicit opt-in; default/manual behavior is unchanged |

The primary machine-readable summary is
`artifacts/backend_codegen/aarch64_schedule_final/summary.json`; the reviewer
summary is `artifacts/backend_codegen/aarch64_schedule_final/summary.md`.

## What This Proves

- A project-owned Transform-dialect schedule-unroll choice can be materialized
  into real AArch64 objects through unmodified MLIR/LLVM.
- The selected schedule identity is reflected in the compiled object, not just
  in a report.
- Exact-domain Raspberry Pi calibration is necessary for reliable winner
  selection in this slice.
- Static backend evidence is useful for diagnosis and risk labeling, but should
  not be used as a hard rejection rule without hardware confirmation.

## Truth Boundary

This is a scoped compiler/backend-codegen result, not a universal AArch64
autotuning claim.

- LLVM owns instruction selection, machine scheduling, register allocation,
  spill insertion, and assembly emission. The project does not implement a
  custom LLVM scheduler or register allocator.
- The measured result covers one operation family, one dtype, one CPU target,
  and the tested tile/unroll candidate set.
- No larger tile than `8x8x8`, no non-FP32 dtype, and no non-Cortex-A76 target
  is validated by this evidence.
- Calibrated selection is opt-in and evidence-gated. It is not the default
  compiler policy.
- The path is not a production runtime dispatch integration; it is compiler
  materialization plus hardware validation evidence.

## Reproduction Pointers

- Canonical summary and manifest:
  `artifacts/backend_codegen/aarch64_schedule_final/`
- Schedule/unroll result artifacts:
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/`,
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/`,
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model/`,
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/`,
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_multidomain/`,
  `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/`
- Main selector:
  `tools/select_and_compile_aarch64_matmul_schedule.py`
- Candidate/evidence model:
  `tools/aarch64_schedule_candidate_model.py`
- Raspberry Pi validation runner:
  `tools/run_aarch64_schedule_pi_validation.py`
- Focused tests:
  `tests/test_aarch64_schedule_comparison.py`,
  `tests/test_aarch64_schedule_pi_validation.py`,
  `tests/test_aarch64_schedule_candidate_model.py`,
  `tests/test_aarch64_schedule_selection.py`,
  `tests/test_aarch64_schedule_multidomain.py`,
  `tests/test_aarch64_schedule_boundary.py`
