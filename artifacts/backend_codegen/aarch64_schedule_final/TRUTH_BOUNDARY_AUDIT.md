# Truth Boundary Audit (Stage 18C)

Audit method: grepped every Markdown file across all 6 stage artifact
directories plus the consolidated final report and summary for each
forbidden claim pattern, then manually inspected every match's
surrounding context.

## Forbidden claims -- verified absent as affirmative statements

| Claim | Matches found | Context |
|---|---|---|
| Universal uk4 optimality | 1 | Denial: "it does not claim uk4 is universally optimal" |
| Automatic production autotuning | 3 | Denial: "This is not: general production autotuning..." |
| Custom LLVM scheduler | 2 | Denial: "A custom LLVM scheduler or scheduling pass" (listed under "NOT claimed") |
| Custom register allocator | 1 | Denial: same list |
| Arbitrary loop scheduling/software pipelining | 2 each | Denial: same list |
| General loop interchange | 2 | Denial: same list |
| Calibrated mode or uk4 enabled by default | 3 | Denial: "Default behavior is unchanged" / "defaults to manual everywhere" |

Zero affirmative occurrences of any forbidden claim found in any
documentation file across Stages 10-18.

## Required affirmative claim -- verified present, correctly scoped

Located in `artifacts/backend_codegen/aarch64_schedule_final/summary.md`
and `DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md`:

> "For the tested Raspberry Pi 5 Cortex-A76 path, FP32, this tiled
> AArch64 matmul-bias-relu NEON microkernel, `schedule-unroll-k=4` was
> the measured fastest candidate in all 6 independently tested shape/tile
> domains."

Every required qualifier is present: target (Raspberry Pi 5 / Cortex-A76),
dtype (FP32), kernel (tested tiled NEON microkernel), scope (six
validated domains). No unqualified "uk4 is the best" statement exists
anywhere in the evidence chain.

## Files audited

All `.md` files under:
- `artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/`
- `artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/`
- `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model/`
- `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/`
- `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_multidomain/`
- `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/`
- `artifacts/backend_codegen/aarch64_schedule_final/`
- `DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md`

## Result: PASS
