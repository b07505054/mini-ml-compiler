# A2 P1D.1 Unified Candidate Migration

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler base before A2: `b58aeb2ac4a76b29976ecb324a55736ce6c6ae31`
Runtime reference: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6`

## Verdict

`PASSED_P1D1_UNIFIED_CANDIDATE_MIGRATION`

A2 migrates the P1D.1 Raspberry Pi thread-schedule decision into the A1
compiler-internal `ImplementationCandidate` core. Runtime production code,
Runtime schemas, kernel code, target profiles, policy values, and evidence JSON
are unchanged.

## Pre-A2 Bypass Path

Before A2, `KernelSelectionPass` selected the concrete portable CPU kernel and
then called `resolveThreadSchedule()`. That function:

1. read the offline thread policy attrs,
2. found serial and 4-thread split-M declarations directly in the selected
   RuntimeKernelDescriptor,
3. computed `matmul_mnk`,
4. branched directly on threshold `262144`, and
5. wrote `thread_schedule.*` attrs.

No `ImplementationCandidate` represented the serial or parallel schedule
choices, and no `PolicyResult` owned the selected schedule identity.

## Candidate Identity

A2 hardens candidate identity for current schedule decisions. A thread-schedule
candidate ID deterministically includes:

- semantic target: current fused op short name, e.g. `fused_matmul_bias_relu`
- scope: `fused_region`
- implementation kind: `portable_cpu_opaque_kernel_thread_schedule`
- kernel ID: `portable_fused_matmul_bias_relu_bm32_bn128_bk32`
- schedule tuple: `thread_count`, `partition_axis`, `partition_strategy`
- target profile ID where needed for this profile-local policy

Example serial:

`fused_matmul_bias_relu:portable_cpu_opaque_kernel_thread_schedule:kernel=portable_fused_matmul_bias_relu_bm32_bn128_bk32:threads=1:axis=none:strategy=serial:target=raspberry-pi5-cortex-a76-cpu`

Example parallel:

`fused_matmul_bias_relu:portable_cpu_opaque_kernel_thread_schedule:kernel=portable_fused_matmul_bias_relu_bm32_bn128_bk32:threads=4:axis=m:strategy=contiguous_chunks:target=raspberry-pi5-cortex-a76-cpu`

Cost, feasibility, evidence, and selection state are not part of candidate ID.

## IR Rooting

Current rooting uses the existing deterministic op short name from the MLIR op:

- IR op: `hir.fused_matmul_bias_relu`
- semantic target: `fused_matmul_bias_relu`
- scope kind: `fused_region`

This is sufficient for the current single fused-op policy. External provider ID
bridging remains unresolved.

## Candidate Enumeration

A2 enumerates only the two live P1D.1 policy candidates:

| Candidate | Kernel | Threads | Axis | Strategy |
|---|---|---:|---|---|
| Serial | `portable_fused_matmul_bias_relu_bm32_bn128_bk32` | 1 | `none` | `serial` |
| Parallel | `portable_fused_matmul_bias_relu_bm32_bn128_bk32` | 4 | `m` | `contiguous_chunks` |

Declared 2-thread and split-N schedules remain Runtime capabilities but are not
active P1D.1 policy candidates.

## Feasibility

Feasibility is attached before policy selection.

Serial candidate feasibility requires:

- selected kernel descriptor exists
- fused op target matches current IR op
- static shape path reached kernel selection
- dtype/profile/kernel policy contract remains compatible or safe serial
  fallback is allowed
- serial schedule is declared

Parallel candidate feasibility additionally requires:

- policy profile/op/dtype/kernel/metric/boundary rule match
- static `M/N/K` metric is available
- 4-thread split-M schedule is declared
- `physicalComputeUnits >= 4`

When parallel is rejected or deferred, serial is selected only through the
serial candidate if it is declared.

## PolicyResult

The P1D.1 policy now records:

- selected candidate ID
- considered candidate IDs
- rejected candidate IDs and reasons
- policy ID/version
- metric name and value
- threshold
- evidence reference/hash
- truth boundary

The selected `ThreadSchedule` is materialized from the selected candidate's
schedule tuple.

## Materialization Boundary

A2 remains an Execution Contract decision:

- The compiler selects among opaque prebuilt portable CPU kernel schedules.
- The compiler does not generate parallel loop Implementation IR.
- The selected candidate materializes into existing `thread_schedule.*` attrs.
- ExecutionPlan JSON remains semantically equivalent to P1D.1 baseline.
- Runtime validates and executes the exact contract unchanged.

## Duplicate Authority Removal

The old direct threshold branch is no longer a second authority. The resolver now
creates schedule candidates, evaluates candidate feasibility, applies the
unchanged policy to feasible/fallback candidates, and materializes the selected
candidate. The threshold branch no longer independently constructs a schedule
after selection.

## Baseline and Plan Equivalence

Baseline before A2:

- A1 focused CTest: pass.
- P1D.1 Python tests: 15/15 pass.
- Full compiler CTest: 21/21 pass.

Normalized ExecutionPlan hashes before and after A2 are identical:

| Shape | Hash | Schedule |
|---|---|---|
| 8x8x8 | `817e104909ae3780b2db1aa65aa70553ab0029072736b3cc833d62de4911d811` | 1 thread, serial |
| 64x64x64 | `b9dc9ea4ab4b02dd2cc7a10f53c6299e63224815322a9010f0e75ca76f96ce37` | 4-thread split-M |
| 256x256x256 | `5fa7a80f9fffcb3137412afdba0bd081d1cf3c956a0c1c02e576353808c5dc7f` | 4-thread split-M |

## Held-Out Metric Equivalence

The committed `p1d_regret_analysis.json` still records the earlier midpoint
summary (`32768`) and is not the final P1D.1 threshold. A2 therefore recomputed
from raw measurements using the accepted unchanged P1D.1 threshold `262144`.

Results over 30 held-out workload/session rows:

- serial selections: 3
- 4-thread split-M selections: 27
- exact-match rate: 86.67%
- mean regret: 0.0674%
- median regret: 0%
- P95 regret: 0.489%
- max regret: 0.769%
- average speedup over serial: 3.328x
- worst slowdown versus serial: 1.000x

No held-out retuning was performed.

## Test Evidence

Passed:

- `ctest --test-dir build-mlir --output-on-failure`: 21/21
- `tests/test_p1d1_thread_schedule_policy.py`: 15/15
- `tests/test_a2_thread_schedule_candidates.py`: 9/9
- A1 `ImplementationCandidateTest`
- Runtime P1D contract test command found the tests but skipped all 19 in the
  GPU environment; Runtime source was unchanged.

## Remaining Outside Candidate Core

- TilePlan
- KernelSelection as a candidate choice
- QuantizationDecision and QuantizationCoDesign
- BackendDecision
- Triton measured selector
- AWQ/vLLM deployment
- graph partition candidates
- deployment and serving candidates
- external CandidateProvider interfaces
- full feasibility architecture
- complete Implementation IR parallel-loop materialization

## Complexity Review

1. P1D.1 now uses the candidate core for a real decision.
2. There is one authoritative policy selection result.
3. Candidate identity is stable enough for current serial/parallel schedule variants.
4. Feasibility remains separate from ranking.
5. Selected schedule is derived from the selected candidate.
6. ExecutionPlan and Runtime complexity did not increase.
7. No Triton/AWQ/NPU/DMA/provider abstractions were added.
8. The old direct threshold schedule construction is no longer active as a
   second authority.
9. The implementation is clearer than maintaining a parallel threshold branch.
10. Several decision systems still bypass the candidate core, listed above.

## Limitations

A2 is not a full unified policy engine and not a general ARM scheduler. It does
not add schedules, retune the threshold, integrate external providers, modify
Runtime, or generate parallel loop Implementation IR.
