# Current State

Last verified: 2026-07-14.

| Repository | Branch/head | State | Ownership |
|---|---|---|---|
| `ml-graph-compiler-runtime` | `master` `ba388a93eeccd11045f8c1f2eb950ede2601bc88` | clean, ahead 10 / behind 0 | compiler, IR, candidates, feasibility, policy, contracts |
| `heterogeneous-inference-runtime` | `main` `53c80e2c11101ec7b8db2e73f978e220c054d9a1` | clean, ahead 7 / behind 0 | runtime validation/dispatch and evidence |
| `ml-platform-capabilities` | `main` `aac593da0bdde7a95c38c03920fc4d00b73011db` | clean, ahead 1 / behind 0 | declared capability profiles, partial today |
| `Inference-Validation-Platform` | `main` `c80beede31338b5f66831595f56d4dbb8f57335d` | divergent, ahead 2 / behind 2 | validation/control-plane project, unchanged in S1 |
| Raspberry Pi 5 | `edgeaiplatform` | evidence target | real aarch64 execution only |

## Five-Minute Summary

The strongest current path is the Raspberry Pi fused MatMul + Bias + ReLU path: Semantic IR drives complete implementation candidates, feasibility, calibrated policy, `PolicyResult`, ExecutionPlan/contract materialization, Runtime validation, and real Pi execution.

The second strongest current result is E3: a live-Compiler same-XNNPACK comparison against ExecuTorch default using the same `.pte`, runner, XNNPACK source, input bytes, timing boundary, process lifetime, and oracle.

## Phase Timeline

P1A: Pi hardware profile.
P1B: compiler-selected portable CPU kernel execution.
P1C: portable tile candidate discovery.
P1C.1: low-regret static tile default.
P1D: thread decomposition candidate discovery.
P1D.1: offline-calibrated IR-shape-aware thread policy.
A1: unified `ImplementationCandidate` foundation.
A2: P1D.1 migration into candidate architecture.
A3: complete portable CPU candidate identity.
A4: `PortableCPUProvider` extraction.
A5: provider/feasibility/policy/materialization separation.
A6: Triton shadow provider with unresolved IR bridge.
R1: ExecuTorch Compiler reverse engineering.
E0: ExecuTorch comparison audit.
E1: ExecuTorch Pi baseline bring-up.
E2: invalid correctness experiment.
E2.1: corrected implementation-stack comparison.
E2.1A: dispatch-path audit proving live Compiler bypass.
E3: live-Compiler same-XNNPACK comparison repair and result.
S0: Principal Engineer truth audit.
S1: publication canonicalization.

## Verified Results

P1D.1 portable policy over 30 held-out workload/session rows: exact match 86.6667%, mean regret 0.067392%, median regret 0%, P95 regret 0.489076%, max regret 0.768578%, average speedup over serial 3.327722x, worst slowdown versus serial 1.000x.

E2 remains invalid because its correctness predicate required independent absolute and relative gates.

E2.1 has 324 records and zero correctness failures, but it is an implementation-stack comparison. Project portable scalar/native execution was geometrically about 2.631x slower than ExecuTorch/XNNPACK on the frozen scope.

E3 has 162 discovery records and 60 formal held-out records. Candidate-space verdict: `XNNPACK_ONE_STATIC_WINNER`; policy: `static_X1`; formal result: 2 wins, 8 ties, 0 losses; geomean default/project ratio 1.031686x.

## Current Truth Boundary

Production/canonical: Pi portable CPU path and P1D.1 policy for the fixed fused-op/kernel/dtype/profile; E3 evaluation contract path for same-XNNPACK comparison.

Measured: Pi portable evidence, E1/E2/E2.1/E3 ExecuTorch evidence, AWQ/vLLM serving traces, selected Triton artifacts.

Shadow: Triton provider, because committed Triton artifacts lack sufficient source provenance for production IR-root mapping.

Experimental/partial: AWQ quantization planning/materialization, CV/YOLO paths, Apple/Metal demos, LLM serving planning artifacts.

Missing: NPU execution, DMA/local-memory IR, transfer model, unified quantization candidate policy, quantization accuracy/perplexity calibration, universal cross-backend ranking.
