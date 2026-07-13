# Phase A4 - Portable CPU Candidate Provider Extraction

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler baseline HEAD: `43f46b2f71ebe501cbc7017d344cceec809ede22`
Runtime HEAD: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6` (unchanged)
Capabilities HEAD: `aac593da0bdde7a95c38c03920fc4d00b73011db` (unchanged)
Verdict: `PASSED_PORTABLE_CPU_CANDIDATE_PROVIDER_EXTRACTION`

## Scope

A4 is a compiler-only architecture migration. It extracts the active Raspberry
Pi portable CPU candidate enumeration into a dedicated in-process provider
component. It does not change policy, threshold, selected kernel, tile, dtype,
thread schedules, ExecutionPlan semantics, Runtime code, target profiles,
policy artifacts, or evidence.

## Pre-A4 Responsibility Map

Before A4, `KernelSelectionPass` owned:

| Responsibility | Pre-A4 owner |
| --- | --- |
| IR semantic matching | `KernelSelectionPass` |
| target-profile runtime kernel descriptor parsing | `KernelSelectionPass` |
| descriptor legality/matching | `KernelSelectionPass` |
| portable CPU candidate construction | `KernelSelectionPass` local helper |
| candidate feasibility filtering for policy | `KernelSelectionPass` |
| P1D.1 threshold policy | `KernelSelectionPass` |
| selected-candidate materialization | `KernelSelectionPass` |
| rejection/defer diagnostics | `KernelSelectionPass` |

A4 moves only portable CPU candidate enumeration/construction to
`PortableCPUProvider`. The pass remains the orchestrator for descriptor matching,
policy, fallback behavior, and materialization.

## Provider Contract

`mlir_passes/include/serving/PortableCPUProvider.h` defines a small in-process
provider component, not a global plugin framework.

Inputs:

- `PortableCpuProviderContext`: semantic target, scope, target profile ID,
  backend, dtype, truth boundary.
- `PortableCpuRuntimeKernelDescriptor`: parsed kernel ID, op name, backend,
  supported dtypes, supported tile shapes, declared thread schedules, truth
  boundary.

Outputs:

- zero or more `PortableCpuCandidateView` records containing complete
  `ImplementationCandidate`s plus their typed thread schedule.
- diagnostics such as `unsupported_semantic_scope`, `wrong_dtype`,
  `kernel_tile_identity_mismatch`, `missing_serial_schedule`, and
  `missing_parallel_schedule`.

The output does not contain `PolicyResult`, selected candidate, ranking,
threshold outcome, Runtime dispatch result, measured latency, raw evidence, or
fallback execution order.

## Candidate Enumeration

For the fused MatMul + Bias + ReLU scope, the provider emits exactly the active
candidate set when declared components exist:

- Candidate S: CPU, opaque portable native kernel, portable CPU adapter
  contract, `portable_fused_matmul_bias_relu_bm32_bn128_bk32`, tile
  `BM=32 BN=128 BK=32`, dtype `fp32`, schedule `1/none/serial`.
- Candidate P: same implementation identity, schedule
  `4/m/contiguous_chunks`.

The provider does not enumerate inactive P1C tile alternatives, 2-thread
schedules, split-N schedules, other kernels, fallback frameworks, or unsupported
dtypes.

## Provider / Policy Separation

The provider contains no `262144` threshold literal and no P1D.1 ranking logic.
Tests verify that tiny and large shapes expose the same provider candidate set.
Only the policy layer selects serial below threshold and 4-thread split-M at or
above threshold.

Malformed policy state where below-threshold and above-threshold schedules
collapse to the same declaration remains diagnosed in the policy orchestration
layer as `rejected_thread_schedule_candidate_id_collision`.

## Feasibility and Materialization

Feasibility remains visible after provider enumeration. Policy consumes only
candidate views whose feasibility status is `feasible`. The selected candidate
is still materialized into:

- `implementation_candidate.*` compiler attrs
- `kernel_selection.selected_id`
- `thread_schedule.*`
- unchanged ExecutionPlan `kernel_selection` and `thread_schedule` objects

Runtime-facing JSON is unchanged.

## Plan Equivalence

Normalized plan hashes remain identical to A3:

| Shape | Hash |
| --- | --- |
| `8x8x8` | `c2471f3b95708c305c7f26482d88314334224f604a7548bceed871177079822e` |
| `64x64x64` | `9f9d3c8b11f95bd63e2da8c916dac951c138099e4e61fda3b6fc60721e37709a` |
| `256x256x256` | `d1b4b98c77e89e565ac966b82e2b2a22afe186a7251a982d91a7435306fffb0a` |

Selected kernel, tile, dtype, schedule, policy provenance, and ExecutionPlan
semantics are unchanged.

## Held-Out Metric Equivalence

Recomputed from committed P1D post-warmup held-out evidence using the unchanged
threshold:

- rows: 30
- exact match: 86.67%
- mean regret: 0.067392%
- median regret: 0%
- P95 regret: 0.489076%
- maximum regret: 0.768578%
- average speedup over serial: 3.327722x
- worst slowdown versus serial: 1.0x

No performance measurement or retuning was performed.

## Validation

- `ctest --test-dir build-mlir --output-on-failure`: 21/21 pass
- `python tests/test_a2_thread_schedule_candidates.py`: 12/12 pass
- `python tests/test_p1d1_thread_schedule_policy.py`: 15/15 pass
- provider header checked to ensure the threshold literal is absent
- grep confirmed old `buildThreadScheduleCandidate` helper is removed

## Complexity Review

The provider abstraction is justified because one real provider now has one real
consumer and the extracted interface is smaller than the candidate-construction
logic it replaces. No global registry, dynamic loading, external provider API,
Triton/AWQ/NPU fields, deployment scopes, or future-only payloads were added.
The design leaves a path for a second provider without pretending that framework
already exists.

## Remaining Outside A4

- global provider registry
- external provider loading
- Triton provider
- AWQ/vLLM provider
- CUDA provider
- NPU/Hailo provider
- deployment and serving candidate providers
- inactive P1C tile alternatives
- tile x thread joint search
- unified Capability DB
- full Implementation IR materialization
- new optimization policy

Runtime production code, Capability DB, IVP, target profiles, policy artifacts,
raw evidence, native kernels, Runtime adapters, Triton artifacts, and AWQ
artifacts were not modified.
