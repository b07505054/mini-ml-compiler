# Phase A5 - Portable CPU Provider Contract Hardening

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler baseline HEAD: `b81830e6883d6284e867fe5e19cc44ccd85f0e23`
Runtime HEAD: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6` (unchanged)
Capabilities HEAD: `aac593da0bdde7a95c38c03920fc4d00b73011db` (unchanged)
Verdict: `PASSED_PROVIDER_FEASIBILITY_POLICY_SEPARATION`

## Scope

A5 is a compiler-only architecture migration. It hardens the first real
in-process candidate-provider boundary for the Raspberry Pi portable CPU path.
It does not change the candidate set, candidate IDs, threshold, policy metric,
boundary rule, selected kernel, tile, dtype, ExecutionPlan semantics, Runtime
code, target profiles, policy artifacts, raw evidence, or performance claims.

## Pre-A5 Responsibility Map

Before A5, `PortableCPUProvider` owned construction of the two complete active
portable CPU candidates, but provider output already marked those candidates as
`feasible`. `KernelSelectionPass` still mixed provider invocation, implicit
target satisfaction checks, P1D.1 policy, fallback/defer handling, and selected
candidate materialization.

| Component | Pre-A5 responsibilities |
| --- | --- |
| `KernelSelectionPass` | IR scope orchestration, descriptor matching, provider invocation, policy threshold evaluation, fallback/defer diagnostics, inline materialization |
| `PortableCPUProvider` | structural support checks, candidate construction, diagnostics, initial feasible status |
| `ImplementationCandidate` | complete candidate identity, codec, candidate ID construction, minimal feasibility summary, `PolicyResult` |
| `TargetConstraints` | typed target-profile attributes consumed by the pass |
| P1D.1 policy code | `matmul_mnk` metric, threshold `262144`, boundary rule, policy provenance |
| Materialization code | inline attributes for selected kernel, tile, dtype, candidate identity, and thread schedule |

## Provider Context

`PortableCpuProviderContext` now carries only enumeration inputs:

- semantic target identity
- scope kind
- static-shape availability
- target profile ID
- declared backend
- IR dtype
- truth boundary

The provider also consumes typed `PortableCpuRuntimeKernelDescriptor` values:

- kernel ID
- op name
- backend
- supported dtypes
- supported tile shapes
- declared thread schedules
- descriptor truth boundary

The provider context intentionally has no policy threshold, policy metric value,
selected candidate, `PolicyResult`, benchmark evidence, latency, regret,
Runtime execution result, fallback order, or ExecutionPlan output object.

## Provider Output

`PortableCpuProviderResult` contains:

- zero or more complete `ImplementationCandidate` records
- typed candidate views with the declared schedule
- provider diagnostics

Provider output is deterministic for testing, but ordering is not a ranking or
selection policy. Provider-emitted candidates now carry
`CandidateFeasibilityStatus::Unknown` with a reason beginning
`provider_enumerated_requires_feasibility:`. This makes enumeration explicit and
prevents provider construction from silently becoming target/workload
feasibility.

## Legality Versus Feasibility

A5 uses the following ownership split.

Provider-level structural legality:

- semantic scope is the fused MatMul + Bias + ReLU region
- descriptor backend is CPU
- descriptor op and kernel metadata are coherent
- descriptor supports `fp32`
- descriptor exposes tile `BM=32 BN=128 BK=32`
- schedule tuples are structurally valid
- malformed descriptors can prevent candidate construction

Target/workload feasibility:

- semantic target and scope still match the current IR operation
- target profile matches
- backend is available
- dtype remains `fp32`
- kernel and tile identity are present and coherent
- selected schedule tuple is declared and valid
- static shape is available
- parallel schedule has sufficient `physicalComputeUnits`

Evidence is not consulted by either layer to establish legality.

## Feasibility Evaluator

`PortableCPUFeasibilityEvaluator` evaluates each complete candidate against a
typed `PortableCpuFeasibilityContext` and returns a
`CandidateFeasibilitySummary`.

Preserved or made explicit reasons include:

- `wrong_semantic_scope`
- `target_profile_mismatch`
- `backend_unavailable`
- `wrong_dtype`
- `no_matching_kernel_descriptor`
- `kernel_tile_identity_mismatch`
- `missing_thread_schedule`
- `invalid_schedule_tuple`
- `missing_static_shape`
- `deferred_missing_compute_units`
- `rejected_exceeds_compute_units`
- `serial_schedule_declared_legal`
- `parallel_schedule_declared_legal`

The evaluator does not rank candidates, inspect measured evidence, read policy
artifacts, or invoke Runtime behavior.

## Policy Boundary

The P1D.1 policy remains the only owner of:

- `matmul_mnk`
- threshold `262144`
- below-threshold serial selection
- at-or-above-threshold 4-thread split-M selection
- policy ID/version
- evidence reference and hash
- policy truth boundary

`PortableCPUProvider.h` contains no threshold, policy-result, benchmark,
latency, or regret references. Policy consumes candidate views after explicit
feasibility evaluation and emits the only `PolicyResult`.

## Materialization Boundary

A5 adds an explicit selected-candidate materialization helper:

```text
selected complete ImplementationCandidate
+ PolicyResult
-> compiler decision attributes
-> unchanged ExecutionPlan contract
```

Materialization copies the selected candidate's backend, implementation kind,
Runtime contract kind, kernel ID, tile identity, dtype, provider ID, candidate
ID, and thread schedule. It does not reselect the kernel, infer a different
tile, recompute policy, change candidate identity, or silently fall back.

## KernelSelectionPass Orchestration

After A5, `KernelSelectionPass` still orchestrates this narrow path:

1. Identify the eligible fused IR scope.
2. Match target descriptors and prepare typed context.
3. Invoke `PortableCPUProvider`.
4. Invoke `PortableCPUFeasibilityEvaluator`.
5. Invoke the unchanged P1D.1 policy.
6. Materialize the selected complete candidate.
7. Preserve existing fallback/defer diagnostics.

The pass no longer relies on provider output as an implicit feasibility claim
and no longer performs selected-candidate materialization inline for the
migrated portable CPU path. Descriptor matching and the local P1D.1 policy
remain in the pass translation unit; this is intentionally narrower than a
project-wide policy framework.

## Fallback and Deferred Behavior

Existing safe behavior is preserved:

- no candidate: existing explicit rejection/defer diagnostics remain
- serial feasible and parallel rejected: serial remains the safe selection
- missing static shape: the candidate feasibility is deferred
- insufficient compute units: the parallel candidate is rejected and serial may
  remain feasible
- unavailable policy: existing static fallback/defer behavior is preserved
- materialization inconsistency: compilation must fail or preserve existing
  explicit defer/reject semantics rather than silently switching

Runtime fallback is out of scope for A5.

## Plan Equivalence

Normalized plan hashes are unchanged from A4:

| Shape | Hash | Selected schedule |
| --- | --- | --- |
| `8x8x8` | `c2471f3b95708c305c7f26482d88314334224f604a7548bceed871177079822e` | `1/none/serial` |
| `64x64x64` | `9f9d3c8b11f95bd63e2da8c916dac951c138099e4e61fda3b6fc60721e37709a` | `4/m/contiguous_chunks` |
| `256x256x256` | `d1b4b98c77e89e565ac966b82e2b2a22afe186a7251a982d91a7435306fffb0a` | `4/m/contiguous_chunks` |

Selected kernel, tile, dtype, thread schedule, policy provenance, and
ExecutionPlan semantics are unchanged.

## Held-Out Metric Equivalence

No calibration or performance benchmarking was rerun. Recomputed committed P1D
post-warmup held-out metrics remain:

- rows: 30
- exact match: 86.67%
- mean regret: 0.067392%
- median regret: 0%
- P95 regret: 0.489076%
- maximum regret: 0.768578%
- average speedup over serial: 3.327722x
- worst slowdown versus serial: 1.0x

No held-out retuning was performed.

## Validation

- `git diff --check`: pass
- `ctest --test-dir build-mlir --output-on-failure`: 21/21 pass
- `.venv/bin/python tests/test_a2_thread_schedule_candidates.py`: 12/12 pass
- `.venv/bin/python tests/test_p1d1_thread_schedule_policy.py`: 15/15 pass
- normalized plan hashes for `8x8x8`, `64x64x64`, and `256x256x256`: unchanged
- provider header grep: no threshold, policy result, benchmark, latency, or
  regret references

## Complexity Review

A5 adds one small typed feasibility context/evaluator and one materialization
helper. Those additions are used by current code and replace implicit provider
feasibility plus inline materialization. No global registry, dynamic loading,
external provider API, Triton/AWQ/NPU fields, or future-only payloads were
added.

The provider remains local and in-process. Feasibility is separate from
enumeration and ranking. There is still one policy authority and one selected
implementation authority. Candidate IDs and plan hashes remain stable.

## Remaining Limitations

A5 does not implement:

- generic provider registry
- stable cross-provider plugin API
- Triton provider
- AWQ/vLLM provider
- CUDA provider
- NPU/Hailo provider
- external artifact exchange
- full feasibility architecture across all backends
- ExecutionPlan schema unification
- Capability DB canonicalization
- full Implementation IR
- new performance policy

Inactive P1C tile alternatives, generic TilePlan, unrelated KernelSelection
paths, QuantizationDecision, QuantizationCoDesign, Triton private selector,
AWQ/vLLM deployment, graph partition, model deployment, serving candidates, and
external provider interfaces remain outside the A5 provider core.

Runtime production code, Capability DB, IVP, target profiles, policy artifacts,
raw evidence, native kernels, Runtime adapters, Triton artifacts, and AWQ
artifacts were not modified.
