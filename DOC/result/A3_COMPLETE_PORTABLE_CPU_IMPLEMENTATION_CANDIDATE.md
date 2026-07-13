# Phase A3 - Complete Portable CPU ImplementationCandidate Identity

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler baseline HEAD: `b1a7210d8fedc40c95c8b94db98298676d4b5984`
Runtime HEAD: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6` (unchanged)
Capabilities HEAD: `aac593da0bdde7a95c38c03920fc4d00b73011db` (unchanged)
Verdict: `PASSED_COMPLETE_PORTABLE_CPU_CANDIDATE_IDENTITY`

## Scope

A3 is a compiler-internal architecture migration. It does not add candidates,
change policy, retune evidence, alter Runtime behavior, or integrate Triton,
AWQ/vLLM, external CandidateProviders, NEON, DMA, NPU, Hailo, quantization
expansion, or ExecuTorch comparison.

## Baseline Authority Map

Before A3, the active Raspberry Pi path was split:

| Decision | Pre-A3 authority | Materialization |
| --- | --- | --- |
| backend | function-level representation/source backend attrs | ExecutionPlan backend decision |
| kernel | `KernelSelectionPass` descriptor-order match | `kernel_selection.selected_id` |
| tile | P1C.1 static default encoded in kernel ID/profile descriptor | Runtime kernel ID; `tile_plan` remains deferred |
| dtype | IR result type via `OpShapeFacts`, checked against descriptor/policy | tensor/quantization contract attrs |
| thread schedule | A2 `ImplementationCandidate` schedule variants | ExecutionPlan `thread_schedule` |

This meant A2 candidates represented schedule variants under an already-selected
kernel rather than complete executable implementations.

## Complete Candidate Model

A3 extends the compiler-internal `ImplementationCandidate` with current-use
fields only:

- `backend`
- `runtime_contract_kind`
- `dtype`
- `tile.block_m`
- `tile.block_n`
- `tile.block_k`

The active portable CPU candidates now describe one complete opaque native
implementation option:

Candidate S:

- scope: `fused_region`
- semantic target: `fused_matmul_bias_relu`
- backend: `cpu`
- implementation kind: `opaque_portable_cpu_native_kernel`
- Runtime contract kind: `portable_cpu_kernel_adapter_contract`
- kernel ID: `portable_fused_matmul_bias_relu_bm32_bn128_bk32`
- tile: `BM=32, BN=128, BK=32`
- dtype: compiler-normalized `fp32` for IR `f32`
- thread schedule: `1 / none / serial`

Candidate P is identical except:

- thread schedule: `4 / m / contiguous_chunks`

## Candidate Identity

The deterministic fallback candidate ID now includes:

semantic target, scope, backend, implementation kind, Runtime contract kind,
kernel ID, tile identity, dtype, thread schedule, and target profile ID.

It deliberately excludes cost, feasibility, evidence, confidence, and policy
selection state.

## Feasibility and Policy

Feasibility remains separate from ranking. Candidate construction checks that
the selected kernel ID exposes a parseable tile identity and, when a descriptor
declares supported tile shapes, that the candidate tile agrees with the
descriptor. P1D.1 still applies the unchanged `matmul_mnk` threshold policy:

- `< 262144` selects serial
- `>= 262144` selects 4-thread split-M

PolicyResult selects a complete candidate ID. The selected `ThreadSchedule`,
kernel ID, tile identity, dtype, and implementation metadata are derived from
that selected candidate for the migrated path.

## ExecutionPlan Equivalence

A3 adds compiler-internal MLIR `implementation_candidate.*` attrs for audit and
materialization provenance. The Runtime-facing ExecutionPlan schema and
semantics are unchanged.

Normalized plan hashes were identical before and after A3:

| Shape | Hash |
| --- | --- |
| `8x8x8` | `c2471f3b95708c305c7f26482d88314334224f604a7548bceed871177079822e` |
| `64x64x64` | `9f9d3c8b11f95bd63e2da8c916dac951c138099e4e61fda3b6fc60721e37709a` |
| `256x256x256` | `d1b4b98c77e89e565ac966b82e2b2a22afe186a7251a982d91a7435306fffb0a` |

The selected kernel stayed
`portable_fused_matmul_bias_relu_bm32_bn128_bk32`. The selected schedules stayed
serial for `8x8x8` and 4-thread split-M for `64x64x64` and `256x256x256`.

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

No held-out retuning was performed.

## Implementation IR Truth Boundary

A3 still selects an opaque prebuilt native portable CPU implementation. It does
not materialize tiled loops, parallel loops, vector operations, bufferization,
memory spaces, DMA, or synchronization IR. Kernel/tile/dtype/thread identity is
represented in `ImplementationCandidate` and exported through the existing
Execution Contract.

## Remaining Outside the Candidate Core

Still outside A3:

- inactive P1C tile alternatives
- generic TilePlanning decisions
- BackendDecision for unrelated paths
- QuantizationDecision and QuantizationCoDesign
- Triton private selector
- AWQ/vLLM deployment
- graph partition, model deployment, and serving candidates
- external CandidateProvider interface
- Runtime schema canonicalization
- Capability DB canonicalization
- full Implementation IR materialization

## Validation

- `ctest --test-dir build-mlir --output-on-failure`: 21/21 pass
- `python tests/test_a2_thread_schedule_candidates.py`: 10/10 pass
- `python tests/test_p1d1_thread_schedule_policy.py`: 15/15 pass
- Temporary baseline rebuild from `HEAD` source confirmed normalized plan hash
  equivalence for `8x8x8`, `64x64x64`, and `256x256x256`.

Runtime production code, Capability DB, IVP, target profiles, raw evidence,
native kernels, Runtime adapters, P1D.1 threshold, and P1D.1 policy artifact
were not modified.
