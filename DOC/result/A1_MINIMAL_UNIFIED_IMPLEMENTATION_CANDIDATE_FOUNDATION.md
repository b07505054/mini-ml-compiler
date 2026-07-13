# A1 Minimal Unified ImplementationCandidate Foundation

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler base before A1: `5822a04aa600ec41fcca5ef00619cc27d3e37c40`
Runtime reference: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6`

## Verdict

`PASSED_MINIMAL_UNIFIED_CANDIDATE_FOUNDATION`

A1 establishes a minimal compiler-internal candidate foundation for the active
op-level path:

`CandidateGenerationPass -> ServingCostModelPass -> PlanSelectionPass`

It does not implement the full Architecture 3.1 candidate/provider system.

## Baseline Candidate Representations

| Current field | Producing pass | Consuming pass | Semantic meaning | Pre-A1 representation | Truth boundary | Serialization |
|---|---|---|---|---|---|---|
| `candidate_type` | CandidateGenerationPass or fixture MLIR | ServingCostModelPass, PlanSelectionPass | implementation kind such as `direct_lower`, `backend_fallback`, `unsupported` | pass-local string reads in every pass | static compiler candidate, not runtime evidence | `compiler.candidates[]` / `compiler.evaluated_candidates[]` DictionaryAttr |
| `source_op` | CandidateGenerationPass or fixture MLIR | ServingCostModelPass, PlanSelectionPass | IR-rooted semantic target | raw string in dictionaries | op name derived from IR, not stable external provider ID | DictionaryAttr string |
| `required_boundary_ops` | CandidateGenerationPass / AlternativeLoweringPlanningPass | ServingCostModelPass, PlanSelectionPass, BoundaryMaterializationPass indirectly through selected attrs | boundary work required by candidate | raw string array parser in each consumer | compiler static lowering requirement | DictionaryAttr array |
| `fallback_backend` | CandidateGenerationPass or fixture MLIR | ServingCostModelPass | fallback target for backend fallback candidates | raw string parser | declared compiler fallback, not runtime fallback permission | DictionaryAttr string |
| `evaluation.penalty_score` | ServingCostModelPass | PlanSelectionPass | ranking score after static evaluation | local `CandidateEval` then local `CandidateInfo` | static penalty / optional shape estimate, not measured latency | `compiler.evaluated_candidates[]` DictionaryAttr |
| `evaluation.status` | ServingCostModelPass | PlanSelectionPass | evaluated, partially evaluated, or rejected | local string parser | evaluation state, not legality authority | DictionaryAttr string |
| `evaluation.cost.*` | ServingCostModelPass | PlanSelectionPass / ExecutionPlanBuilder via selected attrs | structured static cost components | raw dictionary promotion | static cost model, not runtime measurement | DictionaryAttr fields promoted to `selected_plan.cost.*` |
| `evaluation.shape_cost.*` | ServingCostModelPass | PlanSelectionPass / ExecutionPlanBuilder via selected attrs | shape-derived static cost facts | raw dictionary promotion | shape-derived static estimate from declared profile numbers | DictionaryAttr fields promoted to `selected_plan.shape_cost.*` |
| winner state | PlanSelectionPass | downstream selected-plan consumers | policy decision result | local mutable `CandidateInfo` vector and selected/rejection arrays | static plan selection, not Runtime execution | `selected_plan.*`, `compiler.selected_candidates`, `compiler.selection_rejections` |

Pre-A1 duplicated parsers existed in:

- `CandidateGenerationPass.cpp`: local per-op builders and a separate func-level `Candidate` struct.
- `ServingCostModelPass.cpp`: `readStr`, `readStrs`, local `CandidateEval`.
- `PlanSelectionPass.cpp`: `readStrDict`, `readI64Dict`, `readStrsDict`, local `CandidateInfo`.

## Canonical Type

A1 adds `serving/ImplementationCandidate.h`.

The type is compiler-internal and op-scoped in this phase. It contains:

- identity: `candidateId`, `providerId`
- IR rooting: `scopeKind`, `semanticTargetRef`, optional `functionRef`
- implementation kind: `implementationKind`
- boundary contract: `requiredBoundaryOps`, optional `fallbackBackend`
- minimal feasibility: `feasible`, `deferred`, `rejected`, `unsupported`, `unknown` plus reason
- cost summary: optional penalty score, evaluation status/reason, cost model id, truth boundary
- provenance: candidate truth boundary

The shared codec preserves existing DictionaryAttr storage while adding typed
fields such as `candidate_id`, `scope_kind`, `semantic_target_ref`,
`implementation_kind`, `provider_id`, and `feasibility.status`.

## IR Rooting

A1 uses the current deterministic op-level root:

- `semanticTargetRef` comes from `source_op`.
- Missing `candidate_id` is synthesized as `<source_op>:<candidate_type>`.
- `scopeKind` defaults to `operator` for decoded existing candidate dictionaries.

This is stable across CandidateGeneration, ServingCostModel, and PlanSelection.
External-provider ID bridging, Triton IDs, graph partitions, and deployment
boundaries remain unresolved.

## Feasibility Summary

A1 maps current vocabulary without adding new hardware facts:

- `evaluation.status = evaluated` -> `feasible`
- `evaluation.status = partially_evaluated` -> `deferred`
- `evaluation.status = rejected` + `candidate_type = unsupported` -> `unsupported`
- `rejection_reason` -> `rejected`
- `constraint_status = pass/fail` -> `feasible/rejected`
- generated unsupported sentinel -> `unsupported`

Feasibility remains separate from cost ranking.

## PolicyResult Separation

A1 adds a minimal `PolicyResult` struct. PlanSelection uses it to keep the
selected candidate identity, considered IDs, rejected IDs, policy id, reason,
objective summary, and truth boundary separate from the candidate option.

`selected=true` is not part of `ImplementationCandidate`.

## Pass Migration

CandidateGenerationPass:

- builds op-level `ImplementationCandidate`
- encodes it through `encodeImplementationCandidate`
- preserves existing `compiler.candidates` and `compiler.rejected_candidates`

ServingCostModelPass:

- decodes each candidate once
- uses typed fields for implementation kind, fallback backend, and boundary ops
- re-encodes the candidate while preserving existing evaluation fields

PlanSelectionPass:

- decodes typed candidate views for ranking
- keeps tier, penalty, and tie-break behavior unchanged
- emits `compiler.selected_candidates` and `compiler.selection_rejections`
- adds selected implementation candidate provenance without changing Runtime code

ExecutionPlanBuilder, ExecutionPlanExporter, Runtime schemas, and Runtime
production code are unchanged.

## Behavior Preservation

P1D.1 behavior remains outside the A1 candidate core and is preserved:

- tiny `8x8x8`: serial, metric `512`, threshold `262144`
- boundary `64x64x64`: 4-thread split-M, metric `262144`
- large `256x256x256`: 4-thread split-M, metric `16777216`

The P1D.1 policy artifact, threshold, boundary rule, legality checks, selected
ThreadSchedule, and ExecutionPlan thread-schedule provenance were not changed.

## Tests

Passed:

- `ImplementationCandidateTest`
- `ServingStaticCostModelV1Test`
- `ctest --test-dir build-mlir --output-on-failure`: 21/21 passed
- `tests/test_p1d1_thread_schedule_policy.py`: 15/15 passed
- MLIR FileCheck runs for:
  - `candidate_generation.mlir`
  - `candidate_evaluation.mlir`
  - `plan_selection.mlir`
  - `shape_cost_model.mlir`
  - `tile_planning.mlir`

## Complexity Review

1. Duplication reduced: yes, the active path now has one typed codec.
2. Indirection: acceptable; DictionaryAttr remains the storage boundary.
3. Field usage: fields are used by generation, evaluation, selection, or tests.
4. Future-only fields: excluded.
5. Scope: explicit and minimal: operator scope only for A1.
6. IR root: understandable, based on existing `source_op`.
7. New consumers: can use the header instead of private parsers.
8. Provider leakage: no Triton, AWQ, NPU, DMA, or vLLM fields added.
9. Runtime complexity: unchanged.
10. Smaller alternative: a codec-only helper would reduce parsing but would not
    establish candidate/policy-result separation.

## Remaining Outside A1

- P1D.1 ThreadSchedule selection
- TilePlan and tile default policy
- KernelSelection
- QuantizationDecision / QuantizationCoDesign
- BackendDecision
- Triton private measured selector
- AWQ/vLLM deployment
- graph partition, deployment, and serving candidates
- external CandidateProvider interfaces
- full feasibility architecture
- Runtime schema unification
- Capability DB canonicalization

## Final Truth Boundary

A1 is a compiler-internal architecture foundation. It proves the existing
candidate generation/evaluation/selection path can share one typed representation
without changing Runtime behavior. It does not prove full Architecture 3.1
candidate unification.
