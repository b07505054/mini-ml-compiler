# Phase A6 - IR-Rooted Triton Provider Shadow Integration

Last verified: 2026-07-13
Source host: GPU Linux `/home/allen/Desktop/Project/ml-graph-compiler-runtime`
Compiler baseline HEAD: `112ceeb0901e4a5e45a9fdc0e64b3a989edb54c0`
Runtime HEAD: `a6e2ae8648ee27d8e73396218266e98a0ea0cbc6` (unchanged)
Capabilities HEAD: `aac593da0bdde7a95c38c03920fc4d00b73011db` (unchanged)
Verdict: `PASSED_SHADOW_PROVIDER_WITH_UNRESOLVED_ID_BRIDGE`

## Scope

A6 is a compiler-only, non-authoritative integration phase. It adds a strict
shadow Triton candidate provider adapter over existing measured Triton artifacts.
The adapter may enumerate candidate-shaped records, attach evidence references,
evaluate shadow feasibility, and emit a companion shadow analysis artifact.

A6 does not alter production selection, Portable CPU behavior, P1D.1,
ExecutionPlan output, Runtime dispatch, target profiles, raw Triton evidence,
or policy artifacts.

## Artifact Inventory

| Artifact | Classification | Schema | SHA-256 | Notes |
| --- | --- | --- | --- | --- |
| `trace/matmul_postop_triton_fused_candidate_sweep.json` | authoritative measured candidate evidence | `triton_matmul_bias_relu_fused_candidate_sweep` | `a508642446d6b590053311ab1f2511b54c91366a45376d75d1da8eb00cac6ca8` | 12 workloads, real Triton/CUDA measurements |
| `trace/matmul_postop_triton_fused_config_repair_training_profile.json` | calibration input | `triton_matmul_bias_relu_fused_candidate_sweep` | `0f28c34498c5ee0fc3f723533cade95c05ebfcc61db45978c641f6334e04ad64` | 8 training workloads |
| `trace/matmul_postop_triton_fused_config_repair_training_manifest.json` | calibration split manifest | `matmul_postop_workload_manifest` | `cc571d4073fc87c011e94db784dc5f7c09b1308b2393ae20dd401691e6d307d5` | training workload IDs |
| `trace/matmul_postop_triton_fused_config_repair_oracle_manifest.json` | evaluation manifest | `matmul_postop_workload_manifest` | `57c763334f090de3d2324087b74d6859a516b8cf0ca511291f85eefd41ba2de0` | oracle/evaluation workload IDs |
| `trace/matmul_postop_triton_fused_config_repair_cost_model.json` | calibrated prediction model | `triton_matmul_bias_relu_fused_config_repair_cost_model` | `623d19ce7bd3c5ef9179ac95d36b3b99df9e84eee861f4266caaae7f5a95d119` | repaired analytical model |
| `trace/matmul_postop_triton_fused_config_repair_plans.json` | private shadow selection plan input | `triton_matmul_bias_relu_fused_config_repair_plans` | `d5d5041dfed45e808e75d3f2e8355c4f5b72844fb8906af8efc29223e6c51151` | 7 private plans, not canonical ExecutionPlan |
| `trace/matmul_postop_triton_fused_config_repair_plan_validation.json` | dispatch validation evidence | `triton_matmul_bias_relu_fused_config_repair_plan_validation` | `1fb3a22c396f308e6d87b30629c54711e75409736d4267767ab8219f7e05a23d` | 7 validation workloads |
| `trace/matmul_postop_triton_fused_config_repair_summary.json` | evaluation summary | `triton_matmul_bias_relu_fused_config_repair_summary` | `544cae020c3b033ce93e1c7bb7897a7c30e508cf4a28a30ef7041fea77c49330` | repair summary and collapse audit |
| `trace/matmul_postop_triton_fused_config_repair_fresh_oracle.json` | evaluation-only oracle evidence | `triton_matmul_bias_relu_fused_candidate_sweep` | `6069fb29e1b96bed08a386b6b3826197004f7f7cea26dea44957178ea416b3e7` | not used as predictor input |
| `trace/matmul_postop_triton_analytical_cost_model.json` | superseded/parallel analytical model | `triton_matmul_bias_relu_analytical_cost_model` | `7a48782515598dc5faa60931bfbdd622661cad6de3fc46886fdcd6fca63e6855` | older analytical selector path |
| `trace/matmul_postop_triton_analytical_selection_plans.json` | superseded/parallel private plans | `triton_matmul_bias_relu_analytical_selection_plans` | `faace18364d1efab65bf9f6cf736d5ce5805daffbb2e9f8526a7b3614f062a21` | 12 private plans |
| `trace/matmul_postop_triton_analytical_plan_validation.json` | superseded/parallel validation | `triton_matmul_bias_relu_analytical_plan_validation` | `ad15db0aa0155d6d4de2ffd0698833ae3fd9c2ad400a3b4365079e263bc9badf` | 12 validation workloads |

Producer tools:

- `tools/run_triton_fused_candidate_discovery.py`
- `tools/run_triton_fused_config_selection.py`
- `tools/run_triton_fused_config_repair.py`

Existing consumers before A6 were tests, reports, and private validation tools.
Runtime does not consume these selections.

## IR Identity Mapping

Mapping verdict for existing committed Triton artifacts:
`AMBIGUOUS_MAPPING`.

Reason: the committed Triton private plans contain `op_id`, `workload_id`,
`graph_id`, fused operation kind, shape, dtype, and selected config, but they do
not contain canonical compiler provenance such as:

- `source.graph_node_id`
- `source.imported_node_id`
- `source.dispatch_group`
- `source.onnx_name`
- stable function reference
- model or GenericGraphIR identity

A6 implements three outcomes:

- `VERIFIED_DIRECT_MAPPING`: accepted only when the Triton artifact supplies
  explicit IR mapping provenance that matches the MLIR fused region.
- `VERIFIED_DERIVED_MAPPING`: accepted only for a single-fused-region MLIR case
  where op kind, static shape, and dtype uniquely match. This is shadow-safe and
  not sufficient for multi-region production binding.
- `AMBIGUOUS_MAPPING`: emitted when mapping is not unique or provenance is
  missing. Candidates remain shadow-only and feasibility is deferred with
  `deferred_missing_mapping`.

Shape alone is explicitly insufficient when multiple semantic targets collide.

## Provider Contract

A6 adds `tools/run_triton_shadow_candidate_provider.py`, a standalone compiler
shadow-analysis tool. It deliberately does not add a global provider registry,
dynamic loading, production pass, or Runtime dependency.

Inputs:

- MLIR file containing `hir.fused_matmul_bias_relu`
- existing Triton candidate sweep artifact
- existing repaired cost-model artifact
- existing private repaired plans artifact
- existing plan-validation artifact
- optional summary artifact
- optional expected SHA-256 checks

Outputs:

- companion JSON artifact with schema
  `triton_ir_rooted_candidate_shadow_analysis`
- provider diagnostics through mapping and feasibility fields
- candidate-shaped records using canonical names where possible
- shadow-only policy result

The tool never imports Triton, imports Torch, invokes Triton JIT, benchmarks
hardware, modifies raw evidence, writes canonical ExecutionPlan, or changes
Runtime behavior.

## Candidate Schema

Each shadow Triton candidate records:

- `candidate_id`
- `provider_id = triton_candidate_provider_shadow`
- `scope = fused_region`
- `semantic_target_ref` when mapping is resolved
- `backend = cuda`
- `implementation_kind = triton_generated_fused_kernel`
- `runtime_contract_kind = triton_kernel_config_contract_shadow`
- `kernel_id = triton_matmul_bias_relu_one_pass_f32`
- `selected_kernel = triton_tiled_matmul_bias_relu_one_pass_f32`
- `config_id`
- tile `{block_m, block_n, block_k}`
- `num_warps`
- `num_stages`
- dtype
- target GPU and compute capability
- evidence references
- confidence fields
- feasibility summary
- truth boundary

Candidate identity excludes measured latency, confidence, feasibility status,
evidence freshness, oracle labels, regret, and selection state.

## Evidence Adapter

Evidence is attached by reference only. The adapter preserves distinctions among:

- measured candidate evidence
- calibration model
- shadow selection plan
- plan/dispatch validation
- evaluation summary

Raw timing samples, oracle winners, and regret are not copied into candidates or
MLIR attributes. Artifact hashes are retained and optional expected-hash checks
can reject candidate feasibility with `rejected_artifact`.

## Feasibility

The shadow feasibility states map into the current minimal model:

- resolved mapping plus valid artifacts -> `feasible_predicted` or
  `feasible_verified`, based on profile match
- unresolved mapping -> `deferred_missing_mapping`
- hash mismatch -> `rejected_artifact`

The full cross-backend feasibility model is not implemented in A6.

## Shadow Policy

The shadow policy result adapts the existing repaired Triton selection from the
private plans artifact. It records:

- selected shadow candidate ID
- considered candidate IDs
- selection source
- profile match
- confidence
- mapping status
- `production_plan_affected: false`
- `runtime_dispatch_affected: false`

The result is explicitly `shadow_only_non_authoritative` and is not consumed by
canonical `PlanSelection`, `KernelSelection`, ExecutionPlan export, or Runtime.

## Production Isolation

A6 does not change default `compile-for-target` behavior. Missing or invalid
Triton artifacts affect only explicit shadow analysis. Raspberry Pi plan hashes
remain unchanged:

| Shape | Hash |
| --- | --- |
| `8x8x8` | `c2471f3b95708c305c7f26482d88314334224f604a7548bceed871177079822e` |
| `64x64x64` | `9f9d3c8b11f95bd63e2da8c916dac951c138099e4e61fda3b6fc60721e37709a` |
| `256x256x256` | `d1b4b98c77e89e565ac966b82e2b2a22afe186a7251a982d91a7435306fffb0a` |

ExecutionPlan contains no Triton shadow fields.

## Tests

- `ctest --test-dir build-mlir --output-on-failure`: 22/22 pass
- `.venv/bin/python tests/test_a2_thread_schedule_candidates.py`: 12/12 pass
- `.venv/bin/python tests/test_p1d1_thread_schedule_policy.py`: 15/15 pass
- `python3 tests/test_triton_shadow_candidate_provider.py`: pass
- Existing Triton tests:
  - `test_triton_fused_candidate_discovery.py`
  - `test_triton_fused_config_selection.py`
  - `test_triton_fused_config_repair.py`
  - `test_triton_exact_selection.py`
  - `test_triton_analytical_selection.py`
  - `test_triton_target_sensitivity.py`

The A6 test covers direct mapping, derived mapping, ambiguous mapping, wrong
mapping, candidate ID stability, evidence hash mismatch, provider/policy
separation, raw evidence exclusion, and production isolation.

## Complexity Review

A6 validates the provider architecture with a genuinely different provider, but
keeps the shared contract conceptual rather than introducing a registry. Triton
specific fields remain in the shadow provider output and evidence adapter. The
canonical `ImplementationCandidate` concept remains backend-neutral enough for
shadow candidate-shaped records, but the C++ common type is not expanded with
Triton-specific fields in this phase.

The result is smaller and safer than premature canonical Triton Runtime
integration. Ordinary compilation remains independent of Triton artifacts.

## Limitations

A6 does not implement:

- canonical Triton ExecutionPlan integration
- Runtime Triton dispatch
- CPU-vs-GPU global selection
- production backend selection
- Triton code generation inside MLIR
- Triton JIT during compilation
- unified cross-backend objectives
- AWQ integration
- external provider registry
- full Implementation IR materialization

The missing provenance required for production Triton binding is explicit:
future Triton artifacts need stable compiler IR identity, including graph node
ID or dispatch group, function/model identity, and source operation provenance.
