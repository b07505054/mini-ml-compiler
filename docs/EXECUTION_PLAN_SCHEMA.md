# Execution Plan Schema V2

## Purpose

ExecutionPlan is the canonical compiler output. It replaces the ad-hoc
`vllm_execution_plan` artifact (V1) with a typed, backend-agnostic contract.

Key invariants:
- All decisions use backend-agnostic vocabulary. No field encodes a runtime
  flag name (`max_num_seqs`, `gpu_memory_utilization`, `tensor_parallel_size`, etc.).
- Runtime-specific configuration is produced by **materializers** that read
  the plan and emit runtime-specific artifacts.
- Every decision carries `truth_boundary` at the decision level.
- No measured performance values appear in this plan.
- Per-op bundles may carry a `shape_cost` object (shape_cost_model_v2): a
  **static compiler estimate** derived from tensor shapes, existing dtype /
  quantization metadata, and declared theoretical peak numbers
  (`staticCostProfile` in the target profile). Fields:
  `flops_estimate`, `input_bytes_estimate`, `output_bytes_estimate`,
  `weight_bytes_estimate`, `total_memory_bytes_estimate`,
  `arithmetic_intensity_milli` (FLOPs×1000/bytes), optional
  `estimated_{compute,memory,boundary,total}_cost_nanos` (roofline form,
  present only when the profile declares peak FLOPs/bandwidth), `status`
  (`estimated` | `facts_only_no_profile_numbers`), `cost_model_version`,
  and `cost_truth_boundary =
  static_shape_derived_declared_profile_not_measured_not_runtime_validated`.
  This is **not a measured benchmark and not a runtime latency guarantee**;
  it is the first step toward hardware-aware co-design (shape × dtype ×
  quantization × declared hardware profile). Ops with unknown kinds or
  dynamic shapes have no `shape_cost` — they honestly fall back to the V1
  fixed model.
- Per-op bundles may carry a `tile_plan` object (tile_planning_v1,
  `TilePlanningPass`): **memory-hierarchy-aware static planning** for
  matmul-like ops against the declared local memory capacity
  (`staticCostProfile.localMemoryBytes` — SRAM / shared memory /
  scratchpad). The memory hierarchy is **optional declared metadata** —
  not every backend exposes local memory or DMA details. Tile feasibility
  runs only when the op kind is supported, the capacity is declared, and
  shapes are fully static. Fields: `status` (`planned` |
  `no_feasible_tile` | `deferred_missing_memory_hierarchy` |
  `dynamic_dims_unresolved` | …), `tile_shape_mnk`, `local_memory_bytes`
  (selected tile footprint), `estimated_global_traffic_bytes`
  (reuse-limited bound, reported alongside — never replacing —
  `shape_cost`'s ideal each-byte-once bytes), `double_buffer_fits`,
  `staging_capability` (`async_copy_declared` | `dma_declared` |
  `declared_unavailable` when declared false | `unknown_not_declared` when
  no declaration exists — unknown is not unavailable),
  `rejected_tile_count`, `rejection_reason`, `deferred_reason` (why
  feasibility was deferred, e.g.
  `local_memory_bytes_not_declared_in_target_profile` — absence is
  recorded, never replaced with an invented capacity; the plan stays
  valid), and `truth_boundary =
  tile_planning_static_local_memory_model_not_measured_not_codegen`.
  This is **not measured performance, not DMA/async-copy execution, not
  codegen**, and no claim that the backend kernel uses this tiling.
  Candidate ranking is unchanged by tile planning.
- Per-op bundles may carry a `kernel_selection` object
  (kernel_selection_contract_v1, `KernelSelectionPass`): the concrete
  compiler/runtime kernel contract. `selected_kernel` (+ `source`) appears
  **only** when a declared `RuntimeKernelDescriptor`
  (`runtimeKernels` in the target profile) fully matches the planned op —
  op name × backend × dtype × quant mode × layout × shape staticness ×
  tile plan × local memory. Otherwise `status` carries an explicit
  `rejected_*` / `deferred_*` reason plus per-descriptor
  `rejection_reasons`, including `deferred_no_kernel_library_declared`
  when no registry exists — never silent. Today exactly one kernel is
  declared (Metal RMSNorm f32, `handwritten_runtime`); this is a
  kernel-selection framework, not broad kernel coverage. A selection is a
  contract handed to the runtime, not an execution or performance claim
  (`truth_boundary =
  kernel_selection_static_descriptor_match_not_runtime_execution`). See
  `docs/RUNTIME_KERNEL_CONTRACT.md`.
- Per-op bundles may carry a `quantization_codesign` object
  (quantization_codesign_contract_v1, `QuantizationCoDesignPass` — see
  `docs/QUANTIZATION_CODESIGN.md`): separated quantization facts —
  representation, algorithm declaration status, backend legality, concrete
  kernel support (structured `{status, kernel_id, source}`), accuracy
  evidence (structured `{status, artifact_ref}`), static systems-cost
  comparison (`systems_cost_estimate` with before/after bytes and nanos and
  an explicit `excluded_terms` list), materialization requirement/status,
  scale/zero-point sources, rejection/deferral reasons, policy, and truth
  boundary. Present only when the profile/module opts in via
  `quant.codesign.policy` — existing plans are byte-identical by default.
  Named distinctly from the existing per-op `quantization` object (the
  `QuantizationStrategyPlanningPass` output), which is unchanged. Unknown
  metadata (granularity, group size, axis, symmetric/asymmetric, scale,
  zero point) is omitted, never defaulted. No calibration, measured
  accuracy, or quantized execution is claimed.
- Plans may carry an optional `cv_extension` object when the input module was
  annotated by the real upstream-MLIR CV path. This extension is absent for
  Qwen/LLM plans. It records model family, function name, target profile id,
  graph input/output tensor contracts, CV semantic regions, output roles,
  static tensor byte estimates, postprocess boundary, and a CV truth boundary.
  It is metadata and compiler planning evidence only: no runtime execution,
  no measured performance, no numerical validation, and no memory slot
  allocation.
- Per-op bundles may carry a `layout` decision collected from
  `LayoutPlanningPass` attrs: `selected_layout`, `required_input_layout`
  (the producer's layout — a transform boundary exists when they differ),
  and `requires_layout_transform`. A layout decision alone creates a
  bundle only when a transform is required. Layout transforms remain
  planning-only (`BoundaryMaterializationPass` defers them; no layout op
  exists in the hir dialect yet).
- IR materialization is limited to float-precision cast boundaries:
  `BoundaryMaterializationPass` inserts `hir.cast` where the selected plan
  set `boundary.cast_required = true`, and the per-op bundle then reports
  `materialized_boundary_ops` (inserted into IR) alongside
  `deferred_boundary_ops` (`dequant` / `layout_transform` — planned but not
  yet materializable without inventing metadata the planner does not
  produce). Both fields are omitted when materialization did not run or
  recorded nothing for an op. Materialized casts are compiler IR transforms
  only (`truth_boundary =
  compiler_materialized_boundary_op_not_runtime_executed`), not runtime
  execution claims.

V1 (`ServingExecutionPlan.h`) has been removed; `ExecutionPlan` (this schema,
formerly called "V2" internally) is now the only compiler output. The older
`artifacts/vllm_plans/qwen_0_5b_gtx1650_plan.json` artifact and
`tools/export_vllm_execution_plan.py` are a separate, hand-authored Python
planning illustration that predates this contract and is not produced from
this schema or from `artifacts/qwen/execution_plan.json`.

C++ types: `mlir_passes/include/serving/ExecutionPlan.h`

**Per-layer fidelity (Phase 1):** `per_op_decisions` has no `layer_index` or
`layer_range` field today. When the input graph is real per-layer-expanded
IR (from `qwen-onnx-to-serving-mlir`, see `docs/future_work.md`), each real
layer's ops each get their own `op_N` entry — the list is verbose by design
(e.g. ~170 entries per phase for 24 layers), not compressed. Compression into
layer ranges is deferred future work and would be an additive schema field,
not a change to how the compiler's internal IR represents layers (which stays
fully expanded regardless of export compression).

---

## Complete JSON Example: Qwen 2.5-0.5B on GTX 1650 Max-Q

**This example is a schema illustration, not the current pipeline output.** It
shows what a `weight_only_int4` / AWQ `QuantizationDecision` would look like if
the target profile declared AWQ support. The GTX 1650 Max-Q profile actually
in use (`configs/target_profiles/nvidia_gtx1650_maxq.json`) declares
`supportedQuantModes: ["none"]` for both backends (Turing, cc 7.5, no native
INT4 tensor cores) — it does **not** declare `backend.supported_quantization=[awq]`
as this example's `capability_evidence` shows. The execution plan this
pipeline actually generates for GTX 1650
(`artifacts/qwen/execution_plan.json`, produced by
`tools/run_qwen_compiler_pipeline.sh`) carries `strategy: "fp16_fallback"`
per-op, not this AWQ example. Enabling real AWQ/GPTQ quantization (Phase C)
requires either a different/updated target profile that declares int4/AWQ
support, or an explicit experimental forced-quant profile, plus a real
quantized model artifact — see `docs/future_work.md`.

```json
{
  "schema": "execution_plan",
  "schema_version": "2.0.0",
  "plan_id": "qwen_0_5b_gtx1650_v2_001",

  "provenance": {
    "compiler_tool": "compile-for-target",
    "model_spec_ref": "configs/models/qwen_0_5b_spec.json",
    "capability_bundle": {
      "hardware_profile_ref": "nvidia_gtx1650_maxq",
      "backend_profile_refs": ["vllm"],
      "kernel_profile_refs": ["cublas", "triton", "cutlass"],
      "workload_ref": "qwen_short_to_medium_32",
      "deployment_profile_ref": "nvidia_gtx1650_maxq_deployment"
    },
    "truth_boundary": "execution_planning_declared_profiles_not_measured_runtime"
  },

  "model_identity": {
    "model_id": "qwen2.5-0.5b",
    "model_family": "transformer_decoder",
    "num_layers": 24,
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_kv_heads": 2,
    "attention_mechanism": "gqa",
    "positional_encoding": "rope",
    "truth_boundary": "declared_model_config_not_full_graph_import"
  },

  "global_decisions": {

    "quantization": {
      "decision_id": "qd_global_001",
      "decision_type": "QuantizationDecision",
      "scope": "Global",
      "source_pass": "quantization-strategy-planning",
      "strategy": "weight_only_int4",
      "algorithm": "awq",
      "weight_dtype": "int4",
      "activation_dtype": "fp16",
      "accumulation_dtype": "fp16",
      "granularity": "per_group",
      "requires_calibration": true,
      "skip_norm_layers": true,
      "skip_embedding_layers": true,
      "capability_evidence": [
        "hardware.memory_bytes=4294967296",
        "hardware.fp16_native=true",
        "hardware.int4_native=false",
        "backend.supported_quantization=[awq]"
      ],
      "rejected_alternatives": ["weight_only_int8", "fp16"],
      "reason": "4GB VRAM constraint requires int4; AWQ declared by backend profile; int8 weight-only estimated insufficient for 24-layer model at fp16 activations",
      "truth_boundary": "declared_profile_not_accuracy_calibrated"
    },

    "memory": {
      "decision_id": "md_global_001",
      "decision_type": "MemoryDecision",
      "scope": "Global",
      "source_pass": "kv-layout-planning",
      "memory_budget_fraction": 0.75,
      "kv_cache_layout": "paged",
      "kv_block_size_tokens": 16,
      "estimated_kv_peak_mb": 42.0,
      "capability_evidence": [
        "backend.paged_attention.supported=true",
        "deployment.preferred_serving_topology=colocated"
      ],
      "reason": "paged KV declared compatible on target; 0.75 fraction preserves headroom for activations",
      "truth_boundary": "static_formula_estimate_not_measured_memory"
    },

    "serving": {
      "decision_id": "sd_global_001",
      "decision_type": "ServingDecision",
      "scope": "Global",
      "source_pass": "serving-phase-analysis",
      "topology": "colocated",
      "attention_algorithm": "triton_paged_attention",
      "token_budget_per_step": 2048,
      "prefix_reuse_eligible": true,
      "chunked_prefill_eligible": true,
      "replay_eligible": true,
      "speculative_decoding": "disabled",
      "parallelism_degree": 1,
      "parallelism_kind": "none",
      "colocated_cost_estimate_ms": 38.4,
      "pd_split_cost_estimate_ms": 51.2,
      "capability_evidence": [
        "hardware.device_count=1",
        "backend.chunked_prefill.supported=true",
        "backend.prefix_cache.supported=true",
        "workload.shared_prefix_expected=true"
      ],
      "rejected_alternatives": ["prefill_decode_split"],
      "reason": "single device; prefill-decode split adds coordination overhead (~12.8ms) without bandwidth savings at parallelism_degree=1",
      "truth_boundary": "static_formula_estimate_not_measured_runtime"
    },

    "calibration": {
      "decision_id": "cal_global_001",
      "decision_type": "CalibrationDecision",
      "scope": "Global",
      "source_pass": "quantization-strategy-planning",
      "calibration_kind": "awq",
      "target_layer_patterns": ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
      "skip_layer_patterns": ["embed_tokens", "norm", "lm_head"],
      "calibration_dataset_hint": "c4_subset",
      "num_calibration_samples": 512,
      "weight_group_size": "128",
      "zero_point_required": false,
      "reason": "global QuantizationDecision requires_calibration=true; awq selected; lm_head skipped (accuracy-sensitive output projection)",
      "truth_boundary": "calibration_recipe_planning_artifact_weights_not_loaded_accuracy_not_verified"
    }
  },

  "function_plans": [
    {
      "function_name": "qwen_prefill",
      "serving_phase": "Prefill",

      "backend": {
        "decision_id": "bd_prefill_001",
        "decision_type": "BackendDecision",
        "scope": "Function",
        "source_pass": "execution-provider-planning",
        "selected_backend": "cuda_triton",
        "fallback_backends": ["cuda_cublas", "cpu"],
        "capability_evidence": [
          "kernel.flashattention2.availability=unsupported",
          "kernel.triton.attention.availability=opaque",
          "hardware.compute_capability=7.5"
        ],
        "rejected_alternatives": ["flashattention2"],
        "reason": "FlashAttention2 unsupported on cc 7.5; Triton paged attention declared as observed fallback",
        "truth_boundary": "declared_profile"
      },

      "per_op_decisions": [
        {
          "op_name": "prefill_attention_0",
          "op_type": "llm.attention_prefill",
          "kernel": {
            "decision_id": "kd_prefill_attn_001",
            "decision_type": "KernelDecision",
            "scope": "PerOp",
            "source_pass": "kernel-availability-planning",
            "op_type": "llm.attention_prefill",
            "op_name": "prefill_attention_0",
            "selected_kernel": "triton_paged_attention",
            "kernel_library": "triton",
            "lowering_path": "direct_lower",
            "required_boundary_ops": [],
            "kernel_exists": true,
            "rejected_alternatives": ["flashattention2_kernel"],
            "reason": "flashattention2 requires cc >= 8.0; triton paged attention declared for cc 7.5",
            "truth_boundary": "declared_profile"
          }
        },
        {
          "op_name": "prefill_rmsnorm_0",
          "op_type": "llm.rmsnorm",
          "kernel": {
            "decision_id": "kd_prefill_norm_001",
            "decision_type": "KernelDecision",
            "scope": "PerOp",
            "source_pass": "kernel-availability-planning",
            "op_type": "llm.rmsnorm",
            "op_name": "prefill_rmsnorm_0",
            "selected_kernel": "triton_rmsnorm",
            "kernel_library": "triton",
            "lowering_path": "direct_lower",
            "required_boundary_ops": [],
            "kernel_exists": true,
            "truth_boundary": "declared_profile"
          },
          "quantization": {
            "decision_id": "qd_prefill_norm_001",
            "decision_type": "QuantizationDecision",
            "scope": "PerOp",
            "source_pass": "quantization-strategy-planning",
            "op_type": "llm.rmsnorm",
            "strategy": "fp16",
            "algorithm": "none",
            "weight_dtype": "fp16",
            "activation_dtype": "fp16",
            "accumulation_dtype": "fp32",
            "granularity": "none",
            "requires_calibration": false,
            "skip_norm_layers": true,
            "accuracy_risk": "low",
            "reason": "norm layers excluded from weight quantization by global skip rule",
            "truth_boundary": "declared_skip_rule_not_accuracy_measured"
          }
        },
        {
          "op_name": "prefill_qkv_0",
          "op_type": "llm.qkv_projection",
          "kernel": {
            "decision_id": "kd_prefill_qkv_001",
            "decision_type": "KernelDecision",
            "scope": "PerOp",
            "source_pass": "kernel-availability-planning",
            "op_type": "llm.qkv_projection",
            "op_name": "prefill_qkv_0",
            "selected_kernel": "cublas_gemm_fp16",
            "kernel_library": "cublas",
            "lowering_path": "dequant_then_lower",
            "required_boundary_ops": ["dequant"],
            "kernel_exists": true,
            "reason": "cublas does not have native int4 gemm; dequant boundary required before fp16 matmul",
            "truth_boundary": "declared_profile"
          }
        }
      ]
    },
    {
      "function_name": "qwen_decode",
      "serving_phase": "Decode",

      "backend": {
        "decision_id": "bd_decode_001",
        "decision_type": "BackendDecision",
        "scope": "Function",
        "source_pass": "execution-provider-planning",
        "selected_backend": "cuda_triton",
        "fallback_backends": ["cuda_cublas", "cpu"],
        "reason": "same hardware constraints as prefill; decode uses paged KV attention variant",
        "truth_boundary": "declared_profile"
      },

      "per_op_decisions": [
        {
          "op_name": "decode_attention_0",
          "op_type": "llm.attention_decode",
          "kernel": {
            "decision_id": "kd_decode_attn_001",
            "decision_type": "KernelDecision",
            "scope": "PerOp",
            "source_pass": "kernel-availability-planning",
            "op_type": "llm.attention_decode",
            "op_name": "decode_attention_0",
            "selected_kernel": "triton_paged_attention",
            "kernel_library": "triton",
            "lowering_path": "direct_lower",
            "required_boundary_ops": [],
            "kernel_exists": true,
            "truth_boundary": "declared_profile"
          }
        }
      ]
    }
  ]
}
```

---

## Field Semantics

### `provenance`

| Field | Type | Meaning |
|---|---|---|
| `compiler_tool` | string | Which tool produced this plan |
| `model_spec_ref` | string | Path to model architecture spec (not weights) |
| `capability_bundle` | object | Profile IDs used during compilation |
| `truth_boundary` | string | Weakest truth claim across all decisions in this plan |

### `global_decisions`

Each decision type carries `decision_id`, `decision_type`, `scope`, `source_pass`, `capability_evidence`, `rejected_alternatives`, `reason`, and `truth_boundary` from `DecisionMetadata`, plus type-specific fields.

| Decision type | Scope | Key fields |
|---|---|---|
| `QuantizationDecision` | Global or PerOp | strategy, algorithm, weight_dtype, activation_dtype, granularity, requires_calibration |
| `MemoryDecision` | Global | memory_budget_fraction, kv_cache_layout, kv_block_size_tokens |
| `ServingDecision` | Global | topology, attention_algorithm, token_budget_per_step, prefix_reuse_eligible, chunked_prefill_eligible, replay_eligible, parallelism_degree |
| `CalibrationDecision` | Global | calibration_kind, target_layer_patterns, skip_layer_patterns, num_calibration_samples |
| `BackendDecision` | Function | selected_backend, fallback_backends |
| `KernelDecision` | PerOp | selected_kernel, kernel_library, lowering_path, required_boundary_ops |
| `LayoutDecision` | Global or PerOp | selected_layout, requires_layout_transform |
| `FallbackDecision` | PerOp | fallback_kind, fallback_backend, tried_paths |

### Backend vocabulary

`selected_backend` names an **execution unit**, not a runtime:
- `"cuda_cublas"` — CUDA device via cuBLAS
- `"cuda_triton"` — CUDA device via Triton kernels
- `"arm_compute"` — ARM CPU/GPU via ARM Compute Library
- `"coreml_ane"` — Apple ANE via Core ML
- `"cpu"` — fallback host CPU

`"vllm"` is **not** a valid value. vLLM is a runtime that orchestrates these units. Backend selection in the plan is independent of which runtime will execute the plan.

---

## V1 → V2 Field Mapping

### Fields that stay (already backend-agnostic)

| V1 path | V2 path | Note |
|---|---|---|
| `quantization_policy.dtype` | `global_decisions.quantization.weight_dtype` | renamed |
| `quantization_policy.quantization` | `global_decisions.quantization.algorithm` | renamed |
| `backend_execution_plan.primary_backend` | `function_plans[].backend.selected_backend` | renamed |
| `backend_execution_plan.fallback_chain` | `function_plans[].backend.fallback_backends` | renamed |
| `kv_plan.layout` | `global_decisions.memory.kv_cache_layout` | merged into MemoryDecision |
| `kv_plan.kv_byte_estimate_mb` | `global_decisions.memory.estimated_kv_peak_mb` | renamed |
| `replay_plan.replay_eligible` | `global_decisions.serving.replay_eligible` | merged into ServingDecision |
| `per_op_lowering_decisions[].lowering_decision` | `per_op_decisions[].kernel.lowering_path` | renamed |
| `per_op_lowering_decisions[].requires_dequant` | `per_op_decisions[].kernel.required_boundary_ops` includes `"dequant"` | abstracted |
| `per_op_quantization_decisions[].strategy` | `per_op_decisions[].quantization.strategy` | same |
| `truth_boundary` (all levels) | `truth_boundary` (all levels) | unchanged |
| `source_passes[]` | `meta.source_pass` on each decision | moved to per-decision |

### Fields moved to materializers

These fields contained runtime-specific vocabulary. They are removed from the
plan and produced by materializers that read the backend-agnostic decisions.

| V1 field | Materializer | Derived from |
|---|---|---|
| `batch_policy.max_num_seqs` | vLLM materializer → `--max-num-seqs` | workload.expected_concurrency |
| `batch_policy.max_num_batched_tokens` | vLLM materializer → `--max-num-batched-tokens` | serving.token_budget_per_step |
| `batch_policy.enable_chunked_prefill` | vLLM materializer → `--enable-chunked-prefill` | serving.chunked_prefill_eligible |
| `prefix_policy.enable_prefix_caching` | vLLM materializer → `--enable-prefix-caching` | serving.prefix_reuse_eligible |
| `prefix_policy.group_by_shared_prefix` | vLLM materializer | workload.shared_prefix_expected |
| `memory_policy.gpu_memory_utilization` | vLLM materializer → `--gpu-memory-utilization` | memory.memory_budget_fraction |
| `memory_policy.max_model_len` | vLLM materializer → `--max-model-len` | workload.max_prompt_tokens + max_output_tokens |
| `memory_policy.block_size` | vLLM materializer → `--block-size` | memory.kv_block_size_tokens |
| `memory_policy.swap_space` | vLLM materializer | deployment policy; not a compiler decision |
| `speculative_policy.draft_model` | vLLM materializer | deployment policy; not a compiler decision |
| `speculative_policy.num_speculative_tokens` | vLLM materializer | deployment policy; not a compiler decision |
| `runtime_config.tensor_parallel_size` | vLLM materializer → `--tensor-parallel-size` | serving.parallelism_degree |
| `runtime_config.pipeline_parallel_size` | vLLM materializer | serving.parallelism_kind |
| `runtime_config.served_model_name` | vLLM materializer | deployment concern |
| `feedback.baseline_metrics_artifact` | Benchmark manifest materializer | measurement harness concern |
| `feedback.compiler_plan_metrics_artifact` | Benchmark manifest materializer | measurement harness concern |

---

## Truth Boundary Rules

### Rule 1: Every decision carries its own truth_boundary

Each decision struct includes `meta.truth_boundary`. This encodes the epistemic
status of that specific decision. Values follow the existing compiler convention:

| Value | Meaning |
|---|---|
| `declared_profile` | Derived from profile JSON; not measured |
| `declared_model_config_not_full_graph_import` | From architecture constants; weights not loaded |
| `static_formula_estimate_not_measured_*` | From formula; not from a benchmark |
| `calibration_recipe_planning_artifact_weights_not_loaded_accuracy_not_verified` | Recipe only; calibration not run |
| `declared_skip_rule_not_accuracy_measured` | Skip rule applied; accuracy not verified |
| `candidate_evaluation_static_penalty_not_measured_latency` | Static penalty score; not timed |

### Rule 2: Plan-level truth_boundary is the weakest claim

`provenance.truth_boundary` reflects the weakest epistemic claim across all
decisions in the plan. If any decision is `static_formula_estimate`, the plan
cannot claim `declared_profile`. The plan truth_boundary is derived by the
plan builder, not asserted arbitrarily.

### Rule 3: Materializers tighten, never broaden

A materializer reads the plan's decisions and emits a runtime-specific artifact.
The artifact's truth_boundary must be at least as specific as the plan's:

```
plan.truth_boundary = "declared_profile_not_measured"
    → vLLM materializer output:
      "declared_profile_not_measured_vllm_runtime_config_not_executed"
```

Materializers must also report decisions they cannot express as
`unsupported_decisions[]`, rather than silently dropping them.

---

## Phase 26 CV Additions (optional, additive)

CV full-graph plans (currently YOLO-Seg) additionally carry — all fields
absent for LLM plans, which stay byte-identical:

- `function_plans[].dispatch_units[]` — the runtime-facing execution granule:
  one GenericGraphIR source node (or a materialized fusion) with all helper
  MLIR ops (`tensor.empty`/`linalg.fill`/scalar constants/pads) folded
  inside. Carries source provenance (graph/imported node ids, ONNX names,
  op type, operation family, semantic region), tensor IO
  (`input_tensor_ids`/`output_tensor_ids`/`initializer_tensor_ids` — args
  are `arg_<i>`, inter-unit tensors are `<unit_id>:o<k>`), a
  `backend_intent {backend, intent_basis}` (vocabulary:
  `configured_preference | capability_validated | analytically_selected |
  measured_selected | unavailable`), a `kernel_status` (vocabulary:
  `runtime_registered | library_available | lowering_only | deferred |
  fallback_only | unavailable`), static byte/flop estimates, and an
  `executable` flag that is true only for `runtime_registered`.
- `function_plans[].op_classification` — reconciliation: every top-level
  MLIR op classified exactly once (`dispatch_root`,
  `dispatch_internal_compute`, `tensor_contract_operation`,
  `allocation_helper`, `scalar_helper`, `view_operation`,
  `non_dispatch_metadata`, `unresolved`).
- For CV full-graph functions `per_op_decisions` is empty — internal MLIR
  ops are not top-level runtime decisions (LLM functions keep the per-op
  list).
- Top-level `tensor_bindings[]` — typed ABI: role
  (`model_input | initializer | weight | bias | shape_constant | temporary |
  model_output`), original ONNX name, argument index, shape/dtype/layout/
  byte size, ownership (`caller | model_state | runtime | dispatch_unit`),
  mutability, and `model_artifact_reference` (weights referenced, never
  embedded). `provenance.model_spec_ref` carries the model artifact path.
- `cv_extension.memory_summary` — corrected metrics: `model_input_bytes`,
  `initializer_bytes`, `model_output_bytes`,
  `total_intermediate_tensor_bytes`, `total_intermediate_write_bytes`,
  `peak_live_temporary_bytes` (static lifetime scan), `workspace_bytes`,
  `planned_slot_bytes` (null until a slot allocator exists). The legacy
  `memory_estimates.estimated_temporary_bytes` remains and is explicitly
  labeled `cumulative_ssa_result_write_volume_not_peak_live_deprecated`.
- `cv_extension.postprocess_contract` — detection/prototype tensor ids and
  shapes, traced `detection_channel_groups`, mask-coefficient channel range,
  `nms_required`, `mask_decode_required`, `implementation_status`
  (`runtime_required` today), confidence, and provenance.

See `docs/YOLOSEG_DISPATCH_UNIT_MATERIALIZATION.md`.

## What ExecutionPlan Does Not Contain

- Runtime flag names from any specific runtime (vLLM, TensorRT-LLM, ONNX Runtime)
- Measured performance values (latency, throughput, memory usage)
- Model weights (Phase 26 CV plans reference the model artifact path for
  weight binding; weight data itself is never embedded)
- IR materialization operations (cast/dequant/layout_transform nodes are
  described as `required_boundary_ops` in KernelDecision, but not inserted)
- Deployment policy decisions that are not compiler decisions (swap space,
  served model name, replica count)
- Speculative decoding draft model choice (this is a deployment-time concern;
  the plan records `speculative_decoding: "disabled"` or `"eligible_if_draft_available"`)
