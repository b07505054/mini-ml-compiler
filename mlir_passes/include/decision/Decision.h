#pragma once

// Decision.h — typed compiler decision contract.
//
// Each Decision type represents one planning outcome produced by a compiler
// pass.  All decisions share a common DecisionMetadata header and use
// backend-agnostic vocabulary.  No field in any Decision type encodes
// the flag name of a specific runtime (e.g. vLLM, TensorRT-LLM).
//
// No MLIR headers are included here; this file is usable by tools,
// materializers, and tests that do not link against MLIR.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

enum class DecisionScope { Global, Function, PerOp };

// Structured cost evidence produced by ServingCostModelPass.
// All components are relative static penalties — not measured latency.
// total_cost is the exact sum of all components, computed by the pass.
// The builder reads but never recomputes it; it may assert total == sum (debug).
// truth_boundary distinguishes static estimate from measured profiling.
struct DecisionCost {
  int64_t compute_cost          = 0; // algebraic_decomposition overhead
  int64_t memory_cost           = 0; // dtype/format mismatch (representation_conversion)
  int64_t dequant_cost          = 0; // weight dequantization boundary op
  int64_t requant_cost          = 0; // requantization boundary op
  int64_t layout_transform_cost = 0; // layout_transform boundary or standalone conversion
  int64_t cast_cost             = 0; // cast boundary or standalone cast_conversion
  int64_t backend_switch_cost   = 0; // cross-backend fallback routing
  int64_t launch_overhead_cost  = 0; // extra kernel launches (boundary ops, fallback path)
  int64_t kv_cache_cost         = 0; // KV layout mismatch (reserved: 0 until kv.* attrs exist)
  int64_t transfer_cost         = 0; // device transfer in backend_fallback path
  int64_t unsupported_penalty   = 0; // no viable lowering path sentinel
  int64_t total_cost            = 0; // exact sum of all above (pass-computed, not builder-derived)
  std::string cost_model_id;         // "serving_static_cost_model_v1"
  std::string truth_boundary;        // "serving_static_cost_model_v1_not_measured_latency"
};

// Decision justification container.
// Owns cost evidence and capability/alternative evidence.
struct DecisionEvidence {
  std::vector<std::string>    capability_refs;       // capability facts that constrained this decision
  std::vector<std::string>    rejected_alternatives; // candidates considered and rejected
  std::optional<DecisionCost> cost;                  // present when ServingCostModelPass ran
};

// Fields shared by every Decision type.
struct DecisionMetadata {
  std::string      decision_id;   // unique within plan: "qd_global_001"
  std::string      decision_type; // "QuantizationDecision", "KernelDecision", etc.
  DecisionScope    scope = DecisionScope::Global;
  std::string      source_pass;   // compiler pass that produced this decision
  std::string      reason;
  std::string      truth_boundary;
  DecisionEvidence evidence;      // capability refs, rejected alternatives, and cost
};

// Quantization strategy for the model (Global) or one op (PerOp).
struct QuantizationDecision {
  DecisionMetadata meta;
  // strategy: "weight_only_int4" | "weight_only_int8" | "static_int8" |
  //           "dynamic_int8" | "fp16" | "bf16" | "none"
  std::string strategy;
  // algorithm: "awq" | "gptq" | "ptq" | "none"
  // Not AWQ-specific; CalibrationDecision handles algorithm-specific config.
  std::string algorithm;
  std::string weight_dtype;          // "int4" | "int8" | "fp16" | "bf16"
  std::string activation_dtype;      // "int8" | "fp16" | "bf16"
  std::string accumulation_dtype;    // "fp16" | "fp32"
  // granularity: "per_channel" | "per_tensor" | "per_group"
  std::string granularity;
  bool        requires_calibration  = false;
  bool        skip_norm_layers      = false;
  bool        skip_embedding_layers = false;
  std::string op_type;               // populated only when scope == PerOp
  // accuracy_risk: "low" | "medium" | "high" | ""
  std::string accuracy_risk;
};

// Backend selection for a function (Function) or op (PerOp).
// selected_backend names an execution unit, not a runtime.
// Valid values: "cuda_cublas" | "cuda_triton" | "arm_compute" | "coreml_ane" | "cpu"
// "vllm" is NOT a valid value — vLLM is a runtime, not a backend.
struct BackendDecision {
  DecisionMetadata         meta;
  std::string              selected_backend;
  std::vector<std::string> fallback_backends;
};

// Kernel selection for one op instance.
struct KernelDecision {
  DecisionMetadata         meta;
  std::string              op_type;
  std::string              op_name;
  std::string              selected_kernel;     // "cublas_gemm_fp16" | "triton_paged_attention"
  std::string              kernel_library;      // "cublas" | "triton" | "cutlass" | "coreml_builtin"
  // lowering_path: "direct_lower" | "rewrite_then_lower" | "dequant_then_lower" |
  //                "fallback_backend" | "unsupported"
  std::string              lowering_path;
  // Abstract boundary ops required before kernel dispatch.
  // Vocabulary: "dequant" | "layout_transform" | "cast"
  std::vector<std::string> required_boundary_ops;
  bool                     kernel_exists = false;
};

// Memory budget and KV cache strategy (Global scope).
struct MemoryDecision {
  DecisionMetadata meta;
  double           memory_budget_fraction = 0.0; // fraction of hardware memory for model+kv
  // kv_cache_layout: "paged" | "contiguous"
  std::string      kv_cache_layout;
  int64_t          kv_block_size_tokens   = 0;   // abstract token block unit
  double           estimated_kv_peak_mb   = 0.0; // static formula estimate, not measured
};

// Tensor memory layout for a global preference or one op.
struct LayoutDecision {
  DecisionMetadata meta;
  std::string      op_type;          // empty when scope == Global
  // selected_layout: "nhwc" | "nchw" | "blocked_kc" | "row_major" | "paged_kv"
  std::string      selected_layout;
  bool             requires_layout_transform = false;
};

// Serving topology and scheduling strategy (Global scope).
//
// Backend-agnostic field mapping — for documentation only; NOT encoded here:
//   token_budget_per_step    -> vLLM: --max-num-batched-tokens
//   prefix_reuse_eligible    -> vLLM: --enable-prefix-caching
//   chunked_prefill_eligible -> vLLM: --enable-chunked-prefill
//   parallelism_degree       -> vLLM: --tensor-parallel-size
// These translations are the materializer's responsibility, not the plan's.
struct ServingDecision {
  DecisionMetadata meta;
  // topology: "colocated" | "prefill_decode_split"
  std::string topology;
  // attention_algorithm: "paged_attention" | "flash_attention" | "triton_paged_attention"
  std::string attention_algorithm;
  int64_t     token_budget_per_step     = 0;     // abstract token budget per scheduling step
  bool        prefix_reuse_eligible     = false;
  bool        chunked_prefill_eligible  = false;
  bool        replay_eligible           = false;
  // speculative_decoding: "disabled" | "eligible_if_draft_available"
  std::string speculative_decoding;
  int64_t     parallelism_degree        = 1;
  // parallelism_kind: "none" | "tensor_parallel" | "pipeline_parallel"
  std::string parallelism_kind;
  double      colocated_cost_estimate_ms = 0.0;  // static formula, not measured runtime
  double      pd_split_cost_estimate_ms  = 0.0;  // static formula, not measured runtime
};

// Calibration recipe (Global scope).
// Present only when a QuantizationDecision requires offline calibration.
// calibration_kind is NOT AWQ-specific; this struct handles any calibration
// algorithm uniformly.  Algorithm-specific hyperparameters use abstract names
// (weight_group_size vs. "awq_group_size") so the compiler never hardcodes
// AWQ vocabulary.
struct CalibrationDecision {
  DecisionMetadata         meta;
  // calibration_kind: "awq" | "gptq" | "ptq_static" | "none"
  std::string              calibration_kind;
  std::vector<std::string> target_layer_patterns;     // which layers to calibrate
  std::vector<std::string> skip_layer_patterns;       // which layers to skip
  std::string              calibration_dataset_hint;  // "c4_subset" | "pile_subset" | ""
  int64_t                  num_calibration_samples = 0;
  std::string              weight_group_size;         // "128" | "64" | "" (per-group quant)
  bool                     zero_point_required = false;
};

// Fallback selection for one op when no direct or alternative path works.
struct FallbackDecision {
  DecisionMetadata         meta;
  std::string              op_type;
  std::string              op_name;
  // fallback_kind: "alternative_backend" | "algebraic_decompose" |
  //                "representation_convert" | "unsupported"
  std::string              fallback_kind;
  std::string              fallback_backend;  // non-empty only when fallback_kind == "alternative_backend"
  std::vector<std::string> tried_paths;
};

} // namespace mlir::hir
