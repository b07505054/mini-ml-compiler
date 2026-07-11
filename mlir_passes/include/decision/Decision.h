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

// Shape-derived static cost estimate produced by shape_cost_model_v2
// (ServingCostModelPass) for the selected candidate of one op.
// FLOPs/bytes are derived from static tensor shapes and existing dtype /
// quantization metadata; time estimates exist only when the declared profile
// provided theoretical peak numbers. This is a static compiler estimate —
// not a measured benchmark and not a runtime latency guarantee.
// truth_boundary:
//   static_shape_derived_declared_profile_not_measured_not_runtime_validated
struct ShapeCostEstimate {
  int64_t flops_estimate              = 0;
  int64_t input_bytes_estimate        = 0;
  int64_t output_bytes_estimate       = 0;
  int64_t weight_bytes_estimate       = 0;
  int64_t total_memory_bytes_estimate = 0;
  int64_t arithmetic_intensity_milli  = 0; // flops * 1000 / total bytes
  // Roofline time estimates (integer nanoseconds); absent when the profile
  // declared no peak FLOPs / bandwidth numbers.
  std::optional<int64_t> estimated_compute_cost_nanos;
  std::optional<int64_t> estimated_memory_cost_nanos;
  std::optional<int64_t> estimated_boundary_cost_nanos;
  std::optional<int64_t> estimated_total_cost_nanos;
  std::string status;             // "estimated" | "facts_only_no_profile_numbers"
  std::string cost_model_version; // "shape_cost_model_v2"
  std::string truth_boundary;
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
  // Path/ref to a compiler-produced (or externally produced, compiler-
  // referenced) quantized model artifact, e.g. "artifacts/qwen_awq". Empty
  // when algorithm == "none" or no offline-quantized checkpoint exists.
  // The compiler never loads or inspects this artifact's weights; it only
  // carries the reference through to the runtime materializer.
  std::string quantized_model_artifact_ref;
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
  // Layout arriving from the producer; a transform boundary exists when it
  // differs from selected_layout and requires_layout_transform is true.
  std::string      required_input_layout;
  bool             requires_layout_transform = false;
};

// Compiler-side kernel selection against concrete runtime kernel
// descriptors (kernel_selection_contract_v1, KernelSelectionPass). A
// kernel is selected only when a declared RuntimeKernelDescriptor fully
// matches the planned op; otherwise the status carries an explicit
// rejection/deferral reason. A selection is a contract handed to the
// runtime — the compiler never executes, dispatches, or benchmarks it.
// truth_boundary:
//   kernel_selection_static_descriptor_match_not_runtime_execution
struct KernelSelection {
  // "selected" | "rejected_*" | "deferred_*" (see KernelSelectionPass)
  std::string status;
  std::string selected_kernel_id;            // when status == "selected"
  // Descriptor source when selected: "handwritten_runtime" |
  // "declared_profile" | "measured_runtime" | "fixture"
  std::string source;
  std::vector<std::string> rejection_reasons; // per-descriptor "{id}:{reason}"
  std::string contract_version;               // "kernel_selection_contract_v1"
  std::string truth_boundary;
};

// Quantization co-design evidence (quantization_codesign_contract_v1,
// QuantizationCoDesignPass). Deliberately SEPARATE facts: representation,
// algorithm declaration, backend legality, concrete runtime-kernel support,
// static systems-cost estimate, accuracy evidence, and materialization
// status are never collapsed into one string. Unknown metadata
// (granularity, group size, axis, symmetric/asymmetric, scale, zero point)
// is omitted, never defaulted. Static planning evidence only — no
// calibration, no measured accuracy, no quantized runtime execution.
// truth_boundary:
//   quantization_codesign_static_planning_no_calibration_no_measured_accuracy
struct QuantizationCoDesign {
  std::string status;   // "selected" | "rejected_*" | "deferred_*" | ...
  std::string policy;   // the explicit policy that produced this decision
  // Candidate representation facts (best backend-legal candidate).
  std::string representation;      // "weight_only_int8" | "weight_only_int4"
  std::string weight_dtype;
  std::string activation_dtype;
  std::string accumulator_dtype;   // only when the profile declares one
  // Algorithm declaration (no algorithm is implemented in this repo).
  std::string algorithm_status;    // "not_declared" | "declared_external"
  std::string algorithm_name;      // only when declared (e.g. "awq")
  // Backend capability vs concrete kernel support — kept distinct.
  std::string backend_legality;    // "legal" | "not_legal"
  std::string kernel_support_status;
  std::string kernel_support_kernel_id; // only when runtime_dispatchable
  std::string kernel_support_source;    // only when runtime_dispatchable
  // Accuracy evidence status; artifact_ref only when a quantized artifact
  // is declared by the profile.
  std::string accuracy_evidence_status;
  std::string accuracy_evidence_artifact_ref;
  // Quant parameter provenance (always unavailable today).
  std::string scale_source;
  std::string zero_point_source;
  // Static systems-cost estimate; absent for dynamic shapes / no profile
  // numbers. Units: bytes and integer nanoseconds (roofline form).
  std::optional<int64_t> weight_bytes_before;
  std::optional<int64_t> weight_bytes_after;
  std::optional<int64_t> boundary_bytes;
  std::optional<int64_t> total_cost_before_nanos;
  std::optional<int64_t> total_cost_after_nanos;
  std::optional<int64_t> systems_benefit_nanos;
  std::vector<std::string> excluded_cost_terms; // e.g. inline conversion
  // Materialization facts (the actual IR transform is a separate change).
  bool        materialization_required = false;
  std::string materialization_status;
  std::vector<std::string> rejection_reasons;
  std::string truth_boundary;
  std::string contract_version;
};

// Static local-memory tile plan produced by TilePlanningPass
// (tile_planning_v1) for one matmul-like op. Feasibility against the
// declared local memory capacity plus a reuse-limited traffic estimate.
// Not measured performance, not DMA execution, not codegen, and no claim
// the backend kernel uses this tiling.
// truth_boundary:
//   tile_planning_static_local_memory_model_not_measured_not_codegen
struct TilePlan {
  // "planned" | "no_feasible_tile" | "deferred_missing_memory_hierarchy" |
  // "dynamic_dims_unresolved" | "dtype_unresolved" | "no_tensor_result"
  std::string status;
  int64_t tile_m = 0, tile_n = 0, tile_k = 0;   // valid when status == planned
  int64_t local_memory_bytes = 0;               // selected tile footprint
  int64_t rejected_tile_count = 0;
  int64_t estimated_global_traffic_bytes = 0;   // reuse-limited bound
  bool    double_buffer_fits = false;
  // "async_copy_declared" | "dma_declared" | "declared_unavailable"
  // (declared and false) | "unknown_not_declared" (no declaration)
  std::string staging_capability;
  std::string rejection_reason;                 // when no tile fits
  // Why feasibility was deferred (e.g. the target profile declares no
  // local memory — MemoryHierarchyProfile is optional metadata and absence
  // is never replaced with an invented capacity).
  std::string deferred_reason;
  std::string truth_boundary;
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
