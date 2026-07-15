#pragma once

// ExecutionPlan.h — canonical compiler output contract (schema version 2.0).
//
// ExecutionPlan is the compiler's deliverable: a hardware-aware, backend-
// agnostic set of typed decisions produced by the 15-pass serving pipeline
// and consumed by materializers that translate decisions into runtime-specific
// artifacts.
//
// Design invariants:
//   - All decisions use backend-agnostic vocabulary.
//   - Runtime flag names (max_num_seqs, gpu_memory_utilization, etc.) are NOT
//     present here; they appear only in materializer outputs.
//   - Every decision carries truth_boundary at the decision level.
//   - No measured performance values are stored in this plan.
//   - IR materialization (cast/dequant/layout_transform insertion) is NOT
//     performed; this plan describes planning decisions only.
//
// ExecutionPlan is the sole compiler/runtime contract.

#include "capability/CapabilityBundle.h"
#include "decision/Decision.h"
#include "serving/ServingEnums.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

// References to capability profiles used during compilation.
// The plan carries profile IDs for provenance traceability, not the loaded
// CapabilityBundle struct itself (which is a compiler-session object).
struct CapabilityBundleRefs {
  std::string              hardware_profile_ref;   // "nvidia_gtx1650_maxq"
  std::vector<std::string> backend_profile_refs;   // ["vllm"]
  std::vector<std::string> kernel_profile_refs;    // ["cublas", "triton", "cutlass"]
  std::string              workload_ref;            // "qwen_short_to_medium_32"
  std::string              deployment_profile_ref;  // "nvidia_gtx1650_maxq_deployment"
};

// Model identity snapshot embedded in the plan for self-documentation.
// Populated from ModelCapability at plan-build time.
struct ModelIdentity {
  std::string model_id;               // "qwen2.5-0.5b"
  std::string model_family;           // "transformer_decoder"
  int64_t     num_layers          = 0;
  int64_t     hidden_size         = 0;
  int64_t     num_attention_heads = 0;
  int64_t     num_kv_heads        = 0;
  // attention_mechanism: "gqa" | "mha" | "mqa"
  std::string attention_mechanism;
  // positional_encoding: "rope" | "alibi" | "absolute" | "none"
  std::string positional_encoding;
  std::string truth_boundary;
};

struct PlanProvenanceV2 {
  std::string          compiler_tool;   // "compile-for-target"
  std::string          model_spec_ref;  // "configs/models/qwen_0_5b_spec.json"
  CapabilityBundleRefs capability_bundle;
  std::string          truth_boundary;
};

// Global decisions apply to the entire model compilation.
// All fields are optional because not every compilation produces every global
// decision (e.g., CalibrationDecision is absent when quantization == "none").
struct GlobalDecisions {
  std::optional<QuantizationDecision> quantization;
  std::optional<MemoryDecision>       memory;
  std::optional<ServingDecision>      serving;
  std::optional<CalibrationDecision>  calibration;
};

// All decisions applicable to one op instance within a function.

struct BufferPlacement {
  std::string buffer_id;
  std::string role;
  std::string memory_space;
  int64_t byte_count = 0;
  int64_t alignment = 1;
};

struct TransferOperation {
  std::string transfer_id;
  std::string source_buffer;
  std::string destination_buffer;
  std::string source_memory_space;
  std::string destination_memory_space;
  int64_t byte_count = 0;
  int64_t alignment = 1;
  std::string mode;
  std::vector<std::string> dependency_ids;
  std::string completion_token;
};

struct MemoryPlacementPlan {
  std::string status;
  std::string compute_unit;
  std::string selected_memory_space;
  int64_t input_tile_bytes = 0;
  int64_t weight_tile_bytes = 0;
  int64_t output_tile_bytes = 0;
  int64_t scratch_bytes = 0;
  int64_t padding_bytes = 0;
  int64_t single_buffer_bytes = 0;
  int64_t additional_double_buffer_bytes = 0;
  int64_t total_required_local_memory_bytes = 0;
  std::vector<BufferPlacement> buffer_placements;
  std::vector<TransferOperation> transfer_operations;
  std::vector<std::string> compute_dependency_ids;
  std::string rejection_reason;
  std::string truth_boundary;
};

struct AttentionExecutionContract {
  std::string execution_unit, backend, phase, candidate_id, kernel_id, entry_point;
  std::string artifact_ref, artifact_sha256, artifact_version;
  std::string dtype, input_layout, output_layout, required_isa;
  std::string fallback_identity, truth_boundary;
  int64_t batch = 0, query_length = 0, context_length = 0;
  int64_t num_query_heads = 0, num_kv_heads = 0, head_dim = 0;
  int64_t workspace_bytes = 0, alignment_bytes = 0;
  bool causal = false, runtime_no_redecision = false;
};

struct PerOpDecisionBundle {
  std::string op_name;
  std::string op_type;
  std::optional<KernelDecision>       kernel;
  std::optional<LayoutDecision>       layout;
  std::optional<QuantizationDecision> quantization;  // per-op override; scope = PerOp
  std::optional<FallbackDecision>     fallback;
  // Boundary ops BoundaryMaterializationPass actually inserted into IR for
  // this op (currently only "cast"). Empty when materialization did not run
  // or nothing was materialized.
  std::vector<std::string> materialized_boundary_ops;
  // Boundary requirements materialization explicitly deferred
  // ("dequant" | "layout_transform"): planned but not yet insertable
  // without inventing metadata the planner does not produce.
  std::vector<std::string> deferred_boundary_ops;
  // Shape-derived static cost estimate for the selected candidate
  // (shape_cost_model_v2). Absent when the op fell back to the V1 fixed
  // model (unknown op kind, dynamic shapes) or the pass did not run.
  std::optional<ShapeCostEstimate> shape_cost;
  // Static local-memory tile plan (tile_planning_v1). Absent for
  // non-matmul ops or when the profile declares no local memory.
  std::optional<TilePlan> tile_plan;
  // Concrete runtime-kernel contract selection
  // (kernel_selection_contract_v1). Absent when KernelSelectionPass did
  // not run.
  std::optional<KernelSelection> kernel_selection;
  // Quantization co-design evidence (quantization_codesign_contract_v1).
  // Absent when the co-design pass/policy did not run — existing plans
  // stay byte-identical by default.
  std::optional<QuantizationCoDesign> quantization_codesign;
  // Thread-decomposition schedule (Phase P1D, thread_schedule_contract_v1).
  // A decision SEPARATE from kernel_selection: which kernel/tile runs vs.
  // how many threads and what partitioning it uses. Absent when
  // kernel_selection did not select a kernel, or the selected kernel
  // declares no supported_thread_schedules — existing P1B/P1C plans (no
  // thread schedules declared anywhere) stay byte-identical by default.
  std::optional<ThreadSchedule> thread_schedule;
  // Slice 2 memory placement and transfer contract. Runtime consumes this
  // exactly; absence means no compiler-owned memory plan was produced.
  std::optional<MemoryPlacementPlan> memory_placement;
  std::optional<AttentionExecutionContract> attention_execution;
};

// ---------------------------------------------------------------------------
// Dispatch units (Phase 26, CV full-graph functions only).
//
// A DispatchUnit is the runtime-consumable execution granule: one GenericGraphIR
// source node (or a materialized fusion of several) together with all helper
// MLIR ops (tensor.empty / linalg.fill / scalar constants / pads) emitted while
// lowering it. Helper ops never appear as top-level runtime decisions; they are
// internal implementation detail of their unit.
// ---------------------------------------------------------------------------

// Backend intent basis vocabulary (Phase 26 truth model):
//   "configured_preference" — declared target-profile policy, not validated
//   "capability_validated"  — op-level capability match performed
//   "analytically_selected" — static cost model chose among validated options
//   "measured_selected"     — measured evidence chose the backend
//   "unavailable"           — no backend intent could be formed
struct DispatchBackendIntent {
  std::string backend;        // "coreml" | "metal" | "cpu" | ...
  std::string intent_basis;   // vocabulary above
};

// Kernel status vocabulary (Phase 26 truth model):
//   "runtime_registered" — a concrete runtime kernel descriptor matched
//   "library_available"  — third-party library capability declared, no descriptor
//   "lowering_only"      — analytic lowering path exists, no kernel claim
//   "deferred"           — selection explicitly deferred with a reason
//   "fallback_only"      — only an alternative-backend fallback was planned
//   "unavailable"        — nothing dispatchable or plannable exists
struct DispatchUnit {
  std::string              dispatch_unit_id;        // "du_42"
  std::vector<int64_t>     source_graph_node_ids;   // GenericGraphIR node ids
  std::vector<int64_t>     source_imported_node_ids;// ImportedGraphIR node ids
  std::vector<std::string> source_onnx_node_names;  // "/model.0/conv/Conv"
  std::string              source_op_type;          // ONNX op type, "Conv"
  std::string              operation_family;        // generic op, "nn.conv2d"
  std::string              semantic_region_id;      // "" when not in a CV region
  // Positional references into the annotated MLIR function body
  // ("op_<index>:<mlir_op_name>"), for diagnostics only.
  std::vector<std::string> mlir_operation_refs;
  std::vector<std::string> input_tensor_ids;
  std::vector<std::string> output_tensor_ids;
  std::vector<std::string> initializer_tensor_ids;
  DispatchBackendIntent    backend_intent;
  std::string              execution_domain;        // "unassigned" until validated
  std::string              kernel_status;           // vocabulary above
  std::string              selected_kernel_id;      // "" unless runtime_registered
  std::vector<std::string> fallback_backends;
  std::string              dtype;
  std::string              layout;
  int64_t                  estimated_compute_flops = 0;   // 0 = no estimate
  int64_t                  estimated_read_bytes = 0;
  int64_t                  estimated_write_bytes = 0;
  int64_t                  workspace_bytes = 0;           // no kernel contracts yet
  std::string              decision_provenance;     // source passes / attrs
  bool                     executable = false;
  std::string              non_executable_reason;   // "" when executable
};

// Per-op classification reconciliation for one CV function. Every top-level
// MLIR op receives exactly one classification; the totals must reconcile.
struct DispatchOpClassification {
  int64_t total_mlir_operations = 0;
  int64_t dispatch_root = 0;
  int64_t dispatch_internal_compute = 0;
  int64_t tensor_contract_operation = 0;
  int64_t allocation_helper = 0;
  int64_t scalar_helper = 0;
  int64_t view_operation = 0;
  int64_t non_dispatch_metadata = 0;
  int64_t unresolved = 0;
  int64_t operations_assigned_to_units = 0;
  int64_t source_graph_node_count = 0;
};

// Typed tensor binding ABI (Phase 26). Distinguishes the model image input
// from initializers/weights/biases and identifies outputs, so a runtime can
// bind tensors without out-of-band knowledge. Weights are NOT embedded; the
// model artifact reference locates them.
struct TensorBinding {
  std::string          tensor_id;        // "arg_0", "result_0"
  std::string          original_name;    // "images", "model.0.conv.weight"
  std::string          source_value_id;  // GenericGraphIR value name
  // role: model_input | initializer | weight | bias | shape_constant |
  //       temporary | model_output
  std::string          role;
  int64_t              argument_index = -1;  // -1 when not a function argument
  std::vector<int64_t> shape;
  std::string          dtype;
  std::string          layout;
  int64_t              byte_size = 0;
  // ownership: caller | model_state | runtime | dispatch_unit
  std::string          ownership;
  bool                 is_mutable = false;
  std::string          external_data_reference;   // "" — weights live in model file
  std::string          model_artifact_reference;  // "models/yolo-seg.onnx"
};

// Plan for one serving function (prefill or decode).
struct FunctionPlan {
  std::string                      function_name;   // "qwen_prefill", "qwen_decode"
  ServingPhase                     serving_phase;   // reuses existing ServingPhase enum
  BackendDecision                  backend;
  std::vector<PerOpDecisionBundle> per_op_decisions;
  // Phase 26: populated only for CV full-graph functions; empty for LLM
  // functions, whose plans stay byte-identical.
  std::vector<DispatchUnit>               dispatch_units;
  std::optional<DispatchOpClassification> op_classification;
};

struct TensorContract {
  std::string              tensor_id;
  std::vector<int64_t>     shape;
  std::string              dtype;
  std::string              layout;
  std::string              role;
};

struct CVSemanticRegion {
  std::string              region_id;
  std::string              semantic_role;
  std::string              recognition_confidence;
  int64_t                  operation_count = 0;
  std::vector<std::string> feature_scales;
};

// Corrected memory metrics (Phase 26). The legacy estimated_temporary_bytes
// field was cumulative SSA-result write volume — NOT peak live memory — and is
// preserved for schema compatibility with an explicit deprecated definition.
struct CVMemorySummary {
  int64_t model_input_bytes = 0;             // model image inputs only
  int64_t initializer_bytes = 0;             // weights/biases/constants as args
  int64_t model_output_bytes = 0;
  // Cumulative static byte size of intermediate SSA tensor results that are
  // not function outputs (no deduplication across empty/fill/compute chains).
  int64_t total_intermediate_tensor_bytes = 0;
  // Cumulative static write volume: every top-level op result, incl. outputs.
  int64_t total_intermediate_write_bytes = 0;
  // Static lifetime-scan peak of concurrently live temporaries
  // (Phase 23 algorithm; no slot allocation, no runtime validation).
  int64_t peak_live_temporary_bytes = 0;
  int64_t workspace_bytes = 0;               // no kernel workspace contracts yet
  std::optional<int64_t> planned_slot_bytes; // absent until a slot allocator exists
  std::string truth_boundary;
};

// Runtime-facing CV postprocess contract (Phase 26). Describes what the
// runtime must do beyond the model output boundary; never invents thresholds
// or algorithms the compiler has not proven.
struct CVPostprocessContract {
  std::string detection_tensor_id;
  std::vector<int64_t> detection_shape;
  std::string prototype_tensor_id;
  std::vector<int64_t> prototype_shape;
  // Channel decomposition of the detection tensor, traced from static
  // insert_slice offsets. Empty when the trace failed.
  struct ChannelGroup {
    int64_t     channel_start = 0;
    int64_t     channel_count = 0;
    std::string semantic;          // "box_regression" | "class_scores" | "mask_coefficients"
    std::string source_region_id;  // "" when untraced
  };
  std::vector<ChannelGroup> detection_channel_groups;
  // [-1,-1] when unproven; otherwise [start, end) channel range of mask
  // coefficients within the detection tensor.
  int64_t mask_coefficient_channel_start = -1;
  int64_t mask_coefficient_channel_end = -1;
  std::string nms_required;            // "true" | "false" | "unknown"
  bool mask_decode_required = false;
  // "runtime_required" | "external_framework_required" | "implemented" | "unavailable"
  std::string implementation_status;
  std::string expected_output_semantics;
  std::string confidence;
  std::string provenance;
};

struct CVPlanExtension {
  std::string                    model_family;
  std::string                    function_name;
  std::string                    target_profile_id;
  std::vector<TensorContract>    inputs;
  std::vector<TensorContract>    outputs;
  std::vector<CVSemanticRegion>  semantic_regions;
  int64_t                        estimated_input_bytes = 0;
  int64_t                        estimated_output_bytes = 0;
  int64_t                        estimated_temporary_bytes = 0;
  int64_t                        estimated_total_tensor_bytes = 0;
  std::string                    postprocess_boundary;
  std::string                    truth_boundary;
  // Phase 26 additions; optional so pre-Phase-26 plans stay serializable.
  std::optional<CVMemorySummary>       memory_summary;
  std::optional<CVPostprocessContract> postprocess_contract;
};

// ExecutionPlan — the canonical compiler output.
//
// JSON representation: docs/EXECUTION_PLAN_SCHEMA.md
struct ExecutionPlan {
  std::string                 schema         = "execution_plan";
  std::string                 schema_version = "2.0.0";
  std::string                 plan_id;
  PlanProvenanceV2            provenance;
  ModelIdentity               model_identity;
  GlobalDecisions             global_decisions;
  std::vector<FunctionPlan>   function_plans;
  std::optional<CVPlanExtension> cv_extension;
  // Phase 26: typed tensor ABI. Empty for LLM plans (byte-stable Qwen output).
  std::vector<TensorBinding>  tensor_bindings;
};

} // namespace mlir::hir
