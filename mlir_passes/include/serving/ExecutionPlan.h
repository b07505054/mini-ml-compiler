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
};

// Plan for one serving function (prefill or decode).
struct FunctionPlan {
  std::string                      function_name;   // "qwen_prefill", "qwen_decode"
  ServingPhase                     serving_phase;   // reuses existing ServingPhase enum
  BackendDecision                  backend;
  std::vector<PerOpDecisionBundle> per_op_decisions;
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
};

} // namespace mlir::hir
