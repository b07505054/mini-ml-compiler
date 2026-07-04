#pragma once

// No serialization headers. No JSON. No Python.
// This is the typed in-memory compiler product produced by the serving pass
// pipeline and consumed by ServingExecutionPlanBuilder.

#include "serving/ServingEnums.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::hir {

// Compiler provenance annotation: identifies the reliability and origin of
// compiler decisions. Not a deployment SLO or validation expectation.
struct CompilerProvenance {
  std::string truth_boundary; // e.g. "estimated_cost_not_measured_latency"
  std::string cost_source;    // e.g. "formula_synthetic"
};

struct CostSummary {
  double colocated_total_ms  = 0.0;
  double pd_split_total_ms   = 0.0;
  double decision_margin_ms  = 0.0;
  double decision_margin_pct = 0.0;
  Confidence confidence      = Confidence::Low;
};

// KV cache layout plan. Populated by KVLayoutPlanningPass when implemented.
// Defaults to Unknown until that pass runs.
struct KVPlan {
  KVLayout layout            = KVLayout::Unknown;
  double kv_byte_estimate_mb = 0.0;
};

// CUDA graph replay eligibility. Populated by ReplayEligibilityPass when
// implemented. Defaults to false (not eligible) until that pass runs.
struct ReplayPlan {
  bool replay_eligible           = false;
  std::string cuda_graph_bucket; // empty if not eligible
};

// Compiler-side quantization decision. Populated by the builder from
// quantization.effective_dtype / quantization.dtype_bytes on func ops
// (emitted by ServingPhaseAnalysisPass) and from module-level
// quantization.plan_* attrs (emitted by QuantizationPlanningPass when it ran).
// dtype_bytes and plan_dtype are used by downstream KV cost estimates.
struct QuantizationPlan {
  std::string plan_dtype;        // e.g. "fp16", "int8"
  std::string plan_source;       // e.g. "target_profile", "default_dtype"
  double dtype_bytes = 2.0;      // bytes per element for cost/memory formulas
  std::string truth_boundary;
};

// Compiler-produced execution provider plan. Populated by
// ExecutionProviderPlanningPass. Encodes static backend selection decisions
// derived from target.* module attrs and prior-pass kv.* / replay.* attrs.
// Runtime is responsible for dynamic dispatch, availability checks, and
// thermal overrides.
struct BackendExecutionPlan {
  std::string primary_backend;                   // "coreml", "metal", "cuda", "cpu", etc.
  std::vector<std::string> fallback_chain;       // ordered fallbacks, excluding primary
  std::string decision_source;                   // closed-set tag
  std::string required_precision;                // "fp16", "fp32", "int8", etc.
  std::string required_kv_layout;                // "paged", "contiguous", or "unknown"
  bool requires_replay = false;                  // copy of replay.eligible
};

// Per-op quantization strategy decision exported from quant.* MLIR attrs.
// Populated by ServingExecutionPlanBuilder when QuantizationStrategyPlanningPass
// has run. Fields mirror quant.* attrs; absent attrs leave fields empty/false.
struct OpQuantizationDecision {
  std::string op_name;                    // SSA result name or "op_N" index
  std::string op_type;                    // e.g. "hir.matmul"
  std::string backend;
  std::string strategy;                   // "weight_only_int8" | "fp16_fallback" | ...
  std::string weight_dtype;
  std::string activation_dtype;
  std::string accumulation_dtype;
  std::string output_dtype;
  std::string granularity;
  std::string activation_quant_mode;
  std::string weight_quant_mode;
  bool requires_dequant_boundary  = false;
  bool requires_requant_boundary  = false;
  std::string decision_reason;
  std::string fallback_reason;
  std::string accuracy_risk;
  std::string truth_boundary;
};

// Per-op lowering decision exported from lowering.* MLIR attrs.
// Populated by ServingExecutionPlanBuilder when LoweringDecisionPlanningPass
// has run. Fields mirror lowering.* attrs; absent attrs leave fields empty/false.
// kernel.* fields are read from the same op for provenance.
struct OpLoweringDecision {
  std::string op_name;              // "op_N" index
  std::string op_type;              // e.g. "hir.matmul"
  std::string backend;              // from kernel.backend
  std::string kernel_library;       // from kernel.library
  std::string kernel_name;          // from kernel.name
  bool kernel_exists                = false;  // from kernel.exists
  std::string kernel_lowering_status;         // from kernel.lowering_status
  std::string lowering_decision;              // direct_lower | rewrite_then_lower |
                                              // dequant_then_lower | fallback_backend | unsupported
  std::string target_backend;       // from lowering.target_backend
  std::string target_kernel_library; // from lowering.target_kernel_library
  std::string target_kernel;        // from lowering.target_kernel
  std::string required_rewrite;     // from lowering.required_rewrite
  bool requires_dequant             = false;  // from lowering.requires_dequant
  bool requires_layout_transform    = false;  // from lowering.requires_layout_transform
  bool requires_cast                = false;  // from lowering.requires_cast
  std::string reason;               // from lowering.reason
  std::string truth_boundary;       // from lowering.truth_boundary
};

struct FunctionExecutionPlan {
  std::string function_name;
  ServingPhase serving_phase   = ServingPhase::Unknown;
  ExecutionMode execution_mode = ExecutionMode::Unknown;
  CostSummary cost_summary;
  KVPlan kv_plan;
  ReplayPlan replay_plan;
  QuantizationPlan quantization_plan;
  BackendExecutionPlan backend_execution_plan;
  CompilerProvenance provenance;
  std::vector<std::string> source_passes;                    // passes that contributed attrs
  std::vector<OpQuantizationDecision> per_op_quant_decisions; // empty if pass did not run
  std::vector<OpLoweringDecision> per_op_lowering_decisions;  // empty if pass did not run
};

struct ServingExecutionPlan {
  std::string target_profile_id; // "" if no target.profile_id module attr present
  std::string model_name;
  int64_t num_layers           = 0;
  int64_t hidden_size          = 0;
  int64_t num_attention_heads  = 0; // 0 if absent (e.g. tiny-gpt fixture)
  int64_t num_kv_heads         = 0; // 0 if absent; non-zero signals GQA
  std::vector<FunctionExecutionPlan> function_plans;
};

} // namespace mlir::hir
