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

struct FunctionExecutionPlan {
  std::string function_name;
  ServingPhase serving_phase   = ServingPhase::Unknown;
  ExecutionMode execution_mode = ExecutionMode::Unknown;
  CostSummary cost_summary;
  KVPlan kv_plan;
  ReplayPlan replay_plan;
  BackendExecutionPlan backend_execution_plan;
  CompilerProvenance provenance;
  std::vector<std::string> source_passes; // passes that contributed attrs
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
