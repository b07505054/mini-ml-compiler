#pragma once

// Typed in-memory CV execution plan produced from cv.* compiler annotations.
// No serialization, JSON, artifact generation, or backend assignment lives here.

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::hir {

struct CVExecutionStep {
  std::string op_name;
  std::string source_op;
  std::string execution_domain;
  std::string execution_domain_reason;
  std::string memory_plan_reason;

  int64_t buffer_id = -1;
  int64_t buffer_offset = -1;
  int64_t lifetime_begin = -1;
  int64_t lifetime_end = -1;
  int64_t reuse_group = -1;
  int64_t bytes_estimate = -1;
  int64_t num_elements = -1;
};

struct CVMemoryPlanSummary {
  int64_t buffer_count = -1;
  int64_t peak_memory_bytes = -1;
  int64_t planned_ops = -1;
  int64_t reused_buffer_count = -1;
  int64_t skipped_ops = -1;
  int64_t total_allocated_bytes = -1;
  std::string status = "absent";
  std::string truth_boundary = "absent";
};

struct CVExecutionDomainSummary {
  int64_t accelerated_ops = -1;
  int64_t host_ops = -1;
  int64_t fallback_ops = -1;
  int64_t planned_ops = -1;
  std::string status = "absent";
  std::string truth_boundary = "absent";
};

struct CVTruthBoundaries {
  std::string frontend = "absent";
  std::string shape_inference = "absent";
  std::string memory_plan = "absent";
  std::string execution_domain = "absent";
};

struct CVGraphPlan {
  std::string function_name;
  CVMemoryPlanSummary memory_plan;
  CVExecutionDomainSummary execution_domain_plan;
  CVTruthBoundaries truth_boundaries;
  std::vector<CVExecutionStep> steps;
};

struct CVExecutionPlan {
  std::string model_name = "unknown";
  std::string target_profile_id = "portable_cv_pseudo_target";
  std::vector<CVGraphPlan> graph_plans;
};

} // namespace mlir::hir
