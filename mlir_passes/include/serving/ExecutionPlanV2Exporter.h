#pragma once

// ExecutionPlanV2Exporter — serializes ExecutionPlanV2 to JSON.
//
// Output schema matches the runtime parse_execution_plan_v2 contract:
//   schema: "execution_plan", schema_version: "2.0.0"
//   Required top-level fields: plan_id, provenance, provenance.capability_bundle,
//     model_identity, global_decisions, function_plans.
//   No measured fields (measured_latency_ms, speedup, metrics, etc.).
//   No V1 fields (artifact_type: "serving_execution_plan", schema_version: "1.0.0").

#include "serving/ExecutionPlanV2.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::hir {

class ExecutionPlanV2Exporter {
public:
  // Serialize plan to JSON at outPath.
  // Creates parent directories as needed.
  // Returns llvm::Error on write failure.
  static llvm::Error exportToFile(const ExecutionPlanV2 &plan,
                                  llvm::StringRef outPath);
};

} // namespace mlir::hir
