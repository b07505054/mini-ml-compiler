#pragma once

// ExecutionPlanExporter — serializes ExecutionPlan to JSON.
//
// Output schema matches the runtime parse_execution_plan contract:
//   schema: "execution_plan", schema_version: "2.0.0"
//   Required top-level fields: plan_id, provenance, provenance.capability_bundle,
//     model_identity, global_decisions, function_plans.
//   No measured fields (measured_latency_ms, speedup, metrics, etc.).

#include "serving/ExecutionPlan.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::hir {

class ExecutionPlanExporter {
public:
  // Serialize plan to JSON at outPath.
  // Creates parent directories as needed.
  // Returns llvm::Error on write failure.
  static llvm::Error exportToFile(const ExecutionPlan &plan,
                                  llvm::StringRef outPath);

  // Phase 26: reconciliation report for dispatch-unit materialization
  // (op counts, classification totals, tensor binding roles, memory metric
  // reconciliation). Diagnostic artifact; not part of the runtime contract.
  static llvm::Error exportDispatchUnitReport(const ExecutionPlan &plan,
                                              llvm::StringRef outPath);
};

} // namespace mlir::hir
