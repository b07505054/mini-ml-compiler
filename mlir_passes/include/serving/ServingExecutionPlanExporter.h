#pragma once

// ServingExecutionPlanExporter: tool-boundary JSON serializer.
//
// NOT part of compiler core. No MLIR passes or analysis logic here.
// Owned entirely at the compile-for-target tool boundary.
// Uses llvm::json (LLVMSupport) — compiler core never sees JSON types.
//
// Two outputs from a single ServingExecutionPlan instance:
//   exportToFile()        — canonical runtime artifact consumed by heterogeneous-inference-runtime
//   exportSummaryToFile() — human/demo artifact, NEVER consumed by runtime

#include "serving/ServingExecutionPlan.h"

#include "llvm/Support/Error.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::hir {

struct ServingExecutionPlanExporter {
  // Emit the canonical runtime artifact (schema_version "1.0.0",
  // artifact_type "serving_execution_plan").
  // Must match heterogeneous-inference-runtime RuntimeExecutionPlanAdapter contract.
  static llvm::Error exportToFile(const ServingExecutionPlan &plan,
                                  llvm::StringRef outPath);

  // Emit the human-readable summary artifact (artifact_type
  // "serving_execution_plan_summary"). Runtime must never read this file.
  // Both methods must be called on the same ServingExecutionPlan instance.
  static llvm::Error exportSummaryToFile(const ServingExecutionPlan &plan,
                                         llvm::StringRef summaryPath);
};

} // namespace mlir::hir
