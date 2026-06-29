#pragma once

// CVExecutionPlanExporter: JSON serializer for the typed CV execution plan.
//
// This is not a pass and does not perform backend mapping or artifact
// generation beyond writing the requested JSON file.

#include "serving/CVExecutionPlan.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::hir {

struct CVExecutionPlanExporter {
  static llvm::Error exportToFile(const CVExecutionPlan &plan,
                                  llvm::StringRef outPath);
};

} // namespace mlir::hir
