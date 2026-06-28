#pragma once

// CVExecutionPlanBuilder reads cv.* attrs from MLIR and produces a typed
// in-memory plan. Missing attrs are represented as "absent" or -1.

#include "serving/CVExecutionPlan.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir::hir {

struct CVExecutionPlanBuilder {
  static CVExecutionPlan build(mlir::ModuleOp module);
};

} // namespace mlir::hir
