#pragma once

// ServingExecutionPlanBuilder: reads MLIR attributes annotated by the serving
// pass pipeline and populates a ServingExecutionPlan C++ struct.
//
// Ownership boundary: compiler core only. No serialization. No Python.
// Serialization belongs in a separate export tool outside compiler core.

#include "serving/ServingExecutionPlan.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::hir {

struct ServingExecutionPlanBuilder {
  // Walk module, read serving.* attrs from annotated func.func ops, and
  // return the in-memory compiler product. Missing optional attrs (kv.*,
  // replay.*) are left at struct defaults until the corresponding passes run.
  static ServingExecutionPlan build(mlir::ModuleOp module);
};

} // namespace mlir::hir
