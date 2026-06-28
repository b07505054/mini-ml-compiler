#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::hir {

// Returns the compiler-decided plan dtype from quantization.plan_dtype on the
// module op (emitted by QuantizationPlanningPass). Falls back to "fp16" when
// the attribute is absent or the module pointer is null.
inline llvm::StringRef getPlanDtype(Operation *module) {
  if (!module) return "fp16";
  if (auto a = module->getAttrOfType<StringAttr>("quantization.plan_dtype"))
    return a.getValue();
  return "fp16";
}

// Maps the plan dtype string to an element byte count for cost and memory
// estimates. Unknown or missing dtypes default to 2.0 (fp16).
inline double dtypeBytesFromPlan(Operation *module) {
  llvm::StringRef dtype = getPlanDtype(module);
  if (dtype == "fp32") return 4.0;
  if (dtype == "fp16") return 2.0;
  if (dtype == "bf16") return 2.0;
  if (dtype == "int8") return 1.0;
  if (dtype == "fp8")  return 1.0;
  return 2.0;
}

} // namespace mlir::hir
