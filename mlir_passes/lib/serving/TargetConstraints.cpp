#include "serving/TargetConstraints.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::hir {

TargetConstraints TargetConstraints::fromModule(mlir::ModuleOp module) {
  TargetConstraints tc;
  mlir::Operation *op = module.getOperation();

  if (auto a = op->getAttrOfType<mlir::StringAttr>("target.profile_id"))
    tc.profile_id = a.getValue().str();

  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.memory_budget_mb")) {
    tc.memory_budget_mb  = a.getValueAsDouble();
    tc.has_memory_budget = true;
  }

  if (auto a = op->getAttrOfType<mlir::BoolAttr>("target.static_shape_support")) {
    tc.static_shape_support  = a.getValue();
    tc.has_static_shape_support = true;
  }

  if (auto a = op->getAttrOfType<mlir::FloatAttr>("target.frame_latency_budget_ms")) {
    tc.frame_latency_budget_ms = a.getValueAsDouble();
    tc.has_frame_latency_budget = true;
  }

  if (auto a = op->getAttrOfType<mlir::StringAttr>("target.preferred_backend"))
    tc.preferred_backend = a.getValue().str();

  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("target.allowed_backends")) {
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        tc.allowed_backends.push_back(s.getValue().str());
  }

  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("target.supported_precisions")) {
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        tc.supported_precisions.push_back(s.getValue().str());
  }

  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("target.paged_kv_compatible_backends")) {
    for (mlir::Attribute elem : a)
      if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
        tc.paged_kv_compatible_backends.push_back(s.getValue().str());
  }

  return tc;
}

void TargetConstraints::attachToModule(mlir::ModuleOp module,
                                       mlir::MLIRContext *ctx) const {
  mlir::Operation *op = module.getOperation();
  mlir::Type f64 = mlir::Float64Type::get(ctx);

  if (!profile_id.empty())
    op->setAttr("target.profile_id", mlir::StringAttr::get(ctx, profile_id));

  if (has_memory_budget)
    op->setAttr("target.memory_budget_mb",
                mlir::FloatAttr::get(f64, memory_budget_mb));

  if (has_static_shape_support)
    op->setAttr("target.static_shape_support",
                mlir::BoolAttr::get(ctx, static_shape_support));

  if (has_frame_latency_budget)
    op->setAttr("target.frame_latency_budget_ms",
                mlir::FloatAttr::get(f64, frame_latency_budget_ms));

  if (!preferred_backend.empty())
    op->setAttr("target.preferred_backend",
                mlir::StringAttr::get(ctx, preferred_backend));

  if (!allowed_backends.empty()) {
    llvm::SmallVector<mlir::Attribute> elems;
    for (const auto &b : allowed_backends)
      elems.push_back(mlir::StringAttr::get(ctx, b));
    op->setAttr("target.allowed_backends", mlir::ArrayAttr::get(ctx, elems));
  }

  if (!supported_precisions.empty()) {
    llvm::SmallVector<mlir::Attribute> elems;
    for (const auto &p : supported_precisions)
      elems.push_back(mlir::StringAttr::get(ctx, p));
    op->setAttr("target.supported_precisions",
                mlir::ArrayAttr::get(ctx, elems));
  }

  // Always emit target.paged_kv_compatible_backends — even when empty — so
  // passes can distinguish "profile says no paged-KV backends" from
  // "profile was not lowered" (which leaves the attr absent).
  {
    llvm::SmallVector<mlir::Attribute> elems;
    for (const auto &b : paged_kv_compatible_backends)
      elems.push_back(mlir::StringAttr::get(ctx, b));
    op->setAttr("target.paged_kv_compatible_backends",
                mlir::ArrayAttr::get(ctx, elems));
  }
}

} // namespace mlir::hir
