#include "serving/CVExecutionPlanBuilder.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::hir {
namespace {

std::string getStringAttr(mlir::Operation *op, llvm::StringRef name) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(name))
    return attr.getValue().str();
  return "absent";
}

int64_t getIntAttr(mlir::Operation *op, llvm::StringRef name) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getInt();
  return -1;
}

bool hasCompletedExecutionDomainPlan(mlir::func::FuncOp funcOp) {
  auto status = funcOp->getAttrOfType<mlir::StringAttr>(
      "cv.execution_domain_plan.status");
  return status && status.getValue() == "completed";
}

} // namespace

CVExecutionPlan CVExecutionPlanBuilder::build(mlir::ModuleOp module) {
  CVExecutionPlan plan;

  if (auto attr =
          module->getAttrOfType<mlir::StringAttr>("cv.target.profile_id"))
    plan.target_profile_id = attr.getValue().str();

  module.walk([&](mlir::func::FuncOp funcOp) {
    if (!hasCompletedExecutionDomainPlan(funcOp))
      return;

    CVGraphPlan graphPlan;
    graphPlan.function_name = funcOp.getName().str();

    if (plan.model_name == "unknown") {
      if (auto attr =
              funcOp->getAttrOfType<mlir::StringAttr>("cv.frontend.model_family"))
        plan.model_name = attr.getValue().str();
    }

    graphPlan.memory_plan.buffer_count =
        getIntAttr(funcOp, "cv.memory_plan.buffer_count");
    graphPlan.memory_plan.peak_memory_bytes =
        getIntAttr(funcOp, "cv.memory_plan.peak_memory_bytes");
    graphPlan.memory_plan.planned_ops =
        getIntAttr(funcOp, "cv.memory_plan.planned_ops");
    graphPlan.memory_plan.reused_buffer_count =
        getIntAttr(funcOp, "cv.memory_plan.reused_buffer_count");
    graphPlan.memory_plan.skipped_ops =
        getIntAttr(funcOp, "cv.memory_plan.skipped_ops");
    graphPlan.memory_plan.total_allocated_bytes =
        getIntAttr(funcOp, "cv.memory_plan.total_allocated_bytes");
    graphPlan.memory_plan.status =
        getStringAttr(funcOp, "cv.memory_plan.status");
    graphPlan.memory_plan.truth_boundary =
        getStringAttr(funcOp, "cv.memory_plan.truth_boundary");

    graphPlan.execution_domain_plan.accelerated_ops =
        getIntAttr(funcOp, "cv.execution_domain_plan.accelerated_ops");
    graphPlan.execution_domain_plan.host_ops =
        getIntAttr(funcOp, "cv.execution_domain_plan.host_ops");
    graphPlan.execution_domain_plan.fallback_ops =
        getIntAttr(funcOp, "cv.execution_domain_plan.fallback_ops");
    graphPlan.execution_domain_plan.planned_ops =
        getIntAttr(funcOp, "cv.execution_domain_plan.planned_ops");
    graphPlan.execution_domain_plan.status =
        getStringAttr(funcOp, "cv.execution_domain_plan.status");
    graphPlan.execution_domain_plan.truth_boundary =
        getStringAttr(funcOp, "cv.execution_domain_plan.truth_boundary");

    graphPlan.truth_boundaries.frontend =
        getStringAttr(funcOp, "cv.frontend.truth_boundary");
    graphPlan.truth_boundaries.shape_inference =
        getStringAttr(funcOp, "cv.shape_inference.truth_boundary");
    graphPlan.truth_boundaries.memory_plan =
        getStringAttr(funcOp, "cv.memory_plan.truth_boundary");
    graphPlan.truth_boundaries.execution_domain =
        getStringAttr(funcOp, "cv.execution_domain_plan.truth_boundary");

    funcOp.walk([&](mlir::Operation *op) {
      if (!op->getName().getStringRef().starts_with("cv."))
        return;

      CVExecutionStep step;
      step.op_name = op->getName().getStringRef().str();
      step.source_op = getStringAttr(op, "cv.source_op");
      step.execution_domain = getStringAttr(op, "cv.execution_domain");
      step.execution_domain_reason =
          getStringAttr(op, "cv.execution_domain_reason");
      step.memory_plan_reason = getStringAttr(op, "cv.memory_plan_reason");
      step.buffer_id = getIntAttr(op, "cv.buffer_id");
      step.buffer_offset = getIntAttr(op, "cv.buffer_offset");
      step.lifetime_begin = getIntAttr(op, "cv.lifetime_begin");
      step.lifetime_end = getIntAttr(op, "cv.lifetime_end");
      step.reuse_group = getIntAttr(op, "cv.reuse_group");
      step.bytes_estimate = getIntAttr(op, "cv.bytes_estimate");
      step.num_elements = getIntAttr(op, "cv.num_elements");

      graphPlan.steps.push_back(std::move(step));
    });

    plan.graph_plans.push_back(std::move(graphPlan));
  });

  return plan;
}

} // namespace mlir::hir
