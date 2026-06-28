#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CVEXECUTIONDOMAINPLANNING
#include "FusionPasses.h.inc"

// Pipeline context:
//
//   Commit D (this pass):  cv-execution-domain-planning
//     classifies each cv.* op into a portable execution domain using op name
//     only: accelerated / host / fallback.
//
//   Commit E (future):     CVExecutionPlanBuilder
//     maps (execution_domain + target_profile) to a concrete backend:
//       accelerated + apple_profile  -> metal
//       accelerated + cuda_profile   -> cuda
//       accelerated + mobile_profile -> npu
//       host + any profile           -> cpu
//
//   This pass never performs that target mapping.

static constexpr llvm::StringLiteral kTruthBoundary =
    "static_execution_domain_classification_not_target_mapping";

enum class ExecutionDomain { Accelerated, Host, Fallback };

// classifyExecutionDomain is the only place containing domain policy.
// Replacing this function changes the policy without touching the pass body.
static ExecutionDomain classifyExecutionDomain(llvm::StringRef opName) {
  static const llvm::StringLiteral kAcceleratedOps[] = {
    "cv.conv2d", "cv.batch_norm", "cv.silu", "cv.upsample", "cv.concat"
  };
  static const llvm::StringLiteral kHostOps[] = {
    "cv.detect_head", "cv.prototype_head"
  };
  for (auto s : kAcceleratedOps)
    if (opName == s) return ExecutionDomain::Accelerated;
  for (auto s : kHostOps)
    if (opName == s) return ExecutionDomain::Host;
  return ExecutionDomain::Fallback;
}

struct CVExecutionDomainPlanningPass
    : impl::CVExecutionDomainPlanningBase<CVExecutionDomainPlanningPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    // Gate: require at least one cv.* op.
    bool hasCVOps = false;
    funcOp.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef().starts_with("cv.")) {
        hasCVOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasCVOps) return;

    int64_t acceleratedOps = 0, hostOps = 0, fallbackOps = 0;

    funcOp.walk([&](Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      if (!name.starts_with("cv.")) return;

      ExecutionDomain domain = classifyExecutionDomain(name);

      llvm::StringRef domainStr;
      llvm::StringRef reasonStr;
      if (domain == ExecutionDomain::Accelerated) {
        domainStr = "accelerated";
        reasonStr = "accelerated_tensor_policy";
        ++acceleratedOps;
      } else if (domain == ExecutionDomain::Host) {
        domainStr = "host";
        reasonStr = "host_postprocess_policy";
        ++hostOps;
      } else {
        domainStr = "fallback";
        reasonStr = "fallback_unknown_cv_op";
        ++fallbackOps;
      }

      op->setAttr("cv.execution_domain", StringAttr::get(ctx, domainStr));
      op->setAttr("cv.execution_domain.truth_boundary",
                  StringAttr::get(ctx, kTruthBoundary));
      op->setAttr("cv.execution_domain_reason", StringAttr::get(ctx, reasonStr));
    });

    auto i64 = IntegerType::get(ctx, 64);
    int64_t plannedOps = acceleratedOps + hostOps + fallbackOps;

    funcOp->setAttr("cv.execution_domain_plan.accelerated_ops",
                    IntegerAttr::get(i64, acceleratedOps));
    funcOp->setAttr("cv.execution_domain_plan.fallback_ops",
                    IntegerAttr::get(i64, fallbackOps));
    funcOp->setAttr("cv.execution_domain_plan.host_ops",
                    IntegerAttr::get(i64, hostOps));
    funcOp->setAttr("cv.execution_domain_plan.planned_ops",
                    IntegerAttr::get(i64, plannedOps));
    funcOp->setAttr("cv.execution_domain_plan.status",
                    StringAttr::get(ctx, "completed"));
    funcOp->setAttr("cv.execution_domain_plan.truth_boundary",
                    StringAttr::get(ctx, kTruthBoundary));
  }
};

} // namespace

std::unique_ptr<Pass> createCVExecutionDomainPlanningPass() {
  return std::make_unique<CVExecutionDomainPlanningPass>();
}

} // namespace mlir::hir
