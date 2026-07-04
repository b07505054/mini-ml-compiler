#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_REPRESENTATIONPLANNING
#include "FusionPasses.h.inc"

static std::vector<std::string> readStringArr(Operation *op, StringRef name) {
  std::vector<std::string> result;
  if (auto a = op->getAttrOfType<ArrayAttr>(name))
    for (Attribute elem : a)
      if (auto s = dyn_cast<StringAttr>(elem))
        result.push_back(s.getValue().str());
  return result;
}

static bool containsStr(const std::vector<std::string> &v, const std::string &s) {
  for (const auto &e : v)
    if (e == s) return true;
  return false;
}

struct RepresentationPlanningPass
    : impl::RepresentationPlanningBase<RepresentationPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Operation *module = funcOp->getParentOp();

    // Skip funcs that have no execution provider selection (non-attention funcs).
    auto primaryAttr = funcOp->getAttrOfType<StringAttr>("execution_provider.primary");
    if (!primaryAttr)
      return;
    std::string backend = primaryAttr.getValue().str();

    // Read plan_dtype from module-level attr; default "fp16" if absent.
    std::string planDtype = "fp16";
    if (module)
      if (auto a = module->getAttrOfType<StringAttr>("quantization.plan_dtype"))
        planDtype = a.getValue().str();

    static constexpr StringLiteral kTruthBoundary =
        "representation_plan_static_not_validated_against_kernel_performance";

    std::string effectiveDtype;
    std::string dtypeSource;
    std::string activationLayout = "unknown";
    std::string weightLayout     = "unknown";
    std::string conflictReason;
    std::string conflictFallback;

    // Check whether capability data is registered for this backend.
    std::vector<std::string> capNames;
    if (module)
      capNames = readStringArr(module, "target.backend_capability_names");
    bool capPresent = containsStr(capNames, backend);

    if (!capPresent) {
      effectiveDtype = planDtype;
      dtypeSource    = "quantization_plan_no_backend_capability_data";
      conflictReason = "backend_capability_missing";
    } else {
      std::string prefix = "target.backend_capabilities." + backend + ".";
      std::vector<std::string> supportedDtypes;
      std::vector<std::string> activationLayouts;
      std::vector<std::string> weightLayouts;
      if (module) {
        supportedDtypes   = readStringArr(module, prefix + "supported_dtypes");
        activationLayouts = readStringArr(module, prefix + "preferred_activation_layouts");
        weightLayouts     = readStringArr(module, prefix + "preferred_weight_layouts");
      }

      if (!activationLayouts.empty())
        activationLayout = activationLayouts[0];
      if (!weightLayouts.empty())
        weightLayout = weightLayouts[0];

      if (supportedDtypes.empty()) {
        // Capability entry present but no dtype constraint declared.
        effectiveDtype = planDtype;
        dtypeSource    = "quantization_plan_no_dtype_constraint";
      } else if (containsStr(supportedDtypes, planDtype)) {
        effectiveDtype = planDtype;
        dtypeSource    = "target_profile";
      } else {
        // plan_dtype unsupported — fall back to first listed dtype.
        effectiveDtype   = supportedDtypes[0];
        dtypeSource      = "fallback_first_supported";
        conflictReason   = "backend_lacks_plan_dtype";
        conflictFallback = effectiveDtype;
      }
    }

    funcOp->setAttr("representation.effective_dtype",
                    StringAttr::get(ctx, effectiveDtype));
    funcOp->setAttr("representation.effective_dtype_source",
                    StringAttr::get(ctx, dtypeSource));
    funcOp->setAttr("representation.preferred_activation_layout",
                    StringAttr::get(ctx, activationLayout));
    funcOp->setAttr("representation.preferred_weight_layout",
                    StringAttr::get(ctx, weightLayout));
    funcOp->setAttr("representation.source_backend",
                    StringAttr::get(ctx, backend));
    funcOp->setAttr("representation.truth_boundary",
                    StringAttr::get(ctx, kTruthBoundary));

    if (!conflictReason.empty()) {
      funcOp->setAttr("representation.conflict_reason",
                      StringAttr::get(ctx, conflictReason));
      if (!conflictFallback.empty())
        funcOp->setAttr("representation.conflict_fallback_dtype",
                        StringAttr::get(ctx, conflictFallback));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createRepresentationPlanningPass() {
  return std::make_unique<RepresentationPlanningPass>();
}

} // namespace mlir::hir
