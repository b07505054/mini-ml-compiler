#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_LAYOUTPLANNING
#include "FusionPasses.h.inc"

static std::vector<std::string> readStringArr(Operation *op, StringRef name) {
  std::vector<std::string> result;
  if (auto a = op->getAttrOfType<ArrayAttr>(name))
    for (Attribute elem : a)
      if (auto s = dyn_cast<StringAttr>(elem))
        result.push_back(s.getValue().str());
  return result;
}

static bool containsStr(const std::vector<std::string> &v,
                        const std::string &s) {
  for (const auto &e : v)
    if (e == s) return true;
  return false;
}

// "cv.relu" -> "relu", "llm.attention_prefill" -> "attention_prefill"
static std::string shortOpName(Operation *op) {
  StringRef full = op->getName().getStringRef();
  auto pos = full.rfind('.');
  if (pos != StringRef::npos)
    return full.drop_front(pos + 1).str();
  return full.str();
}

struct LayoutPlanningPass
    : impl::LayoutPlanningBase<LayoutPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Operation *module = funcOp->getParentOp();

    // Skip funcs that have no representation planning annotation.
    auto prefLayoutAttr =
        funcOp->getAttrOfType<StringAttr>("representation.preferred_activation_layout");
    if (!prefLayoutAttr)
      return;
    std::string targetLayout = prefLayoutAttr.getValue().str();

    // Read source backend.
    std::string backend;
    if (auto a = funcOp->getAttrOfType<StringAttr>("representation.source_backend"))
      backend = a.getValue().str();

    // Read layout-agnostic ops from module backend capability.
    std::vector<std::string> agnosticOps;
    if (module && !backend.empty()) {
      std::string prefix = "target.backend_capabilities." + backend + ".";
      agnosticOps = readStringArr(module, prefix + "layout_agnostic_ops");
    }

    // Read initial activation layout from func attr; default "NCHW".
    std::string currentLayout = "NCHW";
    if (auto a = funcOp->getAttrOfType<StringAttr>("layout.initial_layout"))
      currentLayout = a.getValue().str();

    static constexpr StringLiteral kTruthBoundary =
        "layout_planning_static_cost_model_not_measured_kernel_performance";

    if (funcOp.getBody().empty())
      return;
    Block &entry = funcOp.getBody().front();

    for (Operation &op : entry.without_terminator()) {
      std::string opShort = shortOpName(&op);
      bool isAgnostic = containsStr(agnosticOps, opShort);

      std::string effectiveLayout;
      std::string layoutSource;
      bool transformRequired;

      if (isAgnostic) {
        effectiveLayout  = currentLayout;
        layoutSource     = "layout_agnostic_propagation";
        transformRequired = false;
      } else {
        effectiveLayout  = targetLayout;
        layoutSource     = "backend_preference";
        transformRequired = (currentLayout != targetLayout);
      }

      op.setAttr("layout.required_input_layout",
                 StringAttr::get(ctx, currentLayout));
      op.setAttr("layout.effective_layout",
                 StringAttr::get(ctx, effectiveLayout));
      op.setAttr("layout.layout_source",
                 StringAttr::get(ctx, layoutSource));
      op.setAttr("layout.transform_required", BoolAttr::get(ctx, transformRequired));
      op.setAttr("layout.truth_boundary",
                 StringAttr::get(ctx, kTruthBoundary));

      currentLayout = effectiveLayout;
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createLayoutPlanningPass() {
  return std::make_unique<LayoutPlanningPass>();
}

} // namespace mlir::hir
