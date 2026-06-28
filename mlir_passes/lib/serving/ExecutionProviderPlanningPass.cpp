#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringSet.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_EXECUTIONPROVIDERPLANNING
#include "FusionPasses.h.inc"

struct ExecutionProviderPlanningPass
    : impl::ExecutionProviderPlanningBase<ExecutionProviderPlanningPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    // Only annotate functions that contain an attention op.
    bool hasAttentionOp = false;
    funcOp.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "llm.attention_prefill" || name == "llm.attention_decode")
        hasAttentionOp = true;
    });
    if (!hasAttentionOp)
      return;

    Operation *parent = funcOp->getParentOp();

    // Read precision early — used in both the normal and constraint-conflict paths.
    std::string precision = "fp16";
    if (parent) {
      if (auto a = parent->getAttrOfType<ArrayAttr>("target.supported_precisions")) {
        for (Attribute elem : a) {
          if (auto s = dyn_cast<StringAttr>(elem)) {
            precision = s.getValue().str();
            break;
          }
        }
      }
    }

    // Copy replay eligibility from ReplayEligibilityPass output (if present).
    bool requiresReplay = false;
    if (auto a = funcOp->getAttrOfType<BoolAttr>("replay.eligible"))
      requiresReplay = a.getValue();

    // Copy KV layout from KVLayoutPlanningPass output (if present).
    std::string kvLayout = "unknown";
    if (auto a = funcOp->getAttrOfType<StringAttr>("kv.layout"))
      kvLayout = a.getValue().str();

    // -----------------------------------------------------------------------
    // Step 1: candidate list from target.allowed_backends.
    // Absent or empty -> ["cpu"], decision_source = "default_policy".
    // -----------------------------------------------------------------------
    std::vector<std::string> candidates;
    bool hasAllowedBackends = false;
    if (parent) {
      if (auto a = parent->getAttrOfType<ArrayAttr>("target.allowed_backends")) {
        for (Attribute elem : a) {
          if (auto s = dyn_cast<StringAttr>(elem))
            candidates.push_back(s.getValue().str());
        }
        hasAllowedBackends = !candidates.empty();
      }
    }
    std::string decisionSource;
    if (!hasAllowedBackends) {
      candidates = {"cpu"};
      decisionSource = "default_policy";
    }

    // -----------------------------------------------------------------------
    // Step 2: preferred backend from target.preferred_backend.
    // -----------------------------------------------------------------------
    std::string preferred;
    if (parent) {
      if (auto a = parent->getAttrOfType<StringAttr>("target.preferred_backend"))
        preferred = a.getValue().str();
    }

    // Track whether preferred appeared in the pre-filter candidate list.
    bool preferredInOriginal = false;
    for (const auto &c : candidates) {
      if (c == preferred) { preferredInOriginal = true; break; }
    }

    // -----------------------------------------------------------------------
    // Step 3: KV layout compatibility filter.
    // Paged KV requires a backend declared in target.paged_kv_compatible_backends.
    // Compatibility is a target property, not a compiler invariant: the set
    // comes from the device profile and is absent for targets where no paged-KV
    // runtime exists (e.g. the Apple A17 Pro currently has an empty set).
    // Absent attr → empty set → constraint_conflict for any paged-KV function.
    // -----------------------------------------------------------------------
    if (kvLayout == "paged") {
      llvm::StringSet<> pagedKVSet;
      if (parent) {
        if (auto a = parent->getAttrOfType<mlir::ArrayAttr>(
                "target.paged_kv_compatible_backends")) {
          for (mlir::Attribute elem : a)
            if (auto s = mlir::dyn_cast<mlir::StringAttr>(elem))
              pagedKVSet.insert(s.getValue());
        }
      }

      std::vector<std::string> filtered;
      for (const auto &c : candidates) {
        if (pagedKVSet.count(c))
          filtered.push_back(c);
      }
      if (filtered.empty()) {
        emitPlan(funcOp, context, "cpu", {}, "constraint_conflict",
                 precision, kvLayout, requiresReplay);
        return;
      }
      candidates = std::move(filtered);
    }

    // -----------------------------------------------------------------------
    // Step 4: primary selection.
    // -----------------------------------------------------------------------
    bool preferredInFiltered = false;
    for (const auto &c : candidates) {
      if (c == preferred) { preferredInFiltered = true; break; }
    }

    std::string primary;
    if (!preferred.empty() && preferredInFiltered) {
      primary       = preferred;
      decisionSource = "target_preferred";
    } else if (!preferred.empty() && preferredInOriginal && !preferredInFiltered) {
      // preferred was in allowed_backends but removed by the KV layout filter.
      primary       = candidates[0];
      decisionSource = "kv_layout_constraint";
    } else {
      primary = candidates[0];
      if (decisionSource.empty())
        decisionSource = "allowed_backends_default";
      // else: keep "default_policy"
    }

    // -----------------------------------------------------------------------
    // Step 5: fallback chain = remaining filtered candidates, excluding primary.
    // -----------------------------------------------------------------------
    std::vector<std::string> fallbackChain;
    for (const auto &c : candidates) {
      if (c != primary)
        fallbackChain.push_back(c);
    }

    emitPlan(funcOp, context, primary, fallbackChain, decisionSource,
             precision, kvLayout, requiresReplay);
  }

private:
  static void emitPlan(func::FuncOp funcOp, MLIRContext *context,
                       llvm::StringRef primary,
                       const std::vector<std::string> &fallbackChain,
                       llvm::StringRef decisionSource,
                       llvm::StringRef precision,
                       llvm::StringRef kvLayout,
                       bool requiresReplay) {
    funcOp->setAttr("execution_provider.primary",
                    StringAttr::get(context, primary));

    SmallVector<Attribute> fallbackAttrs;
    for (const auto &f : fallbackChain)
      fallbackAttrs.push_back(StringAttr::get(context, f));
    funcOp->setAttr("execution_provider.fallback_chain",
                    ArrayAttr::get(context, fallbackAttrs));

    funcOp->setAttr("execution_provider.decision_source",
                    StringAttr::get(context, decisionSource));
    funcOp->setAttr("execution_provider.required_precision",
                    StringAttr::get(context, precision));
    funcOp->setAttr("execution_provider.required_kv_layout",
                    StringAttr::get(context, kvLayout));
    funcOp->setAttr("execution_provider.requires_replay",
                    BoolAttr::get(context, requiresReplay));
    funcOp->setAttr("execution_provider.truth_boundary",
                    StringAttr::get(context,
                        "compiler_execution_provider_plan_not_runtime_dispatch"));
  }
};

} // namespace

std::unique_ptr<Pass> createExecutionProviderPlanningPass() {
  return std::make_unique<ExecutionProviderPlanningPass>();
}

} // namespace mlir::hir
