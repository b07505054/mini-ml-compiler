#include "FusionPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_QUANTIZATIONPLANNING
#include "FusionPasses.h.inc"

struct QuantizationPlanningPass
    : impl::QuantizationPlanningBase<QuantizationPlanningPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = module.getContext();

    // Read model dtype intent from the module.
    std::string llmDtype;
    if (auto a = module->getAttrOfType<StringAttr>("llm.dtype"))
      llmDtype = a.getValue().str();

    // Read target precision support list from the module.
    std::vector<std::string> supported;
    if (auto a = module->getAttrOfType<ArrayAttr>("target.supported_precisions"))
      for (Attribute elem : a)
        if (auto s = dyn_cast<StringAttr>(elem))
          supported.push_back(s.getValue().str());

    // Decide plan_dtype using priority rules:
    //   Rule 1: llm.dtype in target.supported_precisions -> use it.
    //   Rule 2: target.supported_precisions non-empty    -> use first element.
    //   Rule 3: llm.dtype non-empty (no supported list)  -> use llm.dtype.
    //   Rule 4: fallback                                  -> "fp16".
    bool llmDtypeSupported = false;
    for (const auto &p : supported)
      if (p == llmDtype) { llmDtypeSupported = true; break; }

    std::string planDtype;
    std::string planSource;
    if (!llmDtype.empty() && llmDtypeSupported) {
      planDtype = llmDtype;
      planSource = "target_profile";
    } else if (!supported.empty()) {
      planDtype = supported[0];
      planSource = "target_profile";
    } else if (!llmDtype.empty()) {
      planDtype = llmDtype;
      planSource = "default_dtype";
    } else {
      planDtype = "fp16";
      planSource = "default_dtype";
    }

    // Emit the quantization planning decision as module attributes.
    module->setAttr("quantization.plan_dtype",
                    StringAttr::get(context, planDtype));
    module->setAttr("quantization.plan_source",
                    StringAttr::get(context, planSource));
    module->setAttr("quantization.truth_boundary",
                    StringAttr::get(context,
                        "precision_selection_from_target_profile_not_calibrated"));
    module->setAttr("quantization.planning_complete",
                    BoolAttr::get(context, true));

    // The native codegen bridge accepts canonical Linalg input. Record the
    // selected precision on those operations at the stage where they exist;
    // MatMulBiasReluFusionPass consumes this exact attribute before replacing
    // the Linalg pattern with its HIR fused operation.
    if (planDtype == "int8" || planDtype == "fp32") {
      module.walk([&](Operation *op) {
        if (op->getName().getStringRef() == "linalg.matmul") {
          op->setAttr("quantization.candidate",
                      StringAttr::get(context, planDtype));
          if (planDtype == "int8")
            op->setAttr("profile.quantized_path",
                        StringAttr::get(context, "faster"));
        }
      });
    }
  }
};

} // namespace

std::unique_ptr<Pass> createQuantizationPlanningPass() {
  return std::make_unique<QuantizationPlanningPass>();
}

} // namespace mlir::hir
