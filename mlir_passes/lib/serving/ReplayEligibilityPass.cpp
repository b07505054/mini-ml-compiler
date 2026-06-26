#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_REPLAYELIGIBILITY
#include "FusionPasses.h.inc"

static bool hasDynamicShape(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && !shaped.hasStaticShape();
}

struct ReplayEligibilityPass
    : impl::ReplayEligibilityBase<ReplayEligibilityPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    bool hasPrefill     = false;
    bool hasDecode      = false;
    bool hasDynamicDims = false;

    // Check function argument types.
    for (BlockArgument arg : funcOp.getArguments()) {
      if (hasDynamicShape(arg.getType()))
        hasDynamicDims = true;
    }

    // Walk body: detect attention op kind and any dynamic result types.
    funcOp.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name == "llm.attention_prefill") hasPrefill = true;
      if (name == "llm.attention_decode")  hasDecode  = true;
      for (Value result : op->getResults()) {
        if (hasDynamicShape(result.getType()))
          hasDynamicDims = true;
      }
    });

    // Only annotate functions containing an attention op.
    if (!hasPrefill && !hasDecode)
      return;

    // Eligibility rules (prefill check takes priority over decode).
    bool eligible = false;
    StringRef bucket = "";
    if (!hasPrefill && hasDecode && !hasDynamicDims) {
      eligible = true;
      bucket   = "decode_static";
    }

    funcOp->setAttr("replay.eligible", BoolAttr::get(context, eligible));
    funcOp->setAttr("replay.cuda_graph_bucket",
                    StringAttr::get(context, bucket));
    funcOp->setAttr("replay.truth_boundary",
                    StringAttr::get(context,
                        "static_shape_replay_eligibility_not_cuda_graph_capture"));
  }
};

} // namespace

std::unique_ptr<Pass> createReplayEligibilityPass() {
  return std::make_unique<ReplayEligibilityPass>();
}

} // namespace mlir::hir
