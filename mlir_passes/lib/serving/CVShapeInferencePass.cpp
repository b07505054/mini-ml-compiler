#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CVSHAPEINFERENCE
#include "FusionPasses.h.inc"

static constexpr llvm::StringLiteral kTruthBoundary =
    "static_ranked_tensor_shape_metadata_not_runtime_allocation";
static constexpr int64_t kFp16Bytes = 2;

struct CVShapeInferencePass
    : impl::CVShapeInferenceBase<CVShapeInferencePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    // Gate: return early if the function contains no cv.* ops.
    bool hasCVOps = false;
    funcOp.walk([&](Operation *op) -> WalkResult {
      if (op->getName().getStringRef().starts_with("cv.")) {
        hasCVOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasCVOps)
      return;

    auto i64 = IntegerType::get(ctx, 64);
    int64_t annotated  = 0;
    int64_t skipped    = 0;
    int64_t totalBytes = 0;

    funcOp.walk([&](Operation *op) {
      if (!op->getName().getStringRef().starts_with("cv."))
        return;
      if (op->getNumResults() != 1) { ++skipped; return; }

      auto rt = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!rt) { ++skipped; return; }

      for (int64_t d : rt.getShape()) {
        if (d == ShapedType::kDynamic) { ++skipped; return; }
      }

      int64_t numElements = 1;
      for (int64_t d : rt.getShape())
        numElements *= d;
      int64_t bytesEst = numElements * kFp16Bytes;

      op->setAttr("cv.bytes_estimate",
                  IntegerAttr::get(i64, bytesEst));
      op->setAttr("cv.num_elements",
                  IntegerAttr::get(i64, numElements));
      op->setAttr("cv.shape_inference.truth_boundary",
                  StringAttr::get(ctx, kTruthBoundary));

      totalBytes += bytesEst;
      ++annotated;
    });

    funcOp->setAttr("cv.shape_inference.annotated_ops",
                    IntegerAttr::get(i64, annotated));
    funcOp->setAttr("cv.shape_inference.skipped_ops",
                    IntegerAttr::get(i64, skipped));
    funcOp->setAttr("cv.shape_inference.status",
                    StringAttr::get(ctx, "completed"));
    funcOp->setAttr("cv.shape_inference.total_bytes_estimate",
                    IntegerAttr::get(i64, totalBytes));
    funcOp->setAttr("cv.shape_inference.truth_boundary",
                    StringAttr::get(ctx, kTruthBoundary));
  }
};

} // namespace

std::unique_ptr<Pass> createCVShapeInferencePass() {
  return std::make_unique<CVShapeInferencePass>();
}

} // namespace mlir::hir
