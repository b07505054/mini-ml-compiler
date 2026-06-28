#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CVFRONTENDNORMALIZATION
#include "FusionPasses.h.inc"

// Pseudo-CV op names counted by this pass.
static constexpr llvm::StringLiteral kConv2d        = "cv.conv2d";
static constexpr llvm::StringLiteral kBatchNorm     = "cv.batch_norm";
static constexpr llvm::StringLiteral kSiLU          = "cv.silu";
static constexpr llvm::StringLiteral kUpsample      = "cv.upsample";
static constexpr llvm::StringLiteral kConcat        = "cv.concat";
static constexpr llvm::StringLiteral kDetectHead    = "cv.detect_head";
static constexpr llvm::StringLiteral kPrototypeHead = "cv.prototype_head";

struct CVOpCounts {
  int64_t conv2d        = 0;
  int64_t batchNorm     = 0;
  int64_t silu          = 0;
  int64_t upsample      = 0;
  int64_t concat        = 0;
  int64_t detectHead    = 0;
  int64_t prototypeHead = 0;
};

static CVOpCounts countCVOps(func::FuncOp funcOp) {
  CVOpCounts c;
  funcOp.walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if      (name == kConv2d)        ++c.conv2d;
    else if (name == kBatchNorm)     ++c.batchNorm;
    else if (name == kSiLU)          ++c.silu;
    else if (name == kUpsample)      ++c.upsample;
    else if (name == kConcat)        ++c.concat;
    else if (name == kDetectHead)    ++c.detectHead;
    else if (name == kPrototypeHead) ++c.prototypeHead;
  });
  return c;
}

struct CVFrontendNormalizationPass
    : impl::CVFrontendNormalizationBase<CVFrontendNormalizationPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    CVOpCounts c = countCVOps(funcOp);

    // Gate: require the three anchor op types that identify a YOLOSeg graph.
    if (c.conv2d == 0 || c.detectHead == 0 || c.prototypeHead == 0)
      return;

    auto i64 = IntegerType::get(ctx, 64);

    funcOp->setAttr("cv.frontend.kind",
                    StringAttr::get(ctx, "raw_pseudo_cv_mlir"));
    funcOp->setAttr("cv.frontend.model_family",
                    StringAttr::get(ctx, "yoloseg"));
    funcOp->setAttr("cv.frontend.normalization_status",
                    StringAttr::get(ctx, "detected_not_rewritten"));
    funcOp->setAttr("cv.frontend.truth_boundary",
                    StringAttr::get(ctx,
                      "raw_pseudo_cv_mlir_not_full_onnx_importer"));
    funcOp->setAttr("cv.frontend.conv2d_count",
                    IntegerAttr::get(i64, c.conv2d));
    funcOp->setAttr("cv.frontend.batch_norm_count",
                    IntegerAttr::get(i64, c.batchNorm));
    funcOp->setAttr("cv.frontend.silu_count",
                    IntegerAttr::get(i64, c.silu));
    funcOp->setAttr("cv.frontend.upsample_count",
                    IntegerAttr::get(i64, c.upsample));
    funcOp->setAttr("cv.frontend.concat_count",
                    IntegerAttr::get(i64, c.concat));
    funcOp->setAttr("cv.frontend.detect_head_count",
                    IntegerAttr::get(i64, c.detectHead));
    funcOp->setAttr("cv.frontend.prototype_head_count",
                    IntegerAttr::get(i64, c.prototypeHead));
  }
};

} // namespace

std::unique_ptr<Pass> createCVFrontendNormalizationPass() {
  return std::make_unique<CVFrontendNormalizationPass>();
}

} // namespace mlir::hir
