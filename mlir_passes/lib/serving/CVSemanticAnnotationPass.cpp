#include "FusionPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CVSEMANTICANNOTATION
#include "FusionPasses.h.inc"

static constexpr llvm::StringLiteral kTruthBoundary =
    "cv_semantic_annotation_only_no_backend_selection_no_memory_plan_no_kernel_selection_no_execution_plan_generation";

static RankedTensorType rankedTensor(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

static RankedTensorType firstRankedResult(Operation *op) {
  if (!op || op->getNumResults() == 0)
    return {};
  return rankedTensor(op->getResult(0));
}

static bool hasShape(RankedTensorType type, ArrayRef<int64_t> shape) {
  if (!type || type.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [actual, expected] : llvm::zip_equal(type.getShape(), shape)) {
    if (actual != expected)
      return false;
  }
  return true;
}

static bool isRank4Spatial(RankedTensorType type, int64_t h, int64_t w) {
  return type && type.getRank() == 4 && type.getDimSize(2) == h &&
         type.getDimSize(3) == w;
}

static bool hasAnchorDim(RankedTensorType type) {
  return type && type.getRank() >= 3 && type.getDimSize(type.getRank() - 1) == 8400;
}

static bool isHeadSpatialTensor(RankedTensorType type) {
  if (!type || type.getRank() != 4)
    return false;
  int64_t h = type.getDimSize(2);
  int64_t w = type.getDimSize(3);
  return h == w && (h == 20 || h == 40 || h == 80);
}

static bool isPrototypeSpatialTensor(RankedTensorType type) {
  if (!type || type.getRank() != 4)
    return false;
  int64_t c = type.getDimSize(1);
  int64_t h = type.getDimSize(2);
  int64_t w = type.getDimSize(3);
  return h == w && (h == 80 || h == 160) && (c == 32 || c == 64);
}

static bool resultHasManyUsers(Operation *op, unsigned maxUsers) {
  if (!op)
    return false;
  for (Value result : op->getResults()) {
    unsigned users = 0;
    for (Operation *user : result.getUsers()) {
      (void)user;
      ++users;
      if (users > maxUsers)
        return true;
    }
  }
  return false;
}

static ArrayAttr evidence(MLIRContext *ctx, ArrayRef<StringRef> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (StringRef value : values)
    attrs.push_back(StringAttr::get(ctx, value));
  return ArrayAttr::get(ctx, attrs);
}

static void annotate(Operation *op, StringRef role, StringRef regionId,
                     StringRef confidence, ArrayRef<StringRef> evidenceValues) {
  if (!op)
    return;
  MLIRContext *ctx = op->getContext();
  op->setAttr("cv.semantic_role", StringAttr::get(ctx, role));
  op->setAttr("cv.region_id", StringAttr::get(ctx, regionId));
  op->setAttr("cv.recognition_confidence", StringAttr::get(ctx, confidence));
  op->setAttr("cv.recognition_evidence", evidence(ctx, evidenceValues));

  if (RankedTensorType type = firstRankedResult(op)) {
    if (type.getRank() == 4) {
      int64_t h = type.getDimSize(2);
      int64_t w = type.getDimSize(3);
      if (h == w && (h == 20 || h == 40 || h == 80 || h == 160)) {
        op->setAttr("cv.feature_scale",
                    StringAttr::get(ctx, (Twine(h) + "x" + Twine(w)).str()));
      }
    }
  }
}

static bool detectionRegionOp(Operation *op) {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  RankedTensorType type = firstRankedResult(op);
  if (name == "tensor.insert_slice" || name == "tensor.extract_slice" ||
      name == "tensor.collapse_shape" || name == "tensor.expand_shape" ||
      name == "linalg.transpose")
    return hasAnchorDim(type) || isHeadSpatialTensor(type);
  if (name == "linalg.generic")
    return hasAnchorDim(type) || isHeadSpatialTensor(type);
  if (name == "linalg.conv_2d_nchw_fchw")
    return isHeadSpatialTensor(type) || hasAnchorDim(type);
  return false;
}

static bool prototypeRegionOp(Operation *op) {
  if (!op)
    return false;
  StringRef name = op->getName().getStringRef();
  RankedTensorType type = firstRankedResult(op);
  if (name == "linalg.generic" || name == "linalg.conv_2d_nchw_fchw" ||
      name == "tensor.pad" || name == "linalg.fill")
    return isPrototypeSpatialTensor(type);
  // Phase 20 ConvTranspose is represented as a stride-2 linalg.generic whose
  // output has the prototype 160x160 spatial contract.
  return false;
}

static void collectBackward(Value value,
                            bool (*predicate)(Operation *),
                            llvm::SmallPtrSetImpl<Operation *> &region,
                            unsigned depth = 0) {
  if (depth > 64)
    return;
  Operation *op = value.getDefiningOp();
  if (!op || !predicate(op))
    return;
  if (!region.insert(op).second)
    return;

  StringRef name = op->getName().getStringRef();
  if (name == "linalg.conv_2d_nchw_fchw" && predicate == detectionRegionOp)
    return;

  if (predicate == prototypeRegionOp && resultHasManyUsers(op, 2))
    return;

  for (Value operand : op->getOperands()) {
    if (!rankedTensor(operand))
      continue;
    Operation *producer = operand.getDefiningOp();
    if (!producer)
      continue;
    if (predicate == prototypeRegionOp && resultHasManyUsers(producer, 2))
      continue;
    collectBackward(operand, predicate, region, depth + 1);
  }
}

static Operation *sourceProducer(Operation *insertSlice) {
  if (!insertSlice || insertSlice->getName().getStringRef() != "tensor.insert_slice" ||
      insertSlice->getNumOperands() == 0)
    return nullptr;
  return insertSlice->getOperand(0).getDefiningOp();
}

static bool isMaskCoefficientInsert(Operation *op) {
  if (!op || op->getName().getStringRef() != "tensor.insert_slice" ||
      op->getNumOperands() == 0)
    return false;
  RankedTensorType src = rankedTensor(op->getOperand(0));
  RankedTensorType result = firstRankedResult(op);
  return hasShape(src, {1, 32, 8400}) && hasShape(result, {1, 116, 8400});
}

static bool isTwoXGenerate(Operation *op) {
  if (!op || op->getName().getStringRef() != "tensor.generate")
    return false;
  RankedTensorType outType = firstRankedResult(op);
  if (!outType || outType.getRank() != 4)
    return false;

  bool foundTwoXExtract = false;
  op->walk([&](Operation *nested) {
    if (nested->getName().getStringRef() != "tensor.extract" ||
        nested->getNumOperands() == 0)
      return;
    RankedTensorType inType = rankedTensor(nested->getOperand(0));
    if (!inType || inType.getRank() != 4)
      return;
    if (inType.getDimSize(0) == outType.getDimSize(0) &&
        inType.getDimSize(1) == outType.getDimSize(1) &&
        inType.getDimSize(2) * 2 == outType.getDimSize(2) &&
        inType.getDimSize(3) * 2 == outType.getDimSize(3))
      foundTwoXExtract = true;
  });
  return foundTwoXExtract;
}

static bool feedsInsertSlice(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->getName().getStringRef() == "tensor.insert_slice")
      return true;
  }
  return false;
}

struct CVSemanticAnnotationPass
    : impl::CVSemanticAnnotationBase<CVSemanticAnnotationPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    auto returnOp = dyn_cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());
    if (!returnOp)
      return;

    Operation *detectionRoot = nullptr;
    Operation *prototypeRoot = nullptr;
    int64_t detectionOutputIndex = -1;
    int64_t prototypeOutputIndex = -1;

    for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
      RankedTensorType type = rankedTensor(operand);
      Operation *producer = operand.getDefiningOp();
      if (!type || !producer)
        continue;
      if (hasShape(type, {1, 116, 8400}) &&
          producer->getName().getStringRef() == "tensor.insert_slice") {
        detectionRoot = producer;
        detectionOutputIndex = static_cast<int64_t>(index);
      }
      if (hasShape(type, {1, 32, 160, 160}) &&
          producer->getName().getStringRef() == "linalg.generic") {
        prototypeRoot = producer;
        prototypeOutputIndex = static_cast<int64_t>(index);
      }
    }

    SmallVector<Attribute> outputRoles;
    if (detectionRoot) {
      detectionRoot->setAttr("cv.output_role", StringAttr::get(ctx, "detection"));
      detectionRoot->setAttr("cv.postprocess_boundary",
                             StringAttr::get(ctx, "model_output_boundary"));
      annotate(detectionRoot, "detection_output", "cv.region.detection_head",
               "high",
               {"rank3 output contract [1,116,8400]",
                "producer is tensor.insert_slice assembly of bbox/class/mask tensors",
                "reachable from func.return"});
      outputRoles.push_back(StringAttr::get(ctx, "detection"));
    }
    if (prototypeRoot) {
      prototypeRoot->setAttr("cv.output_role",
                             StringAttr::get(ctx, "segmentation_prototype"));
      prototypeRoot->setAttr("cv.postprocess_boundary",
                             StringAttr::get(ctx, "model_output_boundary"));
      annotate(prototypeRoot, "segmentation_prototype",
               "cv.region.segmentation_prototype", "high",
               {"rank4 output contract [1,32,160,160]",
                "producer is linalg.generic final activation",
                "reachable from func.return"});
      outputRoles.push_back(StringAttr::get(ctx, "segmentation_prototype"));
    }

    llvm::SmallPtrSet<Operation *, 32> detectionRegion;
    if (detectionRoot) {
      for (Value operand : detectionRoot->getOperands())
        collectBackward(operand, detectionRegionOp, detectionRegion);
      for (Operation *op : detectionRegion) {
        if (op == detectionRoot)
          continue;
        annotate(op, "detection_head", "cv.region.detection_head", "high",
                 {"backward slice from detection output",
                  "tensor contract includes anchor dimension 8400 or 20/40/80 head scale",
                  "operation family participates in reshape/concat/softmax/conv detection branch"});
      }
    }

    int64_t maskOps = 0;
    if (detectionRoot) {
      funcOp.walk([&](Operation *op) {
        if (!isMaskCoefficientInsert(op))
          return;
        if (op == detectionRoot) {
          op->setAttr("cv.contains_mask_coefficients",
                      StringAttr::get(ctx, "true"));
        } else {
          ++maskOps;
          annotate(op, "mask_coefficient_branch",
                   "cv.region.mask_coefficient_branch", "high",
                   {"tensor.insert_slice contributes [1,32,8400] into [1,116,8400]",
                    "channel placement matches mask coefficient output contract",
                    "source reaches detection output"});
        }
        if (Operation *producer = sourceProducer(op)) {
          annotate(producer, "mask_coefficient_branch",
                   "cv.region.mask_coefficient_branch", "high",
                   {"producer tensor contract [1,32,8400]",
                    "consumed by detection output assembly",
                    "distinct from bbox and class score slices"});
          ++maskOps;
        }
      });
    }

    llvm::SmallPtrSet<Operation *, 32> prototypeRegion;
    if (prototypeRoot) {
      for (Value operand : prototypeRoot->getOperands())
        collectBackward(operand, prototypeRegionOp, prototypeRegion);
      for (Operation *op : prototypeRegion) {
        if (op == prototypeRoot)
          continue;
        annotate(op, "segmentation_prototype",
                 "cv.region.segmentation_prototype", "high",
                 {"exclusive backward slice from prototype output",
                  "rank4 prototype tensor contracts at 80x80 or 160x160",
                  "operation family is upstream linalg/tensor computation"});
      }
    }

    int64_t featurePyramidOps = 0;
    if (detectionRoot) {
      funcOp.walk([&](Operation *op) {
        if (isTwoXGenerate(op) && op->getNumResults() == 1 &&
            feedsInsertSlice(op->getResult(0))) {
          ++featurePyramidOps;
          annotate(op, "feature_pyramid", "cv.region.feature_pyramid",
                   "medium",
                   {"tensor.generate performs rank4 2x spatial remap",
                    "generated tensor feeds tensor.insert_slice fusion",
                    "detection output contract is present in same function"});
          for (Operation *user : op->getResult(0).getUsers()) {
            if (user->getName().getStringRef() == "tensor.insert_slice") {
              ++featurePyramidOps;
              annotate(user, "feature_pyramid", "cv.region.feature_pyramid",
                       "medium",
                       {"tensor.insert_slice consumes a 2x resize tensor",
                        "rank4 concat/fusion data movement",
                        "detection output contract is present in same function"});
            }
          }
        }
      });
    }

    SmallVector<Attribute> unresolved;
    if (maskOps == 0) {
      unresolved.push_back(StringAttr::get(
          ctx, "mask_coefficient_branch_unresolved_no_[1,32,8400]_detection_insert"));
    }
    if (!detectionRoot)
      unresolved.push_back(StringAttr::get(ctx, "detection_output_not_recognized"));
    if (!prototypeRoot)
      unresolved.push_back(StringAttr::get(ctx, "prototype_output_not_recognized"));

    bool fullYoloSegContract = detectionRoot && prototypeRoot;
    if (fullYoloSegContract) {
      funcOp->setAttr("cv.model_family", StringAttr::get(ctx, "yoloseg"));
      funcOp->setAttr("cv.recognition_confidence", StringAttr::get(ctx, "high"));
      funcOp->setAttr(
          "cv.recognition_evidence",
          evidence(ctx, {"function has detection output [1,116,8400]",
                         "function has segmentation prototype output [1,32,160,160]",
                         "output producers match upstream tensor/linalg topology"}));
    }

    auto i64 = IntegerType::get(ctx, 64);
    funcOp->setAttr("cv.semantic_annotation.status",
                    StringAttr::get(ctx, "completed"));
    funcOp->setAttr("cv.semantic_annotation.truth_boundary",
                    StringAttr::get(ctx, kTruthBoundary));
    funcOp->setAttr("cv.semantic_annotation.source_name_dependency",
                    StringAttr::get(ctx, "none"));
    funcOp->setAttr("cv.semantic_annotation.detection_output_index",
                    IntegerAttr::get(i64, detectionOutputIndex));
    funcOp->setAttr("cv.semantic_annotation.prototype_output_index",
                    IntegerAttr::get(i64, prototypeOutputIndex));
    funcOp->setAttr("cv.semantic_annotation.detection_region_ops",
                    IntegerAttr::get(i64, detectionRegion.size() +
                                              (detectionRoot ? 1 : 0)));
    funcOp->setAttr("cv.semantic_annotation.prototype_region_ops",
                    IntegerAttr::get(i64, prototypeRegion.size() +
                                              (prototypeRoot ? 1 : 0)));
    funcOp->setAttr("cv.semantic_annotation.mask_coefficient_ops",
                    IntegerAttr::get(i64, maskOps));
    funcOp->setAttr("cv.semantic_annotation.feature_pyramid_ops",
                    IntegerAttr::get(i64, featurePyramidOps));
    funcOp->setAttr("cv.semantic_annotation.unresolved",
                    ArrayAttr::get(ctx, unresolved));
    if (!outputRoles.empty())
      funcOp->setAttr("cv.semantic_annotation.output_roles",
                      ArrayAttr::get(ctx, outputRoles));
  }
};

} // namespace

std::unique_ptr<Pass> createCVSemanticAnnotationPass() {
  return std::make_unique<CVSemanticAnnotationPass>();
}

} // namespace mlir::hir
