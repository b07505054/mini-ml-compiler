#include "FusionPasses.h"
#include "HIR/IR/HIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_HIRQUANTCANONICALIZATION
#define GEN_PASS_DEF_HIRQUANTPROPAGATION
#define GEN_PASS_DEF_HIRINT8OPERATORSELECTION
#include "FusionPasses.h.inc"

bool sameAttr(Operation *lhs, Operation *rhs, StringRef name) {
  Attribute lhsAttr = lhs->getAttr(name);
  Attribute rhsAttr = rhs->getAttr(name);
  return lhsAttr && rhsAttr && lhsAttr == rhsAttr;
}

bool sameQuantizationMetadata(Operation *lhs, Operation *rhs) {
  return sameAttr(lhs, rhs, "scale") &&
         sameAttr(lhs, rhs, "zero_point") &&
         sameAttr(lhs, rhs, "quantized_dtype") &&
         sameAttr(lhs, rhs, "quantization.mode");
}

struct DequantizeQuantizeElimination
    : OpRewritePattern<DequantizeOp> {
  using OpRewritePattern<DequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DequantizeOp dequantize,
                                PatternRewriter &rewriter) const override {
    auto quantize = dequantize.getInput().getDefiningOp<QuantizeOp>();
    if (!quantize)
      return failure();
    if (!sameQuantizationMetadata(quantize.getOperation(),
                                  dequantize.getOperation()))
      return failure();
    if (quantize.getInput().getType() != dequantize.getOutput().getType())
      return failure();

    rewriter.replaceOp(dequantize, quantize.getInput());
    return success();
  }
};

struct QuantizeDequantizeElimination
    : OpRewritePattern<QuantizeOp> {
  using OpRewritePattern<QuantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(QuantizeOp quantize,
                                PatternRewriter &rewriter) const override {
    auto dequantize = quantize.getInput().getDefiningOp<DequantizeOp>();
    if (!dequantize)
      return failure();
    if (!sameQuantizationMetadata(dequantize.getOperation(),
                                  quantize.getOperation()))
      return failure();
    if (dequantize.getInput().getType() != quantize.getOutput().getType())
      return failure();

    rewriter.replaceOp(quantize, dequantize.getInput());
    return success();
  }
};

struct HIRQuantCanonicalizationPass
    : impl::HIRQuantCanonicalizationBase<HIRQuantCanonicalizationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DequantizeQuantizeElimination,
                 QuantizeDequantizeElimination>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

bool isInt8Candidate(Operation *op) {
  auto candidate = op->getAttrOfType<StringAttr>("quantization.candidate");
  return candidate && candidate.getValue() == "int8";
}

bool isReluMap(linalg::MapOp mapOp) {
  bool hasMaximum = false;
  mapOp.getBody()->walk([&](arith::MaximumFOp) {
    hasMaximum = true;
  });
  return hasMaximum;
}

std::optional<StringRef> islandFor(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return std::nullopt;
  auto island = definingOp->getAttrOfType<StringAttr>("quant.island");
  if (!island)
    return std::nullopt;
  return island.getValue();
}

std::optional<StringRef> singleUseIncomingIsland(Operation *op) {
  std::optional<StringRef> found;
  for (Value operand : op->getOperands()) {
    auto incoming = islandFor(operand);
    if (!incoming)
      continue;
    if (!operand.hasOneUse())
      return std::nullopt;
    if (found && *found != *incoming)
      return std::nullopt;
    found = *incoming;
  }
  return found;
}

void markIsland(Operation *op, StringRef island, StringRef reason) {
  MLIRContext *context = op->getContext();
  op->setAttr("quant.island", StringAttr::get(context, island));
  op->setAttr("quant.state", StringAttr::get(context, "int8_capable"));
  op->setAttr("quant.propagation", StringAttr::get(context, reason));
}

struct HIRQuantPropagationPass
    : impl::HIRQuantPropagationBase<HIRQuantPropagationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    HIRDialect,
                    linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    int nextIsland = 0;

    funcOp.walk([&](Operation *op) {
      if (isa<linalg::MatmulOp>(op) && isInt8Candidate(op)) {
        std::optional<StringRef> incoming = singleUseIncomingIsland(op);
        std::string island =
            incoming ? incoming->str()
                     : "int8_island_" + std::to_string(nextIsland++);
        markIsland(op, island, incoming ? "matmul_join" : "matmul_seed");
        return;
      }

      if (auto mapOp = dyn_cast<linalg::MapOp>(op)) {
        if (!isReluMap(mapOp))
          return;
        std::optional<StringRef> incoming = singleUseIncomingIsland(op);
        if (incoming)
          markIsland(op, *incoming, "relu");
        return;
      }

      if (isa<tensor::ExpandShapeOp,
              tensor::CollapseShapeOp,
              tensor::CastOp>(op)) {
        std::optional<StringRef> incoming = singleUseIncomingIsland(op);
        if (incoming)
          markIsland(op, *incoming, "reshape");
      }
    });
  }
};

struct QuantCapability {
  StringRef opName;
  StringRef backend;
  StringRef precision;
  bool requiresLayout;
  bool requiresAlignedShape;
};

constexpr QuantCapability kCapabilities[] = {
    {"matmul_bias_relu", "cpu", "int8", true, true},
    {"matmul", "cpu", "int8", true, true},
    {"relu", "cpu", "int8", false, false},
    {"reshape", "cpu", "int8", false, false},
    {"matmul_bias_relu", "metal", "fp16", false, false},
    {"matmul", "metal", "fp16", false, false},
    {"relu", "metal", "fp16", false, false},
    {"reshape", "metal", "fp16", false, false},
};

std::optional<QuantCapability> findCapability(StringRef opName,
                                              StringRef backend,
                                              StringRef precision) {
  for (const QuantCapability &capability : kCapabilities) {
    if (capability.opName == opName && capability.backend == backend &&
        capability.precision == precision)
      return capability;
  }
  return std::nullopt;
}

std::string backendFor(func::FuncOp funcOp) {
  if (auto attr = funcOp->getAttrOfType<StringAttr>("target.backend"))
    return attr.getValue().str();
  if (Operation *parent = funcOp->getParentOp()) {
    if (auto attr = parent->getAttrOfType<StringAttr>("target.backend"))
      return attr.getValue().str();
    if (auto attr = parent->getAttrOfType<StringAttr>("target.preferred_backend"))
      return attr.getValue().str();
  }
  return "cpu";
}

std::optional<StringRef> quantOpName(Operation *op) {
  if (isa<linalg::MatmulOp>(op))
    return StringRef("matmul");
  if (isa<FusedMatMulBiasReluOp, FusedQMatMulBiasReluOp>(op))
    return StringRef("matmul_bias_relu");
  if (auto mapOp = dyn_cast<linalg::MapOp>(op))
    if (isReluMap(mapOp))
      return StringRef("relu");
  if (isa<tensor::ExpandShapeOp,
          tensor::CollapseShapeOp,
          tensor::CastOp>(op))
    return StringRef("reshape");
  return std::nullopt;
}

bool isMarkedForInt8(Operation *op) {
  return op->getAttrOfType<StringAttr>("quant.island") || isInt8Candidate(op);
}

bool hasLegalLayout(Operation *op, StringRef opName) {
  if (opName != "matmul" && opName != "matmul_bias_relu")
    return true;

  auto inputLayout = op->getAttrOfType<StringAttr>("input_layout");
  auto weightLayout = op->getAttrOfType<StringAttr>("weight_layout");
  return inputLayout && inputLayout.getValue() == "NHWC" &&
         weightLayout && weightLayout.getValue() == "blocked_kc";
}

bool rankedMatmulShape(Value lhs, Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return false;
  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  int64_t rhsN = rhsType.getDimSize(1);
  if (lhsK != ShapedType::kDynamic && rhsK != ShapedType::kDynamic &&
      lhsK != rhsK)
    return false;
  if (lhsK != ShapedType::kDynamic && lhsK % 32 != 0)
    return false;
  if (rhsK != ShapedType::kDynamic && rhsK % 32 != 0)
    return false;
  if (rhsN != ShapedType::kDynamic && rhsN % 32 != 0)
    return false;
  return true;
}

bool hasLegalShape(Operation *op, StringRef opName) {
  if (opName != "matmul" && opName != "matmul_bias_relu")
    return true;

  if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
    if (matmul.getInputs().size() < 2)
      return false;
    return rankedMatmulShape(matmul.getInputs()[0], matmul.getInputs()[1]);
  }
  if (auto fused = dyn_cast<FusedMatMulBiasReluOp>(op))
    return rankedMatmulShape(fused.getLhs(), fused.getRhs());
  if (auto fusedQ = dyn_cast<FusedQMatMulBiasReluOp>(op))
    return rankedMatmulShape(fusedQ.getLhs(), fusedQ.getRhs());
  return false;
}

bool profileAllowsInt8(Operation *op) {
  auto profile = op->getAttrOfType<StringAttr>("profile.quantized_path");
  return !profile || profile.getValue() == "faster";
}

void annotateSelection(Operation *op, StringRef opName, StringRef backend,
                       StringRef selection, StringRef reason) {
  MLIRContext *context = op->getContext();
  op->setAttr("quant.op", StringAttr::get(context, opName));
  op->setAttr("quant.backend", StringAttr::get(context, backend));
  op->setAttr("quant.selection", StringAttr::get(context, selection));
  op->setAttr("quant.selection_reason", StringAttr::get(context, reason));
}

struct HIRINT8OperatorSelectionPass
    : impl::HIRINT8OperatorSelectionBase<HIRINT8OperatorSelectionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect,
                    linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    std::string backend = backendFor(funcOp);

    funcOp.walk([&](Operation *op) {
      std::optional<StringRef> opName = quantOpName(op);
      if (!opName || !isMarkedForInt8(op))
        return;

      std::optional<QuantCapability> capability =
          findCapability(*opName, backend, "int8");
      if (!capability) {
        annotateSelection(op, *opName, backend, "fallback",
                          "backend_lacks_int8");
        return;
      }

      if (capability->requiresLayout && !hasLegalLayout(op, *opName)) {
        annotateSelection(op, *opName, backend, "fallback",
                          "illegal_layout");
        return;
      }

      if (capability->requiresAlignedShape && !hasLegalShape(op, *opName)) {
        annotateSelection(op, *opName, backend, "fallback",
                          "illegal_shape");
        return;
      }

      if (!profileAllowsInt8(op)) {
        annotateSelection(op, *opName, backend, "fallback",
                          "profile_not_faster");
        return;
      }

      annotateSelection(op, *opName, backend, "int8",
                        "capability_table_int8");
    });
  }
};

} // namespace

std::unique_ptr<Pass> createHIRQuantCanonicalizationPass() {
  return std::make_unique<HIRQuantCanonicalizationPass>();
}

std::unique_ptr<Pass> createHIRQuantPropagationPass() {
  return std::make_unique<HIRQuantPropagationPass>();
}

std::unique_ptr<Pass> createHIRINT8OperatorSelectionPass() {
  return std::make_unique<HIRINT8OperatorSelectionPass>();
}

} // namespace mlir::hir
