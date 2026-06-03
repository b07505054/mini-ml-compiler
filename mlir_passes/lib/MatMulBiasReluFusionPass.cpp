#include "FusionPasses.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"

#include <string>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_HIRCANONICALIZATION
#define GEN_PASS_DEF_MATMULBIASRELUFUSION
#define GEN_PASS_DEF_RMSNORMKERNELSELECTION
#define GEN_PASS_DEF_HIRFUSIONLOWERING
#define GEN_PASS_DEF_HIRFUSEDOPVERIFIER
#include "FusionPasses.h.inc"

bool isAddMap(linalg::MapOp mapOp) {
  bool foundAdd = false;
  mapOp.getBody()->walk([&](arith::AddFOp) { foundAdd = true; });
  return foundAdd;
}

bool isZeroConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant) {
    return false;
  }

  auto floatAttr = dyn_cast<FloatAttr>(constant.getValue());
  return floatAttr && floatAttr.getValueAsDouble() == 0.0;
}

bool isReluMap(linalg::MapOp mapOp) {
  bool foundRelu = false;

  mapOp.getBody()->walk([&](arith::MaximumFOp maximum) {
    if (isZeroConstant(maximum.getLhs()) ||
        isZeroConstant(maximum.getRhs())) {
      foundRelu = true;
    }
  });

  return foundRelu;
}

bool isAddZeroMap(linalg::MapOp mapOp) {
  bool foundAddZero = false;

  mapOp.getBody()->walk([&](arith::AddFOp add) {
    if (isZeroConstant(add.getLhs()) ||
        isZeroConstant(add.getRhs())) {
      foundAddZero = true;
    }
  });

  return foundAddZero;
}

Value passthroughInputForAddZero(linalg::MapOp mapOp) {
  if (mapOp.getInputs().empty()) {
    return {};
  }

  return mapOp.getInputs().front();
}

Value biasInputForAdd(linalg::MapOp addMap, Value matmulResult) {
  for (Value input : addMap.getInputs()) {
    if (input != matmulResult) {
      return input;
    }
  }
  return {};
}

constexpr int64_t kTargetTileM = 16;
constexpr int64_t kTargetTileN = 16;
constexpr int64_t kTargetTileK = 32;
constexpr int64_t kTargetAlignmentBytes = 128;
constexpr int64_t kTargetSramKb = 256;

bool isStaticMultiple(int64_t dim, int64_t multiple) {
  return dim != ShapedType::kDynamic && dim % multiple == 0;
}

bool isSupportedMatMulElementType(Type type) {
  return type.isF32() || type.isF16();
}

struct MatMulBiasReluLegality {
  bool legal = false;
  StringRef reason = "unknown";
  Value bias;
};

MatMulBiasReluLegality checkMatMulBiasReluLegality(linalg::MatmulOp matmul,
                                                   linalg::MapOp addMap,
                                                   linalg::MapOp reluMap) {
  if (matmul->getNumResults() != 1) {
    return {false, "matmul_result_count", {}};
  }
  if (!matmul->getResult(0).hasOneUse()) {
    return {false, "matmul_result_not_one_use", {}};
  }
  if (addMap->getNumResults() != 1 || !addMap->getResult(0).hasOneUse()) {
    return {false, "bias_add_not_one_use", {}};
  }
  if (reluMap->getNumResults() != 1) {
    return {false, "relu_result_count", {}};
  }

  auto lhsType =
      dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
  auto rhsType =
      dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
  auto resultType =
      dyn_cast<RankedTensorType>(reluMap->getResult(0).getType());
  if (!lhsType || !rhsType || !resultType) {
    return {false, "dynamic_or_unranked_shape", {}};
  }
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      resultType.getRank() != 2) {
    return {false, "rank_not_2", {}};
  }
  if (!isSupportedMatMulElementType(lhsType.getElementType()) ||
      lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != resultType.getElementType()) {
    return {false, "unsupported_dtype", {}};
  }

  int64_t m = lhsType.getDimSize(0);
  int64_t k = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  int64_t n = rhsType.getDimSize(1);
  if (rhsK == ShapedType::kDynamic || k == ShapedType::kDynamic ||
      rhsK != k) {
    return {false, "matmul_k_mismatch_or_dynamic", {}};
  }
  if (resultType.getDimSize(0) != m || resultType.getDimSize(1) != n) {
    return {false, "result_shape_mismatch", {}};
  }
  if (!isStaticMultiple(m, kTargetTileM) ||
      !isStaticMultiple(n, kTargetTileN) ||
      !isStaticMultiple(k, kTargetTileK)) {
    return {false, "target_tile_multiple", {}};
  }

  Value bias = biasInputForAdd(addMap, matmul->getResult(0));
  auto biasType = bias ? dyn_cast<RankedTensorType>(bias.getType()) : nullptr;
  if (!biasType || biasType.getRank() != 2) {
    return {false, "bias_not_rank2", {}};
  }
  int64_t biasM = biasType.getDimSize(0);
  int64_t biasN = biasType.getDimSize(1);
  bool legalBiasM = biasM == 1 || biasM == m;
  bool legalBiasN = biasN == n;
  if (!legalBiasM || !legalBiasN) {
    return {false, "bias_broadcast_illegal", {}};
  }

  return {true, "target_legal", bias};
}

void attachSparseCoreTargetAttrs(Operation *op, OpBuilder &builder) {
  MLIRContext *context = op->getContext();
  op->setAttr("target.model",
              StringAttr::get(context, "sparsecore_like_v1"));
  op->setAttr("target.memory_hierarchy",
              StringAttr::get(context, "global_sram_register"));
  op->setAttr("target.sram_kb",
              builder.getI32IntegerAttr(kTargetSramKb));
  op->setAttr("target.tile_m",
              builder.getI32IntegerAttr(kTargetTileM));
  op->setAttr("target.tile_n",
              builder.getI32IntegerAttr(kTargetTileN));
  op->setAttr("target.tile_k",
              builder.getI32IntegerAttr(kTargetTileK));
  op->setAttr("target.vector_bytes",
              builder.getI32IntegerAttr(kTargetAlignmentBytes));
  op->setAttr("target.alignment",
              builder.getI32IntegerAttr(kTargetAlignmentBytes));
  op->setAttr("target.sparse_layout",
              StringAttr::get(context, "dense_or_2_4"));
  op->setAttr("target.collective",
              StringAttr::get(context, "none"));
}

struct AddZeroCanonicalizationPattern : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp mapOp,
                                PatternRewriter &rewriter) const override {
    if (mapOp->getNumResults() != 1 || !isAddZeroMap(mapOp)) {
      return failure();
    }

    Value replacement = passthroughInputForAddZero(mapOp);
    if (!replacement || replacement.getType() != mapOp->getResult(0).getType()) {
      return failure();
    }

    rewriter.replaceOp(mapOp, replacement);
    return success();
  }
};

struct NestedReluCanonicalizationPattern : OpRewritePattern<linalg::MapOp> {
  using OpRewritePattern<linalg::MapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MapOp outerRelu,
                                PatternRewriter &rewriter) const override {
    if (!isReluMap(outerRelu) || outerRelu->getNumResults() != 1 ||
        outerRelu.getInputs().empty()) {
      return failure();
    }

    Value input = outerRelu.getInputs().front();
    auto innerRelu = input.getDefiningOp<linalg::MapOp>();
    if (!innerRelu || !isReluMap(innerRelu) ||
        innerRelu->getNumResults() != 1 ||
        innerRelu->getResult(0).getType() != outerRelu->getResult(0).getType()) {
      return failure();
    }

    rewriter.replaceOp(outerRelu, innerRelu->getResult(0));
    return success();
  }
};

struct HIRCanonicalizationPass
    : impl::HIRCanonicalizationBase<HIRCanonicalizationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddZeroCanonicalizationPattern,
                 NestedReluCanonicalizationPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

struct MatMulBiasReluFusionPass
    : impl::MatMulBiasReluFusionBase<MatMulBiasReluFusionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    func.walk([&](linalg::MatmulOp matmul) {
      if (matmul->getNumResults() == 0) {
        return;
      }

      Value matmulResult = matmul->getResult(0);
      if (!matmulResult.hasOneUse()) {
        return;
      }

      for (Operation *matmulUser : matmulResult.getUsers()) {
        auto addMap = dyn_cast<linalg::MapOp>(matmulUser);
        if (!addMap || !isAddMap(addMap) || addMap->getNumResults() == 0) {
          continue;
        }

        Value addResult = addMap->getResult(0);

        for (Operation *addUser : addResult.getUsers()) {
          auto reluMap = dyn_cast<linalg::MapOp>(addUser);
          if (!reluMap || !isReluMap(reluMap)) {
            continue;
          }

          MatMulBiasReluLegality legality =
              checkMatMulBiasReluLegality(matmul, addMap, reluMap);
          if (!legality.legal) {
            matmul->setAttr(
                "fusion.reject_reason",
                StringAttr::get(matmul.getContext(), legality.reason));
            continue;
          }

          auto *context = matmul.getContext();
          auto group = StringAttr::get(context, "matmul_bias_relu_0");

          matmul->setAttr(
              "fusion.candidate",
              StringAttr::get(context, "matmul_bias_relu"));
          matmul->setAttr("fusion.group", group);
          matmul->setAttr("fusion.role", StringAttr::get(context, "producer"));

          addMap->setAttr("fusion.group", group);
          addMap->setAttr("fusion.role", StringAttr::get(context, "bias_add"));

          reluMap->setAttr("fusion.group", group);
          reluMap->setAttr("fusion.role", StringAttr::get(context, "activation"));
        }
      }
    });
  }
};

struct RMSNormKernelSelectionPass
    : impl::RMSNormKernelSelectionBase<RMSNormKernelSelectionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    int groupIndex = 0;

    func.walk([&](Operation *op) {
      if (op->getName().getStringRef() != "llm.rmsnorm") {
        return;
      }

      auto *context = op->getContext();
      auto group = StringAttr::get(
          context,
          "rmsnorm_" + std::to_string(groupIndex++));

      op->setAttr("fusion.candidate", StringAttr::get(context, "rmsnorm"));
      op->setAttr("fusion.group", group);
      op->setAttr("fusion.role", StringAttr::get(context, "normalization"));
      op->setAttr("lowering.hir_op", StringAttr::get(context, "hir.fused_rmsnorm"));
      op->setAttr("kernel.selection", StringAttr::get(context, "runtime_profile"));
    });
  }
};

struct HIRTypeConverter : TypeConverter {
  HIRTypeConverter() {
    addConversion([](Type type) { return type; });
  }
};

bool shouldUseQuantizedMatMul(linalg::MatmulOp matmul) {
  auto quantizationCandidate =
      matmul->getAttrOfType<StringAttr>("quantization.candidate");
  auto profileDecision =
      matmul->getAttrOfType<StringAttr>("profile.quantized_path");
  return quantizationCandidate && quantizationCandidate.getValue() == "int8" &&
         profileDecision && profileDecision.getValue() == "faster";
}

struct MatMulBiasReluToHIRConversionPattern
    : OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      linalg::MatmulOp matmul, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (matmul->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(matmul, "expected one matmul result");
    }

    auto candidate = matmul->getAttrOfType<StringAttr>("fusion.candidate");
    if (!candidate || candidate.getValue() != "matmul_bias_relu") {
      return rewriter.notifyMatchFailure(matmul, "missing MatMul-Bias-ReLU fusion marker");
    }

    Value matmulResult = matmul->getResult(0);
    if (!matmulResult.hasOneUse()) {
      return rewriter.notifyMatchFailure(matmul, "fused matmul result must have one user");
    }

    auto addMap = dyn_cast<linalg::MapOp>(*matmulResult.getUsers().begin());
    if (!addMap || !isAddMap(addMap) || addMap->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(matmul, "expected bias-add linalg.map user");
    }

    Value addResult = addMap->getResult(0);
    if (!addResult.hasOneUse()) {
      return rewriter.notifyMatchFailure(matmul, "bias-add result must have one user");
    }

    auto reluMap = dyn_cast<linalg::MapOp>(*addResult.getUsers().begin());
    if (!reluMap || !isReluMap(reluMap) || reluMap->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(matmul, "expected ReLU linalg.map user");
    }

    Value bias = biasInputForAdd(addMap, matmulResult);
    if (!bias || matmul.getInputs().size() < 2) {
      return rewriter.notifyMatchFailure(matmul, "failed to identify fused bias input");
    }

    MatMulBiasReluLegality legality =
        checkMatMulBiasReluLegality(matmul, addMap, reluMap);
    if (!legality.legal) {
      return rewriter.notifyMatchFailure(matmul, legality.reason);
    }

    Type loweredType = getTypeConverter()->convertType(reluMap->getResult(0).getType());
    if (!loweredType) {
      return rewriter.notifyMatchFailure(matmul, "failed to convert fused result type");
    }

    rewriter.setInsertionPoint(reluMap);
    Operation *fused = nullptr;
    if (shouldUseQuantizedMatMul(matmul)) {
      auto quantized = rewriter.create<FusedQMatMulBiasReluOp>(
          reluMap.getLoc(), loweredType,
          matmul.getInputs()[0], matmul.getInputs()[1], bias);
      quantized->setAttr("fusion.candidate",
                         StringAttr::get(matmul.getContext(), "qmatmul_bias_relu"));
      quantized->setAttr("fusion.group", matmul->getAttr("fusion.group"));
      quantized->setAttr("kernel.selection",
                         StringAttr::get(matmul.getContext(), "runtime_profile"));
      quantized->setAttr("lowering.source",
                         StringAttr::get(matmul.getContext(), "profile_guided_int8_matmul_add_relu"));
      quantized->setAttr("quantized_dtype",
                         StringAttr::get(matmul.getContext(), "i8"));
      quantized->setAttr("quantization.mode",
                         StringAttr::get(matmul.getContext(), "per_channel"));
      quantized->setAttr("input_layout",
                         StringAttr::get(matmul.getContext(), "NHWC"));
      quantized->setAttr("weight_layout",
                         StringAttr::get(matmul.getContext(), "blocked_kc"));
      quantized->setAttr("alignment",
                         rewriter.getI32IntegerAttr(128));
      quantized->setAttr("lhs_scale",
                         rewriter.getF32FloatAttr(0.01f));
      quantized->setAttr("rhs_scale",
                         rewriter.getF32FloatAttr(0.01f));
      quantized->setAttr("lhs_zero_point",
                         rewriter.getI32IntegerAttr(0));
      quantized->setAttr("rhs_zero_point",
                         rewriter.getI32IntegerAttr(0));
      attachSparseCoreTargetAttrs(quantized.getOperation(), rewriter);
      fused = quantized.getOperation();
    } else {
      auto fp32 = rewriter.create<FusedMatMulBiasReluOp>(
          reluMap.getLoc(), loweredType,
          matmul.getInputs()[0], matmul.getInputs()[1], bias);
      fp32->setAttr("fusion.candidate", candidate);
      fp32->setAttr("fusion.group", matmul->getAttr("fusion.group"));
      fp32->setAttr("kernel.selection",
                    StringAttr::get(matmul.getContext(), "runtime_profile"));
      fp32->setAttr("lowering.source",
                    StringAttr::get(matmul.getContext(), "linalg.matmul_add_relu"));
      attachSparseCoreTargetAttrs(fp32.getOperation(), rewriter);
      fused = fp32.getOperation();
    }

    rewriter.replaceOp(reluMap, fused->getResults());
    rewriter.eraseOp(addMap);
    rewriter.eraseOp(matmul);
    (void)adaptor;
    return success();
  }
};

struct RMSNormToHIRConversionPattern : ConversionPattern {
  RMSNormToHIRConversionPattern(TypeConverter &typeConverter,
                                MLIRContext *context)
      : ConversionPattern(typeConverter, "llm.rmsnorm", 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(op, "expected one RMSNorm result");
    }

    auto candidate = op->getAttrOfType<StringAttr>("fusion.candidate");
    if (!candidate || candidate.getValue() != "rmsnorm") {
      return rewriter.notifyMatchFailure(op, "missing RMSNorm fusion marker");
    }

    Type loweredType = getTypeConverter()->convertType(op->getResult(0).getType());
    if (!loweredType) {
      return rewriter.notifyMatchFailure(op, "failed to convert RMSNorm result type");
    }

    auto lowered = rewriter.create<FusedRMSNormOp>(op->getLoc(), loweredType,
                                                  operands.front());
    lowered->setAttr("fusion.candidate", candidate);
    lowered->setAttr("fusion.group", op->getAttr("fusion.group"));
    lowered->setAttr("kernel.selection",
                     StringAttr::get(op->getContext(), "runtime_profile"));
    lowered->setAttr("lowering.source",
                     StringAttr::get(op->getContext(), "llm.rmsnorm"));

    rewriter.replaceOp(op, lowered->getResults());
    return success();
  }
};

struct HIRFusionLoweringPass
    : impl::HIRFusionLoweringBase<HIRFusionLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    HIRTypeConverter typeConverter;
    RewritePatternSet patterns(func.getContext());
    patterns.add<MatMulBiasReluToHIRConversionPattern,
                 RMSNormToHIRConversionPattern>(typeConverter, func.getContext());

    ConversionTarget target(*func.getContext());
    target.addLegalDialect<arith::ArithDialect,
                           func::FuncDialect,
                           HIRDialect,
                           linalg::LinalgDialect,
                           tensor::TensorDialect>();
    target.addDynamicallyLegalOp<linalg::MatmulOp>([](linalg::MatmulOp op) {
      auto candidate = op->getAttrOfType<StringAttr>("fusion.candidate");
      return !candidate || candidate.getValue() != "matmul_bias_relu";
    });
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getStringRef() != "llm.rmsnorm";
    });

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

LogicalResult verifyFusedRMSNorm(Operation *op) {
  if (op->getNumOperands() != 1) {
    return op->emitOpError("expects exactly one input operand");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expects exactly one result");
  }
  if (op->getOperand(0).getType() != op->getResult(0).getType()) {
    return op->emitOpError("expects input and result types to match");
  }

  auto candidate = op->getAttrOfType<StringAttr>("fusion.candidate");
  if (!candidate || candidate.getValue() != "rmsnorm") {
    return op->emitOpError("requires fusion.candidate = \"rmsnorm\"");
  }
  if (!op->getAttrOfType<StringAttr>("kernel.selection")) {
    return op->emitOpError("requires kernel.selection metadata");
  }
  return success();
}

LogicalResult verifyFusedMatMulBiasRelu(Operation *op) {
  if (op->getNumOperands() != 3) {
    return op->emitOpError("expects lhs, rhs, and bias operands");
  }
  if (op->getNumResults() != 1) {
    return op->emitOpError("expects exactly one result");
  }

  auto candidate = op->getAttrOfType<StringAttr>("fusion.candidate");
  if (!candidate || candidate.getValue() != "matmul_bias_relu") {
    return op->emitOpError("requires fusion.candidate = \"matmul_bias_relu\"");
  }
  if (!op->getAttrOfType<StringAttr>("kernel.selection")) {
    return op->emitOpError("requires kernel.selection metadata");
  }
  return success();
}

struct HIRFusedOpVerifierPass
    : impl::HIRFusedOpVerifierBase<HIRFusedOpVerifierPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect>();
  }

  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](Operation *op) -> WalkResult {
      StringRef opName = op->getName().getStringRef();
      if (opName == "hir.fused_rmsnorm") {
        return failed(verifyFusedRMSNorm(op)) ? WalkResult::interrupt()
                                             : WalkResult::advance();
      }
      if (opName == "hir.fused_matmul_bias_relu") {
        return failed(verifyFusedMatMulBiasRelu(op)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createHIRCanonicalizationPass() {
  return std::make_unique<HIRCanonicalizationPass>();
}

std::unique_ptr<Pass> createMatMulBiasReluFusionPass() {
  return std::make_unique<MatMulBiasReluFusionPass>();
}

std::unique_ptr<Pass> createRMSNormKernelSelectionPass() {
  return std::make_unique<RMSNormKernelSelectionPass>();
}

std::unique_ptr<Pass> createHIRFusionLoweringPass() {
  return std::make_unique<HIRFusionLoweringPass>();
}

std::unique_ptr<Pass> createHIRFusedOpVerifierPass() {
  return std::make_unique<HIRFusedOpVerifierPass>();
}

void registerFusionPasses() {
  PassRegistration<HIRCanonicalizationPass>();
  PassRegistration<HIRFusedOpVerifierPass>();

  static PassPipelineRegistration<> pipeline(
      "matmul-bias-relu-fusion",
      "Detect MatMul + bias add + ReLU fusion candidates",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createMatMulBiasReluFusionPass());
      });

  static PassPipelineRegistration<> canonicalizePipeline(
      "hir-canonicalize",
      "Canonicalize HIR-friendly tensor MLIR patterns",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createHIRCanonicalizationPass());
      });

  static PassPipelineRegistration<> rmsnormPipeline(
      "rmsnorm-kernel-selection",
      "Annotate RMSNorm ops for runtime-aware kernel selection",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createRMSNormKernelSelectionPass());
      });

  static PassPipelineRegistration<> loweringPipeline(
      "hir-fusion-lowering",
      "Lower fusion candidates to HIR generic MLIR ops",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createHIRFusionLoweringPass());
      });

  static PassPipelineRegistration<> verifierPipeline(
      "hir-verify-fused-ops",
      "Verify generic HIR fused op invariants",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createHIRFusedOpVerifierPass());
      });
}

} // namespace mlir::hir
extern "C" ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "MatMulBiasReluFusionPass",
      LLVM_VERSION_STRING,
      []() {
        mlir::hir::registerFusionPasses();
      }};
}

extern "C" ::mlir::DialectPluginLibraryInfo
mlirGetDialectPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "HIRDialect",
      LLVM_VERSION_STRING,
      [](mlir::DialectRegistry *registry) {
        registry->insert<mlir::hir::HIRDialect>();
      }};
}
