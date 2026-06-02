#include "FusionPasses.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
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

    Type loweredType = getTypeConverter()->convertType(reluMap->getResult(0).getType());
    if (!loweredType) {
      return rewriter.notifyMatchFailure(matmul, "failed to convert fused result type");
    }

    rewriter.setInsertionPoint(reluMap);
    auto fused = rewriter.create<FusedMatMulBiasReluOp>(
        reluMap.getLoc(), loweredType,
        matmul.getInputs()[0], matmul.getInputs()[1], bias);
    fused->setAttr("fusion.candidate", candidate);
    fused->setAttr("fusion.group", matmul->getAttr("fusion.group"));
    fused->setAttr("kernel.selection",
                   StringAttr::get(matmul.getContext(), "runtime_profile"));
    fused->setAttr("lowering.source",
                   StringAttr::get(matmul.getContext(), "linalg.matmul_add_relu"));

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
