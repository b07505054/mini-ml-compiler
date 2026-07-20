#include "FusionPasses.h"
#include "CV/IR/CVDialect.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"
#include "MatMulLoweringCostModelV2.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
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

#include <cstdint>
#include <string>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_HIRCANONICALIZATION
#define GEN_PASS_DEF_MATMULBIASRELUFUSION
#define GEN_PASS_DEF_RMSNORMKERNELSELECTION
#define GEN_PASS_DEF_HIRFUSIONLOWERING
#define GEN_PASS_DEF_HIRFUSEDOPVERIFIER
#define GEN_PASS_DEF_HIRRMSNORMTOLINALG
#define GEN_PASS_DEF_HIRMATMULBIASRELUTOLINALG
#define GEN_PASS_DEF_STABLEHLOCOMPATIBLERMSNORMIMPORT
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

int64_t ceilDiv(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

int64_t roundUpToMultiple(int64_t value, int64_t multiple) {
  return ceilDiv(value, multiple) * multiple;
}

bool isSupportedMatMulElementType(Type type) {
  return type.isF32() || type.isF16();
}

struct MatMulBiasReluLegality {
  bool legal = false;
  StringRef reason = "unknown";
  Value bias;
  bool requiresPaddingCrop = false;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t paddedM = 0;
  int64_t paddedN = 0;
  int64_t paddedK = 0;
  double paddingComputeOverhead = 1.0;
  double paddingOutputOverhead = 1.0;
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
  if (m == ShapedType::kDynamic || n == ShapedType::kDynamic) {
    return {false, "dynamic_target_shape", {}};
  }
  if (resultType.getDimSize(0) != m || resultType.getDimSize(1) != n) {
    return {false, "result_shape_mismatch", {}};
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

  int64_t paddedM = roundUpToMultiple(m, kTargetTileM);
  int64_t paddedN = roundUpToMultiple(n, kTargetTileN);
  int64_t paddedK = roundUpToMultiple(k, kTargetTileK);
  bool requiresPaddingCrop =
      paddedM != m || paddedN != n || paddedK != k;
  double computeOverhead =
      static_cast<double>(paddedM) * static_cast<double>(paddedN) *
      static_cast<double>(paddedK) /
      (static_cast<double>(m) * static_cast<double>(n) *
       static_cast<double>(k));
  double outputOverhead =
      static_cast<double>(paddedM) * static_cast<double>(paddedN) /
      (static_cast<double>(m) * static_cast<double>(n));
  return {true,
          requiresPaddingCrop ? "target_legal_with_padding_crop"
                              : "target_legal",
          bias,
          requiresPaddingCrop,
          m,
          n,
          k,
          paddedM,
          paddedN,
          paddedK,
          computeOverhead,
          outputOverhead};
}

NativeCostCalibration calibrationFor(ModuleOp module) {
  NativeCostCalibration calibration;
  if (auto profile =
          module->getAttrOfType<StringAttr>("target.profile_id");
      profile && profile.getValue() == "raspberry-pi5-cortex-a76-cpu")
    calibration = raspberryPi5CortexA76F32Calibration();
  if (auto value = module->getAttrOfType<FloatAttr>(
          "target.native_cost_v2.effective_scalar_ops_per_ns"))
    calibration.effectiveScalarOpsPerNs = value.getValueAsDouble();
  if (auto value = module->getAttrOfType<FloatAttr>(
          "target.native_cost_v2.effective_vector_ops_per_ns"))
    calibration.effectiveVectorOpsPerNs = value.getValueAsDouble();
  if (auto value = module->getAttrOfType<FloatAttr>(
          "target.native_cost_v2.effective_bandwidth_bytes_per_ns"))
    calibration.effectiveBandwidthBytesPerNs = value.getValueAsDouble();
  if (auto value = module->getAttrOfType<FloatAttr>(
          "target.native_cost_v2.vector_utilization"))
    calibration.vectorUtilization = value.getValueAsDouble();
  if (auto value = module->getAttrOfType<BoolAttr>(
          "target.native_cost_v2.supports_vector"))
    calibration.supportsVector = value.getValue();
  if (auto value = module->getAttrOfType<BoolAttr>(
          "target.native_cost_v2.vector_lowering_available"))
    calibration.supportsVector =
        calibration.supportsVector && value.getValue();
  return calibration;
}

DictionaryAttr candidateEvidence(MLIRContext *context, OpBuilder &builder,
                                 const MatMulLoweringCandidate &candidate,
                                 bool selected) {
  auto i64 = [&](int64_t value) { return builder.getI64IntegerAttr(value); };
  auto f64 = [&](double value) { return builder.getF64FloatAttr(value); };
  SmallVector<NamedAttribute> attrs;
  attrs.emplace_back(builder.getStringAttr("candidate_id"),
                     builder.getStringAttr(candidate.id));
  attrs.emplace_back(builder.getStringAttr("fusion_mode"),
                     builder.getStringAttr(fusionName(candidate.fusion)));
  attrs.emplace_back(builder.getStringAttr("schedule_mode"),
                     builder.getStringAttr(scheduleName(candidate.schedule)));
  attrs.emplace_back(builder.getStringAttr("m_tail_strategy"),
                     builder.getStringAttr(tailName(candidate.mTailStrategy)));
  attrs.emplace_back(builder.getStringAttr("n_tail_strategy"),
                     builder.getStringAttr(tailName(candidate.nTailStrategy)));
  attrs.emplace_back(builder.getStringAttr("k_tail_strategy"),
                     builder.getStringAttr(tailName(candidate.kTailStrategy)));
  attrs.emplace_back(builder.getStringAttr("original_shape"),
                     builder.getArrayAttr({i64(candidate.originalM),
                                           i64(candidate.originalN),
                                           i64(candidate.originalK)}));
  attrs.emplace_back(builder.getStringAttr("executed_shape"),
                     builder.getArrayAttr({i64(candidate.executedM),
                                           i64(candidate.executedN),
                                           i64(candidate.executedK)}));
  attrs.emplace_back(builder.getStringAttr("requires_padding"),
                     builder.getBoolAttr(candidate.requiresPadding));
  attrs.emplace_back(builder.getStringAttr("requires_crop"),
                     builder.getBoolAttr(candidate.requiresCrop));
  attrs.emplace_back(builder.getStringAttr("uses_materialized_padding"),
                     builder.getBoolAttr(candidate.usesMaterializedPadding));
  attrs.emplace_back(builder.getStringAttr("uses_masked_loads"),
                     builder.getBoolAttr(candidate.usesMaskedLoads));
  attrs.emplace_back(builder.getStringAttr("uses_masked_stores"),
                     builder.getBoolAttr(candidate.usesMaskedStores));
  attrs.emplace_back(builder.getStringAttr("uses_scalar_cleanup"),
                     builder.getBoolAttr(candidate.usesScalarCleanup));
  attrs.emplace_back(builder.getStringAttr("vector_width"),
                     i64(candidate.vectorWidth));
  attrs.emplace_back(builder.getStringAttr("m_remainder"),
                     i64(candidate.mRemainder));
  attrs.emplace_back(builder.getStringAttr("n_remainder"),
                     i64(candidate.nRemainder));
  attrs.emplace_back(builder.getStringAttr("k_remainder"),
                     i64(candidate.kRemainder));
  attrs.emplace_back(builder.getStringAttr("m_tile_count"),
                     i64(candidate.mTileCount));
  attrs.emplace_back(builder.getStringAttr("n_tile_count"),
                     i64(candidate.nTileCount));
  attrs.emplace_back(builder.getStringAttr("k_tile_count"),
                     i64(candidate.kTileCount));
  attrs.emplace_back(builder.getStringAttr("tail_invocation_count"),
                     i64(candidate.estimatedTailInvocationCount));
  attrs.emplace_back(builder.getStringAttr("full_tile_count"),
                     i64(candidate.fullTileCount));
  attrs.emplace_back(builder.getStringAttr("edge_tile_count"),
                     i64(candidate.edgeTileCount));
  attrs.emplace_back(builder.getStringAttr("tail_elements"),
                     i64(candidate.tailElementCount));
  attrs.emplace_back(builder.getStringAttr("materialized_padding_bytes"),
                     i64(candidate.materializedPaddingBytes));
  attrs.emplace_back(builder.getStringAttr("zero_fill_bytes"),
                     i64(candidate.zeroFillBytes));
  attrs.emplace_back(builder.getStringAttr("copy_bytes"),
                     i64(candidate.copyBytes));
  attrs.emplace_back(builder.getStringAttr("masked_lane_count"),
                     i64(candidate.maskedLaneCount));
  attrs.emplace_back(builder.getStringAttr("scalar_cleanup_ops"),
                     i64(candidate.scalarCleanupOperations));
  attrs.emplace_back(builder.getStringAttr("direct_vector_cleanup_ops"),
                     i64(candidate.directVectorCleanupOperations));
  attrs.emplace_back(builder.getStringAttr("estimated_code_size_bytes"),
                     i64(candidate.estimatedCodeSizeBytes));
  attrs.emplace_back(builder.getStringAttr("code_size_penalty_ns"),
                     f64(candidate.codeSizePenaltyNs));
  attrs.emplace_back(builder.getStringAttr("register_pressure_risk"),
                     f64(candidate.registerPressureRisk));
  attrs.emplace_back(builder.getStringAttr("spill_risk_penalty_ns"),
                     f64(candidate.spillRiskPenaltyNs));
  attrs.emplace_back(builder.getStringAttr("lowering_complete"),
                     builder.getBoolAttr(candidate.loweringComplete));
  attrs.emplace_back(builder.getStringAttr("temporary_allocation_count"),
                     i64(candidate.temporaryAllocationCount));
  attrs.emplace_back(builder.getStringAttr("temporary_allocated_bytes"),
                     i64(candidate.temporaryAllocatedBytes));
  attrs.emplace_back(builder.getStringAttr("intermediate_read_count"),
                     i64(candidate.intermediateReadCount));
  attrs.emplace_back(builder.getStringAttr("intermediate_write_count"),
                     i64(candidate.intermediateWriteCount));
  attrs.emplace_back(builder.getStringAttr("compute_operations"),
                     i64(candidate.computeOperations));
  attrs.emplace_back(builder.getStringAttr("compute_ns"),
                     f64(candidate.cost.computeNs));
  attrs.emplace_back(builder.getStringAttr("memory_traffic_ns"),
                     f64(candidate.cost.memoryTrafficNs));
  attrs.emplace_back(builder.getStringAttr("allocation_ns"),
                     f64(candidate.cost.allocationNs));
  attrs.emplace_back(builder.getStringAttr("padding_compute_ns"),
                     f64(candidate.cost.paddingComputeNs));
  attrs.emplace_back(builder.getStringAttr("padding_copy_ns"),
                     f64(candidate.cost.paddingCopyNs));
  attrs.emplace_back(builder.getStringAttr("crop_copy_ns"),
                     f64(candidate.cost.cropCopyNs));
  attrs.emplace_back(builder.getStringAttr("control_overhead_ns"),
                     f64(candidate.cost.controlOverheadNs));
  attrs.emplace_back(builder.getStringAttr("interaction_correction_ns"),
                     f64(candidate.cost.interactionCorrectionNs));
  attrs.emplace_back(builder.getStringAttr("total_ns"),
                     f64(candidate.cost.totalNs));
  attrs.emplace_back(builder.getStringAttr("selected"),
                     builder.getBoolAttr(selected));
  attrs.emplace_back(builder.getStringAttr("rejection_reason"),
                     builder.getStringAttr(candidate.rejectionReason));
  return DictionaryAttr::get(context, attrs);
}

void attachTargetProfileAttrs(Operation *op, OpBuilder &builder) {
  MLIRContext *context = op->getContext();
  op->setAttr("target.model",
              StringAttr::get(context, "portable_accelerator_v1"));
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
  op->setAttr("target.collective",
              StringAttr::get(context, "none"));
}

void attachPaddingAttrs(Operation *op, OpBuilder &builder,
                        const MatMulBiasReluLegality &legality) {
  MLIRContext *context = op->getContext();
  if (!legality.requiresPaddingCrop) {
    op->setAttr("target.padding", StringAttr::get(context, "none"));
    return;
  }
  op->setAttr("target.padding",
              StringAttr::get(context, "pad_to_tile_with_crop"));
  op->setAttr("target.valid_region",
              StringAttr::get(context, "original_m_n"));
  op->setAttr("target.original_m", builder.getI64IntegerAttr(legality.m));
  op->setAttr("target.original_n", builder.getI64IntegerAttr(legality.n));
  op->setAttr("target.original_k", builder.getI64IntegerAttr(legality.k));
  op->setAttr("target.padded_m", builder.getI64IntegerAttr(legality.paddedM));
  op->setAttr("target.padded_n", builder.getI64IntegerAttr(legality.paddedN));
  op->setAttr("target.padded_k", builder.getI64IntegerAttr(legality.paddedK));
  op->setAttr("target.pad_m",
              builder.getI64IntegerAttr(legality.paddedM - legality.m));
  op->setAttr("target.pad_n",
              builder.getI64IntegerAttr(legality.paddedN - legality.n));
  op->setAttr("target.pad_k",
              builder.getI64IntegerAttr(legality.paddedK - legality.k));
  op->setAttr("target.padding_compute_overhead_ratio",
              builder.getF64FloatAttr(legality.paddingComputeOverhead));
  op->setAttr("target.padding_output_overhead_ratio",
              builder.getF64FloatAttr(legality.paddingOutputOverhead));
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

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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

      // A completed native precision plan is causal: only a type-compatible
      // selected candidate is eligible for this fused lowering. Pipelines
      // that do not run quantization planning retain the historical
      // standalone fusion behavior.
      if (auto module = func->getParentOfType<ModuleOp>();
          module && module->getAttrOfType<BoolAttr>(
                        "quantization.planning_complete")) {
        auto selected =
            matmul->getAttrOfType<StringAttr>("quantization.candidate");
        auto lhsType =
            dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
        auto rhsType =
            dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
        bool selectedF32 =
            selected && selected.getValue() == "fp32" && lhsType && rhsType &&
            lhsType.getElementType().isF32() &&
            rhsType.getElementType().isF32();
        bool selectedI8 =
            selected && selected.getValue() == "int8" && lhsType && rhsType &&
            lhsType.getElementType().isInteger(8) &&
            rhsType.getElementType().isInteger(8);
        if (!selectedF32 && !selectedI8)
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
          OpBuilder builder(context);
          ModuleOp module = func->getParentOfType<ModuleOp>();
          bool useCostModel =
              module &&
              (module->hasAttr("quantization.planning_complete") ||
               module->hasAttr("target.native_cost_v2.supports_vector"));
          if (useCostModel) {
            MatMulLoweringProblem problem;
            problem.m = legality.m;
            problem.n = legality.n;
            problem.k = legality.k;
            problem.dtype =
                cast<RankedTensorType>(matmul.getInputs()[0].getType())
                        .getElementType()
                        .isF16()
                    ? "f16"
                    : "f32";
            problem.fusionLegal = true;
            problem.tileM = kTargetTileM;
            problem.tileN = kTargetTileN;
            problem.tileK = kTargetTileK;
            NativeCostCalibration calibration = calibrationFor(module);
            MatMulLoweringSelection selection =
                selectMatMulLowering(problem, calibration);
            SmallVector<Attribute> evidence;
            for (int i = 0,
                     e = static_cast<int>(selection.candidates.size());
                 i < e; ++i)
              evidence.push_back(candidateEvidence(
                  context, builder, selection.candidates[i],
                  i == selection.selectedIndex));
            matmul->setAttr("native.cost_model.version",
                            builder.getStringAttr("static_cost_model_v2"));
            matmul->setAttr("native.cost_model.target",
                            builder.getStringAttr(calibration.target));
            matmul->setAttr("native.cost_model.candidates",
                            builder.getArrayAttr(evidence));
            const MatMulLoweringCandidate *winner = selection.winner();
            if (!winner) {
              matmul->setAttr(
                  "fusion.reject_reason",
                  builder.getStringAttr("no_legal_native_lowering_candidate"));
              continue;
            }
            matmul->setAttr("native.cost_model.selected_candidate",
                            builder.getStringAttr(winner->id));
            matmul->setAttr(
                "native.cost_model.selected_schedule",
                builder.getStringAttr(scheduleName(winner->schedule)));
            matmul->setAttr("native.cost_model.selected_total_ns",
                            builder.getF64FloatAttr(winner->cost.totalNs));
            if (winner->fusion != FusionMode::Fused)
              continue;
          }

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
  if (!quantizationCandidate || quantizationCandidate.getValue() != "int8" ||
      !profileDecision || profileDecision.getValue() != "faster") {
    return false;
  }
  // hir.fused_qmatmul_bias_relu requires i8 lhs/rhs operands (its own
  // verifier enforces this). The quantization.candidate/profile.quantized_path
  // attrs are only a planning-time *hint* that int8 execution would be
  // profitable -- they do not themselves quantize the operands. Real
  // quantization (inserting hir.quantize) is a separate, later stage
  // (QuantizationMaterializationPass). If this matmul's operands are not
  // already i8-typed, honoring the hint here would construct a
  // hir.fused_qmatmul_bias_relu whose own verifier immediately rejects it
  // (a real bug found while validating this pipeline: the hint was trusted
  // without checking operand types, producing IR that failed its own op
  // verifier). Fall back to the fp32 fused op instead -- a safe, explicit
  // choice, not a silent miscompile.
  auto lhsType = dyn_cast<RankedTensorType>(matmul.getInputs()[0].getType());
  auto rhsType = dyn_cast<RankedTensorType>(matmul.getInputs()[1].getType());
  return lhsType && rhsType && lhsType.getElementType().isInteger(8) &&
         rhsType.getElementType().isInteger(8);
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
      auto quantized = FusedQMatMulBiasReluOp::create(
          rewriter, reluMap.getLoc(), loweredType,
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
      attachTargetProfileAttrs(quantized.getOperation(), rewriter);
      attachPaddingAttrs(quantized.getOperation(), rewriter, legality);
      fused = quantized.getOperation();
    } else {
      auto fp32 = FusedMatMulBiasReluOp::create(
          rewriter, reluMap.getLoc(), loweredType,
          matmul.getInputs()[0], matmul.getInputs()[1], bias);
      fp32->setAttr("fusion.candidate", candidate);
      fp32->setAttr("fusion.group", matmul->getAttr("fusion.group"));
      fp32->setAttr("kernel.selection",
                    StringAttr::get(matmul.getContext(), "runtime_profile"));
      fp32->setAttr("lowering.source",
                    StringAttr::get(matmul.getContext(), "linalg.matmul_add_relu"));
      if (auto planned =
              matmul->getAttrOfType<StringAttr>("quantization.candidate"))
        fp32->setAttr("quantization.candidate", planned);
      attachTargetProfileAttrs(fp32.getOperation(), rewriter);
      attachPaddingAttrs(fp32.getOperation(), rewriter, legality);
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

    auto lowered = FusedRMSNormOp::create(rewriter, op->getLoc(), loweredType,
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

struct HIRRMSNormToLinalgPattern : OpRewritePattern<FusedRMSNormOp> {
  using OpRewritePattern<FusedRMSNormOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedRMSNormOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!inputType || !outputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (inputType != outputType) {
      return rewriter.notifyMatchFailure(op, "input and output types must match");
    }
    if (inputType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "RMSNorm lowering expects rank-2 tensors");
    }
    if (!inputType.getElementType().isF32()) {
      return rewriter.notifyMatchFailure(op, "CPU linalg RMSNorm lowering supports f32");
    }
    if (inputType.isDynamicDim(0) || inputType.isDynamicDim(1)) {
      return rewriter.notifyMatchFailure(op, "CPU linalg RMSNorm lowering expects static shapes");
    }

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type elementType = inputType.getElementType();
    int64_t rows = inputType.getDimSize(0);
    int64_t hidden = inputType.getDimSize(1);

    auto rowType = RankedTensorType::get({rows}, elementType);
    auto zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));
    auto epsilon = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, 1.0e-6));
    auto hiddenConstant = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, static_cast<double>(hidden)));

    auto rowEmpty = tensor::EmptyOp::create(
        rewriter, loc, ArrayRef<int64_t>{rows}, elementType);
    auto rowInit = linalg::FillOp::create(
        rewriter, loc, ValueRange{zero.getResult()}, ValueRange{rowEmpty.getResult()});
    Value rowAccumulator = rowInit.getResult(0);

    SmallVector<AffineMap> reductionMaps = {
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
                       context),
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0)}, context)};
    SmallVector<utils::IteratorType> reductionIterators = {
        utils::IteratorType::parallel,
        utils::IteratorType::reduction};

    auto sumSquares = linalg::GenericOp::create(
        rewriter, loc, rowType, ValueRange{op.getInput()},
        ValueRange{rowAccumulator}, reductionMaps, reductionIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value squared = arith::MulFOp::create(
              nestedBuilder, nestedLoc, args[0], args[0]);
          Value sum = arith::AddFOp::create(
              nestedBuilder, nestedLoc, args[1], squared);
          linalg::YieldOp::create(nestedBuilder, nestedLoc, sum);
        });

    auto outputEmpty = tensor::EmptyOp::create(
        rewriter, loc, inputType.getShape(), elementType);
    SmallVector<AffineMap> normalizeMaps = {
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
                       context),
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0)}, context),
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
                       context)};
    SmallVector<utils::IteratorType> normalizeIterators = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel};

    auto normalized = linalg::GenericOp::create(
        rewriter, loc, outputType, ValueRange{op.getInput(), sumSquares.getResult(0)},
        ValueRange{outputEmpty.getResult()}, normalizeMaps, normalizeIterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value mean = arith::DivFOp::create(
              nestedBuilder, nestedLoc, args[1], hiddenConstant.getResult());
          Value variance = arith::AddFOp::create(
              nestedBuilder, nestedLoc, mean, epsilon.getResult());
          Value inv = math::RsqrtOp::create(nestedBuilder, nestedLoc, variance);
          Value result = arith::MulFOp::create(
              nestedBuilder, nestedLoc, args[0], inv);
          linalg::YieldOp::create(nestedBuilder, nestedLoc, result);
        });

    rewriter.replaceOp(op, normalized.getResults());
    return success();
  }
};

struct HIRRMSNormToLinalgPass
    : impl::HIRRMSNormToLinalgBase<HIRRMSNormToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    HIRDialect,
                    linalg::LinalgDialect,
                    math::MathDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<HIRRMSNormToLinalgPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

int64_t intAttr(Operation *op, StringRef name) {
  return op->getAttrOfType<IntegerAttr>(name).getInt();
}

AffineMap biasMapFor(OpBuilder &builder, RankedTensorType biasType,
                     int64_t expectedM) {
  MLIRContext *context = builder.getContext();
  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  if (biasType.getDimSize(0) == 1 && expectedM != 1) {
    return AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                          {builder.getAffineConstantExpr(0), d1}, context);
  }
  return AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0, {d0, d1},
                        context);
}

Value createZeroPaddedTensor(PatternRewriter &rewriter, Location loc,
                             Value source, RankedTensorType sourceType,
                             ArrayRef<int64_t> paddedShape) {
  auto paddedType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  SmallVector<OpFoldResult> low(sourceType.getRank(),
                                rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> high;
  for (auto [original, padded] :
       llvm::zip_equal(sourceType.getShape(), paddedShape)) {
    high.push_back(rewriter.getIndexAttr(padded - original));
  }

  auto zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getFloatAttr(sourceType.getElementType(), 0.0));
  auto pad = tensor::PadOp::create(rewriter, loc, paddedType, source,
                                   low, high, zero.getResult(), false, {});
  return pad.getResult();
}

struct HIRMatMulBiasReluToLinalgPattern
    : OpRewritePattern<FusedMatMulBiasReluOp> {
  using OpRewritePattern<FusedMatMulBiasReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedMatMulBiasReluOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    auto biasType = dyn_cast<RankedTensorType>(op.getBias().getType());
    auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!lhsType || !rhsType || !biasType || !outputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        biasType.getRank() != 2 || outputType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "MatMul-Bias-ReLU lowering expects rank-2 tensors");
    }
    if (!lhsType.getElementType().isF32() ||
        rhsType.getElementType() != lhsType.getElementType() ||
        biasType.getElementType() != lhsType.getElementType() ||
        outputType.getElementType() != lhsType.getElementType()) {
      return rewriter.notifyMatchFailure(op, "CPU linalg MatMul-Bias-ReLU lowering supports f32");
    }
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
        outputType.hasStaticShape()) {
      int64_t m = lhsType.getDimSize(0);
      int64_t k = lhsType.getDimSize(1);
      int64_t rhsK = rhsType.getDimSize(0);
      int64_t n = rhsType.getDimSize(1);
      if (k != rhsK || outputType.getDimSize(0) != m ||
          outputType.getDimSize(1) != n) {
        return rewriter.notifyMatchFailure(op, "matmul/output shape mismatch");
      }
      bool legalBias = (biasType.getDimSize(0) == m || biasType.getDimSize(0) == 1) &&
                       biasType.getDimSize(1) == n;
      if (!legalBias) {
        return rewriter.notifyMatchFailure(op, "bias broadcast shape mismatch");
      }
    } else {
      return rewriter.notifyMatchFailure(op, "CPU linalg MatMul-Bias-ReLU lowering expects static shapes");
    }

    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type elementType = outputType.getElementType();
    auto outputShape = outputType.getShape();
    auto padding = op->getAttrOfType<StringAttr>("target.padding");
    bool usePaddingCrop =
        padding && padding.getValue() == "pad_to_tile_with_crop";

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value bias = op.getBias();
    RankedTensorType matmulOutputType = outputType;
    RankedTensorType addReluOutputType = outputType;
    int64_t biasExpectedM = outputType.getDimSize(0);
    if (usePaddingCrop) {
      int64_t paddedM = intAttr(op.getOperation(), "target.padded_m");
      int64_t paddedN = intAttr(op.getOperation(), "target.padded_n");
      int64_t paddedK = intAttr(op.getOperation(), "target.padded_k");
      lhs = createZeroPaddedTensor(rewriter, loc, lhs, lhsType,
                                   {paddedM, paddedK});
      rhs = createZeroPaddedTensor(rewriter, loc, rhs, rhsType,
                                   {paddedK, paddedN});
      int64_t paddedBiasM =
          biasType.getDimSize(0) == 1 ? 1 : paddedM;
      bias = createZeroPaddedTensor(rewriter, loc, bias, biasType,
                                    {paddedBiasM, paddedN});
      matmulOutputType =
          RankedTensorType::get({paddedM, paddedN}, elementType);
      addReluOutputType = matmulOutputType;
      biasType = cast<RankedTensorType>(bias.getType());
      biasExpectedM = paddedM;
    }

    auto zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));
    auto matmulEmpty = tensor::EmptyOp::create(
        rewriter, loc, matmulOutputType.getShape(), elementType);
    // linalg.matmul accumulates into its "outs" operand (it has a reduction
    // iterator over K), so that operand must start at zero. tensor.empty()
    // does not guarantee zero-initialized content -- its result is
    // documented as undefined, and in practice the backing buffer is only
    // accidentally zero on a fresh, never-reused heap allocation. On repeat
    // invocation (heap address reuse), the stale contents of a *previous*
    // call's own output buffer get accumulated into, producing incorrect,
    // growing results (found via a real/AArch64 repeated-invocation
    // regression: bufferized output equals (call_count+1) x correct value
    // for the affected elements). Explicit linalg.fill matches the existing,
    // correct pattern already used for the RMSNorm row accumulator above.
    auto matmulInit = linalg::FillOp::create(
        rewriter, loc, ValueRange{zero.getResult()},
        ValueRange{matmulEmpty.getResult()});
    auto matmul = linalg::MatmulOp::create(
        rewriter, loc, matmulOutputType,
        ValueRange{lhs, rhs},
        ValueRange{matmulInit.getResult(0)});

    auto outputEmpty = tensor::EmptyOp::create(
        rewriter, loc, addReluOutputType.getShape(), elementType);
    SmallVector<AffineMap> maps = {
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
                       context),
        biasMapFor(rewriter, biasType, biasExpectedM),
        AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                       {rewriter.getAffineDimExpr(0), rewriter.getAffineDimExpr(1)},
                       context)};
    SmallVector<utils::IteratorType> iterators = {
        utils::IteratorType::parallel,
        utils::IteratorType::parallel};

    auto addRelu = linalg::GenericOp::create(
        rewriter, loc, addReluOutputType,
        ValueRange{matmul->getResult(0), bias},
        ValueRange{outputEmpty.getResult()}, maps, iterators,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          Value added = arith::AddFOp::create(
              nestedBuilder, nestedLoc, args[0], args[1]);
          Value relu = arith::MaximumFOp::create(
              nestedBuilder, nestedLoc, added, zero.getResult());
          linalg::YieldOp::create(nestedBuilder, nestedLoc, relu);
        });

    if (usePaddingCrop) {
      auto cropped = tensor::ExtractSliceOp::create(
          rewriter, loc, outputType, addRelu->getResult(0),
          ValueRange{}, ValueRange{}, ValueRange{},
          ArrayRef<int64_t>{0, 0}, outputShape, ArrayRef<int64_t>{1, 1});
      rewriter.replaceOp(op, cropped.getResult());
      return success();
    }

    rewriter.replaceOp(op, addRelu.getResults());
    return success();
  }
};

struct HIRMatMulBiasReluToLinalgPass
    : impl::HIRMatMulBiasReluToLinalgBase<HIRMatMulBiasReluToLinalgPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    HIRDialect,
                    linalg::LinalgDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<HIRMatMulBiasReluToLinalgPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    WalkResult result =
        getOperation().walk([&](FusedMatMulBiasReluOp op) -> WalkResult {
          auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
          if (!outputType)
            return op.emitError()
                       << "hir-native-codegen cannot lower '" << op->getName()
                       << "': result is not a ranked tensor",
                   WalkResult::interrupt();
          if (!outputType.getElementType().isF32())
            return op.emitError()
                       << "hir-native-codegen cannot lower '" << op->getName()
                       << "': element type " << outputType.getElementType()
                       << " is unsupported by HIRMatMulBiasReluToLinalg "
                          "lowering; supported element type: f32; shape: "
                       << outputType,
                   WalkResult::interrupt();
          if (!outputType.hasStaticShape())
            return op.emitError()
                       << "hir-native-codegen cannot lower '" << op->getName()
                       << "': dynamic shape is unsupported by "
                          "HIRMatMulBiasReluToLinalg lowering; shape: "
                       << outputType,
                   WalkResult::interrupt();
          return op.emitError()
                     << "hir-native-codegen cannot lower '" << op->getName()
                     << "': operands do not satisfy the rank-2 "
                        "MatMul/Bias/ReLU shape contract; result: "
                     << outputType,
                 WalkResult::interrupt();
        });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

bool hasRsqrtInBody(linalg::GenericOp generic) {
  bool foundRsqrt = false;
  generic.getBody()->walk([&](math::RsqrtOp) { foundRsqrt = true; });
  return foundRsqrt;
}

bool hasMulInBody(linalg::GenericOp generic) {
  bool foundMul = false;
  generic.getBody()->walk([&](arith::MulFOp) { foundMul = true; });
  return foundMul;
}

bool allParallelIterators(linalg::GenericOp generic) {
  for (utils::IteratorType iteratorType : generic.getIteratorTypesArray()) {
    if (iteratorType != utils::IteratorType::parallel) {
      return false;
    }
  }
  return true;
}

struct StableHLOCompatibleRMSNormPattern
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp normalize,
                                PatternRewriter &rewriter) const override {
    if (normalize->getNumResults() != 1 || normalize.getInputs().size() != 2 ||
        normalize.getNumDpsInits() != 1) {
      return failure();
    }
    if (!allParallelIterators(normalize) || !hasRsqrtInBody(normalize) ||
        !hasMulInBody(normalize)) {
      return failure();
    }

    Value input = normalize.getInputs()[0];
    Value reduction = normalize.getInputs()[1];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto reductionType = dyn_cast<RankedTensorType>(reduction.getType());
    auto outputType =
        dyn_cast<RankedTensorType>(normalize->getResult(0).getType());
    if (!inputType || !reductionType || !outputType) {
      return failure();
    }
    if (inputType != outputType || inputType.getRank() != 2 ||
        reductionType.getRank() != 1 ||
        reductionType.getDimSize(0) != inputType.getDimSize(0) ||
        !inputType.getElementType().isF32()) {
      return failure();
    }

    auto reductionProducer = reduction.getDefiningOp<linalg::GenericOp>();
    if (!reductionProducer || !hasMulInBody(reductionProducer)) {
      return failure();
    }

    auto fused = FusedRMSNormOp::create(
        rewriter, normalize.getLoc(), outputType, input);
    fused->setAttr("fusion.candidate",
                   StringAttr::get(normalize.getContext(), "rmsnorm"));
    fused->setAttr("fusion.group",
                   StringAttr::get(normalize.getContext(),
                                   "stablehlo_compatible_rmsnorm_0"));
    fused->setAttr("kernel.selection",
                   StringAttr::get(normalize.getContext(), "runtime_profile"));
    fused->setAttr("lowering.source",
                   StringAttr::get(normalize.getContext(), "llm.rmsnorm"));
    fused->setAttr("frontend.source",
                   StringAttr::get(
                       normalize.getContext(),
                       "stablehlo_compatible_rmsnorm_decomposition"));

    rewriter.replaceOp(normalize, fused->getResults());
    return success();
  }
};

struct StableHLOCompatibleRMSNormImportPass
    : impl::StableHLOCompatibleRMSNormImportBase<
          StableHLOCompatibleRMSNormImportPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect,
                    HIRDialect,
                    linalg::LinalgDialect,
                    math::MathDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<StableHLOCompatibleRMSNormPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
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

std::unique_ptr<Pass> createHIRRMSNormToLinalgPass() {
  return std::make_unique<HIRRMSNormToLinalgPass>();
}

std::unique_ptr<Pass> createHIRMatMulBiasReluToLinalgPass() {
  return std::make_unique<HIRMatMulBiasReluToLinalgPass>();
}

std::unique_ptr<Pass> createStableHLOCompatibleRMSNormImportPass() {
  return std::make_unique<StableHLOCompatibleRMSNormImportPass>();
}

void registerFusionPasses() {
  PassRegistration<HIRCanonicalizationPass>();
  PassRegistration<HIRFusedOpVerifierPass>();
  PassRegistration<HIRRMSNormToLinalgPass>();
  PassRegistration<HIRMatMulBiasReluToLinalgPass>();
  PassRegistration<StableHLOCompatibleRMSNormImportPass>();

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

  static PassPipelineRegistration<> rmsnormToLinalgPipeline(
      "hir-rmsnorm-to-linalg",
      "Lower hir.fused_rmsnorm to executable Linalg/Math tensor IR",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createHIRRMSNormToLinalgPass());
      });

  static PassPipelineRegistration<> matmulToLinalgPipeline(
      "hir-matmul-bias-relu-to-linalg",
      "Lower hir.fused_matmul_bias_relu to executable Linalg tensor IR",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(
            createHIRMatMulBiasReluToLinalgPass());
      });

  static PassPipelineRegistration<> stableHLOCompatibleRMSNormPipeline(
      "stablehlo-compatible-rmsnorm-import",
      "Import StableHLO-compatible RMSNorm decomposition into HIR",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(
            createStableHLOCompatibleRMSNormImportPass());
      });

  static PassPipelineRegistration<> servingPhaseAnalysisPipeline(
      "serving-phase-analysis",
      "Annotate serving functions with colocated vs pd_split policy decisions",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createServingPhaseAnalysisPass());
      });

  static PassPipelineRegistration<> llmFrontendNormPipeline(
      "llm-frontend-normalization",
      "Normalize simplified LLM attention graph to canonical serving-aware IR",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createLLMFrontendNormalizationPass());
      });

  static PassPipelineRegistration<> cvFrontendNormPipeline(
      "cv-frontend-normalization",
      "Detect and annotate pseudo-CV YOLOSeg-like graph patterns",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCVFrontendNormalizationPass());
      });

  static PassPipelineRegistration<> cvShapeInferencePipeline(
      "cv-shape-inference",
      "Annotate pseudo-CV ops with static shape-derived size metadata",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCVShapeInferencePass());
      });

  static PassPipelineRegistration<> cvMemoryPlanningPipeline(
      "cv-memory-planning",
      "Assign compiler buffer slots to pseudo-CV ops using linear lifetime reuse",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCVMemoryPlanningPass());
      });

  static PassPipelineRegistration<> cvExecutionDomainPlanningPipeline(
      "cv-execution-domain-planning",
      "Classify pseudo-CV ops into portable accelerated/host/fallback execution domains",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCVExecutionDomainPlanningPass());
      });

  static PassPipelineRegistration<> cvSemanticAnnotationPipeline(
      "cv-semantic-annotation",
      "Annotate real upstream YOLO-Seg MLIR with CV semantic attrs",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCVSemanticAnnotationPass());
      });

  static PassPipelineRegistration<> cvExecutionPlanAttrsPipeline(
      "cv-execution-plan-attrs",
      "Attach minimum CV planning attrs for canonical ExecutionPlan export",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createCVExecutionPlanAttrsPass());
      });

  // QuantizationPlanningPass is module-scoped: it reads/writes module attrs
  // and optionally annotates linalg.matmul ops inside functions.
  static PassPipelineRegistration<> quantizationPlanningPipeline(
      "quantization-planning",
      "Compiler-side precision selection for the HIR serving pipeline",
      [](OpPassManager &pm) {
        pm.addPass(createQuantizationPlanningPass());
      });

  static PassPipelineRegistration<> quantCanonicalizationPipeline(
      "hir-quant-canonicalize",
      "Eliminate redundant HIR quantize/dequantize boundaries",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(
            createHIRQuantCanonicalizationPass());
      });

  static PassPipelineRegistration<> quantPropagationPipeline(
      "hir-quant-propagate",
      "Annotate conservative INT8-capable islands",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(createHIRQuantPropagationPass());
      });

  static PassPipelineRegistration<> int8OperatorSelectionPipeline(
      "hir-int8-operator-selection",
      "Select INT8 execution metadata using a small capability table",
      [](OpPassManager &pm) {
        pm.addNestedPass<func::FuncOp>(
            createHIRINT8OperatorSelectionPass());
      });

  // Registers kv-layout-planning, replay-eligibility, execution-provider-planning
  // standalone wrappers and serving-optimization-pipeline (four-pass).
  registerServingOptimizationPipeline();
}

} // namespace mlir::hir
