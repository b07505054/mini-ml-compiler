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
#define GEN_PASS_DEF_QUANTIZATIONMATERIALIZATION
#define GEN_PASS_DEF_QUANTIZEDKERNELLOWERING
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

std::string strAttr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<StringAttr>(name))
    return attr.getValue().str();
  return {};
}

FloatAttr floatAttr(Operation *op, StringRef name) {
  return op->getAttrOfType<FloatAttr>(name);
}

IntegerAttr intAttr(Operation *op, StringRef name) {
  return op->getAttrOfType<IntegerAttr>(name);
}

bool boolAttr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<BoolAttr>(name))
    return attr.getValue();
  return false;
}

LogicalResult requireSelectedInt8Fact(Operation *op, StringRef name) {
  if (auto attr = op->getAttr(name))
    return success();
  return op->emitError("selected INT8 materialization requires attribute '")
         << name << "'";
}

bool isSelectedPackedInt8(FusedMatMulBiasReluOp op) {
  Operation *raw = op.getOperation();
  std::string scheme = strAttr(raw, "quant.scheme");
  if (scheme.empty())
    scheme = strAttr(raw, "quant.strategy");
  if (scheme != "int8_static_symmetric")
    return false;
  return strAttr(raw, "quant.required_kernel_capability") ==
             "quant_kernel.int8_static_symmetric.packed_b_transposed" ||
         boolAttr(raw, "quant.kernel_requires_packed_weight") ||
         strAttr(raw, "quant.packed_layout") == "packed_b_transposed_nxk";
}

LogicalResult validateMaterializationInputs(FusedMatMulBiasReluOp op) {
  Operation *raw = op.getOperation();
  for (StringRef name : {
           "quant.selected_candidate_id",
           "quant.selected_complete_candidate_id",
           "quant.activation_scale",
           "quant.weight_scale",
           "quant.activation_zero_point",
           "quant.weight_zero_point",
           "quant.calibration_artifact_ref",
           "quant.calibration_artifact_id",
           "quant.calibration_artifact_sha256",
           "quant.packed_weight_artifact_ref",
           "quant.packed_weight_artifact_id",
           "quant.packed_weight_sha256",
           "quant.source_weight_sha256",
           "quant.packed_layout",
           "quant.packing_scheme",
           "quant.required_kernel_capability",
           "quant.kernel_id",
           "quant.codegen_target_id",
           "quant.binary_sha256",
           "quant.workload_id",
       }) {
    if (failed(requireSelectedInt8Fact(raw, name)))
      return failure();
  }
  if (strAttr(raw, "quant.packed_layout") != "packed_b_transposed_nxk")
    return raw->emitError("selected INT8 materialization requires packed_b_transposed_nxk layout");
  if (strAttr(raw, "quant.packing_scheme") != "b_transposed_nxk_contiguous")
    return raw->emitError("selected INT8 materialization requires b_transposed_nxk_contiguous packing");
  if (strAttr(raw, "quant.required_kernel_capability") !=
      "quant_kernel.int8_static_symmetric.packed_b_transposed")
    return raw->emitError("selected INT8 materialization requires packed INT8 kernel capability");
  if (strAttr(raw, "quant.kernel_id") !=
      "portable_fused_matmul_bias_relu_int8_symmetric_packed_b")
    return raw->emitError("selected INT8 materialization requires the packed portable CPU INT8 kernel id");
  if (intAttr(raw, "quant.activation_zero_point").getInt() != 0 ||
      intAttr(raw, "quant.weight_zero_point").getInt() != 0)
    return raw->emitError("selected INT8 materialization requires symmetric zero points equal to 0");

  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
  auto biasType = dyn_cast<RankedTensorType>(op.getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!lhsType || !rhsType || !biasType || !outputType ||
      lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      outputType.getRank() != 2 ||
      (biasType.getRank() != 1 && biasType.getRank() != 2))
    return raw->emitError("selected INT8 materialization requires ranked fused MatMul+Bias+ReLU tensors");
  if (lhsType.getDimSize(1) != ShapedType::kDynamic &&
      rhsType.getDimSize(0) != ShapedType::kDynamic &&
      lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return raw->emitError("selected INT8 materialization shape mismatch: lhs K != rhs K");
  return success();
}

void copyAttrs(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getAttrs())
    to->setAttr(attr.getName(), attr.getValue());
}

void copyAttrsWithPrefixes(Operation *from, Operation *to,
                           ArrayRef<StringRef> prefixes) {
  for (NamedAttribute attr : from->getAttrs()) {
    StringRef name = attr.getName().strref();
    for (StringRef prefix : prefixes) {
      if (name.starts_with(prefix)) {
        to->setAttr(attr.getName(), attr.getValue());
        break;
      }
    }
  }
}

void addCommonQuantAttrs(Operation *source, SmallVectorImpl<NamedAttribute> &attrs,
                         Builder &builder) {
  auto addString = [&](StringRef dst, StringRef src) {
    std::string v = strAttr(source, src);
    if (!v.empty())
      attrs.push_back(builder.getNamedAttr(dst, builder.getStringAttr(v)));
  };
  auto addFloat = [&](StringRef dst, StringRef src) {
    if (auto v = floatAttr(source, src))
      attrs.push_back(builder.getNamedAttr(dst, v));
  };
  auto addInt = [&](StringRef dst, StringRef src) {
    if (auto v = intAttr(source, src))
      attrs.push_back(builder.getNamedAttr(dst, v));
  };
  addString("selected_complete_candidate_id",
            "quant.selected_complete_candidate_id");
  addString("selected_candidate_id", "quant.selected_candidate_id");
  addString("kernel_id", "quant.kernel_id");
  addString("codegen_target_id", "quant.codegen_target_id");
  addString("binary_sha256", "quant.binary_sha256");
  addString("packed_weight_artifact_ref", "quant.packed_weight_artifact_ref");
  addString("packed_weight_artifact_id", "quant.packed_weight_artifact_id");
  addString("packed_weight_sha256", "quant.packed_weight_sha256");
  addString("calibration_artifact_ref", "quant.calibration_artifact_ref");
  addString("calibration_artifact_id", "quant.calibration_artifact_id");
  addString("calibration_artifact_sha256",
            "quant.calibration_artifact_sha256");
  addString("workload_id", "quant.workload_id");
  addString("target_architecture", "quant.target_architecture");
  addString("target_microarchitecture", "quant.target_microarchitecture");
  addString("measurement_artifact_ref", "quant.measurement_artifact_ref");
  addString("build_manifest_ref", "quant.build_manifest_ref");
  addFloat("activation_scale", "quant.activation_scale");
  addFloat("weight_scale", "quant.weight_scale");
  addInt("activation_zero_point", "quant.activation_zero_point");
  addInt("weight_zero_point", "quant.weight_zero_point");
  attrs.push_back(builder.getNamedAttr("scheme",
                                       builder.getStringAttr("int8_static_symmetric")));
  attrs.push_back(builder.getNamedAttr("packed_layout",
                                       builder.getStringAttr("packed_b_transposed_nxk")));
  attrs.push_back(builder.getNamedAttr("packing_scheme",
                                       builder.getStringAttr("b_transposed_nxk_contiguous")));
  attrs.push_back(builder.getNamedAttr("accumulator_dtype",
                                       builder.getStringAttr("int32")));
  attrs.push_back(builder.getNamedAttr("output_dtype",
                                       builder.getStringAttr("fp32")));
}

ArrayAttr buildExecutionStages(Builder &builder, Operation *source) {
  auto str = [&](StringRef v) { return builder.getStringAttr(v); };
  auto strArray = [&](ArrayRef<StringRef> values) {
    SmallVector<Attribute> attrs;
    for (StringRef v : values)
      attrs.push_back(str(v));
    return builder.getArrayAttr(attrs);
  };
  auto stage = [&](StringRef id, StringRef op, ArrayRef<StringRef> deps,
                   StringRef produces) {
    SmallVector<NamedAttribute> attrs;
    attrs.push_back(builder.getNamedAttr("stage_id", str(id)));
    attrs.push_back(builder.getNamedAttr("op", str(op)));
    attrs.push_back(builder.getNamedAttr("dependency_ids", strArray(deps)));
    attrs.push_back(builder.getNamedAttr("produces", str(produces)));
    if (id == "quantize_activation") {
      if (auto scale = floatAttr(source, "quant.activation_scale"))
        attrs.push_back(builder.getNamedAttr("scale", scale));
      if (auto zp = intAttr(source, "quant.activation_zero_point"))
        attrs.push_back(builder.getNamedAttr("zero_point", zp));
      attrs.push_back(builder.getNamedAttr("rounding_mode",
                                           str("round_nearest_even")));
      attrs.push_back(builder.getNamedAttr("clamp_min",
          builder.getI64IntegerAttr(-127)));
      attrs.push_back(builder.getNamedAttr("clamp_max",
          builder.getI64IntegerAttr(127)));
      attrs.push_back(builder.getNamedAttr("source_dtype", str("fp32")));
      attrs.push_back(builder.getNamedAttr("destination_dtype", str("int8")));
    }
    if (id == "load_packed_weight") {
      attrs.push_back(builder.getNamedAttr("artifact_ref",
          str(strAttr(source, "quant.packed_weight_artifact_ref"))));
      attrs.push_back(builder.getNamedAttr("artifact_sha256",
          str(strAttr(source, "quant.packed_weight_sha256"))));
      attrs.push_back(builder.getNamedAttr("packed_layout",
          str("packed_b_transposed_nxk")));
    }
    if (id == "execute_int8_kernel") {
      attrs.push_back(builder.getNamedAttr("kernel_id",
          str(strAttr(source, "quant.kernel_id"))));
      attrs.push_back(builder.getNamedAttr("fused_postprocess",
          str("dequantize_bias_relu")));
      attrs.push_back(builder.getNamedAttr("binary_sha256",
          str(strAttr(source, "quant.binary_sha256"))));
    }
    return DictionaryAttr::get(builder.getContext(), attrs);
  };
  return builder.getArrayAttr({
      stage("quantize_activation", "hir.quantize", {},
            "quantized_activation_ready"),
      stage("load_packed_weight", "hir.load_quantized_weight", {},
            "packed_weight_ready"),
      stage("execute_int8_kernel",
            "hir.portable_cpu_int8_fused_matmul_bias_relu",
            {"quantized_activation_ready", "packed_weight_ready"},
            "fp32_output_ready"),
      stage("return_fp32_output", "runtime.return",
            {"fp32_output_ready"}, "return_ready"),
  });
}

void stampCpuVisibleMemoryPlacement(Operation *op, Builder &builder,
                                    RankedTensorType inputType,
                                    RankedTensorType packedType,
                                    RankedTensorType outputType) {
  int64_t m = inputType.getDimSize(0);
  int64_t k = inputType.getDimSize(1);
  int64_t n = packedType.getDimSize(0);
  if (m == ShapedType::kDynamic || n == ShapedType::kDynamic ||
      k == ShapedType::kDynamic)
    return;
  int64_t inputBytes = m * k * 4;
  int64_t weightBytes = k * n * 4;
  int64_t outputBytes = outputType.getDimSize(0) * outputType.getDimSize(1) * 4;
  int64_t total = inputBytes + weightBytes + outputBytes;
  auto i64 = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
  auto str = [&](StringRef v) { return builder.getStringAttr(v); };
  auto placement = [&](StringRef id, StringRef role, int64_t bytes) {
    return DictionaryAttr::get(builder.getContext(), {
        builder.getNamedAttr("buffer_id", str(id)),
        builder.getNamedAttr("role", str(role)),
        builder.getNamedAttr("memory_space", str("cpu_visible_host_memory")),
        builder.getNamedAttr("byte_count", i64(bytes)),
        builder.getNamedAttr("alignment", i64(64)),
    });
  };
  op->setAttr("memory_placement.status", str("selected"));
  op->setAttr("memory_placement.compute_unit", str("cpu"));
  op->setAttr("memory_placement.selected_memory_space",
              str("cpu_visible_host_memory"));
  op->setAttr("memory_placement.input_tile_bytes", i64(inputBytes));
  op->setAttr("memory_placement.weight_tile_bytes", i64(weightBytes));
  op->setAttr("memory_placement.output_tile_bytes", i64(outputBytes));
  op->setAttr("memory_placement.scratch_bytes", i64(0));
  op->setAttr("memory_placement.padding_bytes", i64(0));
  op->setAttr("memory_placement.single_buffer_bytes", i64(total));
  op->setAttr("memory_placement.additional_double_buffer_bytes", i64(0));
  op->setAttr("memory_placement.total_required_local_memory_bytes", i64(total));
  op->setAttr("memory_placement.buffer_placements", builder.getArrayAttr({
      placement("input_tile", "input", inputBytes),
      placement("weight_tile", "weight", weightBytes),
      placement("output_tile", "output", outputBytes),
      placement("scratch", "scratch", 0),
  }));
  op->setAttr("memory_placement.transfer_operations", builder.getArrayAttr({}));
  op->setAttr("memory_placement.compute_dependency_ids", builder.getArrayAttr({}));
  op->setAttr("memory_placement.truth_boundary",
              str("slice3d_cpu_visible_source_buffers_and_explicit_packed_weight_artifact_stage_not_runtime_allocation"));
}

Operation *createGenericOp(OpBuilder &builder, Location loc, StringRef name,
                           ValueRange operands, TypeRange results,
                           ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return builder.create(state);
}

struct QuantizationMaterializationPass
    : impl::QuantizationMaterializationBase<QuantizationMaterializationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<FusedMatMulBiasReluOp> worklist;
    funcOp.walk([&](FusedMatMulBiasReluOp op) {
      if (isSelectedPackedInt8(op))
        worklist.push_back(op);
    });

    for (FusedMatMulBiasReluOp op : worklist) {
      if (failed(validateMaterializationInputs(op))) {
        signalPassFailure();
        return;
      }
      OpBuilder builder(op);
      Location loc = op.getLoc();
      Operation *raw = op.getOperation();
      auto lhsType = cast<RankedTensorType>(op.getLhs().getType());
      auto rhsType = cast<RankedTensorType>(op.getRhs().getType());
      auto outType = cast<RankedTensorType>(op.getOutput().getType());
      auto i8 = builder.getIntegerType(8);
      auto i32 = builder.getIntegerType(32);
      auto qaType = RankedTensorType::get(lhsType.getShape(), i8);
      int64_t originalK = rhsType.getDimSize(0);
      int64_t originalN = rhsType.getDimSize(1);
      auto packedType = RankedTensorType::get({originalN, originalK}, i8);
      auto accType = RankedTensorType::get(outType.getShape(), i32);

      SmallVector<NamedAttribute> qAttrs;
      qAttrs.push_back(builder.getNamedAttr("scale",
                                            floatAttr(raw, "quant.activation_scale")));
      qAttrs.push_back(builder.getNamedAttr("zero_point",
                                            intAttr(raw, "quant.activation_zero_point")));
      qAttrs.push_back(builder.getNamedAttr("quantized_dtype",
                                            builder.getStringAttr("i8")));
      qAttrs.push_back(builder.getNamedAttr("quantization.mode",
                                            builder.getStringAttr("per_tensor")));
      qAttrs.push_back(builder.getNamedAttr("rounding_mode",
                                            builder.getStringAttr("round_nearest_even")));
      qAttrs.push_back(builder.getNamedAttr("clamp_min",
                                            builder.getI64IntegerAttr(-127)));
      qAttrs.push_back(builder.getNamedAttr("clamp_max",
                                            builder.getI64IntegerAttr(127)));
      qAttrs.push_back(builder.getNamedAttr("source_dtype",
                                            builder.getStringAttr("fp32")));
      qAttrs.push_back(builder.getNamedAttr("destination_dtype",
                                            builder.getStringAttr("int8")));
      qAttrs.push_back(builder.getNamedAttr("artifact_ref",
          builder.getStringAttr(strAttr(raw, "quant.calibration_artifact_ref"))));
      Operation *quant = createGenericOp(builder, loc, "hir.quantize",
                                         {op.getLhs()}, {qaType}, qAttrs);

      SmallVector<NamedAttribute> loadAttrs;
      loadAttrs.push_back(builder.getNamedAttr("artifact_ref",
          builder.getStringAttr(strAttr(raw, "quant.packed_weight_artifact_ref"))));
      loadAttrs.push_back(builder.getNamedAttr("artifact_id",
          builder.getStringAttr(strAttr(raw, "quant.packed_weight_artifact_id"))));
      loadAttrs.push_back(builder.getNamedAttr("artifact_sha256",
          builder.getStringAttr(strAttr(raw, "quant.packed_weight_sha256"))));
      loadAttrs.push_back(builder.getNamedAttr("source_weight_sha256",
          builder.getStringAttr(strAttr(raw, "quant.source_weight_sha256"))));
      loadAttrs.push_back(builder.getNamedAttr("packed_layout",
          builder.getStringAttr("packed_b_transposed_nxk")));
      loadAttrs.push_back(builder.getNamedAttr("packing_scheme",
          builder.getStringAttr("b_transposed_nxk_contiguous")));
      loadAttrs.push_back(builder.getNamedAttr("dtype",
          builder.getStringAttr("int8")));
      loadAttrs.push_back(builder.getNamedAttr("kernel_capability",
          builder.getStringAttr("quant_kernel.int8_static_symmetric.packed_b_transposed")));
      loadAttrs.push_back(builder.getNamedAttr("original_k",
          builder.getI64IntegerAttr(originalK)));
      loadAttrs.push_back(builder.getNamedAttr("original_n",
          builder.getI64IntegerAttr(originalN)));
      loadAttrs.push_back(builder.getNamedAttr("packed_n",
          builder.getI64IntegerAttr(originalN)));
      loadAttrs.push_back(builder.getNamedAttr("packed_k",
          builder.getI64IntegerAttr(originalK)));
      Operation *packed = createGenericOp(builder, loc,
                                          "hir.load_quantized_weight",
                                          ValueRange{}, {packedType}, loadAttrs);

      SmallVector<NamedAttribute> qmAttrs;
      qmAttrs.push_back(builder.getNamedAttr("quantized_dtype",
                                             builder.getStringAttr("i8")));
      qmAttrs.push_back(builder.getNamedAttr("lhs_scale",
                                             floatAttr(raw, "quant.activation_scale")));
      qmAttrs.push_back(builder.getNamedAttr("rhs_scale",
                                             floatAttr(raw, "quant.weight_scale")));
      qmAttrs.push_back(builder.getNamedAttr("lhs_zero_point",
                                             intAttr(raw, "quant.activation_zero_point")));
      qmAttrs.push_back(builder.getNamedAttr("rhs_zero_point",
                                             intAttr(raw, "quant.weight_zero_point")));
      qmAttrs.push_back(builder.getNamedAttr("packed_layout",
          builder.getStringAttr("packed_b_transposed_nxk")));
      qmAttrs.push_back(builder.getNamedAttr("accumulator_dtype",
          builder.getStringAttr("int32")));
      qmAttrs.push_back(builder.getNamedAttr("output_dtype",
          builder.getStringAttr("fp32")));
      qmAttrs.push_back(builder.getNamedAttr("selected_candidate_id",
          builder.getStringAttr(strAttr(raw, "quant.selected_candidate_id"))));
      qmAttrs.push_back(builder.getNamedAttr("kernel_id",
          builder.getStringAttr(strAttr(raw, "quant.kernel_id"))));
      Operation *qmatmul = createGenericOp(builder, loc, "hir.qmatmul",
                                           {quant->getResult(0),
                                            packed->getResult(0)},
                                           accType, qmAttrs);

      double deqScale =
          floatAttr(raw, "quant.activation_scale").getValueAsDouble() *
          floatAttr(raw, "quant.weight_scale").getValueAsDouble();
      SmallVector<NamedAttribute> deqAttrs;
      deqAttrs.push_back(builder.getNamedAttr("scale",
          builder.getF64FloatAttr(deqScale)));
      deqAttrs.push_back(builder.getNamedAttr("zero_point",
          builder.getI64IntegerAttr(0)));
      deqAttrs.push_back(builder.getNamedAttr("quantized_dtype",
          builder.getStringAttr("i8")));
      deqAttrs.push_back(builder.getNamedAttr("source_accumulator_dtype",
          builder.getStringAttr("int32")));
      deqAttrs.push_back(builder.getNamedAttr("output_dtype",
          builder.getStringAttr("fp32")));
      createGenericOp(builder, loc, "hir.dequantize", {qmatmul->getResult(0)},
                      {op.getOutput().getType()}, deqAttrs);

      SmallVector<NamedAttribute> fusedAttrs;
      addCommonQuantAttrs(raw, fusedAttrs, builder);
      fusedAttrs.push_back(builder.getNamedAttr("fusion.candidate",
          builder.getStringAttr("qmatmul_bias_relu")));
      fusedAttrs.push_back(builder.getNamedAttr("quantized_dtype",
          builder.getStringAttr("i8")));
      fusedAttrs.push_back(builder.getNamedAttr("quantization.mode",
          builder.getStringAttr("per_tensor")));
      fusedAttrs.push_back(builder.getNamedAttr("input_layout",
          builder.getStringAttr("row_major")));
      fusedAttrs.push_back(builder.getNamedAttr("weight_layout",
          builder.getStringAttr("packed_b_transposed_nxk")));
      fusedAttrs.push_back(builder.getNamedAttr("lhs_scale",
          floatAttr(raw, "quant.activation_scale")));
      fusedAttrs.push_back(builder.getNamedAttr("rhs_scale",
          floatAttr(raw, "quant.weight_scale")));
      fusedAttrs.push_back(builder.getNamedAttr("lhs_zero_point",
          intAttr(raw, "quant.activation_zero_point")));
      fusedAttrs.push_back(builder.getNamedAttr("rhs_zero_point",
          intAttr(raw, "quant.weight_zero_point")));
      fusedAttrs.push_back(builder.getNamedAttr("alignment",
          builder.getI64IntegerAttr(1)));
      fusedAttrs.push_back(builder.getNamedAttr("materialization.stage",
          builder.getStringAttr("slice3d_quantized_implementation_ir")));
      Operation *fused = createGenericOp(builder, loc,
          "hir.fused_qmatmul_bias_relu",
          {quant->getResult(0), packed->getResult(0), op.getBias()},
          {op.getOutput().getType()}, fusedAttrs);
      copyAttrsWithPrefixes(raw, fused,
                            {"quant.", "kernel_selection.",
                             "memory_placement.", "thread_schedule.",
                             "tile_plan."});
      op->replaceAllUsesWith(fused->getResults());
      op->erase();
    }
  }
};

struct QuantizedKernelLoweringPass
    : impl::QuantizedKernelLoweringBase<QuantizedKernelLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HIRDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<FusedQMatMulBiasReluOp> worklist;
    funcOp.walk([&](FusedQMatMulBiasReluOp op) { worklist.push_back(op); });

    for (FusedQMatMulBiasReluOp op : worklist) {
      Operation *raw = op.getOperation();
      if (strAttr(raw, "kernel_id") !=
          "portable_fused_matmul_bias_relu_int8_symmetric_packed_b") {
        raw->emitError("quantized kernel lowering requires selected packed INT8 kernel_id");
        signalPassFailure();
        return;
      }
      OpBuilder builder(op);
      SmallVector<NamedAttribute> attrs;
      addCommonQuantAttrs(raw, attrs, builder);
      attrs.push_back(builder.getNamedAttr("kernel_id",
          builder.getStringAttr("portable_fused_matmul_bias_relu_int8_symmetric_packed_b")));
      attrs.push_back(builder.getNamedAttr("lowered.stage",
          builder.getStringAttr("slice3d_portable_cpu_int8_kernel_contract")));
      attrs.push_back(builder.getNamedAttr("quant.execution_stages",
          buildExecutionStages(builder, raw)));
      Operation *lowered = createGenericOp(builder, op.getLoc(),
          "hir.portable_cpu_int8_fused_matmul_bias_relu",
          {op.getLhs(), op.getRhs(), op.getBias()},
          {op.getOutput().getType()}, attrs);
      copyAttrs(raw, lowered);
      lowered->setAttr("kernel_id",
          builder.getStringAttr("portable_fused_matmul_bias_relu_int8_symmetric_packed_b"));
      lowered->setAttr("lowered.stage",
          builder.getStringAttr("slice3d_portable_cpu_int8_kernel_contract"));
      lowered->setAttr("quant.execution_stages",
          buildExecutionStages(builder, raw));
      lowered->setAttr("kernel_selection.status",
          builder.getStringAttr("selected"));
      lowered->setAttr("kernel_selection.selected_id",
          builder.getStringAttr("portable_fused_matmul_bias_relu_int8_symmetric_packed_b"));
      lowered->setAttr("kernel_selection.source",
          builder.getStringAttr("slice3d_lowered_selected_complete_candidate"));
      lowered->setAttr("kernel_selection.contract_version",
          builder.getStringAttr("kernel_selection_contract_v1"));
      lowered->setAttr("kernel_selection.truth_boundary",
          builder.getStringAttr("lowered_from_compiler_selected_slice3c_int8_candidate_not_runtime_search"));
      lowered->removeAttr("kernel_selection.rejection_reasons");
      lowered->setAttr("thread_schedule.status", builder.getStringAttr("selected"));
      lowered->setAttr("thread_schedule.thread_count", builder.getI64IntegerAttr(1));
      lowered->setAttr("thread_schedule.partition_axis", builder.getStringAttr("none"));
      lowered->setAttr("thread_schedule.partition_strategy", builder.getStringAttr("serial"));
      lowered->setAttr("thread_schedule.source",
          builder.getStringAttr("slice3d_int8_kernel_contract"));
      lowered->setAttr("quant.strategy", builder.getStringAttr("int8_static_symmetric"));
      lowered->setAttr("quant.scheme", builder.getStringAttr("int8_static_symmetric"));
      lowered->setAttr("quant.activation_dtype", builder.getStringAttr("int8"));
      lowered->setAttr("quant.weight_dtype", builder.getStringAttr("int8"));
      lowered->setAttr("quant.accumulation_dtype", builder.getStringAttr("int32"));
      lowered->setAttr("quant.output_dtype", builder.getStringAttr("fp32"));
      lowered->setAttr("quant.granularity", builder.getStringAttr("per_tensor"));
      lowered->setAttr("quant.activation_granularity", builder.getStringAttr("per_tensor"));
      lowered->setAttr("quant.weight_granularity", builder.getStringAttr("per_tensor"));
      lowered->setAttr("quant.required_kernel_capability",
          builder.getStringAttr("quant_kernel.int8_static_symmetric.packed_b_transposed"));
      lowered->setAttr("quant.kernel_requires_packed_weight",
          builder.getBoolAttr(true));
      lowered->setAttr("quant.decision_reason",
          builder.getStringAttr("slice3d_lowered_from_selected_int8_materialized_ir"));
      lowered->setAttr("quant.truth_boundary",
          builder.getStringAttr("explicit_quantized_ir_lowered_to_selected_portable_cpu_int8_kernel_contract"));
      stampCpuVisibleMemoryPlacement(
          lowered, builder,
          cast<RankedTensorType>(op.getLhs().getType()),
          cast<RankedTensorType>(op.getRhs().getType()),
          cast<RankedTensorType>(op.getOutput().getType()));
      op->replaceAllUsesWith(lowered->getResults());
      op->erase();
    }
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

std::unique_ptr<Pass> createQuantizationMaterializationPass() {
  return std::make_unique<QuantizationMaterializationPass>();
}

std::unique_ptr<Pass> createQuantizedKernelLoweringPass() {
  return std::make_unique<QuantizedKernelLoweringPass>();
}

} // namespace mlir::hir
