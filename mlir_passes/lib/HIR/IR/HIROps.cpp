#include "HIR/IR/HIROps.h"

#include "mlir/IR/BuiltinTypes.h"

#include <optional>

using namespace mlir;
using namespace mlir::hir;

static LogicalResult requireStringAttr(Operation *op, StringRef name,
                                       StringRef expected = {}) {
  auto attr = op->getAttrOfType<StringAttr>(name);
  if (!attr) {
    return op->emitOpError("requires string attribute '") << name << "'";
  }
  if (!expected.empty() && attr.getValue() != expected) {
    return op->emitOpError("requires '") << name << "' = \"" << expected << "\"";
  }
  return success();
}

static LogicalResult requireFloatAttr(Operation *op, StringRef name) {
  if (!op->getAttrOfType<FloatAttr>(name)) {
    return op->emitOpError("requires float attribute '") << name << "'";
  }
  return success();
}

static LogicalResult requireIntegerAttr(Operation *op, StringRef name) {
  if (!op->getAttrOfType<IntegerAttr>(name)) {
    return op->emitOpError("requires integer attribute '") << name << "'";
  }
  return success();
}

static LogicalResult requireRankedTensor(Operation *op, Type type,
                                         StringRef role,
                                         RankedTensorType &ranked) {
  ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked)
    return op->emitOpError("expects ranked tensor ") << role;
  return success();
}

static bool isI8(Type type) {
  return type.isInteger(8);
}

static bool isI32(Type type) {
  return type.isInteger(32);
}

static bool isFloat(Type type) {
  return isa<FloatType>(type);
}

static LogicalResult requireFloatAttrNamed(Operation *op, StringRef name) {
  if (!op->getAttrOfType<FloatAttr>(name)) {
    return op->emitOpError("requires float attribute '") << name << "'";
  }
  return success();
}

static std::optional<int64_t> integerAttrValue(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
}

static LogicalResult verifySparseCoreTargetAttrs(Operation *op,
                                                RankedTensorType lhsType,
                                                RankedTensorType rhsType) {
  auto target = op->getAttrOfType<StringAttr>("target.model");
  if (!target) {
    return success();
  }
  if (target.getValue() != "sparsecore_like_v1") {
    return op->emitOpError("requires target.model = \"sparsecore_like_v1\"");
  }
  if (failed(requireStringAttr(op, "target.memory_hierarchy",
                               "global_sram_register")) ||
      failed(requireStringAttr(op, "target.sparse_layout")) ||
      failed(requireStringAttr(op, "target.collective")) ||
      failed(requireIntegerAttr(op, "target.tile_m")) ||
      failed(requireIntegerAttr(op, "target.tile_n")) ||
      failed(requireIntegerAttr(op, "target.tile_k")) ||
      failed(requireIntegerAttr(op, "target.vector_bytes")) ||
      failed(requireIntegerAttr(op, "target.alignment")) ||
      failed(requireIntegerAttr(op, "target.sram_kb"))) {
    return failure();
  }

  auto tileM = integerAttrValue(op, "target.tile_m");
  auto tileN = integerAttrValue(op, "target.tile_n");
  auto tileK = integerAttrValue(op, "target.tile_k");
  auto vectorBytes = integerAttrValue(op, "target.vector_bytes");
  auto alignment = integerAttrValue(op, "target.alignment");
  if (!tileM || *tileM != 16 || !tileN || *tileN != 16 ||
      !tileK || *tileK != 32) {
    return op->emitOpError("requires SparseCore-like tile shape 16x16x32");
  }
  if (!vectorBytes || *vectorBytes != 128 || !alignment || *alignment != 128) {
    return op->emitOpError("requires 128-byte target vector/alignment");
  }
  auto sparseLayout = op->getAttrOfType<StringAttr>("target.sparse_layout");
  if (!sparseLayout ||
      (sparseLayout.getValue() != "dense_or_2_4" &&
       sparseLayout.getValue() != "structured_2_4")) {
    return op->emitOpError("requires target.sparse_layout = \"dense_or_2_4\" or \"structured_2_4\"");
  }
  if (sparseLayout.getValue() == "structured_2_4") {
    if (failed(requireStringAttr(op, "target.sparse_axis", "rhs_k")) ||
        failed(requireIntegerAttr(op, "target.sparse_group_size")) ||
        failed(requireIntegerAttr(op, "target.sparse_max_nonzero"))) {
      return failure();
    }
    auto groupSize = integerAttrValue(op, "target.sparse_group_size");
    auto maxNonzero = integerAttrValue(op, "target.sparse_max_nonzero");
    if (!groupSize || *groupSize != 4 || !maxNonzero || *maxNonzero != 2) {
      return op->emitOpError("requires structured 2:4 sparse metadata");
    }
    if (rhsType.getDimSize(0) != ShapedType::kDynamic &&
        rhsType.getDimSize(0) % *groupSize != 0) {
      return op->emitOpError("requires rhs K dimension to be a multiple of structured sparse group size");
    }
  }

  auto padding = op->getAttrOfType<StringAttr>("target.padding");
  if (!padding || padding.getValue() == "none") {
    if (lhsType.getDimSize(0) != ShapedType::kDynamic &&
        lhsType.getDimSize(0) % *tileM != 0) {
      return op->emitOpError("requires lhs M dimension to be a multiple of target.tile_m");
    }
    if (lhsType.getDimSize(1) != ShapedType::kDynamic &&
        lhsType.getDimSize(1) % *tileK != 0) {
      return op->emitOpError("requires lhs K dimension to be a multiple of target.tile_k");
    }
    if (rhsType.getDimSize(1) != ShapedType::kDynamic &&
        rhsType.getDimSize(1) % *tileN != 0) {
      return op->emitOpError("requires rhs N dimension to be a multiple of target.tile_n");
    }
    return success();
  }

  if (padding.getValue() != "pad_to_tile_with_crop") {
    return op->emitOpError("requires target.padding = \"none\" or \"pad_to_tile_with_crop\"");
  }
  if (failed(requireStringAttr(op, "target.valid_region", "original_m_n")) ||
      failed(requireIntegerAttr(op, "target.original_m")) ||
      failed(requireIntegerAttr(op, "target.original_n")) ||
      failed(requireIntegerAttr(op, "target.original_k")) ||
      failed(requireIntegerAttr(op, "target.padded_m")) ||
      failed(requireIntegerAttr(op, "target.padded_n")) ||
      failed(requireIntegerAttr(op, "target.padded_k")) ||
      failed(requireIntegerAttr(op, "target.pad_m")) ||
      failed(requireIntegerAttr(op, "target.pad_n")) ||
      failed(requireIntegerAttr(op, "target.pad_k")) ||
      failed(requireFloatAttrNamed(op, "target.padding_compute_overhead_ratio")) ||
      failed(requireFloatAttrNamed(op, "target.padding_output_overhead_ratio"))) {
    return failure();
  }

  auto originalM = integerAttrValue(op, "target.original_m");
  auto originalN = integerAttrValue(op, "target.original_n");
  auto originalK = integerAttrValue(op, "target.original_k");
  auto paddedM = integerAttrValue(op, "target.padded_m");
  auto paddedN = integerAttrValue(op, "target.padded_n");
  auto paddedK = integerAttrValue(op, "target.padded_k");
  auto padM = integerAttrValue(op, "target.pad_m");
  auto padN = integerAttrValue(op, "target.pad_n");
  auto padK = integerAttrValue(op, "target.pad_k");
  if (!originalM || !originalN || !originalK || !paddedM || !paddedN ||
      !paddedK || !padM || !padN || !padK) {
    return failure();
  }
  if (lhsType.getDimSize(0) != *originalM ||
      lhsType.getDimSize(1) != *originalK ||
      rhsType.getDimSize(0) != *originalK ||
      rhsType.getDimSize(1) != *originalN) {
    return op->emitOpError("requires padded metadata original shape to match lhs/rhs dimensions");
  }
  if (*paddedM < *originalM || *paddedN < *originalN ||
      *paddedK < *originalK) {
    return op->emitOpError("requires padded dimensions to cover original dimensions");
  }
  if (*padM != *paddedM - *originalM || *padN != *paddedN - *originalN ||
      *padK != *paddedK - *originalK) {
    return op->emitOpError("requires pad dimensions to equal padded minus original dimensions");
  }
  if (*paddedM % *tileM != 0 || *paddedN % *tileN != 0 ||
      *paddedK % *tileK != 0) {
    return op->emitOpError("requires padded dimensions to be multiples of target tile shape");
  }
  return success();
}

static LogicalResult requireRankedTensor(Operation *op, Type type, StringRef label) {
  if (!isa<RankedTensorType>(type)) {
    return op->emitOpError("expects ranked tensor ") << label;
  }
  return success();
}

LogicalResult FusedMatMulBiasReluOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!lhsType || !rhsType || !biasType || !outputType) {
    return emitOpError("expects ranked tensor operands and result");
  }

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      (biasType.getRank() != 1 && biasType.getRank() != 2) ||
      outputType.getRank() != 2) {
    return emitOpError("expects rank-2 lhs/rhs/result and rank-1 or rank-2 bias tensors");
  }

  if (lhsType.getDimSize(1) != ShapedType::kDynamic &&
      rhsType.getDimSize(0) != ShapedType::kDynamic &&
      lhsType.getDimSize(1) != rhsType.getDimSize(0)) {
    return emitOpError("expects lhs K dimension to match rhs K dimension");
  }

  if (outputType.getDimSize(0) != ShapedType::kDynamic &&
      lhsType.getDimSize(0) != ShapedType::kDynamic &&
      outputType.getDimSize(0) != lhsType.getDimSize(0)) {
    return emitOpError("expects result M dimension to match lhs M dimension");
  }

  if (outputType.getDimSize(1) != ShapedType::kDynamic &&
      rhsType.getDimSize(1) != ShapedType::kDynamic &&
      outputType.getDimSize(1) != rhsType.getDimSize(1)) {
    return emitOpError("expects result N dimension to match rhs N dimension");
  }

  if (biasType.getRank() == 1) {
    if (biasType.getDimSize(0) != ShapedType::kDynamic &&
        outputType.getDimSize(1) != ShapedType::kDynamic &&
        biasType.getDimSize(0) != outputType.getDimSize(1)) {
      return emitOpError("expects rank-1 bias N dimension to match result N dimension");
    }
  } else {
    if (biasType.getDimSize(1) != ShapedType::kDynamic &&
        outputType.getDimSize(1) != ShapedType::kDynamic &&
        biasType.getDimSize(1) != outputType.getDimSize(1)) {
      return emitOpError("expects bias N dimension to match result N dimension");
    }
    if (biasType.getDimSize(0) != ShapedType::kDynamic &&
        outputType.getDimSize(0) != ShapedType::kDynamic &&
        biasType.getDimSize(0) != 1 &&
        biasType.getDimSize(0) != outputType.getDimSize(0)) {
      return emitOpError("expects bias M dimension to be 1 or match result M dimension");
    }
  }

  auto padding = getOperation()->getAttrOfType<StringAttr>("target.padding");
  if (padding && padding.getValue() == "pad_to_tile_with_crop") {
    auto originalM = integerAttrValue(getOperation(), "target.original_m");
    auto originalN = integerAttrValue(getOperation(), "target.original_n");
    if (!originalM || !originalN) {
      return emitOpError("requires padded metadata original M/N dimensions");
    }
    if (outputType.getDimSize(0) != *originalM ||
        outputType.getDimSize(1) != *originalN) {
      return emitOpError("requires padded metadata original shape to match result dimensions");
    }
    bool biasMatches =
        biasType.getRank() == 1
            ? biasType.getDimSize(0) == *originalN
            : (biasType.getDimSize(1) == *originalN &&
               (biasType.getDimSize(0) == 1 ||
                biasType.getDimSize(0) == *originalM));
    if (!biasMatches) {
      return emitOpError("requires padded metadata original shape to match bias broadcast dimensions");
    }
  }

  bool hasRuntimeMetadata =
      getOperation()->getAttr("fusion.candidate") ||
      getOperation()->getAttr("kernel.selection") ||
      getOperation()->getAttr("lowering.source");
  if (hasRuntimeMetadata) {
    if (failed(requireStringAttr(getOperation(), "fusion.candidate",
                                 "matmul_bias_relu")) ||
        failed(requireStringAttr(getOperation(), "kernel.selection")) ||
        failed(requireStringAttr(getOperation(), "lowering.source",
                                 "linalg.matmul_add_relu"))) {
      return failure();
    }
  }
  if (failed(verifySparseCoreTargetAttrs(getOperation(), lhsType, rhsType))) {
    return failure();
  }
  return success();
}

LogicalResult FusedRMSNormOp::verify() {
  if (getInput().getType() != getOutput().getType()) {
    return emitOpError("expects input and result types to match");
  }

  if (!isa<RankedTensorType>(getInput().getType())) {
    return emitOpError("expects ranked tensor input and result");
  }

  if (failed(requireStringAttr(getOperation(), "fusion.candidate", "rmsnorm")) ||
      failed(requireStringAttr(getOperation(), "kernel.selection")) ||
      failed(requireStringAttr(getOperation(), "lowering.source", "llm.rmsnorm"))) {
    return failure();
  }
  return success();
}

LogicalResult CastOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !outputType) {
    return emitOpError("expects ranked tensor input and result");
  }
  if (inputType.getShape() != outputType.getShape()) {
    return emitOpError("expects input and result shapes to match");
  }
  if (!isa<FloatType>(inputType.getElementType()) ||
      !isa<FloatType>(outputType.getElementType())) {
    return emitOpError(
        "expects float element types; int<->float boundaries use "
        "hir.quantize/hir.dequantize");
  }
  if (inputType.getElementType() == outputType.getElementType()) {
    return emitOpError("expects input and result element types to differ");
  }
  if (failed(requireStringAttr(getOperation(), "materialized.from_decision")) ||
      failed(requireStringAttr(getOperation(), "materialized.truth_boundary"))) {
    return failure();
  }
  return success();
}

LogicalResult QuantizeOp::verify() {
  RankedTensorType inputType, outputType;
  if (failed(requireRankedTensor(getOperation(), getInput().getType(), "input", inputType)) ||
      failed(requireRankedTensor(getOperation(), getOutput().getType(), "output", outputType)) ||
      failed(requireFloatAttr(getOperation(), "scale")) ||
      failed(requireIntegerAttr(getOperation(), "zero_point")) ||
      failed(requireStringAttr(getOperation(), "quantized_dtype", "i8")) ||
      failed(requireStringAttr(getOperation(), "quantization.mode"))) {
    return failure();
  }
  if (inputType.getShape() != outputType.getShape())
    return emitOpError("expects input and output shapes to match");
  if (!isFloat(inputType.getElementType()))
    return emitOpError("expects floating-point input element type");
  if (!isI8(outputType.getElementType()))
    return emitOpError("expects i8 output element type");
  if (auto rounding = getOperation()->getAttrOfType<StringAttr>("rounding_mode"))
    if (rounding.getValue() != "round_nearest_even" &&
        rounding.getValue() != "round_nearest")
      return emitOpError("has unsupported rounding_mode");
  if (failed(requireIntegerAttr(getOperation(), "clamp_min")) ||
      failed(requireIntegerAttr(getOperation(), "clamp_max")))
    return failure();
  auto clampMin = integerAttrValue(getOperation(), "clamp_min");
  auto clampMax = integerAttrValue(getOperation(), "clamp_max");
  if (!clampMin || !clampMax || *clampMin != -127 || *clampMax != 127)
    return emitOpError("expects symmetric int8 clamp range [-127, 127]");
  return success();
}

LogicalResult DequantizeOp::verify() {
  RankedTensorType inputType, outputType;
  if (failed(requireRankedTensor(getOperation(), getInput().getType(), "input", inputType)) ||
      failed(requireRankedTensor(getOperation(), getOutput().getType(), "output", outputType)) ||
      failed(requireFloatAttr(getOperation(), "scale")) ||
      failed(requireIntegerAttr(getOperation(), "zero_point")) ||
      failed(requireStringAttr(getOperation(), "quantized_dtype", "i8"))) {
    return failure();
  }
  if (inputType.getShape() != outputType.getShape())
    return emitOpError("expects input and output shapes to match");
  if (!isI8(inputType.getElementType()) && !isI32(inputType.getElementType()))
    return emitOpError("expects integer input element type");
  if (!isFloat(outputType.getElementType()))
    return emitOpError("expects floating-point output element type");
  return success();
}

LogicalResult QMatMulOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!lhsType || !rhsType || !outputType) {
    return emitOpError("expects ranked tensor lhs, rhs, and result");
  }
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outputType.getRank() != 2) {
    return emitOpError("expects rank-2 lhs, rhs, and result tensors");
  }
  if (!isI8(lhsType.getElementType()))
    return emitOpError("expects i8 lhs element type");
  if (!isI8(rhsType.getElementType()))
    return emitOpError("expects i8 rhs element type");
  if (!isI32(outputType.getElementType()))
    return emitOpError("expects i32 accumulator result element type");
  if (lhsType.getDimSize(1) != ShapedType::kDynamic &&
      rhsType.getDimSize(1) != ShapedType::kDynamic &&
      lhsType.getDimSize(1) != rhsType.getDimSize(1)) {
    return emitOpError("expects packed rhs shape [N,K] with K matching lhs K");
  }
  if (lhsType.getDimSize(0) != ShapedType::kDynamic &&
      outputType.getDimSize(0) != ShapedType::kDynamic &&
      lhsType.getDimSize(0) != outputType.getDimSize(0)) {
    return emitOpError("expects output M to match lhs M");
  }
  if (rhsType.getDimSize(0) != ShapedType::kDynamic &&
      outputType.getDimSize(1) != ShapedType::kDynamic &&
      rhsType.getDimSize(0) != outputType.getDimSize(1)) {
    return emitOpError("expects output N to match packed rhs N");
  }
  if (failed(requireStringAttr(getOperation(), "quantized_dtype", "i8")) ||
      failed(requireFloatAttr(getOperation(), "lhs_scale")) ||
      failed(requireFloatAttr(getOperation(), "rhs_scale")) ||
      failed(requireIntegerAttr(getOperation(), "lhs_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "rhs_zero_point"))) {
    return failure();
  }
  auto lhsZp = integerAttrValue(getOperation(), "lhs_zero_point");
  auto rhsZp = integerAttrValue(getOperation(), "rhs_zero_point");
  if (!lhsZp || !rhsZp || *lhsZp != 0 || *rhsZp != 0)
    return emitOpError("expects symmetric zero points equal to 0");
  if (failed(requireStringAttr(getOperation(), "packed_layout",
                               "packed_b_transposed_nxk")) ||
      failed(requireStringAttr(getOperation(), "accumulator_dtype", "int32")) ||
      failed(requireStringAttr(getOperation(), "output_dtype", "fp32")))
    return failure();
  return success();
}

LogicalResult LoadQuantizedWeightOp::verify() {
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType)
    return emitOpError("expects ranked tensor output");
  if (outputType.getRank() != 2)
    return emitOpError("expects rank-2 packed weight output");
  if (!isI8(outputType.getElementType()))
    return emitOpError("expects i8 packed weight element type");
  if (failed(requireStringAttr(getOperation(), "artifact_ref")) ||
      failed(requireStringAttr(getOperation(), "artifact_id")) ||
      failed(requireStringAttr(getOperation(), "artifact_sha256")) ||
      failed(requireStringAttr(getOperation(), "source_weight_sha256")) ||
      failed(requireStringAttr(getOperation(), "packed_layout",
                               "packed_b_transposed_nxk")) ||
      failed(requireStringAttr(getOperation(), "packing_scheme",
                               "b_transposed_nxk_contiguous")) ||
      failed(requireStringAttr(getOperation(), "dtype", "int8")) ||
      failed(requireStringAttr(getOperation(), "kernel_capability",
                               "quant_kernel.int8_static_symmetric.packed_b_transposed"))) {
    return failure();
  }
  if (failed(requireIntegerAttr(getOperation(), "original_k")) ||
      failed(requireIntegerAttr(getOperation(), "original_n")) ||
      failed(requireIntegerAttr(getOperation(), "packed_n")) ||
      failed(requireIntegerAttr(getOperation(), "packed_k")))
    return failure();
  auto originalK = integerAttrValue(getOperation(), "original_k");
  auto originalN = integerAttrValue(getOperation(), "original_n");
  auto packedN = integerAttrValue(getOperation(), "packed_n");
  auto packedK = integerAttrValue(getOperation(), "packed_k");
  if (!originalK || !originalN || !packedN || !packedK ||
      *originalK != *packedK || *originalN != *packedN)
    return emitOpError("expects packed shape [N,K] to match original [K,N]");
  if (outputType.getDimSize(0) != ShapedType::kDynamic &&
      outputType.getDimSize(0) != *packedN)
    return emitOpError("packed output N dimension does not match packed_n");
  if (outputType.getDimSize(1) != ShapedType::kDynamic &&
      outputType.getDimSize(1) != *packedK)
    return emitOpError("packed output K dimension does not match packed_k");
  return success();
}

LogicalResult FusedQMatMulBiasReluOp::verify() {
  auto lhsType = dyn_cast<RankedTensorType>(getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(getRhs().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!lhsType || !rhsType || !biasType || !outputType) {
    return emitOpError("expects ranked tensor operands and result");
  }
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      (biasType.getRank() != 1 && biasType.getRank() != 2) ||
      outputType.getRank() != 2) {
    return emitOpError("expects rank-2 lhs/rhs/result and rank-1 or rank-2 bias tensors");
  }
  if (!isI8(lhsType.getElementType()))
    return emitOpError("expects i8 lhs element type");
  if (!isI8(rhsType.getElementType()))
    return emitOpError("expects i8 rhs element type");
  if (!isFloat(biasType.getElementType()) || !isFloat(outputType.getElementType()))
    return emitOpError("expects floating-point bias and output element types");
  if (failed(requireStringAttr(getOperation(), "fusion.candidate",
                               "qmatmul_bias_relu")) ||
      failed(requireStringAttr(getOperation(), "quantized_dtype", "i8")) ||
      failed(requireFloatAttr(getOperation(), "lhs_scale")) ||
      failed(requireFloatAttr(getOperation(), "rhs_scale")) ||
      failed(requireIntegerAttr(getOperation(), "lhs_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "rhs_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "alignment"))) {
    return failure();
  }
  auto quantMode = getOperation()->getAttrOfType<StringAttr>("quantization.mode");
  if (!quantMode ||
      (quantMode.getValue() != "per_channel" &&
       quantMode.getValue() != "per_tensor")) {
    return emitOpError(
        "requires quantization.mode = \"per_channel\" or \"per_tensor\"");
  }
  auto inputLayout = getOperation()->getAttrOfType<StringAttr>("input_layout");
  if (!inputLayout ||
      (inputLayout.getValue() != "NHWC" &&
       inputLayout.getValue() != "row_major")) {
    return emitOpError("requires input_layout = \"NHWC\" or \"row_major\"");
  }
  auto weightLayout = getOperation()->getAttrOfType<StringAttr>("weight_layout");
  if (!weightLayout ||
      (weightLayout.getValue() != "blocked_kc" &&
       weightLayout.getValue() != "packed_b_transposed_nxk")) {
    return emitOpError(
        "requires weight_layout = \"blocked_kc\" or "
        "\"packed_b_transposed_nxk\"");
  }
  auto lhsZp = integerAttrValue(getOperation(), "lhs_zero_point");
  auto rhsZp = integerAttrValue(getOperation(), "rhs_zero_point");
  if (!lhsZp || !rhsZp || *lhsZp != 0 || *rhsZp != 0)
    return emitOpError("expects symmetric zero points equal to 0");
  auto alignment = integerAttrValue(getOperation(), "alignment");
  if (!alignment || *alignment <= 0) {
    return emitOpError("requires positive activation alignment");
  }
  if (weightLayout.getValue() == "blocked_kc") {
    if (*alignment != 128) {
      return emitOpError("blocked_kc path requires 128-byte activation alignment");
    }
    if (lhsType.getDimSize(1) != ShapedType::kDynamic &&
        lhsType.getDimSize(1) % 32 != 0) {
      return emitOpError("requires lhs K dimension to be a multiple of 32");
    }
    if (rhsType.getDimSize(0) != ShapedType::kDynamic &&
        rhsType.getDimSize(0) % 32 != 0) {
      return emitOpError("requires weight K dimension to be a multiple of 32");
    }
    if (rhsType.getDimSize(1) != ShapedType::kDynamic &&
        rhsType.getDimSize(1) % 32 != 0) {
      return emitOpError("requires INT8 output channel dimension to be a multiple of 32");
    }
  } else {
    if (failed(requireStringAttr(getOperation(), "packed_layout",
                                 "packed_b_transposed_nxk")) ||
        failed(requireStringAttr(getOperation(), "packing_scheme",
                                 "b_transposed_nxk_contiguous")) ||
        failed(requireStringAttr(getOperation(), "accumulator_dtype", "int32")) ||
        failed(requireStringAttr(getOperation(), "output_dtype", "fp32")))
      return failure();
    if (lhsType.getDimSize(1) != ShapedType::kDynamic &&
        rhsType.getDimSize(1) != ShapedType::kDynamic &&
        lhsType.getDimSize(1) != rhsType.getDimSize(1)) {
      return emitOpError("expects packed rhs shape [N,K] with K matching lhs K");
    }
    if (lhsType.getDimSize(0) != ShapedType::kDynamic &&
        outputType.getDimSize(0) != ShapedType::kDynamic &&
        lhsType.getDimSize(0) != outputType.getDimSize(0)) {
      return emitOpError("expects output M to match lhs M");
    }
    if (rhsType.getDimSize(0) != ShapedType::kDynamic &&
        outputType.getDimSize(1) != ShapedType::kDynamic &&
        rhsType.getDimSize(0) != outputType.getDimSize(1)) {
      return emitOpError("expects output N to match packed rhs N");
    }
    if (biasType.getRank() == 1 &&
        rhsType.getDimSize(0) != ShapedType::kDynamic &&
        biasType.getDimSize(0) != ShapedType::kDynamic &&
        rhsType.getDimSize(0) != biasType.getDimSize(0)) {
      return emitOpError("expects rank-1 bias N to match packed rhs N");
    }
  }
  return success();
}

LogicalResult PortableCPUINT8FusedMatMulBiasReluOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto packedType = dyn_cast<RankedTensorType>(getPackedWeight().getType());
  auto biasType = dyn_cast<RankedTensorType>(getBias().getType());
  auto outputType = dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !packedType || !biasType || !outputType)
    return emitOpError("expects ranked tensor operands and result");
  if (inputType.getRank() != 2 || packedType.getRank() != 2 ||
      (biasType.getRank() != 1 && biasType.getRank() != 2) ||
      outputType.getRank() != 2)
    return emitOpError("expects quantized input [M,K], packed weight [N,K], bias [N] or [1,N], output [M,N]");
  if (!isI8(inputType.getElementType()))
    return emitOpError("expects i8 quantized activation input tensor");
  if (!isFloat(biasType.getElementType()) || !isFloat(outputType.getElementType()))
    return emitOpError("expects FP32 bias and output tensors");
  if (!isI8(packedType.getElementType()))
    return emitOpError("expects i8 packed weight tensor");
  int64_t m = inputType.getDimSize(0);
  int64_t k = inputType.getDimSize(1);
  int64_t n = packedType.getDimSize(0);
  int64_t packedK = packedType.getDimSize(1);
  if (k != ShapedType::kDynamic && packedK != ShapedType::kDynamic && k != packedK)
    return emitOpError("expects packed weight K to match input K");
  if (m != ShapedType::kDynamic && outputType.getDimSize(0) != ShapedType::kDynamic &&
      m != outputType.getDimSize(0))
    return emitOpError("expects output M to match input M");
  if (n != ShapedType::kDynamic && outputType.getDimSize(1) != ShapedType::kDynamic &&
      n != outputType.getDimSize(1))
    return emitOpError("expects output N to match packed weight N");
  if (biasType.getRank() == 1) {
    if (n != ShapedType::kDynamic && biasType.getDimSize(0) != ShapedType::kDynamic &&
        n != biasType.getDimSize(0))
      return emitOpError("expects rank-1 bias N to match packed weight N");
  } else {
    if (n != ShapedType::kDynamic && biasType.getDimSize(1) != ShapedType::kDynamic &&
        n != biasType.getDimSize(1))
      return emitOpError("expects rank-2 bias N to match packed weight N");
    if (biasType.getDimSize(0) != ShapedType::kDynamic &&
        biasType.getDimSize(0) != 1 &&
        inputType.getDimSize(0) != ShapedType::kDynamic &&
        biasType.getDimSize(0) != inputType.getDimSize(0))
      return emitOpError("expects rank-2 bias M to be 1 or match input M");
  }
  if (failed(requireStringAttr(getOperation(), "kernel_id",
                               "portable_fused_matmul_bias_relu_int8_symmetric_packed_b")) ||
      failed(requireStringAttr(getOperation(), "scheme",
                               "int8_static_symmetric")) ||
      failed(requireStringAttr(getOperation(), "packed_layout",
                               "packed_b_transposed_nxk")) ||
      failed(requireStringAttr(getOperation(), "packing_scheme",
                               "b_transposed_nxk_contiguous")) ||
      failed(requireStringAttr(getOperation(), "accumulator_dtype", "int32")) ||
      failed(requireStringAttr(getOperation(), "output_dtype", "fp32")) ||
      failed(requireStringAttr(getOperation(), "codegen_target_id",
                               "cortex_a76_dotprod")) ||
      failed(requireStringAttr(getOperation(), "binary_sha256")) ||
      failed(requireStringAttr(getOperation(), "selected_complete_candidate_id")) ||
      failed(requireStringAttr(getOperation(), "packed_weight_artifact_ref")) ||
      failed(requireStringAttr(getOperation(), "packed_weight_artifact_id")) ||
      failed(requireStringAttr(getOperation(), "packed_weight_sha256")) ||
      failed(requireStringAttr(getOperation(), "calibration_artifact_ref")) ||
      failed(requireStringAttr(getOperation(), "calibration_artifact_id")) ||
      failed(requireStringAttr(getOperation(), "calibration_artifact_sha256")) ||
      failed(requireFloatAttr(getOperation(), "activation_scale")) ||
      failed(requireFloatAttr(getOperation(), "weight_scale")) ||
      failed(requireIntegerAttr(getOperation(), "activation_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "weight_zero_point")))
    return failure();
  auto azp = integerAttrValue(getOperation(), "activation_zero_point");
  auto wzp = integerAttrValue(getOperation(), "weight_zero_point");
  if (!azp || !wzp || *azp != 0 || *wzp != 0)
    return emitOpError("expects symmetric zero points equal to 0");
  return success();
}

static LogicalResult verifyAttentionContract(Operation *op, Value q, Value k,
                                             Value v, Value out,
                                             bool lowered) {
  auto qt = dyn_cast<RankedTensorType>(q.getType());
  auto kt = dyn_cast<RankedTensorType>(k.getType());
  auto vt = dyn_cast<RankedTensorType>(v.getType());
  auto ot = dyn_cast<RankedTensorType>(out.getType());
  if (!qt || !kt || !vt || !ot || qt.getRank() != 4 || kt.getRank() != 4 ||
      vt.getRank() != 4 || ot.getRank() != 4)
    return op->emitOpError("expects rank-4 contiguous [B,H,S,D] tensors");
  for (Type t : {qt.getElementType(), kt.getElementType(), vt.getElementType(),
                 ot.getElementType()})
    if (!t.isF32()) return op->emitOpError("supports FP32 tensors only");
  for (StringRef name : {"phase", "dtype", "input_layout", "output_layout",
                         "truth_boundary"})
    if (failed(requireStringAttr(op, name))) return failure();
  for (StringRef name : {"batch", "query_length", "context_length",
                         "num_query_heads", "num_kv_heads", "head_dim",
                         "workspace_bytes", "alignment_bytes"})
    if (failed(requireIntegerAttr(op, name))) return failure();
  auto phase = op->getAttrOfType<StringAttr>("phase").getValue();
  auto ql = integerAttrValue(op, "query_length");
  auto cl = integerAttrValue(op, "context_length");
  auto qh = integerAttrValue(op, "num_query_heads");
  auto kh = integerAttrValue(op, "num_kv_heads");
  auto hd = integerAttrValue(op, "head_dim");
  auto alignment = integerAttrValue(op, "alignment_bytes");
  auto causal = op->getAttrOfType<BoolAttr>("causal");
  if ((phase != "prefill" && phase != "decode") || !causal || !causal.getValue())
    return op->emitOpError("requires phase prefill/decode and causal=true");
  if (!ql || !cl || !qh || !kh || !hd || !alignment || *ql <= 0 || *cl <= 0 ||
      *qh <= 0 || *kh <= 0 || *hd <= 0 || *alignment != alignof(float) ||
      *qh != *kh)
    return op->emitOpError("requires positive static dimensions and equal Q/KV heads");
  if ((phase == "prefill" && (*ql <= 1 || *ql != *cl)) ||
      (phase == "decode" && *ql != 1))
    return op->emitOpError("query/context dimensions violate phase contract");
  if (op->getAttrOfType<StringAttr>("dtype").getValue() != "fp32" ||
      op->getAttrOfType<StringAttr>("input_layout").getValue() != "bhsd_contiguous" ||
      op->getAttrOfType<StringAttr>("output_layout").getValue() != "bhsd_contiguous")
    return op->emitOpError("requires fp32 bhsd_contiguous layout");
  bool runtimeOwnedDecode = !lowered && phase == "decode" &&
      op->getAttrOfType<BoolAttr>("kv_runtime_owned") &&
      op->getAttrOfType<BoolAttr>("kv_runtime_owned").getValue();
  if (qt.getDimSize(0) != *integerAttrValue(op, "batch") ||
      qt.getDimSize(1) != *qh || qt.getDimSize(2) != *ql ||
      qt.getDimSize(3) != *hd ||
      kt.getDimSize(2) != (runtimeOwnedDecode ? 1 : *cl) ||
      vt.getShape() != kt.getShape() || ot.getShape() != qt.getShape())
    return op->emitOpError("tensor types do not match declared attention dimensions");
  if (lowered) {
    for (StringRef name : {"candidate_id", "kernel_id", "entry_point",
                           "artifact_ref", "artifact_sha256", "artifact_version",
                           "fallback_identity", "runtime_execution_unit"})
      if (failed(requireStringAttr(op, name))) return failure();
    auto noRedecision = op->getAttrOfType<BoolAttr>("runtime_no_redecision");
    if (!noRedecision || !noRedecision.getValue())
      return op->emitOpError("requires runtime_no_redecision=true");
  }
  return success();
}

LogicalResult AttentionOp::verify() {
  return verifyAttentionContract(getOperation(), getQuery(), getKey(), getValue(),
                                 getOutput(), false);
}

LogicalResult CPUAttentionOp::verify() {
  return verifyAttentionContract(getOperation(), getQuery(), getKey(), getValue(),
                                 getOutput(), true);
}

static LogicalResult verifyKVAttrs(Operation *op) {
  for (StringRef name : {"kv_candidate_id", "kv_dtype", "kv_layout",
                         "kv_artifact_version", "kv_cache_id"})
    if (failed(requireStringAttr(op, name))) return failure();
  for (StringRef name : {"batch", "num_kv_heads", "head_dim",
                         "capacity_tokens", "alignment_bytes"}) {
    if (failed(requireIntegerAttr(op, name))) return failure();
    auto v = integerAttrValue(op, name);
    if (!v || *v <= 0) return op->emitOpError("requires positive static KV dimensions");
  }
  if (op->getAttrOfType<StringAttr>("kv_dtype").getValue() != "fp32" ||
      op->getAttrOfType<StringAttr>("kv_layout").getValue() != "bhcd_contiguous")
    return op->emitOpError("supports only FP32 bhcd_contiguous KV storage");
  return success();
}

LogicalResult KVCacheCreateOp::verify() { return verifyKVAttrs(getOperation()); }
LogicalResult KVCachePrefillWriteOp::verify() { return verifyKVAttrs(getOperation()); }
LogicalResult KVCacheAppendOp::verify() {
  if (failed(verifyKVAttrs(getOperation()))) return failure();
  auto kt = dyn_cast<RankedTensorType>(getKey().getType());
  auto vt = dyn_cast<RankedTensorType>(getValue().getType());
  if (!kt || kt.getRank() != 4 || kt != vt || kt.getDimSize(2) != 1)
    return emitOpError("requires one-token equal K/V tensors");
  return success();
}
LogicalResult KVCacheViewOp::verify() { return verifyKVAttrs(getOperation()); }
LogicalResult KVCacheResetOp::verify() { return verifyKVAttrs(getOperation()); }
static LogicalResult verifyPagedKV(Operation *op) {
  for(StringRef n:{"kv_candidate_id","kv_layout_kind","kv_dtype","block_table_element_type"})
    if(failed(requireStringAttr(op,n)))return failure();
  for(StringRef n:{"page_tokens","num_physical_pages","maximum_logical_tokens","block_table_length","num_kv_heads","head_dim"}){
    if(failed(requireIntegerAttr(op,n)))return failure();auto v=integerAttrValue(op,n);if(!v||*v<=0)return op->emitOpError("requires positive static paged KV dimensions");}
  if(op->getAttrOfType<StringAttr>("kv_layout_kind").getValue()!="paged_phd_contiguous"||op->getAttrOfType<StringAttr>("kv_dtype").getValue()!="fp32"||op->getAttrOfType<StringAttr>("block_table_element_type").getValue()!="int32")return op->emitOpError("unsupported paged KV layout/dtype/table type");
  return success();
}
LogicalResult PagedKVPoolCreateOp::verify(){return verifyPagedKV(getOperation());}
LogicalResult PagedKVBindBlockOp::verify(){return verifyPagedKV(getOperation());}
LogicalResult PagedKVPrefillWriteOp::verify(){return verifyPagedKV(getOperation());}
LogicalResult PagedKVAppendOp::verify(){return verifyPagedKV(getOperation());}
LogicalResult PagedKVViewOp::verify(){return verifyPagedKV(getOperation());}
LogicalResult PagedKVResetOp::verify(){return verifyPagedKV(getOperation());}

#define GET_OP_CLASSES
#include "HIR/IR/HIROps.cpp.inc"
