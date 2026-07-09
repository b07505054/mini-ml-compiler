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
      biasType.getRank() != 2 || outputType.getRank() != 2) {
    return emitOpError("expects rank-2 lhs, rhs, bias, and result tensors");
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
    if (biasType.getDimSize(1) != *originalN ||
        (biasType.getDimSize(0) != 1 && biasType.getDimSize(0) != *originalM)) {
      return emitOpError("requires padded metadata original shape to match bias broadcast dimensions");
    }
  }

  if (failed(requireStringAttr(getOperation(), "fusion.candidate",
                               "matmul_bias_relu")) ||
      failed(requireStringAttr(getOperation(), "kernel.selection")) ||
      failed(requireStringAttr(getOperation(), "lowering.source",
                               "linalg.matmul_add_relu"))) {
    return failure();
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
  if (failed(requireRankedTensor(getOperation(), getInput().getType(), "input")) ||
      failed(requireRankedTensor(getOperation(), getOutput().getType(), "output")) ||
      failed(requireFloatAttr(getOperation(), "scale")) ||
      failed(requireIntegerAttr(getOperation(), "zero_point")) ||
      failed(requireStringAttr(getOperation(), "quantized_dtype", "i8")) ||
      failed(requireStringAttr(getOperation(), "quantization.mode"))) {
    return failure();
  }
  return success();
}

LogicalResult DequantizeOp::verify() {
  if (failed(requireRankedTensor(getOperation(), getInput().getType(), "input")) ||
      failed(requireRankedTensor(getOperation(), getOutput().getType(), "output")) ||
      failed(requireFloatAttr(getOperation(), "scale")) ||
      failed(requireIntegerAttr(getOperation(), "zero_point")) ||
      failed(requireStringAttr(getOperation(), "quantized_dtype", "i8"))) {
    return failure();
  }
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
  if (failed(requireStringAttr(getOperation(), "quantized_dtype", "i8")) ||
      failed(requireFloatAttr(getOperation(), "lhs_scale")) ||
      failed(requireFloatAttr(getOperation(), "rhs_scale")) ||
      failed(requireIntegerAttr(getOperation(), "lhs_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "rhs_zero_point"))) {
    return failure();
  }
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
      biasType.getRank() != 2 || outputType.getRank() != 2) {
    return emitOpError("expects rank-2 lhs, rhs, bias, and result tensors");
  }
  if (failed(requireStringAttr(getOperation(), "fusion.candidate",
                               "qmatmul_bias_relu")) ||
      failed(requireStringAttr(getOperation(), "quantized_dtype", "i8")) ||
      failed(requireStringAttr(getOperation(), "quantization.mode",
                               "per_channel")) ||
      failed(requireStringAttr(getOperation(), "input_layout", "NHWC")) ||
      failed(requireStringAttr(getOperation(), "weight_layout", "blocked_kc")) ||
      failed(requireFloatAttr(getOperation(), "lhs_scale")) ||
      failed(requireFloatAttr(getOperation(), "rhs_scale")) ||
      failed(requireIntegerAttr(getOperation(), "lhs_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "rhs_zero_point")) ||
      failed(requireIntegerAttr(getOperation(), "alignment"))) {
    return failure();
  }
  auto alignment = integerAttrValue(getOperation(), "alignment");
  if (!alignment || *alignment != 128) {
    return emitOpError("requires 128-byte activation alignment");
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
  return success();
}

#define GET_OP_CLASSES
#include "HIR/IR/HIROps.cpp.inc"
