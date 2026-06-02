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

static std::optional<int64_t> integerAttrValue(Operation *op, StringRef name) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
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

  if (failed(requireStringAttr(getOperation(), "fusion.candidate",
                               "matmul_bias_relu")) ||
      failed(requireStringAttr(getOperation(), "kernel.selection")) ||
      failed(requireStringAttr(getOperation(), "lowering.source",
                               "linalg.matmul_add_relu"))) {
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
