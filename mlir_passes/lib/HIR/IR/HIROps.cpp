#include "HIR/IR/HIROps.h"

#include "mlir/IR/BuiltinTypes.h"

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

#define GET_OP_CLASSES
#include "HIR/IR/HIROps.cpp.inc"
