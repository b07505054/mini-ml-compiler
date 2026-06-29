#include "CV/IR/CVOps.h"

using namespace mlir;
using namespace mlir::cv;

LogicalResult ConcatOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one input");
  return success();
}

LogicalResult DetectHeadOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one input");
  return success();
}

#define GET_OP_CLASSES
#include "CV/IR/CVOps.cpp.inc"
