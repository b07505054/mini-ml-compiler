#include "CV/IR/CVDialect.h"

#include "CV/IR/CVOps.h"

using namespace mlir;
using namespace mlir::cv;

#include "CV/IR/CVOpsDialect.cpp.inc"

void CVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CV/IR/CVOps.cpp.inc"
      >();
}
