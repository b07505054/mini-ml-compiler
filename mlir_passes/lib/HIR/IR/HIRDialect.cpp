#include "HIR/IR/HIRDialect.h"

#include "HIR/IR/HIROps.h"

using namespace mlir;
using namespace mlir::hir;

#include "HIR/IR/HIROpsDialect.cpp.inc"

void HIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "HIR/IR/HIROps.cpp.inc"
      >();
}
