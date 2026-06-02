#ifndef HIR_DIALECT
#define HIR_DIALECT

include "mlir/IR/OpBase.td"

def HIR_Dialect : Dialect {
  let name = "hir";
  let summary = "High-level runtime-facing fused operation dialect";
  let description = [{
    The HIR dialect carries compiler-selected fused operations after MLIR
    canonicalization and fusion detection, before the runtime bridge exports
    executable dispatch plans.
  }];
  let cppNamespace = "::mlir::hir";
}

class HIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<HIR_Dialect, mnemonic, traits>;

#endif // HIR_DIALECT
