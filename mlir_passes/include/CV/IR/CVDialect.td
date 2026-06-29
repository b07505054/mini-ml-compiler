#ifndef CV_DIALECT
#define CV_DIALECT

include "mlir/IR/OpBase.td"

def CV_Dialect : Dialect {
  let name = "cv";
  let summary = "Lightweight compiler-internal computer vision dialect";
  let description = [{
    The CV dialect models the small pseudo-CV operator surface currently used by
    the compiler's CV planning pipeline. It is not an ONNX, StableHLO, or
    backend dialect.
  }];
  let cppNamespace = "::mlir::cv";
}

class CV_Op<string mnemonic, list<Trait> traits = []>
    : Op<CV_Dialect, mnemonic, traits>;

#endif // CV_DIALECT
