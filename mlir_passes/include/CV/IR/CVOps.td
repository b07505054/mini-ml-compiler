#ifndef CV_OPS
#define CV_OPS

include "CV/IR/CVDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class CV_UnaryTensorOp<string mnemonic>
    : CV_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def CV_Conv2dOp : CV_UnaryTensorOp<"conv2d"> {
  let summary = "Compiler-internal CV convolution marker";
}

def CV_BatchNormOp : CV_UnaryTensorOp<"batch_norm"> {
  let summary = "Compiler-internal CV batch normalization marker";
}

def CV_SiLUOp : CV_UnaryTensorOp<"silu"> {
  let summary = "Compiler-internal CV SiLU activation marker";
}

def CV_UpsampleOp : CV_UnaryTensorOp<"upsample"> {
  let summary = "Compiler-internal CV upsample marker";
}

def CV_DetectHeadOp : CV_Op<"detect_head", [Pure]> {
  let summary = "Compiler-internal CV detection head marker";

  let arguments = (ins Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type(operands, results)
  }];
}

def CV_PrototypeHeadOp : CV_UnaryTensorOp<"prototype_head"> {
  let summary = "Compiler-internal CV prototype head marker";
}

def CV_ConcatOp : CV_Op<"concat", [Pure]> {
  let summary = "Compiler-internal CV tensor concatenation marker";

  let arguments = (ins Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type(operands, results)
  }];
}

#endif // CV_OPS
