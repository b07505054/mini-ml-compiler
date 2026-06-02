#ifndef HIR_OPS
#define HIR_OPS

include "HIR/IR/HIRDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HIR_FusedMatMulBiasReluOp
    : HIR_Op<"fused_matmul_bias_relu", [Pure]> {
  let summary = "Fused matrix multiply, bias add, and ReLU";
  let description = [{
    Runtime-facing fused op emitted from the pattern:

      linalg.matmul -> linalg.map addf -> linalg.map maximumf

    The op keeps runtime-aware metadata in the attribute dictionary, including
    fusion.candidate, fusion.group, kernel.selection, and lowering.source.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $bias attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_FusedRMSNormOp
    : HIR_Op<"fused_rmsnorm", [Pure, SameOperandsAndResultType]> {
  let summary = "Fused RMSNorm";
  let description = [{
    Runtime-facing fused op emitted from an LLM RMSNorm source op. Runtime
    profile metadata determines whether the downstream executor dispatches a
    custom CUDA kernel or the fallback RMSNorm implementation.
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

#endif // HIR_OPS
