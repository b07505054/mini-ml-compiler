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

def HIR_QuantizeOp
    : HIR_Op<"quantize", [Pure]> {
  let summary = "Quantize floating-point activations to INT8";
  let description = [{
    Quantizes an activation tensor using runtime-visible quantization metadata.
    Metadata includes scale, zero point, quantized dtype, and per-channel mode.
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_DequantizeOp
    : HIR_Op<"dequantize", [Pure]> {
  let summary = "Dequantize INT8 activations to floating point";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_QMatMulOp
    : HIR_Op<"qmatmul", [Pure]> {
  let summary = "INT8 quantized matrix multiply";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_FusedQMatMulBiasReluOp
    : HIR_Op<"fused_qmatmul_bias_relu", [Pure]> {
  let summary = "Fused INT8 matmul, bias add, and ReLU";
  let description = [{
    Qualcomm/mobile-oriented fused op for an INT8 matmul path with per-channel
    scale, zero point, activation quantization metadata, and layout/alignment
    constraints carried as attributes.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $bias attr-dict `:` functional-type(operands, results)
  }];
}

#endif // HIR_OPS
