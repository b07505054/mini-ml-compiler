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

def HIR_MatMulOp
    : HIR_Op<"matmul", [Pure]> {
  let summary = "Metadata-only HIR matmul compatibility op";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_SoftmaxOp
    : HIR_Op<"softmax", [Pure]> {
  let summary = "Metadata-only HIR softmax compatibility op";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_Conv2dOp
    : HIR_Op<"conv2d", [Pure]> {
  let summary = "Metadata-only HIR conv2d compatibility op";
  let arguments = (ins AnyTensor:$input, AnyTensor:$filter);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_GeluOp
    : HIR_Op<"gelu", [Pure]> {
  let summary = "Metadata-only HIR gelu compatibility op";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_CastOp
    : HIR_Op<"cast", [Pure]> {
  let summary = "Compiler-materialized float precision cast boundary";
  let description = [{
    Float-to-float precision cast inserted by BoundaryMaterializationPass
    where the selected execution plan requires a cast boundary
    (boundary.cast_required = true). Input and result must be ranked tensors
    with identical shapes and different float element types; int<->float
    boundaries use hir.quantize/hir.dequantize instead.

    The verifier requires provenance attributes so every materialized cast
    is traceable to the planning decision that caused insertion:
    materialized.from_decision and materialized.truth_boundary. The op
    records a compiler IR transform only; it makes no runtime execution
    claim.
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

def HIR_LoadQuantizedWeightOp
    : HIR_Op<"load_quantized_weight", [Pure]> {
  let summary = "Compiler-owned quantized weight artifact reference";
  let description = [{
    Represents an offline-produced packed INT8 weight artifact. The runtime
    may load and validate the referenced artifact, but must not quantize,
    transpose, repack, or choose a substitute artifact.
  }];

  let arguments = (ins);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    attr-dict `:` type($output)
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

def HIR_PortableCPUINT8FusedMatMulBiasReluOp
    : HIR_Op<"portable_cpu_int8_fused_matmul_bias_relu", [Pure]> {
  let summary = "Explicit portable CPU INT8 fused MatMul+Bias+ReLU runtime-call contract";
  let description = [{
    Final Slice 3D lowered representation for the selected portable CPU
    int8_static_symmetric packed-B Cortex-A76 implementation. It consumes
    the explicit INT8 activation produced by hir.quantize, the compiler-owned
    packed INT8 weight artifact value, and FP32 bias, and returns FP32 output.
    Quantization scales, zero points, packed artifact identity, selected
    kernel, codegen target, and binary identity are attributes owned by the
    compiler plan.
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$packed_weight, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $input `,` $packed_weight `,` $bias attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_AttentionOp : HIR_Op<"attention", [Pure]> {
  let summary = "Semantic operator-level causal multi-head attention";
  let description = [{
    A deliberately narrow semantic contract retaining real Q/K/V SSA values.
    Required attributes describe phase, static B/H/Q/C/D dimensions, FP32
    contiguous layout, and causal masking.  This is not a general transformer
    importer and does not imply a KV-cache lifetime implementation.
  }];
  let arguments = (ins AnyTensor:$query, AnyTensor:$key, AnyTensor:$value);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $query `,` $key `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_CPUAttentionOp : HIR_Op<"cpu_attention", [Pure]> {
  let summary = "Concrete callable FP32 CPU attention kernel contract";
  let description = [{
    Final operator-level lowering for an already selected prefill or decode
    implementation.  Runtime must validate and call the exact artifact and
    entry point recorded in the attributes; it may not reselect.
  }];
  let arguments = (ins AnyTensor:$query, AnyTensor:$key, AnyTensor:$value);
  let results = (outs AnyTensor:$output);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $query `,` $key `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

def HIR_KVCacheCreateOp : HIR_Op<"kv_cache_create"> {
  let summary = "Create a runtime-owned contiguous KV cache handle";
  let arguments = (ins);
  let results = (outs I64:$handle);
  let hasVerifier = 1;
}

def HIR_KVCachePrefillWriteOp : HIR_Op<"kv_cache_prefill_write"> {
  let summary = "Write prompt K/V tensors into an empty contiguous cache";
  let arguments = (ins I64:$handle, AnyTensor:$key, AnyTensor:$value);
  let results = (outs);
  let hasVerifier = 1;
}

def HIR_KVCacheAppendOp : HIR_Op<"kv_cache_append"> {
  let summary = "Append one token of K/V to a contiguous cache";
  let arguments = (ins I64:$handle, AnyTensor:$key, AnyTensor:$value);
  let results = (outs);
  let hasVerifier = 1;
}

def HIR_KVCacheViewOp : HIR_Op<"kv_cache_view", [Pure]> {
  let summary = "Bind the valid contiguous K/V cache range for attention";
  let arguments = (ins I64:$handle);
  let results = (outs AnyTensor:$key, AnyTensor:$value);
  let hasVerifier = 1;
}

def HIR_KVCacheResetOp : HIR_Op<"kv_cache_reset"> {
  let summary = "Reset logical validity of a contiguous KV cache";
  let arguments = (ins I64:$handle);
  let results = (outs);
  let hasVerifier = 1;
}

def HIR_PagedKVPoolCreateOp : HIR_Op<"paged_kv_pool_create"> {
  let summary = "Create runtime-owned physical KV pages and block table";
  let arguments = (ins); let results = (outs I64:$pool); let hasVerifier = 1;
}
def HIR_PagedKVBindBlockOp : HIR_Op<"paged_kv_bind_block"> {
  let summary = "Declare live logical-block to physical-page binding";
  let arguments = (ins I64:$pool); let results = (outs); let hasVerifier = 1;
}
def HIR_PagedKVPrefillWriteOp : HIR_Op<"paged_kv_prefill_write"> {
  let summary = "Write prompt K/V across mapped physical pages";
  let arguments = (ins I64:$pool, AnyTensor:$key, AnyTensor:$value); let results = (outs); let hasVerifier = 1;
}
def HIR_PagedKVAppendOp : HIR_Op<"paged_kv_append"> {
  let summary = "Append one K/V token through the block table";
  let arguments = (ins I64:$pool, AnyTensor:$key, AnyTensor:$value); let results = (outs); let hasVerifier = 1;
}
def HIR_PagedKVViewOp : HIR_Op<"paged_kv_view", [Pure]> {
  let summary = "Bind paged KV storage and block table to paged attention";
  let arguments = (ins I64:$pool); let results = (outs AnyTensor:$key, AnyTensor:$value); let hasVerifier = 1;
}
def HIR_PagedKVResetOp : HIR_Op<"paged_kv_reset"> {
  let summary = "Release request pages and reset block table";
  let arguments = (ins I64:$pool); let results = (outs); let hasVerifier = 1;
}

#endif // HIR_OPS
