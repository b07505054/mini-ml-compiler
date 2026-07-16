// vectorize_matmul_bias_relu.mlir
//
// Project-owned MLIR Transform-dialect script for the vectorized AArch64
// backend-codegen variant (see mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh
// --variant vectorized).
//
// This is project-owned instruction-selection PREPARATION, not machine
// instruction selection: it rewrites the tensor-level Linalg form of the
// fused MatMul-Bias-ReLU kernel (produced by the existing, unchanged
// hir-matmul-bias-relu-to-linalg pass) into MLIR Vector-dialect operations
// (vector.transfer_read / vector.contract / vector.transfer_write, plus
// vectorized arith.addf/arith.maximumf for the bias-add+ReLU stage). LLVM's
// own AArch64 backend performs the actual machine instruction selection that
// turns this vector IR into NEON instructions (observed to produce `fmla`
// when combined with the `vector-contract-lowering=outerproduct` option on
// convert-vector-to-llvm later in the pipeline) -- that step is LLVM-owned,
// not implemented by this project.
//
// Applied via:
//   mlir-opt <linalg-form>.mlir \
//     --pass-pipeline='builtin.module(
//       transform-preload-library{transform-library-paths=mlir_passes/transforms/vectorize_matmul_bias_relu.mlir},
//       transform-interpreter{entry-point=__transform_main},
//       ...)'
//
// See mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh for the full
// pipeline this is one stage of.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %vectorized = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
