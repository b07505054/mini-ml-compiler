// Phase P1B: minimal single-op MatMul+Bias+ReLU compiler input.
//
// One function, one hir.fused_matmul_bias_relu op, static f32 128x128x128
// shape (M=N=K=128, matching the existing "exact-shape profile match"
// convention already used by mlir_passes/test/matmul_bias_relu_128x128x128.mlir).
// Written in MLIR generic op syntax so it parses under compile-for-target's
// allowUnregisteredDialects(true) context, which does not link the HIR
// dialect. hir.fused_matmul_bias_relu itself is a real, existing, verified
// HIR op (mlir_passes/include/HIR/IR/HIROps.td: HIR_FusedMatMulBiasReluOp) —
// this file hand-authors its post-fusion form directly rather than running
// the separate matmul-bias-relu-fusion mlir-opt pass plugin, which is out of
// scope for this PR (no new fusion-materialization work).
//
// cv.semantic_annotation.status = "completed" activates the CV execution
// plan path (CVExecutionPlanAttrsPass -> serving.policy = "cv_full_graph"),
// the same gate YOLO-Seg uses, so ExecutionPlanBuilder collects a function
// plan (it only collects functions carrying serving.policy).
//
// weight.is_constant = true on the weight/bias producer ops marks them as
// constant (WeightClassificationPlanningPass Rule 4) so quantization
// strategy planning resolves to "none"/f32 for this generic, no-quant-policy
// profile -- matching real FC-layer semantics (weight/bias are trained
// constants; only the input activation is a runtime value) and avoiding an
// unrelated fp16-fallback path meant for non-constant-weight cases.
module {
  func.func @matmul_bias_relu_main(%a: tensor<128x128xf32>)
      -> tensor<128x128xf32> attributes {cv.semantic_annotation.status = "completed"} {
    %b = "test.weight"() {weight.is_constant = true} : () -> tensor<128x128xf32>
    %bias = "test.weight"() {weight.is_constant = true} : () -> tensor<128x128xf32>
    %0 = "hir.fused_matmul_bias_relu"(%a, %b, %bias)
        : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
  }
}
