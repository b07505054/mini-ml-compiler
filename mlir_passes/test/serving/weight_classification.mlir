// RUN: mlir-opt --allow-unregistered-dialect \
// RUN:   --load-pass-plugin=%plugin \
// RUN:   --pass-pipeline='builtin.module(weight-classification-planning-pipeline)' \
// RUN:   --split-input-file %s | FileCheck %s
//
// WeightClassificationPlanningPass tests (Commit M).
// "compute.*" ops used for quantizable ops so the HIR dialect does not reject
// them as unknown ops. The pass strips the dialect prefix and matches on the
// short name, so "compute.matmul" -> "matmul" -> isQuantizableOp = true.
//
// Truth boundary: weight_classification_static_def_use_analysis_not_runtime_validated

// Test 1: RHS defined by arith.constant -> constant_weight, satisfied=true.

// CHECK-LABEL: @matmul_constant_rhs
// CHECK: "compute.matmul"
// CHECK-SAME: weight.classification = "constant_weight"
// CHECK-SAME: weight.classification_reason = "arith_constant_op"
// CHECK-SAME: weight.constant_required = true
// CHECK-SAME: weight.constant_satisfied = true
// CHECK-SAME: weight.truth_boundary = "weight_classification_static_def_use_analysis_not_runtime_validated"

func.func @matmul_constant_rhs(%x: tensor<1x256xf16>) -> tensor<1x256xf16> {
  %w = arith.constant dense<1.0> : tensor<256x256xf16>
  %r = "compute.matmul"(%x, %w) : (tensor<1x256xf16>, tensor<256x256xf16>) -> tensor<1x256xf16>
  return %r : tensor<1x256xf16>
}

// -----

// Test 2: RHS is a function argument (BlockArgument) -> runtime_activation,
//         constant_satisfied=false.

// CHECK-LABEL: @matmul_func_arg_rhs
// CHECK: "compute.matmul"
// CHECK-SAME: weight.classification = "runtime_activation"
// CHECK-SAME: weight.classification_reason = "function_argument_is_runtime"
// CHECK-SAME: weight.constant_required = true
// CHECK-SAME: weight.constant_satisfied = false

func.func @matmul_func_arg_rhs(%x: tensor<1x256xf16>,
                                %w: tensor<256x256xf16>) -> tensor<1x256xf16> {
  %r = "compute.matmul"(%x, %w) : (tensor<1x256xf16>, tensor<256x256xf16>) -> tensor<1x256xf16>
  return %r : tensor<1x256xf16>
}

// -----

// Test 3: RHS produced by llm.attention_prefill (known llm.* compute op) ->
//         runtime_activation, reason = "compute_op_produces_runtime_value".

// CHECK-LABEL: @matmul_attention_rhs
// CHECK: "compute.matmul"
// CHECK-SAME: weight.classification = "runtime_activation"
// CHECK-SAME: weight.classification_reason = "compute_op_produces_runtime_value"
// CHECK-SAME: weight.constant_satisfied = false

func.func @matmul_attention_rhs(%x: tensor<1x256xf16>) -> tensor<1x256xf16> {
  %attn = "llm.attention_prefill"(%x, %x, %x) {
    kv_cache.role = "producer",
    serving.phase  = "prefill"
  } : (tensor<1x256xf16>, tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
  %r = "compute.matmul"(%x, %attn) : (tensor<1x256xf16>, tensor<1x256xf16>) -> tensor<1x256xf16>
  return %r : tensor<1x256xf16>
}

// -----

// Test 4: conv2d with weight.is_constant=true on the op itself (Rule 2 op-level
//         override). %w is a function arg which would otherwise be runtime_activation,
//         but the explicit attribute short-circuits def-use analysis.

// CHECK-LABEL: @conv2d_declared_constant
// CHECK: "compute.conv2d"
// CHECK-SAME: weight.classification = "constant_weight"
// CHECK-SAME: weight.classification_reason = "declared_constant_weight_attr_on_op"
// CHECK-SAME: weight.constant_satisfied = true

func.func @conv2d_declared_constant(%x: tensor<1x64xf16>,
                                     %w: tensor<64x64xf16>) -> tensor<1x64xf16> {
  %r = "compute.conv2d"(%x, %w) { weight.is_constant = true }
      : (tensor<1x64xf16>, tensor<64x64xf16>) -> tensor<1x64xf16>
  return %r : tensor<1x64xf16>
}

// -----

// Test 5: RHS from param.load, which is not in the known-constant or
//         known-runtime sets -> unknown, conservative constant_satisfied=false.

// CHECK-LABEL: @matmul_unknown_producer
// CHECK: "compute.matmul"
// CHECK-SAME: weight.classification = "unknown"
// CHECK-SAME: weight.classification_reason = "unknown_producer_conservative"
// CHECK-SAME: weight.constant_required = true
// CHECK-SAME: weight.constant_satisfied = false

func.func @matmul_unknown_producer(%x: tensor<1x256xf16>) -> tensor<1x256xf16> {
  %w = "param.load"(%x) : (tensor<1x256xf16>) -> tensor<256x256xf16>
  %r = "compute.matmul"(%x, %w) : (tensor<1x256xf16>, tensor<256x256xf16>) -> tensor<1x256xf16>
  return %r : tensor<1x256xf16>
}
