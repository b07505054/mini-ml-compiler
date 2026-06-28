// FileCheck test for QuantizationPlanningPass.
//
// Run with the HIR plugin loaded (split-input-file for multiple modules):
//   mlir-opt %s --split-input-file --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(quantization-planning)' \
//   | FileCheck %s
//
// Pass is module-scoped: it reads llm.dtype and target.supported_precisions
// from the module, emits quantization.plan_dtype / plan_source /
// truth_boundary as module attrs, and (for int8) annotates linalg.matmul ops.
//
// Decision rules (tested in order below):
//   Rule 1: llm.dtype in supported_precisions -> use llm.dtype, target_profile
//   Rule 2: supported_precisions non-empty     -> use first, target_profile
//   Rule 3: llm.dtype set, no supported list   -> use llm.dtype, default_dtype
//   Rule 4: neither set                        -> "fp16", default_dtype
//
// Attr order on module output (MLIR dict, alphabetical).

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 1: llm.dtype = "fp16" is in supported_precisions = ["fp16"].
// Rule 1 fires: plan_dtype = "fp16", plan_source = "target_profile".
// No linalg.matmul ops to annotate.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: module attributes
// CHECK-SAME:  quantization.plan_dtype = "fp16"
// CHECK-SAME:  quantization.plan_source = "target_profile"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"

module attributes {
  llm.dtype = "fp16",
  target.supported_precisions = ["fp16"]
} {
  func.func @noop() { return }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 2: llm.dtype = "int8" is in supported_precisions = ["fp16", "int8"].
// Rule 1 fires: plan_dtype = "int8", plan_source = "target_profile".
// linalg.matmul gets quantization.candidate = "int8" and
// profile.quantized_path = "faster" (enables shouldUseQuantizedMatMul()).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: module attributes
// CHECK-SAME:  quantization.plan_dtype = "int8"
// CHECK-SAME:  quantization.plan_source = "target_profile"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"
// CHECK:       profile.quantized_path = "faster"
// CHECK:       quantization.candidate = "int8"

module attributes {
  llm.dtype = "int8",
  target.supported_precisions = ["fp16", "int8"]
} {
  func.func @matmul_fn(
    %A: tensor<128x128xf32>,
    %B: tensor<128x128xf32>
  ) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
        outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 3: no llm.dtype, no target.supported_precisions.
// Rule 4 fires: plan_dtype = "fp16", plan_source = "default_dtype".
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: module attributes
// CHECK-SAME:  quantization.plan_dtype = "fp16"
// CHECK-SAME:  quantization.plan_source = "default_dtype"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"

module {
  func.func @noop() { return }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 4: llm.dtype = "int8" but supported_precisions = ["fp16"] only.
// llm.dtype NOT in supported_precisions -> Rule 2 fires.
// plan_dtype = "fp16" (first of supported_precisions), plan_source = "target_profile".
// linalg.matmul is NOT annotated (plan_dtype is fp16, not int8).
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: module attributes
// CHECK-SAME:  quantization.plan_dtype = "fp16"
// CHECK-SAME:  quantization.plan_source = "target_profile"
// CHECK-SAME:  quantization.truth_boundary = "precision_selection_from_target_profile_not_calibrated"
// CHECK-NOT:   quantization.candidate

module attributes {
  llm.dtype = "int8",
  target.supported_precisions = ["fp16"]
} {
  func.func @matmul_fn(
    %A: tensor<128x128xf32>,
    %B: tensor<128x128xf32>
  ) -> tensor<128x128xf32> {
    %empty = tensor.empty() : tensor<128x128xf32>
    %result = linalg.matmul
        ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
        outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %result : tensor<128x128xf32>
  }
}
