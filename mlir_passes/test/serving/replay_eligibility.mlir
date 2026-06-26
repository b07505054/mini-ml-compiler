// FileCheck test for ReplayEligibilityPass.
//
// Run with the HIR plugin loaded:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(replay-eligibility)' \
//   | FileCheck %s
//
// Eligibility rules:
//   llm.attention_decode + all-static tensor shapes -> eligible, bucket "decode_static"
//   llm.attention_prefill                           -> not eligible
//   llm.attention_decode + any dynamic tensor shape -> not eligible

// Static decode: eligible for runtime CUDA graph replay.
// CHECK-LABEL: func.func @decode_static
// CHECK-SAME:  replay.cuda_graph_bucket = "decode_static"
// CHECK-SAME:  replay.eligible = true
// CHECK-SAME:  replay.truth_boundary = "static_shape_replay_eligibility_not_cuda_graph_capture"

// Prefill: variable-length sequences are never eligible.
// CHECK-LABEL: func.func @prefill
// CHECK-SAME:  replay.cuda_graph_bucket = ""
// CHECK-SAME:  replay.eligible = false
// CHECK-SAME:  replay.truth_boundary = "static_shape_replay_eligibility_not_cuda_graph_capture"

// Dynamic decode: ineligible due to dynamic tensor dimensions.
// CHECK-LABEL: func.func @decode_dynamic
// CHECK-SAME:  replay.cuda_graph_bucket = ""
// CHECK-SAME:  replay.eligible = false
// CHECK-SAME:  replay.truth_boundary = "static_shape_replay_eligibility_not_cuda_graph_capture"

module {
  // Static-shape decode: token is a rank-1 tensor of size 1.
  func.func @decode_static(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }

  // Prefill with dynamic sequence length.
  func.func @prefill(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }

  // Dynamic decode: dynamic sequence dimension makes it ineligible.
  func.func @decode_dynamic(%token: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }
}
