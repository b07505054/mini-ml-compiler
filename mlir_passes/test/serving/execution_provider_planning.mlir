// FileCheck test for ExecutionProviderPlanningPass.
//
// Run with the HIR plugin loaded (split-input-file allows multiple modules):
//   mlir-opt %s --split-input-file --allow-unregistered-dialect \
//     --load-dialect-plugin=%plugin \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(serving-optimization-pipeline)' \
//   | FileCheck %s
//
// Decision logic under test:
//   1. No target constraints    -> default_policy / constraint_conflict
//   2. Preferred in allowed set -> target_preferred
//   3. Preferred filtered by paged KV (cuda in target.paged_kv_compatible_backends)
//      -> kv_layout_constraint / cuda
//   4. All allowed non-paged-compatible + paged KV -> constraint_conflict
//   5. Apple-style target with target.paged_kv_compatible_backends=["metal"]
//      -> kv_layout_constraint / metal (preferred "coreml" filtered out)
//
// Paged KV compatibility is now a target property (target.paged_kv_compatible_backends).
// Absent attr -> empty set -> constraint_conflict for paged-KV functions.
// No backend (cuda, vllm, or otherwise) is hardcoded in the compiler.
//
// Attr order on func.func (MLIR dict, alphabetical):
//   execution_provider.decision_source
//   execution_provider.fallback_chain
//   execution_provider.primary
//   execution_provider.required_kv_layout
//   execution_provider.required_precision
//   execution_provider.requires_replay
//   execution_provider.truth_boundary
//   kv.* ...  replay.* ...  serving.* ...

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 1a: No target constraints, prefill (producer -> paged KV).
// Paged KV filter removes the only default candidate "cpu" (not paged-compat).
// Result: constraint_conflict, primary = "cpu" as emergency fallback.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @prefill_no_constraints
// CHECK-SAME:  execution_provider.decision_source = "constraint_conflict"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "paged"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.layout = "paged"
// CHECK-SAME:  replay.eligible = false

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 1b: No target constraints, static decode (consumer -> contiguous KV).
// No KV filter. Default cpu candidate used. decision_source = default_policy.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @decode_no_constraints
// CHECK-SAME:  execution_provider.decision_source = "default_policy"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "contiguous"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = true
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.layout = "contiguous"
// CHECK-SAME:  replay.eligible = true

module attributes {
  llm.model = "tiny-gpt",
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64
} {
  func.func @prefill_no_constraints(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }

  func.func @decode_no_constraints(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 2: Preferred backend in allowed set (static decode -> contiguous KV).
// "coreml" is preferred and in allowed_backends.
// Result: target_preferred, fallback_chain = ["metal", "cpu"].
// required_precision = first of supported_precisions = "fp16".
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @decode_preferred
// CHECK-SAME:  execution_provider.decision_source = "target_preferred"
// CHECK-SAME:  execution_provider.fallback_chain = ["metal", "cpu"]
// CHECK-SAME:  execution_provider.primary = "coreml"
// CHECK-SAME:  execution_provider.required_kv_layout = "contiguous"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = true
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"

module attributes {
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.preferred_backend = "coreml",
  target.allowed_backends = ["coreml", "metal", "cpu"],
  target.supported_precisions = ["fp16", "int8"]
} {
  func.func @decode_preferred(%token: tensor<1xi32>) -> tensor<1x768xf16> {
    %0 = "llm.attention_decode"(%token, %token, %token) {
      kv_cache.role = "consumer",
      serving.phase = "decode",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x768xf16>
    return %0 : tensor<1x768xf16>
  }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 3: Paged KV constraint overrides preferred backend.
// Prefill (producer) -> kv.layout = "paged".
// "coreml" is preferred but not in target.paged_kv_compatible_backends.
// "cuda" is in allowed_backends AND in target.paged_kv_compatible_backends.
// Result: kv_layout_constraint, primary = "cuda".
// Note: cuda is paged-compatible because the TARGET PROFILE says so, not
//       because of any hardcoded compiler logic.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @prefill_kv_override
// CHECK-SAME:  execution_provider.decision_source = "kv_layout_constraint"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cuda"
// CHECK-SAME:  execution_provider.required_kv_layout = "paged"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.layout = "paged"

module attributes {
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.preferred_backend = "coreml",
  target.allowed_backends = ["coreml", "cuda", "cpu"],
  target.paged_kv_compatible_backends = ["cuda"]
} {
  func.func @prefill_kv_override(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 4: Constraint conflict.
// Prefill (producer) -> paged KV.
// allowed_backends = ["coreml", "cpu"]: neither in target.paged_kv_compatible_backends.
// target.paged_kv_compatible_backends absent -> empty set default.
// KV filter empties the candidate set.
// Result: constraint_conflict, primary = "cpu" (emergency fallback),
//         fallback_chain = [].
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @prefill_conflict
// CHECK-SAME:  execution_provider.decision_source = "constraint_conflict"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "cpu"
// CHECK-SAME:  execution_provider.required_kv_layout = "paged"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"

module attributes {
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.allowed_backends = ["coreml", "cpu"]
} {
  func.func @prefill_conflict(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }
}

// -----

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Case 5: Apple-style target with target.paged_kv_compatible_backends=["metal"].
// Prefill (producer) -> kv.layout = "paged".
// allowed_backends = ["coreml", "metal", "cpu"], preferred = "coreml".
// paged_kv_compatible_backends = ["metal"]:
//   filter: "coreml" removed, "metal" kept, "cpu" removed -> ["metal"].
//   preferred "coreml" was in original (preferredInOriginal=true) but not
//   in filtered (preferredInFiltered=false) -> kv_layout_constraint.
// Result: kv_layout_constraint, primary = "metal", fallback_chain = [].
//
// This demonstrates the portability fix: an Apple target can declare Metal
// as paged-KV-compatible without any changes to the compiler pass logic.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// CHECK-LABEL: func.func @prefill_apple_metal_paged_kv
// CHECK-SAME:  execution_provider.decision_source = "kv_layout_constraint"
// CHECK-SAME:  execution_provider.fallback_chain = []
// CHECK-SAME:  execution_provider.primary = "metal"
// CHECK-SAME:  execution_provider.required_kv_layout = "paged"
// CHECK-SAME:  execution_provider.required_precision = "fp16"
// CHECK-SAME:  execution_provider.requires_replay = false
// CHECK-SAME:  execution_provider.truth_boundary = "compiler_execution_provider_plan_not_runtime_dispatch"
// CHECK-SAME:  kv.layout = "paged"

module attributes {
  llm.num_layers = 12 : i64,
  llm.hidden_size = 768 : i64,
  target.preferred_backend = "coreml",
  target.allowed_backends = ["coreml", "metal", "cpu"],
  target.supported_precisions = ["fp16"],
  target.paged_kv_compatible_backends = ["metal"]
} {
  func.func @prefill_apple_metal_paged_kv(%tokens: tensor<?xi32>) -> tensor<?x768xf16> {
    %0 = "llm.attention_prefill"(%tokens, %tokens, %tokens) {
      kv_cache.role = "producer",
      serving.phase = "prefill",
      serving.prompt_tokens = 128 : i64,
      serving.output_tokens = 64 : i64
    } : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x768xf16>
    return %0 : tensor<?x768xf16>
  }
}
