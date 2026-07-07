// FileCheck test for llm-frontend-normalization with full per-layer expansion.
//
// Two decoder layers (serving.layer_index = 0, 1) in one flat, unrolled
// function, each with its own raw attention sub-pattern. Verifies the pass
// performs a LOCALIZED per-occurrence rewrite:
//   - each layer's raw pattern collapses to its own llm.attention_prefill,
//     wired to that layer's REAL q/k/v_proj values (kept alive, not erased,
//     not a dummy placeholder)
//   - the correct serving.layer_index/serving.layer_role is propagated onto
//     the new canonical op
//   - surrounding per-layer ops (rmsnorm, o_proj) are left untouched
//   - the now-redundant intermediate ops (attention_scores, softmax,
//     kv_cache_write) are gone, with nothing left between q/k/v_proj and
//     the canonical op, or between the canonical op and o_proj
//
// Run via tools/run_mlir_pass_tests.sh, or directly:
//   mlir-opt %s --allow-unregistered-dialect \
//     --load-pass-plugin=%plugin \
//     --pass-pipeline='builtin.module(llm-frontend-normalization)' \
//   | FileCheck %s
//
// MLIR prints attributes alphabetically; CHECK-SAME directives below follow
// that order (kv_cache.role < serving.layer_index < serving.layer_role).

// CHECK-LABEL: func.func @layered_prefill
// CHECK: %[[N0:.*]] = "llm.rmsnorm"(%arg0)
// CHECK-NEXT: %[[Q0:.*]] = "llm.q_proj"(%[[N0]])
// CHECK-NEXT: %[[K0:.*]] = "llm.k_proj"(%[[N0]])
// CHECK-NEXT: %[[V0:.*]] = "llm.v_proj"(%[[N0]])
// CHECK-NEXT: %[[A0:.*]] = "llm.attention_prefill"(%[[Q0]], %[[K0]], %[[V0]])
// CHECK-SAME:  kv_cache.role = "producer"
// CHECK-SAME:  serving.layer_index = 0 : i64
// CHECK-SAME:  serving.layer_role = "decoder_layer"
// CHECK-NEXT: %[[O0:.*]] = "llm.o_proj"(%[[A0]])
// CHECK-NEXT: %[[N1:.*]] = "llm.rmsnorm"(%[[O0]])
// CHECK-NEXT: %[[Q1:.*]] = "llm.q_proj"(%[[N1]])
// CHECK-NEXT: %[[K1:.*]] = "llm.k_proj"(%[[N1]])
// CHECK-NEXT: %[[V1:.*]] = "llm.v_proj"(%[[N1]])
// CHECK-NEXT: %[[A1:.*]] = "llm.attention_prefill"(%[[Q1]], %[[K1]], %[[V1]])
// CHECK-SAME:  kv_cache.role = "producer"
// CHECK-SAME:  serving.layer_index = 1 : i64
// CHECK-SAME:  serving.layer_role = "decoder_layer"
// CHECK-NEXT: %[[O1:.*]] = "llm.o_proj"(%[[A1]])
// CHECK-NEXT: return %[[O1]]

module attributes {
  llm.num_attention_heads = 2 : i64,
  llm.num_key_value_heads = 1 : i64
} {
  func.func @layered_prefill(%hidden: tensor<?x8xf16>) -> tensor<?x8xf16> {
    // ---- Layer 0 ----
    %n0 = "llm.rmsnorm"(%hidden) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x8xf16>
    %q0 = "llm.q_proj"(%n0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x8xf16>
    %k0 = "llm.k_proj"(%n0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x4xf16>
    %v0 = "llm.v_proj"(%n0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x4xf16>
    %s0 = "llm.attention_scores"(%q0, %k0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>, tensor<?x4xf16>) -> tensor<?x?xf16>
    %p0 = "llm.softmax"(%s0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x?xf16>) -> tensor<?x?xf16>
    %a0 = "llm.attention_output"(%p0, %v0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x?xf16>, tensor<?x4xf16>) -> tensor<?x8xf16>
    "llm.kv_cache_write"(%k0, %v0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x4xf16>, tensor<?x4xf16>) -> ()
    %o0 = "llm.o_proj"(%a0) {serving.layer_index = 0 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x8xf16>

    // ---- Layer 1 ----
    %n1 = "llm.rmsnorm"(%o0) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x8xf16>
    %q1 = "llm.q_proj"(%n1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x8xf16>
    %k1 = "llm.k_proj"(%n1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x4xf16>
    %v1 = "llm.v_proj"(%n1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x4xf16>
    %s1 = "llm.attention_scores"(%q1, %k1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>, tensor<?x4xf16>) -> tensor<?x?xf16>
    %p1 = "llm.softmax"(%s1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x?xf16>) -> tensor<?x?xf16>
    %a1 = "llm.attention_output"(%p1, %v1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x?xf16>, tensor<?x4xf16>) -> tensor<?x8xf16>
    "llm.kv_cache_write"(%k1, %v1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x4xf16>, tensor<?x4xf16>) -> ()
    %o1 = "llm.o_proj"(%a1) {serving.layer_index = 1 : i64, serving.layer_role = "decoder_layer"}
        : (tensor<?x8xf16>) -> tensor<?x8xf16>

    return %o1 : tensor<?x8xf16>
  }
}
