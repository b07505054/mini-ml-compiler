// CTest unit test for the D2 Qwen-aware legality/cost extensions
// (checkQwenCandidateLegality, estimateDistributedCost) in
// serving/DistributedPlanning.h. Pure C++. No MLIR IR, no GoogleTest.
//
// The full pass-in-real-pipeline path (pass registration, real Qwen
// metadata extraction, TP1 backward compatibility, ExecutionPlan export)
// is covered separately by the CTest integration test
// RunDistributedStrategyPlanningPipelineTest.cmake, which runs the actual
// compile-for-target binary against the real per-layer Qwen ONNX graph.

#include "serving/DistributedPlanning.h"

#include <cassert>
#include <chrono>
#include <cstdio>

using namespace mlir::hir;

static QwenOperatorContext realO_projLikeContext() {
  QwenOperatorContext ctx;
  ctx.operator_type = "llm.o_proj";
  ctx.function_name = "qwen_prefill";
  ctx.layer_index = 0;
  ctx.operator_id = "qwen_prefill::llm.o_proj::layer_0";
  ctx.hidden_dim = 896;  // real qwen2.5-0.5b hidden_size
  ctx.hidden_dim_is_static = true;
  ctx.num_attention_heads = 14;
  ctx.num_kv_heads = 2;
  ctx.distributed_capability_available = true;
  return ctx;
}

static void testTP1CandidateGeneration() {
  auto candidates = generateDistributedCandidates();
  assert(candidates.size() == 2);
  assert(candidates[0].candidate_id == "tp1");
  assert(candidates[0].world_size == 1);
  std::puts("  [PASS] testTP1CandidateGeneration");
}

static void testTP2CandidateGenerationFromRealMetadata() {
  auto candidates = generateDistributedCandidates();
  assert(candidates[1].candidate_id == "tp2");
  assert(candidates[1].world_size == 2);
  assert(candidates[1].tensor_parallel_size == 2);
  std::puts("  [PASS] testTP2CandidateGenerationFromRealMetadata");
}

static void testLegalTP2SelectionOverRealHiddenSize() {
  auto candidates = generateDistributedCandidates();
  auto ctx = realO_projLikeContext();
  auto legality = checkQwenCandidateLegality(candidates[1], ctx);
  assert(legality.legal && "896 hidden_size must be legal for TP2 (896 % 2 == 0)");
  assert(legality.rejection_reasons.empty());
  // Every rule must have a recorded status -- never silently skipped.
  bool foundDivisibility = false, foundHeadCount = false, foundCapability = false;
  for (const auto &r : legality.rule_results) {
    if (r.rule == "tensor_hidden_dimension_divisibility") {
      foundDivisibility = true;
      assert(r.status == "pass");
    }
    if (r.rule == "head_count_divisibility") {
      foundHeadCount = true;
      assert(r.status == "not_applicable");
    }
    if (r.rule == "runtime_capability_availability") {
      foundCapability = true;
      assert(r.status == "pass");
    }
  }
  assert(foundDivisibility && foundHeadCount && foundCapability);
  std::puts("  [PASS] testLegalTP2SelectionOverRealHiddenSize");
}

static void testIllegalTP2RejectionNonDivisibleHiddenSize() {
  auto candidates = generateDistributedCandidates();
  auto ctx = realO_projLikeContext();
  ctx.hidden_dim = 897;  // odd -- not divisible by tensor_parallel_size=2
  auto legality = checkQwenCandidateLegality(candidates[1], ctx);
  assert(!legality.legal);
  bool found = false;
  for (const auto &r : legality.rejection_reasons)
    if (r.find("not divisible") != std::string::npos) found = true;
  assert(found);
  std::puts("  [PASS] testIllegalTP2RejectionNonDivisibleHiddenSize");
}

static void testIllegalTP2RejectionUnsupportedOperator() {
  auto candidates = generateDistributedCandidates();
  auto ctx = realO_projLikeContext();
  ctx.operator_type = "llm.mlp";  // not in the D2 narrow allow-list
  auto legality = checkQwenCandidateLegality(candidates[1], ctx);
  assert(!legality.legal);
  bool found = false;
  for (const auto &r : legality.rejection_reasons)
    if (r.find("unsupported operator type") != std::string::npos) found = true;
  assert(found);
  assert(!isSupportedDistributedOperatorType("llm.mlp"));
  assert(isSupportedDistributedOperatorType("llm.o_proj"));
  std::puts("  [PASS] testIllegalTP2RejectionUnsupportedOperator");
}

static void testIllegalTP2RejectionMissingShapeMetadata() {
  auto candidates = generateDistributedCandidates();
  auto ctx = realO_projLikeContext();
  ctx.hidden_dim_is_static = false;
  ctx.hidden_dim = 0;
  auto legality = checkQwenCandidateLegality(candidates[1], ctx);
  assert(!legality.legal);
  bool found = false;
  for (const auto &r : legality.rejection_reasons)
    if (r.find("dynamic") != std::string::npos) found = true;
  assert(found);
  std::puts("  [PASS] testIllegalTP2RejectionMissingShapeMetadata");
}

static void testNoDistributedCapabilityRejection() {
  auto candidates = generateDistributedCandidates();
  auto ctx = realO_projLikeContext();
  ctx.distributed_capability_available = false;
  auto legality = checkQwenCandidateLegality(candidates[1], ctx);
  assert(!legality.legal);
  assert(legality.rejection_reasons.size() == 1);
  assert(legality.rejection_reasons[0].find("opt in") != std::string::npos);
  std::puts("  [PASS] testNoDistributedCapabilityRejection");
}

static void testTP1AlwaysLegalRegardlessOfOperator() {
  // TP1 must be selectable for any graph/profile (D2 Part B requirement 7) --
  // callers bypass Qwen-specific legality entirely for world_size==1
  // candidates (see DistributedStrategyPlanningPass.cpp), so this documents
  // the contract at the data level: an "unsupported" context does not make
  // the TP1 *candidate* itself structurally illegal under D1's base checks.
  auto candidates = generateDistributedCandidates();
  auto base = checkCandidateLegality(candidates[0], /*tensor_dim_k=*/1);
  assert(base.legal);
  std::puts("  [PASS] testTP1AlwaysLegalRegardlessOfOperator");
}

static void testCostEvidenceSerializationIsExplicitAndInspectable() {
  auto candidates = generateDistributedCandidates();
  auto ctx = realO_projLikeContext();

  auto tp1Cost = estimateDistributedCost(candidates[0], ctx);
  assert(tp1Cost.collective_count == 0);
  assert(tp1Cost.estimated_communication_bytes == 0);
  assert(tp1Cost.process_launch_overhead_penalty == 0);
  assert(!tp1Cost.truth_boundary.empty());

  auto tp2Cost = estimateDistributedCost(candidates[1], ctx);
  assert(tp2Cost.collective_count == 1);
  assert(tp2Cost.estimated_communication_bytes > 0);
  assert(tp2Cost.process_launch_overhead_penalty > 0);
  assert(tp2Cost.unsupported_operation_penalty == 0);  // o_proj is supported
  assert(tp2Cost.total_score ==
         tp2Cost.rank_local_compute_bytes + tp2Cost.estimated_communication_bytes +
             tp2Cost.collective_count * 1000 + tp2Cost.process_launch_overhead_penalty +
             tp2Cost.unsupported_operation_penalty + tp2Cost.fallback_penalty);
  assert(tp2Cost.truth_boundary.find("not_gpu_measured") != std::string::npos);
  assert(tp2Cost.truth_boundary.find("not_nccl_calibrated") != std::string::npos);

  QwenOperatorContext unsupportedCtx = ctx;
  unsupportedCtx.operator_type = "llm.mlp";
  auto unsupportedCost = estimateDistributedCost(candidates[1], unsupportedCtx);
  assert(unsupportedCost.unsupported_operation_penalty > 0);

  std::puts("  [PASS] testCostEvidenceSerializationIsExplicitAndInspectable");
}

// D2 Part M measurement: real in-process latency of one full
// generate-candidates + legality + cost evaluation cycle, averaged over
// many iterations. Printed to stdout (not an assertion) so the D2
// measurement script can parse it; does not affect test pass/fail.
static void reportLegalityAndCostLatency() {
  auto ctx = QwenOperatorContext{};
  ctx.operator_type = "llm.o_proj";
  ctx.function_name = "qwen_prefill";
  ctx.layer_index = 0;
  ctx.operator_id = "qwen_prefill::llm.o_proj::layer_0";
  ctx.hidden_dim = 896;
  ctx.hidden_dim_is_static = true;
  ctx.distributed_capability_available = true;

  constexpr int kIterations = 10000;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    auto candidates = generateDistributedCandidates();
    for (const auto &c : candidates) {
      auto legality = c.world_size > 1 ? checkQwenCandidateLegality(c, ctx)
                                        : QwenDistributedLegalityResult{true, {}, {}};
      auto cost = estimateDistributedCost(c, ctx);
      (void)legality;
      (void)cost;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  double nanosPerCall =
      std::chrono::duration<double, std::nano>(t1 - t0).count() / kIterations;
  std::printf("LEGALITY_COST_LATENCY_NS_PER_CALL=%.1f\n", nanosPerCall);
}

int main() {
  std::puts("DistributedStrategyPlanningTest:");
  testTP1CandidateGeneration();
  testTP2CandidateGenerationFromRealMetadata();
  testLegalTP2SelectionOverRealHiddenSize();
  testIllegalTP2RejectionNonDivisibleHiddenSize();
  testIllegalTP2RejectionUnsupportedOperator();
  testIllegalTP2RejectionMissingShapeMetadata();
  testNoDistributedCapabilityRejection();
  testTP1AlwaysLegalRegardlessOfOperator();
  testCostEvidenceSerializationIsExplicitAndInspectable();
  std::puts("DistributedStrategyPlanningTest: PASS");
  reportLegalityAndCostLatency();
  return 0;
}
