// CTest unit test for DistributedPlanning (D1: compiler-planned TP=2
// multi-process simulation). Pure C++. No GoogleTest, no MLIR IR, no Python.
//
// Covers: TP1/TP2 candidate generation, TP2 legality + export, and every
// negative legality rule required by D1 Part C/L.

#include "serving/DistributedPlanning.h"
#include "serving/ExecutionPlanExporter.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cassert>
#include <cstdio>
#include <filesystem>

using namespace mlir::hir;

static void testCandidateGeneration() {
  auto candidates = generateDistributedCandidates();
  assert(candidates.size() == 2 && "expected exactly TP1 and TP2 candidates");

  const auto &tp1 = candidates[0];
  assert(tp1.candidate_id == "tp1");
  assert(tp1.world_size == 1 && tp1.tensor_parallel_size == 1 &&
         tp1.pipeline_parallel_size == 1);

  const auto &tp2 = candidates[1];
  assert(tp2.candidate_id == "tp2");
  assert(tp2.world_size == 2 && tp2.tensor_parallel_size == 2 &&
         tp2.pipeline_parallel_size == 1);

  std::puts("  [PASS] testCandidateGeneration");
}

static void testTP1LegalityAndBuild() {
  auto candidates = generateDistributedCandidates();
  auto legality = checkCandidateLegality(candidates[0], /*tensor_dim_k=*/16);
  assert(legality.legal && "TP1 must be legal for any positive K");
  auto plan = buildDistributedPlan(candidates[0], 16, "partial_output");
  assert(plan.has_value());
  assert(plan->world_size == 1);
  assert(plan->ranks.size() == 1);
  assert(plan->collectives.empty() && "TP1 requires no collective step");
  auto structural = validateDistributedPlan(*plan);
  assert(structural.legal);
  std::puts("  [PASS] testTP1LegalityAndBuild");
}

static void testTP2LegalityAndExport() {
  auto candidates = generateDistributedCandidates();
  const auto &tp2 = candidates[1];
  auto legality = checkCandidateLegality(tp2, /*tensor_dim_k=*/16);
  assert(legality.legal && "TP2 over K=16 must be legal (16 % 2 == 0)");

  auto plan = buildDistributedPlan(tp2, 16, "partial_output");
  assert(plan.has_value());
  assert(plan->world_size == 2);
  assert(plan->ranks.size() == 2);
  assert(plan->ranks[0].rank_id == 0 && plan->ranks[1].rank_id == 1);
  assert(plan->tensor_shards.size() == 2);
  assert(plan->tensor_shards[0].range_start == 0 &&
         plan->tensor_shards[0].range_end == 8);
  assert(plan->tensor_shards[1].range_start == 8 &&
         plan->tensor_shards[1].range_end == 16);
  assert(plan->collectives.size() == 1);
  assert(plan->collectives[0].kind == "all_reduce");
  assert(plan->collectives[0].participants.size() == 2);

  auto structural = validateDistributedPlan(*plan);
  assert(structural.legal && "compiler-built TP2 plan must be structurally legal");

  ExecutionPlan ep;
  ep.plan_id = "d1_distributed_planning_test_tp2";
  ep.provenance.compiler_tool = "DistributedPlanningTest";
  ep.provenance.truth_boundary = "test_fixture";
  ep.model_identity.model_id = "d1_synthetic_matmul";
  ep.model_identity.truth_boundary = "test_fixture";
  ep.distributed = plan;

  std::filesystem::path outPath =
      std::filesystem::temp_directory_path() /
      "distributed_planning_test_tp2_export.json";
  auto err = ExecutionPlanExporter::exportToFile(ep, outPath.string());
  assert(!err && "export must succeed");

  auto buf = llvm::MemoryBuffer::getFile(outPath.string());
  assert(buf && "exported file must be readable");
  auto parsed = llvm::json::parse((*buf)->getBuffer());
  assert(parsed && "exported file must be valid JSON");
  auto *root = parsed->getAsObject();
  assert(root && root->get("distributed") && "distributed key must be present");
  auto *distributed = root->getObject("distributed");
  assert(distributed->getInteger("world_size") == 2);
  assert(distributed->getInteger("tensor_parallel_size") == 2);
  auto *collectives = distributed->getArray("collectives");
  assert(collectives && collectives->size() == 1);

  std::filesystem::remove(outPath);
  std::puts("  [PASS] testTP2LegalityAndExport");
}

static void testInvalidNonDivisibleShardRejection() {
  auto candidates = generateDistributedCandidates();
  const auto &tp2 = candidates[1];
  auto legality = checkCandidateLegality(tp2, /*tensor_dim_k=*/15);
  assert(!legality.legal);
  bool found = false;
  for (const auto &reason : legality.rejection_reasons)
    if (reason.find("divisible") != std::string::npos)
      found = true;
  assert(found && "must record non-divisible rejection reason");
  auto plan = buildDistributedPlan(tp2, 15, "partial_output");
  assert(!plan.has_value() && "illegal candidate must fail closed, no plan built");
  std::puts("  [PASS] testInvalidNonDivisibleShardRejection");
}

static void testInvalidRankTopologyRejection() {
  DistributedPlan plan;
  plan.world_size = 2;
  plan.tensor_parallel_size = 2;
  plan.pipeline_parallel_size = 1;
  // Duplicate rank_id 0, missing rank_id 1 — illegal topology.
  plan.ranks = {DistributedRankPlacement{0, "simulated_cpu_process_0"},
                DistributedRankPlacement{0, "simulated_cpu_process_0b"}};
  plan.tensor_shards = {
      DistributedTensorShard{"partial_output", 0, 2, 0, 0, 8},
      DistributedTensorShard{"partial_output", 0, 2, 1, 8, 16}};

  auto result = validateDistributedPlan(plan);
  assert(!result.legal);
  bool foundDup = false, foundMissing = false;
  for (const auto &r : result.rejection_reasons) {
    if (r.find("duplicate rank_id") != std::string::npos) foundDup = true;
    if (r.find("not contiguous") != std::string::npos) foundMissing = true;
  }
  assert(foundDup && foundMissing);
  std::puts("  [PASS] testInvalidRankTopologyRejection");
}

static void testInvalidCollectiveParticipantsRejection() {
  DistributedPlan plan;
  plan.world_size = 2;
  plan.tensor_parallel_size = 2;
  plan.pipeline_parallel_size = 1;
  plan.ranks = {DistributedRankPlacement{0, "simulated_cpu_process_0"},
                DistributedRankPlacement{1, "simulated_cpu_process_1"}};
  plan.tensor_shards = {
      DistributedTensorShard{"partial_output", 0, 2, 0, 0, 8},
      DistributedTensorShard{"partial_output", 0, 2, 1, 8, 16}};
  DistributedCollectiveStep step;
  step.collective_id = "all_reduce_0";
  step.sequence_id = 0;
  step.kind = "all_reduce";
  // Rank 2 does not exist, and rank 0 is duplicated.
  step.participants = {0, 0, 2};
  step.tensor_id = "partial_output";
  step.reduction = "sum";
  plan.collectives = {step};

  auto result = validateDistributedPlan(plan);
  assert(!result.legal);
  bool foundUndeclared = false, foundDup = false;
  for (const auto &r : result.rejection_reasons) {
    if (r.find("undeclared rank") != std::string::npos) foundUndeclared = true;
    if (r.find("duplicate participant") != std::string::npos) foundDup = true;
  }
  assert(foundUndeclared && foundDup);
  std::puts("  [PASS] testInvalidCollectiveParticipantsRejection");
}

static void testInvalidSequenceOrderingRejection() {
  DistributedPlan plan;
  plan.world_size = 2;
  plan.tensor_parallel_size = 2;
  plan.pipeline_parallel_size = 1;
  plan.ranks = {DistributedRankPlacement{0, "simulated_cpu_process_0"},
                DistributedRankPlacement{1, "simulated_cpu_process_1"}};
  plan.tensor_shards = {
      DistributedTensorShard{"partial_output", 0, 2, 0, 0, 8},
      DistributedTensorShard{"partial_output", 0, 2, 1, 8, 16}};

  DistributedCollectiveStep a;
  a.collective_id = "all_reduce_0";
  a.sequence_id = 0;
  a.kind = "all_reduce";
  a.participants = {0, 1};
  a.tensor_id = "partial_output";
  a.reduction = "sum";

  DistributedCollectiveStep b = a;
  b.collective_id = "all_reduce_1";
  b.sequence_id = 2;  // gap: 1 is skipped

  plan.collectives = {a, b};
  auto result = validateDistributedPlan(plan);
  assert(!result.legal);
  bool found = false;
  for (const auto &r : result.rejection_reasons)
    if (r.find("sequence_ids must be 0..N-1") != std::string::npos)
      found = true;
  assert(found);
  std::puts("  [PASS] testInvalidSequenceOrderingRejection");
}

static void testInvalidShardCoverageRejection() {
  DistributedPlan plan;
  plan.world_size = 2;
  plan.tensor_parallel_size = 2;
  plan.pipeline_parallel_size = 1;
  plan.ranks = {DistributedRankPlacement{0, "simulated_cpu_process_0"},
                DistributedRankPlacement{1, "simulated_cpu_process_1"}};
  // Overlapping shards: [0,10) and [8,16) — not disjoint.
  plan.tensor_shards = {
      DistributedTensorShard{"partial_output", 0, 2, 0, 0, 10},
      DistributedTensorShard{"partial_output", 0, 2, 1, 8, 16}};

  auto result = validateDistributedPlan(plan);
  assert(!result.legal);
  bool found = false;
  for (const auto &r : result.rejection_reasons)
    if (r.find("gap or overlap") != std::string::npos)
      found = true;
  assert(found);
  std::puts("  [PASS] testInvalidShardCoverageRejection");
}

int main() {
  std::puts("DistributedPlanningTest:");
  testCandidateGeneration();
  testTP1LegalityAndBuild();
  testTP2LegalityAndExport();
  testInvalidNonDivisibleShardRejection();
  testInvalidRankTopologyRejection();
  testInvalidCollectiveParticipantsRejection();
  testInvalidSequenceOrderingRejection();
  testInvalidShardCoverageRejection();
  std::puts("DistributedPlanningTest: PASS");
  return 0;
}
