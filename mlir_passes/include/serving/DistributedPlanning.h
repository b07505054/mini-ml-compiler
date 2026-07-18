#pragma once

// DistributedPlanning — D1 compiler-side distributed candidate generation,
// legality filtering, and plan construction.
//
// Scope (D1 "Compiler-Planned TP=2 Multi-Process Simulation"): exactly two
// fixed candidates, TP1 (world_size=1) and TP2 (world_size=2,
// tensor_parallel_size=2, pipeline_parallel_size=1). There is no
// profitability selector here — see ml-graph-compiler-runtime CLAUDE.md on
// selector-v4 status; D1 explicitly does not continue or replace it.
//
// This is a standalone collector/builder in the same architectural role as
// ExecutionPlanBuilder: it does not run measured benchmarks and it does not
// claim real GPU or NCCL execution. The DistributedPlan it produces is
// consumed by heterogeneous-inference-runtime's simulated multi-process
// runtime, never by the real vLLM adapter path.

#include "serving/ExecutionPlan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::hir {

struct DistributedCandidate {
  std::string candidate_id;              // "tp1" | "tp2"
  int64_t     world_size             = 1;
  int64_t     tensor_parallel_size   = 1;
  int64_t     pipeline_parallel_size = 1;
};

// Explicit, total candidate generation: always exactly {TP1, TP2}. Not a
// search — D1 requires generation to be explicit, not implicit in a
// selector.
std::vector<DistributedCandidate> generateDistributedCandidates();

struct DistributedLegalityResult {
  bool                     legal = false;
  std::vector<std::string> rejection_reasons;
};

// Minimum D1 legality rules for a candidate against a concrete problem size:
//   world_size >= 1
//   tensor_parallel_size >= 1
//   pipeline_parallel_size == 1                      (D1 scope)
//   world_size == tensor_parallel_size                (D1 scope)
//   tensor_dim_k % tensor_parallel_size == 0
DistributedLegalityResult
checkCandidateLegality(const DistributedCandidate &candidate,
                        int64_t tensor_dim_k);

// Validates the structural invariants of an already-built DistributedPlan:
//   rank IDs unique and contiguous (0..world_size-1)
//   collective participants are a subset of, and non-empty within, declared
//     ranks, with no duplicate participant
//   collective sequence IDs unique and strictly ordered starting at 0
//   tensor shard coverage per (tensor_id, partition_axis) is complete and
//     non-overlapping
// This is the fail-closed gate: both compiler-built plans (must pass) and
// hand-constructed malformed plans in negative tests (must fail with a
// specific reason) go through it.
DistributedLegalityResult validateDistributedPlan(const DistributedPlan &plan);

// Builds a DistributedPlan for a legal candidate. Caller must have already
// confirmed checkCandidateLegality(candidate, tensor_dim_k).legal; returns
// std::nullopt otherwise rather than emitting an invalid plan (fail closed).
// tensor_id names the sharded/all-reduced tensor — D1 uses a single matmul
// partial-output tensor (see docs: rank-local sharded matmul + all_reduce
// sum, A/B partitioned along the K contraction dimension).
std::optional<DistributedPlan>
buildDistributedPlan(const DistributedCandidate &candidate,
                     int64_t tensor_dim_k, const std::string &tensor_id);

// ---------------------------------------------------------------------------
// D2: Qwen-derived legality and cost evidence.
//
// Extends (does not replace) the D1 candidate/legality/build functions above
// so DistributedStrategyPlanningPass reuses the same structures rather than
// standing up a second planning path (see D2 Part D/E).
// ---------------------------------------------------------------------------

// Real operator metadata read from the annotated Qwen MLIR module by
// DistributedStrategyPlanningPass. hidden_dim/hidden_dim_is_static come from
// the operator's actual (possibly dynamic) tensor type, not an assumed
// constant.
struct QwenOperatorContext {
  std::string operator_id;     // e.g. "qwen_prefill::llm.o_proj::layer_0"
  std::string operator_type;   // e.g. "llm.o_proj"
  std::string function_name;
  int64_t     layer_index = -1;
  int64_t     hidden_dim = 0;              // 0 == unknown
  bool        hidden_dim_is_static = false;
  int64_t     num_attention_heads = 0;
  int64_t     num_kv_heads = 0;
  bool        distributed_capability_available = false;  // profile opt-in
};

// D2's narrow operator allow-list. Only operators in this list are ever
// considered for TP partitioning; every other Qwen operator stays TP1 /
// unpartitioned, never silently assumed tensor-parallel.
bool isSupportedDistributedOperatorType(const std::string &operator_type);

struct QwenLegalityDetail {
  std::string rule;
  std::string status;   // "pass" | "fail" | "not_applicable"
  std::string detail;
};

struct QwenDistributedLegalityResult {
  bool                             legal = false;
  std::vector<QwenLegalityDetail>  rule_results;
  std::vector<std::string>         rejection_reasons;
};

// D2 Part D legality: supported operator type, tensor/hidden dimension
// divisibility, head-count divisibility (reported not_applicable for
// row-parallel-on-hidden-dim operators like o_proj), rank count
// consistency, partition-axis support, required collective support, tensor
// shape availability, static-vs-dynamic shape handling, and distributed
// runtime capability availability (explicit profile opt-in).
QwenDistributedLegalityResult
checkQwenCandidateLegality(const DistributedCandidate &candidate,
                           const QwenOperatorContext &ctx);

// D2 Part E: explicit, inspectable analytical cost estimate. Communication-
// byte and process-launch-overhead terms are calibrated from D1's measured
// local-IPC benchmark medians (results/runtime_paths/
// distributed_d1_tp2_multiprocess/ipc_benchmark.json), not GPU-measured and
// not NCCL-calibrated -- see DistributedCostEstimate::truth_boundary.
struct DistributedCostEstimate {
  int64_t     rank_local_compute_bytes = 0;
  int64_t     estimated_communication_bytes = 0;
  int64_t     collective_count = 0;
  int64_t     process_launch_overhead_penalty = 0;
  int64_t     unsupported_operation_penalty = 0;
  int64_t     fallback_penalty = 0;
  int64_t     total_score = 0;   // lower is better; explicit sum of the above
  std::string truth_boundary;
};

DistributedCostEstimate
estimateDistributedCost(const DistributedCandidate &candidate,
                        const QwenOperatorContext &ctx);

} // namespace mlir::hir
