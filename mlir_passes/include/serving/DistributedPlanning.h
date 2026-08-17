#pragma once

// DistributedPlanning — D1 compiler-side distributed candidate generation,
// legality filtering, and plan construction.
//
// Scope (D1 "Compiler-Planned TP=2 Multi-Process Simulation"): exactly two
// fixed candidates, TP1 (world_size=1) and TP2 (world_size=2,
// tensor_parallel_size=2, pipeline_parallel_size=1). D1 itself still has no
// profitability selector -- see ml-graph-compiler-runtime CLAUDE.md on
// selector-v4 status; D1 explicitly does not continue or replace it.
//
// This is a standalone collector/builder in the same architectural role as
// ExecutionPlanBuilder: it does not run measured benchmarks and it does not
// claim real GPU or NCCL execution. The DistributedPlan it produces is
// consumed by heterogeneous-inference-runtime's simulated multi-process
// runtime, never by the real vLLM adapter path.
//
// D6 update: a real profitability selector now exists one layer up, in
// DistributedStrategyPlanningPass (see estimateDistributedProfitability
// below and DistributedStrategyPlanningPass.cpp). It compares calibrated,
// predicted whole-model throughput for TP1 vs TP2 -- built from real D5
// measurements on a real 2xRTX4090 host -- and is the mechanism that now
// decides which of the two D1 candidates above gets selected when
// distributed.opt_in is set. D1's own generateDistributedCandidates()/
// buildDistributedPlan() remain profitability-free by design (D1's scope
// note above is unchanged); the new selector consumes their output, it
// does not modify them.

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

// ---------------------------------------------------------------------------
// D6: distributed_profitability_contract_v1
//
// A real, whole-model-scope profitability comparison between the TP1 and
// TP2 candidates above, distinct from D2's single-operator
// DistributedCostEstimate (which proxies one o_proj instance's rank-local
// bytes/communication-bytes and is not comparable across TP degrees in
// consistent units -- see estimateDistributedCost's own truth_boundary).
//
// This contract's inputs and coefficient layout are a direct C++
// reimplementation of heterogeneous-inference-runtime's
// deployment/vllm_adapter/tp_cost_model.py (the D5 Python reference/oracle
// model), so a compiler-side decision and the D5 offline regression can be
// numerically cross-checked against each other. Units are declared
// per-field below; the compiler never sums heterogeneous units into one
// undifferentiated score the way D2's total_score does.
// ---------------------------------------------------------------------------

// Real per-model facts needed for whole-model profitability, beyond the
// single-operator QwenOperatorContext above. num_layers/hidden_size/
// num_attention_heads/num_kv_heads are read from the same real module
// attrs ExecutionPlanBuilder::collectModelIdentity already consumes
// (llm.num_layers, llm.hidden_size, llm.num_attention_heads,
// llm.num_key_value_heads) -- never re-derived or guessed here.
// weight_footprint_mb is deployment-time information (the real cached
// checkpoint's measured size in MB, matching heterogeneous-inference-
// runtime's distributed_materializer._estimate_model_footprint_mb()
// exactly) and is supplied via --model-profile, since it is not a graph
// structural fact the MLIR frontend can derive.
struct DistributedModelProfile {
  int64_t num_layers = 0;
  int64_t hidden_size = 0;
  int64_t num_attention_heads = 0;
  int64_t num_kv_heads = 0;
  double  weight_footprint_mb = 0.0;
  bool    available = false;  // true only if every field above was populated
};

// Real, pre-execution workload-shape facts. Supplied via --workload-profile;
// when absent, a conservative declared default is used and
// WorkloadProfile::declared stays false (recorded in evidence, never
// silently treated as if a real workload were declared).
struct DistributedWorkloadProfile {
  int64_t input_tokens = 32;
  int64_t output_tokens = 32;
  int64_t concurrency = 1;
  int64_t max_model_len = 2048;
  int64_t max_num_seqs = 4;
  bool    declared = false;
};

// A single TP degree's linear throughput-prediction coefficients, in the
// exact feature order tp_cost_model.py's FEATURE_NAMES uses:
// [intercept, per_gpu_weight_mb, kv_cache_kb_per_token_per_gpu, gpu_count,
//  input_length, output_length, concurrency]. Units: the predicted value
// is tokens/second (aggregate throughput); per_gpu_weight_mb is megabytes;
// kv_cache_kb_per_token_per_gpu is kilobytes/token; gpu_count/input_length/
// output_length/concurrency are dimensionless counts.
struct DistributedThroughputCoefficients {
  double intercept = 0.0;
  double per_gpu_weight_mb = 0.0;
  double kv_cache_kb_per_token_per_gpu = 0.0;
  double gpu_count = 0.0;
  double input_length = 0.0;
  double output_length = 0.0;
  double concurrency = 0.0;
};

struct DistributedCommunicationPoint {
  int64_t bytes = 0;
  double  time_us = 0.0;
};

struct DistributedCommunicationPrediction {
  bool        valid = false;
  double      time_us = 0.0;
  std::string failure_reason;
};

struct DistributedCommunicationCalibration {
  std::string profile_id;
  std::string topology_class;
  bool        p2p_available = false;
  std::string nccl_transport;
  std::string nccl_version;
  std::string nccl_tests_version;
  std::string collective_kind = "all_reduce";
  std::string predictor_kind;
  std::string mode = "out_of_place";
  double      alpha_us = 0.0;
  double      beta_us_per_byte = 0.0;
  std::vector<DistributedCommunicationPoint> points;
  std::string source_artifact_hashes;
  std::string provenance_hashes;
  bool        valid = false;
};

// Calibration parameters read from a target profile's declared
// distributedProfitability block (never from a raw benchmark file -- see
// Part D). `valid` is false (fail-closed) unless contract_version matches
// exactly and every required numeric field was present.
struct DistributedProfitabilityCalibration {
  std::string contract_version;
  std::string calibration_dataset_hash;
  std::string calibration_hardware_identity;
  std::string calibration_generated_at;
  std::string calibration_compiler_commit;
  std::string calibration_runtime_commit;
  double      gpu_memory_mb_per_device = 0.0;
  double      gpu_memory_utilization = 0.0;
  DistributedThroughputCoefficients tp1_coefficients;
  DistributedThroughputCoefficients tp2_coefficients;
  DistributedCommunicationCalibration communication;
  double      d9_decision_margin_us = 250.0;
  double      d9_runtime_residual_us = 0.0;
  double      d9_compute_reference_weight_mb = 1454.3235168457031;
  double      d9_compute_savings_us_per_weight_mb_above_reference = 0.50;
  std::string d9_overlap_assumption = "zero";
  std::string tie_break_rule;
  std::string truth_boundary;
  bool        valid = false;
};

constexpr const char *kDistributedProfitabilityContractVersion =
    "distributed_profitability_contract_v1";
constexpr const char *kD9BreakEvenPolicyId = "d9_break_even_tp_selector_v1";
constexpr double kD9DecisionMarginUs = 250.0;
constexpr double kD9RuntimeResidualUs = 0.0;
constexpr double kD9ComputeReferenceWeightMb = 1454.3235168457031;
constexpr double kD9ComputeSavingsUsPerWeightMbAboveReference = 0.50;

// Real-unit, comparable-across-TP-degrees profitability estimate for one
// candidate. `predicted_throughput_tokens_per_s` is the sole objective the
// selector compares (higher is better); `required_memory_mb` and
// `feasible` are the hard memory-capacity gate, evaluated independently of
// the throughput objective (a candidate that is not feasible is never
// selected regardless of predicted throughput).
struct DistributedProfitabilityEstimate {
  bool        computed = false;   // false if calibration/model/workload inputs were unavailable
  bool        feasible = false;
  double      predicted_throughput_tokens_per_s = 0.0;  // after D9 communication adjustment
  double      predicted_throughput_before_communication_tokens_per_s = 0.0;
  double      estimated_nccl_comm_time_us = 0.0;
  int64_t     estimated_communication_bytes = 0;
  int64_t     estimated_collective_call_count = 0;
  int64_t     bytes_per_collective_call = 0;
  std::string communication_collective_kind;
  std::string communication_profile_id;
  std::string communication_predictor_kind;
  std::string topology_class;
  bool        p2p_available = false;
  std::string nccl_transport;
  std::string overlap_assumption = "zero";
  double      regression_compute_savings_us = 0.0;
  double      structural_compute_savings_adjustment_us = 0.0;
  double      compute_reference_weight_mb = 1454.3235168457031;
  double      compute_savings_us_per_weight_mb_above_reference = 0.50;
  std::string regression_compute_savings_status = "finite";
  double      estimated_compute_savings_us = 0.0;
  double      estimated_communication_penalty_us = 0.0;
  double      estimated_runtime_residual_us = 0.0;
  double      estimated_net_tp2_benefit_us = 0.0;
  double      decision_margin_us = 250.0;
  double      required_memory_mb = 0.0;
  double      memory_budget_mb = 0.0;
  std::string infeasibility_reason;
  std::string truth_boundary;
};

DistributedCommunicationPrediction
estimateNcclCommunicationTimeUs(int64_t estimated_communication_bytes,
                                const std::string &collective_kind,
                                const DistributedCommunicationCalibration &calibration);

// Real-unit KV-cache-bytes-per-token-per-GPU, matching
// tp_cost_model.kv_cache_bytes_per_token_per_gpu exactly: vLLM's
// tensor-parallel attention shards KV heads across ranks, so each GPU
// stores only 1/tensor_parallel_size of a sequence's KV cache.
double distributedKvCacheBytesPerTokenPerGpu(const DistributedModelProfile &model,
                                             int64_t tensor_parallel_size);

// Real per-GPU weight footprint under this TP degree (weight_footprint_mb
// divided across ranks -- an approximation of vLLM's actual per-rank
// shard size, declared as such, matching
// tp_cost_model.per_gpu_weight_mb exactly).
double distributedPerGpuWeightMb(const DistributedModelProfile &model,
                                 int64_t tensor_parallel_size);

// The D6 profitability contract's core evaluation: hard memory feasibility
// (matching tp_cost_model.is_feasible's worst-case max_num_seqs x
// max_model_len KV reservation) plus, only if feasible and calibration is
// valid, the calibrated linear throughput prediction for this TP degree.
DistributedProfitabilityEstimate
estimateDistributedProfitability(const DistributedCandidate &candidate,
                                 const DistributedModelProfile &model,
                                 const DistributedWorkloadProfile &workload,
                                 const DistributedProfitabilityCalibration &calibration);

} // namespace mlir::hir
