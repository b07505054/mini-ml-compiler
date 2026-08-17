#include "serving/DistributedPlanning.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace mlir::hir {

std::vector<DistributedCandidate> generateDistributedCandidates() {
  return {
      DistributedCandidate{"tp1", /*world_size=*/1, /*tensor_parallel_size=*/1,
                            /*pipeline_parallel_size=*/1},
      DistributedCandidate{"tp2", /*world_size=*/2, /*tensor_parallel_size=*/2,
                            /*pipeline_parallel_size=*/1},
  };
}

DistributedLegalityResult
checkCandidateLegality(const DistributedCandidate &candidate,
                        int64_t tensor_dim_k) {
  DistributedLegalityResult result;
  if (candidate.world_size < 1)
    result.rejection_reasons.push_back("world_size must be >= 1");
  if (candidate.tensor_parallel_size < 1)
    result.rejection_reasons.push_back("tensor_parallel_size must be >= 1");
  if (candidate.pipeline_parallel_size != 1)
    result.rejection_reasons.push_back(
        "pipeline_parallel_size must be == 1 for D1");
  if (candidate.world_size != candidate.tensor_parallel_size)
    result.rejection_reasons.push_back(
        "world_size must equal tensor_parallel_size for D1");
  if (candidate.tensor_parallel_size >= 1 &&
      (tensor_dim_k % candidate.tensor_parallel_size) != 0)
    result.rejection_reasons.push_back(
        "tensor dimension is not divisible by tensor_parallel_size");
  result.legal = result.rejection_reasons.empty();
  return result;
}

DistributedLegalityResult validateDistributedPlan(const DistributedPlan &plan) {
  DistributedLegalityResult result;
  auto &reasons = result.rejection_reasons;

  // Rank IDs unique and contiguous (0..world_size-1).
  std::set<int64_t> rankIds;
  for (const auto &r : plan.ranks) {
    if (!rankIds.insert(r.rank_id).second)
      reasons.push_back("duplicate rank_id: " + std::to_string(r.rank_id));
  }
  if (static_cast<int64_t>(plan.ranks.size()) != plan.world_size)
    reasons.push_back("rank count does not match world_size");
  for (int64_t expected = 0; expected < plan.world_size; ++expected) {
    if (!rankIds.count(expected))
      reasons.push_back("rank ids are not contiguous from 0: missing " +
                         std::to_string(expected));
  }
  for (const auto &id : rankIds) {
    if (id < 0 || id >= plan.world_size)
      reasons.push_back("rank_id out of contiguous [0,world_size) range: " +
                         std::to_string(id));
  }

  // Collective sequence IDs unique and strictly ordered starting at 0;
  // participants must be a non-empty, duplicate-free subset of declared ranks.
  std::set<int64_t> seenSequenceIds;
  std::vector<int64_t> sortedSequenceIds;
  for (const auto &c : plan.collectives) {
    if (!seenSequenceIds.insert(c.sequence_id).second)
      reasons.push_back("duplicate collective sequence_id: " +
                         std::to_string(c.sequence_id));
    sortedSequenceIds.push_back(c.sequence_id);

    std::set<int64_t> participantSet;
    for (int64_t p : c.participants) {
      if (!participantSet.insert(p).second)
        reasons.push_back("duplicate participant in collective " +
                           c.collective_id + ": rank " + std::to_string(p));
      if (!rankIds.count(p))
        reasons.push_back("collective " + c.collective_id +
                           " references undeclared rank " +
                           std::to_string(p));
    }
    if (c.participants.empty())
      reasons.push_back("collective " + c.collective_id +
                         " has no participants");
    if (c.kind != "all_reduce")
      reasons.push_back("collective " + c.collective_id +
                         " has unsupported kind: " + c.kind);
  }
  {
    std::vector<int64_t> ordered(sortedSequenceIds);
    std::sort(ordered.begin(), ordered.end());
    for (size_t i = 0; i < ordered.size(); ++i) {
      if (ordered[i] != static_cast<int64_t>(i)) {
        reasons.push_back(
            "collective sequence_ids must be 0..N-1 with no gaps");
        break;
      }
    }
    if (ordered != sortedSequenceIds && !ordered.empty())
      reasons.push_back(
          "collective steps must be declared in ascending sequence_id order");
  }

  // Tensor shard coverage per (tensor_id, partition_axis): complete and
  // non-overlapping.
  std::set<std::pair<std::string, int64_t>> shardGroups;
  for (const auto &s : plan.tensor_shards)
    shardGroups.insert({s.tensor_id, s.partition_axis});
  for (const auto &group : shardGroups) {
    std::vector<const DistributedTensorShard *> shards;
    for (const auto &s : plan.tensor_shards)
      if (s.tensor_id == group.first && s.partition_axis == group.second)
        shards.push_back(&s);
    std::sort(shards.begin(), shards.end(),
              [](const auto *a, const auto *b) {
                return a->range_start < b->range_start;
              });
    int64_t expectedStart = shards.empty() ? 0 : shards.front()->range_start;
    if (!shards.empty() && shards.front()->range_start != 0)
      reasons.push_back("shard coverage for tensor " + group.first +
                         " does not start at 0");
    for (const auto *s : shards) {
      if (s->range_start >= s->range_end)
        reasons.push_back("shard " + s->tensor_id +
                           " has empty or inverted range");
      if (s->range_start != expectedStart)
        reasons.push_back("shard coverage for tensor " + group.first +
                           " has a gap or overlap at offset " +
                           std::to_string(s->range_start));
      expectedStart = s->range_end;
    }
    if (static_cast<int64_t>(shards.size()) != plan.world_size)
      reasons.push_back("shard count for tensor " + group.first +
                         " does not match world_size");
  }

  result.legal = reasons.empty();
  return result;
}

std::optional<DistributedPlan>
buildDistributedPlan(const DistributedCandidate &candidate,
                     int64_t tensor_dim_k, const std::string &tensor_id) {
  if (!checkCandidateLegality(candidate, tensor_dim_k).legal)
    return std::nullopt;

  DistributedPlan plan;
  plan.strategy = candidate.world_size > 1 ? "tensor_parallel" : "none";
  plan.world_size = candidate.world_size;
  plan.tensor_parallel_size = candidate.tensor_parallel_size;
  plan.pipeline_parallel_size = candidate.pipeline_parallel_size;
  plan.truth_boundary =
      "d1_simulated_localhost_multiprocess_ipc_not_real_gpu_not_nccl_not_"
      "measured_gpu_performance";

  const int64_t chunk = tensor_dim_k / candidate.world_size;
  for (int64_t r = 0; r < candidate.world_size; ++r) {
    plan.ranks.push_back(DistributedRankPlacement{
        r, "simulated_cpu_process_" + std::to_string(r)});
    DistributedTensorShard shard;
    shard.tensor_id = tensor_id;
    shard.partition_axis = 0;  // K contraction dimension
    shard.partition_count = candidate.world_size;
    shard.shard_index = r;
    shard.range_start = r * chunk;
    shard.range_end = (r + 1) * chunk;
    plan.tensor_shards.push_back(shard);
  }

  if (candidate.world_size > 1) {
    DistributedCollectiveStep step;
    step.collective_id = "all_reduce_0";
    step.sequence_id = 0;
    step.kind = "all_reduce";
    for (int64_t r = 0; r < candidate.world_size; ++r)
      step.participants.push_back(r);
    step.tensor_id = tensor_id;
    step.reduction = "sum";
    plan.collectives.push_back(step);
  }

  return plan;
}

// ---------------------------------------------------------------------------
// D2 additions
// ---------------------------------------------------------------------------

bool isSupportedDistributedOperatorType(const std::string &operator_type) {
  // D2 narrow scope: exactly one operator family, matching real vLLM/
  // Megatron row-parallel o_proj (a.k.a. dense/down projection) semantics --
  // the narrowest operator that produces defensible TP2 semantics per D2
  // Part D.
  static const std::set<std::string> kSupported = {"llm.o_proj"};
  return kSupported.count(operator_type) != 0;
}

QwenDistributedLegalityResult
checkQwenCandidateLegality(const DistributedCandidate &candidate,
                           const QwenOperatorContext &ctx) {
  QwenDistributedLegalityResult result;
  auto &rules = result.rule_results;
  auto &reasons = result.rejection_reasons;

  const bool opSupported = isSupportedDistributedOperatorType(ctx.operator_type);
  rules.push_back({"supported_operator_type", opSupported ? "pass" : "fail",
                    ctx.operator_type});
  if (!opSupported)
    reasons.push_back("unsupported operator type: " + ctx.operator_type);

  rules.push_back({"static_shape_availability",
                    ctx.hidden_dim_is_static ? "pass" : "fail",
                    ctx.hidden_dim_is_static ? std::to_string(ctx.hidden_dim)
                                              : "dynamic"});
  if (!ctx.hidden_dim_is_static)
    reasons.push_back("hidden dimension is dynamic; cannot statically shard");

  const bool divisible = ctx.hidden_dim_is_static &&
      candidate.tensor_parallel_size > 0 &&
      (ctx.hidden_dim % candidate.tensor_parallel_size) == 0;
  rules.push_back({"tensor_hidden_dimension_divisibility",
                    divisible ? "pass" : "fail",
                    "hidden_dim=" + std::to_string(ctx.hidden_dim) +
                        " tensor_parallel_size=" +
                        std::to_string(candidate.tensor_parallel_size)});
  if (ctx.hidden_dim_is_static && !divisible)
    reasons.push_back("hidden dimension not divisible by tensor_parallel_size");

  // o_proj partitions hidden_size directly along the contraction (K) axis,
  // not per-attention-head -- head-count divisibility does not apply to
  // this operator's TP semantics. Recorded explicitly, never silently
  // skipped.
  rules.push_back({"head_count_divisibility", "not_applicable",
                    "o_proj partitions hidden_size directly, not per-head"});

  const bool rankConsistent = candidate.world_size == candidate.tensor_parallel_size;
  rules.push_back({"rank_count_consistency", rankConsistent ? "pass" : "fail",
                    ""});
  if (!rankConsistent)
    reasons.push_back("world_size does not equal tensor_parallel_size");

  rules.push_back({"partition_axis_support", "pass",
                    "axis=0 (K / hidden contraction dimension)"});

  rules.push_back({"required_collective_support", "pass",
                    "all_reduce(sum) is implemented by the D1 runtime"});

  rules.push_back({"tensor_shape_availability",
                    ctx.hidden_dim_is_static ? "pass" : "fail", ""});

  rules.push_back({"static_vs_dynamic_shape_handling", "pass",
                    "batch/sequence dimension may remain dynamic; only the "
                    "partitioned hidden dimension must be statically known"});

  rules.push_back({"runtime_capability_availability",
                    ctx.distributed_capability_available ? "pass" : "fail",
                    "requires target profile distributedStrategyOptIn=true"});
  if (!ctx.distributed_capability_available)
    reasons.push_back("distributed capability not declared (profile did not opt in)");

  if (candidate.pipeline_parallel_size != 1)
    reasons.push_back("pipeline_parallel_size must be == 1 for D2");
  if (candidate.world_size < 1)
    reasons.push_back("world_size must be >= 1");

  result.legal = reasons.empty();
  return result;
}

DistributedCostEstimate
estimateDistributedCost(const DistributedCandidate &candidate,
                        const QwenOperatorContext &ctx) {
  DistributedCostEstimate est;
  est.truth_boundary =
      "analytical_and_d1_local_ipc_calibrated_not_gpu_measured_not_nccl_"
      "calibrated_not_multi_gpu_latency_predictor";

  constexpr int64_t kDtypeBytes = 2;  // f16, matches the real Qwen graph dtype
  const int64_t hidden = ctx.hidden_dim_is_static ? ctx.hidden_dim : 0;
  const int64_t worldSize = candidate.world_size > 0 ? candidate.world_size : 1;
  const int64_t perRankHidden = hidden / worldSize;

  // Rank-local compute proxy: bytes of the K-slice this rank touches. Actual
  // FLOPs are not estimated because the batch/sequence (M) dimension is
  // dynamic; this mirrors shape_cost_model_v2's byte-based fallback for
  // operators without a fully static shape.
  est.rank_local_compute_bytes = perRankHidden * kDtypeBytes;

  // Communication-byte estimate: D1's central-coordinator all_reduce moves
  // every rank's full hidden-size contribution to the coordinator, then
  // broadcasts the reduced result back to every rank.
  est.estimated_communication_bytes =
      worldSize > 1 ? worldSize * hidden * kDtypeBytes * 2 : 0;

  est.collective_count = worldSize > 1 ? 1 : 0;

  // Process-launch / distributed-overhead penalty, calibrated from D1's
  // measured results/runtime_paths/distributed_d1_tp2_multiprocess/
  // ipc_benchmark.json world_size=2 process_startup_s median (~0.00624s,
  // i.e. ~3.1ms/rank) converted to nanoseconds.
  constexpr int64_t kD1MeasuredProcessStartupNanosPerRank = 3'100'000;
  est.process_launch_overhead_penalty =
      worldSize > 1 ? worldSize * kD1MeasuredProcessStartupNanosPerRank : 0;

  est.unsupported_operation_penalty =
      isSupportedDistributedOperatorType(ctx.operator_type) ? 0 : 1'000'000'000;
  est.fallback_penalty = 0;

  est.total_score = est.rank_local_compute_bytes +
      est.estimated_communication_bytes + est.collective_count * 1000 +
      est.process_launch_overhead_penalty + est.unsupported_operation_penalty +
      est.fallback_penalty;
  return est;
}

// ---------------------------------------------------------------------------
// D6: distributed_profitability_contract_v1
// ---------------------------------------------------------------------------

namespace {
constexpr double kKvCacheDtypeBytes = 2.0;  // fp16, matches tp_cost_model.BYTES_PER_PARAM_FP16
constexpr double kBytesPerMb = 1024.0 * 1024.0;
}  // namespace

double distributedKvCacheBytesPerTokenPerGpu(const DistributedModelProfile &model,
                                             int64_t tensor_parallel_size) {
  if (tensor_parallel_size <= 0 || model.num_attention_heads <= 0)
    return 0.0;
  const double headDim =
      static_cast<double>(model.hidden_size) / static_cast<double>(model.num_attention_heads);
  const double kvHeadsPerGpu =
      static_cast<double>(model.num_kv_heads) / static_cast<double>(tensor_parallel_size);
  // 2 (K & V) * num_layers * kv_heads_per_gpu * head_dim * dtype_bytes.
  return 2.0 * static_cast<double>(model.num_layers) * kvHeadsPerGpu * headDim * kKvCacheDtypeBytes;
}

double distributedPerGpuWeightMb(const DistributedModelProfile &model,
                                 int64_t tensor_parallel_size) {
  if (tensor_parallel_size <= 0)
    return model.weight_footprint_mb;
  return model.weight_footprint_mb / static_cast<double>(tensor_parallel_size);
}

DistributedCommunicationPrediction
estimateNcclCommunicationTimeUs(int64_t estimated_communication_bytes,
                                const std::string &collective_kind,
                                const DistributedCommunicationCalibration &calibration) {
  DistributedCommunicationPrediction pred;
  if (estimated_communication_bytes < 0) {
    pred.failure_reason = "negative_communication_bytes";
    return pred;
  }
  if (estimated_communication_bytes == 0) {
    pred.valid = true;
    pred.time_us = 0.0;
    return pred;
  }
  if (!calibration.valid) {
    pred.failure_reason = "communication_calibration_missing_or_invalid";
    return pred;
  }
  if (collective_kind != calibration.collective_kind) {
    pred.failure_reason = "communication_collective_kind_mismatch";
    return pred;
  }
  if (calibration.topology_class != "PHB" || calibration.p2p_available ||
      calibration.nccl_transport != "SHM/direct/direct") {
    pred.failure_reason = "communication_topology_mismatch";
    return pred;
  }
  if (calibration.predictor_kind == "alpha_beta") {
    pred.valid = true;
    pred.time_us = calibration.alpha_us +
        calibration.beta_us_per_byte * static_cast<double>(estimated_communication_bytes);
    return pred;
  }
  if (calibration.predictor_kind != "log_size_piecewise_interpolation") {
    pred.failure_reason = "unknown_communication_predictor_kind";
    return pred;
  }
  if (calibration.points.size() < 2) {
    pred.failure_reason = "communication_predictor_has_insufficient_points";
    return pred;
  }
  std::vector<DistributedCommunicationPoint> points = calibration.points;
  std::sort(points.begin(), points.end(), [](const auto &a, const auto &b) {
    return a.bytes < b.bytes;
  });
  for (size_t i = 1; i < points.size(); ++i) {
    if (points[i - 1].bytes >= points[i].bytes) {
      pred.failure_reason = "communication_predictor_points_not_strictly_increasing";
      return pred;
    }
  }
  if (estimated_communication_bytes < points.front().bytes ||
      estimated_communication_bytes > points.back().bytes) {
    pred.failure_reason = "communication_bytes_outside_calibrated_range";
    return pred;
  }
  for (const auto &p : points) {
    if (p.bytes == estimated_communication_bytes) {
      pred.valid = true;
      pred.time_us = p.time_us;
      return pred;
    }
  }
  for (size_t i = 0; i + 1 < points.size(); ++i) {
    const auto &left = points[i];
    const auto &right = points[i + 1];
    if (left.bytes <= estimated_communication_bytes &&
        estimated_communication_bytes <= right.bytes) {
      const double x = std::log2(static_cast<double>(estimated_communication_bytes));
      const double x0 = std::log2(static_cast<double>(left.bytes));
      const double x1 = std::log2(static_cast<double>(right.bytes));
      const double frac = (x - x0) / (x1 - x0);
      pred.valid = true;
      pred.time_us = left.time_us + frac * (right.time_us - left.time_us);
      return pred;
    }
  }
  pred.failure_reason = "communication_interpolation_failed";
  return pred;
}

namespace {

int64_t bytesPerCollectiveCall(const DistributedCandidate &candidate,
                                  const DistributedModelProfile &model) {
  if (candidate.world_size <= 1)
    return 0;
  constexpr int64_t kDtypeBytes = 2;
  return candidate.world_size * model.hidden_size * kDtypeBytes * 2;
}

int64_t estimatedCollectiveCallCount(const DistributedCandidate &candidate,
                                     const DistributedModelProfile &model,
                                     const DistributedWorkloadProfile &workload) {
  if (candidate.world_size <= 1)
    return 0;
  return std::max<int64_t>(1, model.num_layers) * std::max<int64_t>(1, workload.concurrency);
}

double throughputToLatencyUs(double predictedTokensPerSecond) {
  if (predictedTokensPerSecond <= 0.0)
    return INFINITY;
  return 1'000'000.0 / predictedTokensPerSecond;
}

}  // namespace

DistributedProfitabilityEstimate
estimateDistributedProfitability(const DistributedCandidate &candidate,
                                 const DistributedModelProfile &model,
                                 const DistributedWorkloadProfile &workload,
                                 const DistributedProfitabilityCalibration &calibration) {
  DistributedProfitabilityEstimate est;
  est.truth_boundary =
      "linear_regression_calibrated_from_real_d5_measured_throughput_plus_"
      "phase1_nccl_tests_measured_communication_cost_on_2x_rtx4090_phb_"
      "p2p_unavailable_shm_not_a_full_systems_simulator_not_valid_off_"
      "calibration_hardware";

  if (!model.available || !calibration.valid) {
    est.computed = false;
    est.infeasibility_reason = !model.available
        ? "model_profile_unavailable"
        : "calibration_unavailable_or_invalid_version_or_missing_communication_profile";
    return est;
  }
  est.computed = true;

  const int64_t tp = candidate.tensor_parallel_size > 0 ? candidate.tensor_parallel_size : 1;
  const double kvCacheBytesPerTokenPerGpu = distributedKvCacheBytesPerTokenPerGpu(model, tp);
  const double perGpuWeightMb = distributedPerGpuWeightMb(model, tp);

  // Hard memory-feasibility gate: worst-case max_num_seqs x max_model_len
  // KV reservation plus the per-GPU weight shard must fit within the
  // declared per-GPU memory budget. Matches tp_cost_model.is_feasible
  // exactly. This is independent of, and takes priority over, the
  // throughput objective below.
  est.memory_budget_mb = calibration.gpu_memory_mb_per_device * calibration.gpu_memory_utilization;
  const double worstCaseKvMb =
      (static_cast<double>(workload.max_num_seqs) * static_cast<double>(workload.max_model_len) *
       kvCacheBytesPerTokenPerGpu) / kBytesPerMb;
  est.required_memory_mb = perGpuWeightMb + worstCaseKvMb;
  est.feasible = est.required_memory_mb <= est.memory_budget_mb;
  if (!est.feasible) {
    est.infeasibility_reason = "required_memory_mb_exceeds_budget";
    return est;
  }

  const auto &c = (tp == 1) ? calibration.tp1_coefficients : calibration.tp2_coefficients;
  const double kvCacheKbPerTokenPerGpu = kvCacheBytesPerTokenPerGpu / 1024.0;
  est.predicted_throughput_before_communication_tokens_per_s =
      c.intercept + c.per_gpu_weight_mb * perGpuWeightMb +
      c.kv_cache_kb_per_token_per_gpu * kvCacheKbPerTokenPerGpu +
      c.gpu_count * static_cast<double>(tp) +
      c.input_length * static_cast<double>(workload.input_tokens) +
      c.output_length * static_cast<double>(workload.output_tokens) +
      c.concurrency * static_cast<double>(workload.concurrency);
  est.bytes_per_collective_call = bytesPerCollectiveCall(candidate, model);
  est.estimated_collective_call_count = estimatedCollectiveCallCount(candidate, model, workload);
  est.estimated_communication_bytes =
      est.bytes_per_collective_call * est.estimated_collective_call_count;
  est.communication_collective_kind =
      est.estimated_collective_call_count > 0 ? "all_reduce" : "none";
  est.communication_profile_id = calibration.communication.profile_id;
  est.communication_predictor_kind = calibration.communication.predictor_kind;
  est.topology_class = calibration.communication.topology_class;
  est.p2p_available = calibration.communication.p2p_available;
  est.nccl_transport = calibration.communication.nccl_transport;
  auto comm = estimateNcclCommunicationTimeUs(
      est.bytes_per_collective_call,
      est.estimated_collective_call_count > 0 ? "all_reduce" : calibration.communication.collective_kind,
      calibration.communication);
  if (!comm.valid) {
    est.computed = false;
    est.infeasibility_reason = comm.failure_reason;
    return est;
  }
  est.estimated_nccl_comm_time_us = comm.time_us * static_cast<double>(est.estimated_collective_call_count);
  est.estimated_communication_penalty_us = est.estimated_nccl_comm_time_us;
  est.estimated_runtime_residual_us = calibration.d9_runtime_residual_us;
  est.decision_margin_us = calibration.d9_decision_margin_us;
  est.overlap_assumption = calibration.d9_overlap_assumption;
  est.compute_reference_weight_mb = calibration.d9_compute_reference_weight_mb;
  est.compute_savings_us_per_weight_mb_above_reference =
      calibration.d9_compute_savings_us_per_weight_mb_above_reference;
  const double computeLatencyUs = throughputToLatencyUs(
      est.predicted_throughput_before_communication_tokens_per_s);
  const double adjustedLatencyUs = computeLatencyUs + est.estimated_communication_penalty_us +
      est.estimated_runtime_residual_us;
  est.predicted_throughput_tokens_per_s =
      adjustedLatencyUs > 0.0 ? 1'000'000.0 / adjustedLatencyUs : 0.0;
  return est;
}

} // namespace mlir::hir
