// DistributedStrategyPlanningPass — D2/D6: generate, evaluate, and select a
// TP1/TP2 distributed strategy candidate for one real Qwen operator
// instance, as part of the normal compile-for-target pipeline.
//
// Module-scoped (mirrors QuantizationPlanningPass's Pass<..., ModuleOp>
// precedent), because a distributed strategy is a single whole-model
// decision, not a per-function one. See include/FusionPasses.td for the
// full design description and mlir_passes/include/serving/
// DistributedPlanning.h for the reused D1 candidate/legality/build
// functions, the D2 Qwen-aware legality/cost extensions, and the D6
// distributed_profitability_contract_v1 whole-model profitability
// estimator.
//
// D6 selection mechanism (replaces D2's "legal && distributed.opt_in =>
// TP2"): distributed.opt_in now only widens the *candidate space* under
// consideration -- when unset, TP2 is excluded from consideration
// entirely and this is recorded explicitly as
// "tp2_excluded_opt_in_not_set" (never silently, never conflated with
// TP2 being illegal). When set, both legal candidates are evaluated with
// estimateDistributedProfitability (calibrated, real-D5-measured,
// whole-model throughput prediction) and the higher-predicted-throughput,
// memory-feasible candidate wins; a deterministic tie-break prefers the
// lower TP degree. Missing or version-mismatched calibration is a
// conservative fallback to TP1, never a silent TP2 selection and never a
// pipeline crash. See docs/DISTRIBUTED_D6_COMPILER_OWNED_TP_SELECTION.md
// for the full contract.

#include "FusionPasses.h"
#include "serving/DistributedPlanning.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_DISTRIBUTEDSTRATEGYPLANNING
#include "FusionPasses.h.inc"

using llvm::SmallVector;

// ---------------------------------------------------------------------------
// Attribute encoding helpers
// ---------------------------------------------------------------------------

static ArrayAttr stringArrayAttr(MLIRContext *ctx,
                                  const std::vector<std::string> &values) {
  SmallVector<Attribute> attrs;
  for (const auto &v : values)
    attrs.push_back(StringAttr::get(ctx, v));
  return ArrayAttr::get(ctx, attrs);
}

static ArrayAttr int64ArrayAttr(MLIRContext *ctx,
                                 const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  for (int64_t v : values)
    attrs.push_back(IntegerAttr::get(IntegerType::get(ctx, 64), v));
  return ArrayAttr::get(ctx, attrs);
}

// ---------------------------------------------------------------------------
// D6: model/workload/calibration module-attr readers
//
// These read the real per-model, per-workload, and per-target-profile
// facts DistributedStrategyPlanningPass needs for
// estimateDistributedProfitability. All three are additive: absent in any
// pre-D6 module (no --model-profile/--workload-profile passed, no
// distributedProfitability block in the target profile), and their
// absence degrades to the documented conservative fallback (see
// runOnOperation), never a crash and never a silent TP2 selection.
// ---------------------------------------------------------------------------

static DistributedModelProfile readModelProfile(ModuleOp module) {
  DistributedModelProfile mp;
  auto layers = module->getAttrOfType<IntegerAttr>("llm.num_layers");
  auto hidden = module->getAttrOfType<IntegerAttr>("llm.hidden_size");
  auto heads = module->getAttrOfType<IntegerAttr>("llm.num_attention_heads");
  auto kvHeads = module->getAttrOfType<IntegerAttr>("llm.num_key_value_heads");
  auto weightMb = module->getAttrOfType<FloatAttr>("distributed.model.weight_footprint_mb");
  if (layers) mp.num_layers = layers.getInt();
  if (hidden) mp.hidden_size = hidden.getInt();
  if (heads) mp.num_attention_heads = heads.getInt();
  if (kvHeads) mp.num_kv_heads = kvHeads.getInt();
  if (weightMb) mp.weight_footprint_mb = weightMb.getValueAsDouble();
  mp.available = layers && hidden && heads && kvHeads && weightMb;
  return mp;
}

static DistributedWorkloadProfile readWorkloadProfile(ModuleOp module) {
  DistributedWorkloadProfile wp;
  auto inTok = module->getAttrOfType<IntegerAttr>("distributed.workload.input_tokens");
  auto outTok = module->getAttrOfType<IntegerAttr>("distributed.workload.output_tokens");
  auto conc = module->getAttrOfType<IntegerAttr>("distributed.workload.concurrency");
  auto maxLen = module->getAttrOfType<IntegerAttr>("distributed.workload.max_model_len");
  auto maxSeqs = module->getAttrOfType<IntegerAttr>("distributed.workload.max_num_seqs");
  if (inTok) wp.input_tokens = inTok.getInt();
  if (outTok) wp.output_tokens = outTok.getInt();
  if (conc) wp.concurrency = conc.getInt();
  if (maxLen) wp.max_model_len = maxLen.getInt();
  if (maxSeqs) wp.max_num_seqs = maxSeqs.getInt();
  wp.declared = inTok && outTok && conc;
  return wp;
}

static DistributedThroughputCoefficients readCoefficients(DictionaryAttr d) {
  DistributedThroughputCoefficients c;
  if (!d) return c;
  auto get = [&](StringRef key) -> double {
    if (auto a = d.getAs<FloatAttr>(key)) return a.getValueAsDouble();
    return 0.0;
  };
  c.intercept = get("intercept");
  c.per_gpu_weight_mb = get("per_gpu_weight_mb");
  c.kv_cache_kb_per_token_per_gpu = get("kv_cache_kb_per_token_per_gpu");
  c.gpu_count = get("gpu_count");
  c.input_length = get("input_length");
  c.output_length = get("output_length");
  c.concurrency = get("concurrency");
  return c;
}

static DistributedProfitabilityCalibration readCalibration(ModuleOp module) {
  DistributedProfitabilityCalibration cal;
  auto version = module->getAttrOfType<StringAttr>("distributed.profitability.contract_version");
  auto gpuMemMb = module->getAttrOfType<FloatAttr>("distributed.profitability.gpu_memory_mb_per_device");
  auto gpuUtil = module->getAttrOfType<FloatAttr>("distributed.profitability.gpu_memory_utilization");
  auto tp1Coef = module->getAttrOfType<DictionaryAttr>("distributed.profitability.tp1_coefficients");
  auto tp2Coef = module->getAttrOfType<DictionaryAttr>("distributed.profitability.tp2_coefficients");
  auto commPoints = module->getAttrOfType<ArrayAttr>("distributed.profitability.communication.points");

  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.calibration_dataset_hash"))
    cal.calibration_dataset_hash = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.calibration_hardware_identity"))
    cal.calibration_hardware_identity = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.calibration_generated_at"))
    cal.calibration_generated_at = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.calibration_compiler_commit"))
    cal.calibration_compiler_commit = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.calibration_runtime_commit"))
    cal.calibration_runtime_commit = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.tie_break_rule"))
    cal.tie_break_rule = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.truth_boundary"))
    cal.truth_boundary = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.profile_id"))
    cal.communication.profile_id = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.topology_class"))
    cal.communication.topology_class = v.getValue().str();
  if (auto v = module->getAttrOfType<BoolAttr>("distributed.profitability.communication.p2p_available"))
    cal.communication.p2p_available = v.getValue();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.nccl_transport"))
    cal.communication.nccl_transport = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.nccl_version"))
    cal.communication.nccl_version = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.nccl_tests_version"))
    cal.communication.nccl_tests_version = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.collective_kind"))
    cal.communication.collective_kind = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.predictor_kind"))
    cal.communication.predictor_kind = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.mode"))
    cal.communication.mode = v.getValue().str();
  if (auto v = module->getAttrOfType<FloatAttr>("distributed.profitability.communication.alpha_us"))
    cal.communication.alpha_us = v.getValueAsDouble();
  if (auto v = module->getAttrOfType<FloatAttr>("distributed.profitability.communication.beta_us_per_byte"))
    cal.communication.beta_us_per_byte = v.getValueAsDouble();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.source_artifact_hashes"))
    cal.communication.source_artifact_hashes = v.getValue().str();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.communication.provenance_hashes"))
    cal.communication.provenance_hashes = v.getValue().str();
  if (auto v = module->getAttrOfType<FloatAttr>("distributed.profitability.d9.decision_margin_us"))
    cal.d9_decision_margin_us = v.getValueAsDouble();
  if (auto v = module->getAttrOfType<FloatAttr>("distributed.profitability.d9.runtime_residual_us"))
    cal.d9_runtime_residual_us = v.getValueAsDouble();
  if (auto v = module->getAttrOfType<FloatAttr>("distributed.profitability.d9.compute_reference_weight_mb"))
    cal.d9_compute_reference_weight_mb = v.getValueAsDouble();
  if (auto v = module->getAttrOfType<FloatAttr>("distributed.profitability.d9.compute_savings_us_per_weight_mb_above_reference"))
    cal.d9_compute_savings_us_per_weight_mb_above_reference = v.getValueAsDouble();
  if (auto v = module->getAttrOfType<StringAttr>("distributed.profitability.d9.overlap_assumption"))
    cal.d9_overlap_assumption = v.getValue().str();

  if (version) cal.contract_version = version.getValue().str();
  if (gpuMemMb) cal.gpu_memory_mb_per_device = gpuMemMb.getValueAsDouble();
  if (gpuUtil) cal.gpu_memory_utilization = gpuUtil.getValueAsDouble();
  if (tp1Coef) cal.tp1_coefficients = readCoefficients(tp1Coef);
  if (tp2Coef) cal.tp2_coefficients = readCoefficients(tp2Coef);
  if (commPoints) {
    for (auto attr : commPoints) {
      auto d = dyn_cast<DictionaryAttr>(attr);
      if (!d) continue;
      DistributedCommunicationPoint p;
      if (auto b = d.getAs<IntegerAttr>("bytes")) p.bytes = b.getInt();
      if (auto t = d.getAs<FloatAttr>("time_us")) p.time_us = t.getValueAsDouble();
      cal.communication.points.push_back(p);
    }
  }
  cal.communication.valid =
      !cal.communication.profile_id.empty() &&
      cal.communication.topology_class == "PHB" &&
      !cal.communication.p2p_available &&
      cal.communication.nccl_transport == "SHM/direct/direct" &&
      cal.communication.collective_kind == "all_reduce" &&
      (cal.communication.predictor_kind == "log_size_piecewise_interpolation" ||
       cal.communication.predictor_kind == "alpha_beta") &&
      (cal.communication.predictor_kind == "alpha_beta" ||
       cal.communication.points.size() >= 2);

  // Fail-closed: valid only if the contract version matches exactly and
  // every required numeric block was present. A missing block or a
  // version mismatch (e.g. a stale v0 profile) must never be silently
  // treated as valid calibration.
  cal.valid = version && gpuMemMb && gpuUtil && tp1Coef && tp2Coef &&
      cal.contract_version == kDistributedProfitabilityContractVersion &&
      cal.communication.valid;
  return cal;
}

static DictionaryAttr
encodeProfitabilityEvidence(MLIRContext *ctx, const DistributedProfitabilityEstimate &p) {
  SmallVector<NamedAttribute> fields;
  auto add = [&](StringRef key, Attribute value) {
    fields.push_back({StringAttr::get(ctx, key), value});
  };
  add("computed", BoolAttr::get(ctx, p.computed));
  add("feasible", BoolAttr::get(ctx, p.feasible));
  add("predicted_throughput_tokens_per_s",
      FloatAttr::get(Float64Type::get(ctx), p.predicted_throughput_tokens_per_s));
  add("predicted_throughput_before_communication_tokens_per_s",
      FloatAttr::get(Float64Type::get(ctx),
                     p.predicted_throughput_before_communication_tokens_per_s));
  add("estimated_communication_bytes",
      IntegerAttr::get(IntegerType::get(ctx, 64), p.estimated_communication_bytes));
  add("estimated_nccl_comm_time_us",
      FloatAttr::get(Float64Type::get(ctx), p.estimated_nccl_comm_time_us));
  add("estimated_compute_savings_us",
      FloatAttr::get(Float64Type::get(ctx), p.estimated_compute_savings_us));
  add("regression_compute_savings_us",
      FloatAttr::get(Float64Type::get(ctx), p.regression_compute_savings_us));
  add("regression_compute_savings_status",
      StringAttr::get(ctx, p.regression_compute_savings_status));
  add("structural_compute_savings_adjustment_us",
      FloatAttr::get(Float64Type::get(ctx), p.structural_compute_savings_adjustment_us));
  add("compute_reference_weight_mb",
      FloatAttr::get(Float64Type::get(ctx), p.compute_reference_weight_mb));
  add("compute_savings_us_per_weight_mb_above_reference",
      FloatAttr::get(Float64Type::get(ctx), p.compute_savings_us_per_weight_mb_above_reference));
  add("estimated_communication_penalty_us",
      FloatAttr::get(Float64Type::get(ctx), p.estimated_communication_penalty_us));
  add("estimated_runtime_residual_us",
      FloatAttr::get(Float64Type::get(ctx), p.estimated_runtime_residual_us));
  add("estimated_net_tp2_benefit_us",
      FloatAttr::get(Float64Type::get(ctx), p.estimated_net_tp2_benefit_us));
  add("decision_margin_us", FloatAttr::get(Float64Type::get(ctx), p.decision_margin_us));
  add("overlap_assumption", StringAttr::get(ctx, p.overlap_assumption));
  add("estimated_collective_call_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), p.estimated_collective_call_count));
  add("bytes_per_collective_call",
      IntegerAttr::get(IntegerType::get(ctx, 64), p.bytes_per_collective_call));
  add("communication_collective_kind", StringAttr::get(ctx, p.communication_collective_kind));
  add("communication_profile_id", StringAttr::get(ctx, p.communication_profile_id));
  add("communication_predictor_kind", StringAttr::get(ctx, p.communication_predictor_kind));
  add("topology_class", StringAttr::get(ctx, p.topology_class));
  add("p2p_available", BoolAttr::get(ctx, p.p2p_available));
  add("nccl_transport", StringAttr::get(ctx, p.nccl_transport));
  add("required_memory_mb", FloatAttr::get(Float64Type::get(ctx), p.required_memory_mb));
  add("memory_budget_mb", FloatAttr::get(Float64Type::get(ctx), p.memory_budget_mb));
  add("infeasibility_reason", StringAttr::get(ctx, p.infeasibility_reason));
  add("truth_boundary", StringAttr::get(ctx, p.truth_boundary));
  return DictionaryAttr::get(ctx, fields);
}

static DictionaryAttr
encodeCandidateEvidence(MLIRContext *ctx, const DistributedCandidate &c,
                        const QwenDistributedLegalityResult &legality,
                        const DistributedCostEstimate &cost,
                        const QwenOperatorContext &opCtx,
                        const DistributedProfitabilityEstimate &profitability,
                        bool excludedFromConsideration,
                        bool communicationChangedDecision) {
  SmallVector<NamedAttribute> fields;
  auto add = [&](StringRef key, Attribute value) {
    fields.push_back({StringAttr::get(ctx, key), value});
  };

  const bool partitioned = legality.legal && c.world_size > 1;
  add("candidate_id", StringAttr::get(ctx, c.candidate_id));
  add("strategy", StringAttr::get(ctx, c.world_size > 1 ? "tensor_parallel" : "none"));
  add("world_size", IntegerAttr::get(IntegerType::get(ctx, 64), c.world_size));
  add("tensor_parallel_size",
      IntegerAttr::get(IntegerType::get(ctx, 64), c.tensor_parallel_size));
  add("pipeline_parallel_size",
      IntegerAttr::get(IntegerType::get(ctx, 64), c.pipeline_parallel_size));
  add("partitioned_operator_ids",
      stringArrayAttr(ctx, partitioned
                                ? std::vector<std::string>{opCtx.operator_id}
                                : std::vector<std::string>{}));
  add("partition_axis", IntegerAttr::get(IntegerType::get(ctx, 64), 0));
  add("shard_count", IntegerAttr::get(IntegerType::get(ctx, 64), c.world_size));
  add("required_collectives",
      stringArrayAttr(ctx, c.world_size > 1
                                ? std::vector<std::string>{"all_reduce"}
                                : std::vector<std::string>{}));
  add("estimated_communication_bytes",
      IntegerAttr::get(IntegerType::get(ctx, 64),
                       cost.estimated_communication_bytes));
  add("estimated_nccl_comm_time_us",
      FloatAttr::get(Float64Type::get(ctx), profitability.estimated_nccl_comm_time_us));
  add("estimated_compute_savings_us",
      FloatAttr::get(Float64Type::get(ctx), profitability.estimated_compute_savings_us));
  add("estimated_communication_penalty_us",
      FloatAttr::get(Float64Type::get(ctx), profitability.estimated_communication_penalty_us));
  add("estimated_runtime_residual_us",
      FloatAttr::get(Float64Type::get(ctx), profitability.estimated_runtime_residual_us));
  add("estimated_net_tp2_benefit_us",
      FloatAttr::get(Float64Type::get(ctx), profitability.estimated_net_tp2_benefit_us));
  add("estimated_collective_call_count",
      IntegerAttr::get(IntegerType::get(ctx, 64), profitability.estimated_collective_call_count));
  add("collective_kind", StringAttr::get(ctx, profitability.communication_collective_kind));
  add("bytes_per_collective_call",
      IntegerAttr::get(IntegerType::get(ctx, 64), profitability.bytes_per_collective_call));
  add("overlap_assumption", StringAttr::get(ctx, profitability.overlap_assumption));
  add("decision_margin_us", FloatAttr::get(Float64Type::get(ctx), profitability.decision_margin_us));
  add("communication_profile_id", StringAttr::get(ctx, profitability.communication_profile_id));
  add("communication_predictor_kind", StringAttr::get(ctx, profitability.communication_predictor_kind));
  add("nccl_transport", StringAttr::get(ctx, profitability.nccl_transport));
  add("p2p_available", BoolAttr::get(ctx, profitability.p2p_available));
  add("predicted_tp_throughput_before_communication",
      FloatAttr::get(Float64Type::get(ctx),
                     profitability.predicted_throughput_before_communication_tokens_per_s));
  add("predicted_tp_throughput_after_communication",
      FloatAttr::get(Float64Type::get(ctx),
                     profitability.predicted_throughput_tokens_per_s));
  add("communication_changed_tp_decision", BoolAttr::get(ctx, communicationChangedDecision));
  add("estimated_rank_local_compute",
      IntegerAttr::get(IntegerType::get(ctx, 64), cost.rank_local_compute_bytes));
  add("legality_status", StringAttr::get(ctx, legality.legal ? "legal" : "illegal"));
  add("rejection_reasons", stringArrayAttr(ctx, legality.rejection_reasons));
  add("selection_score", IntegerAttr::get(IntegerType::get(ctx, 64), cost.total_score));
  add("truth_boundary", StringAttr::get(ctx, cost.truth_boundary));
  add("excluded_from_consideration", BoolAttr::get(ctx, excludedFromConsideration));
  add("profitability", encodeProfitabilityEvidence(ctx, profitability));

  SmallVector<Attribute> ruleAttrs;
  for (const auto &r : legality.rule_results) {
    SmallVector<NamedAttribute> ruleFields;
    ruleFields.push_back({StringAttr::get(ctx, "rule"), StringAttr::get(ctx, r.rule)});
    ruleFields.push_back({StringAttr::get(ctx, "status"), StringAttr::get(ctx, r.status)});
    ruleFields.push_back({StringAttr::get(ctx, "detail"), StringAttr::get(ctx, r.detail)});
    ruleAttrs.push_back(DictionaryAttr::get(ctx, ruleFields));
  }
  add("legality_rule_results", ArrayAttr::get(ctx, ruleAttrs));

  return DictionaryAttr::get(ctx, fields);
}

static void encodeSelectedDistributedPlan(ModuleOp module, MLIRContext *ctx,
                                          const DistributedPlan &plan) {
  module->setAttr("distributed.strategy", StringAttr::get(ctx, plan.strategy));
  module->setAttr("distributed.world_size",
                  IntegerAttr::get(IntegerType::get(ctx, 64), plan.world_size));
  module->setAttr("distributed.tensor_parallel_size",
                  IntegerAttr::get(IntegerType::get(ctx, 64), plan.tensor_parallel_size));
  module->setAttr("distributed.pipeline_parallel_size",
                  IntegerAttr::get(IntegerType::get(ctx, 64), plan.pipeline_parallel_size));

  SmallVector<Attribute> rankAttrs;
  for (const auto &r : plan.ranks) {
    SmallVector<NamedAttribute> f;
    f.push_back({StringAttr::get(ctx, "rank_id"),
                IntegerAttr::get(IntegerType::get(ctx, 64), r.rank_id)});
    f.push_back({StringAttr::get(ctx, "logical_device"),
                StringAttr::get(ctx, r.logical_device)});
    rankAttrs.push_back(DictionaryAttr::get(ctx, f));
  }
  module->setAttr("distributed.ranks", ArrayAttr::get(ctx, rankAttrs));

  SmallVector<Attribute> shardAttrs;
  for (const auto &s : plan.tensor_shards) {
    SmallVector<NamedAttribute> f;
    f.push_back({StringAttr::get(ctx, "tensor_id"), StringAttr::get(ctx, s.tensor_id)});
    f.push_back({StringAttr::get(ctx, "partition_axis"),
                IntegerAttr::get(IntegerType::get(ctx, 64), s.partition_axis)});
    f.push_back({StringAttr::get(ctx, "partition_count"),
                IntegerAttr::get(IntegerType::get(ctx, 64), s.partition_count)});
    f.push_back({StringAttr::get(ctx, "shard_index"),
                IntegerAttr::get(IntegerType::get(ctx, 64), s.shard_index)});
    f.push_back({StringAttr::get(ctx, "range_start"),
                IntegerAttr::get(IntegerType::get(ctx, 64), s.range_start)});
    f.push_back({StringAttr::get(ctx, "range_end"),
                IntegerAttr::get(IntegerType::get(ctx, 64), s.range_end)});
    shardAttrs.push_back(DictionaryAttr::get(ctx, f));
  }
  module->setAttr("distributed.tensor_shards", ArrayAttr::get(ctx, shardAttrs));

  SmallVector<Attribute> collectiveAttrs;
  for (const auto &c : plan.collectives) {
    SmallVector<NamedAttribute> f;
    f.push_back({StringAttr::get(ctx, "collective_id"), StringAttr::get(ctx, c.collective_id)});
    f.push_back({StringAttr::get(ctx, "sequence_id"),
                IntegerAttr::get(IntegerType::get(ctx, 64), c.sequence_id)});
    f.push_back({StringAttr::get(ctx, "kind"), StringAttr::get(ctx, c.kind)});
    f.push_back({StringAttr::get(ctx, "participants"), int64ArrayAttr(ctx, c.participants)});
    f.push_back({StringAttr::get(ctx, "tensor_id"), StringAttr::get(ctx, c.tensor_id)});
    f.push_back({StringAttr::get(ctx, "reduction"), StringAttr::get(ctx, c.reduction)});
    collectiveAttrs.push_back(DictionaryAttr::get(ctx, f));
  }
  module->setAttr("distributed.collectives", ArrayAttr::get(ctx, collectiveAttrs));
  module->setAttr("distributed.truth_boundary", StringAttr::get(ctx, plan.truth_boundary));
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct DistributedStrategyPlanningPass
    : impl::DistributedStrategyPlanningBase<DistributedStrategyPlanningPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    QwenOperatorContext opCtx;
    if (auto a = module->getAttrOfType<IntegerAttr>("llm.num_attention_heads"))
      opCtx.num_attention_heads = a.getInt();
    if (auto a = module->getAttrOfType<IntegerAttr>("llm.num_key_value_heads"))
      opCtx.num_kv_heads = a.getInt();
    if (auto a = module->getAttrOfType<BoolAttr>("distributed.opt_in"))
      opCtx.distributed_capability_available = a.getValue();

    // Find the first supported operator instance in module walk order
    // (deterministic pre-order over regions/functions). D2 deliberately
    // partitions exactly one operator instance -- not every occurrence
    // across all layers -- see FusionPasses.td description.
    Operation *foundOp = nullptr;
    std::string foundFuncName;
    module.walk([&](func::FuncOp funcOp) {
      if (foundOp)
        return;
      funcOp.walk([&](Operation *op) {
        if (foundOp)
          return;
        if (isSupportedDistributedOperatorType(op->getName().getStringRef().str())) {
          foundOp = op;
          foundFuncName = funcOp.getName().str();
        }
      });
    });

    if (foundOp) {
      opCtx.operator_type = foundOp->getName().getStringRef().str();
      opCtx.function_name = foundFuncName;
      if (auto a = foundOp->getAttrOfType<IntegerAttr>("serving.layer_index"))
        opCtx.layer_index = a.getInt();
      opCtx.operator_id = foundFuncName + "::" + opCtx.operator_type + "::layer_" +
          std::to_string(opCtx.layer_index);

      // Real operator metadata: the op's own result tensor type is the
      // source of truth for the partitioned (trailing) dimension, not an
      // assumed constant. The leading (batch/sequence) dimension may stay
      // dynamic; only the trailing hidden dimension must be static to shard.
      if (foundOp->getNumResults() > 0) {
        if (auto rt = dyn_cast<RankedTensorType>(foundOp->getResult(0).getType())) {
          if (rt.getRank() > 0) {
            int64_t trailing = rt.getShape().back();
            if (!ShapedType::isDynamic(trailing)) {
              opCtx.hidden_dim = trailing;
              opCtx.hidden_dim_is_static = true;
            }
          }
        }
      }
    } else {
      opCtx.operator_type = "none";
      opCtx.operator_id = "none";
    }

    auto candidates = generateDistributedCandidates();
    const DistributedModelProfile modelProfile = readModelProfile(module);
    const DistributedWorkloadProfile workloadProfile = readWorkloadProfile(module);
    const DistributedProfitabilityCalibration calibration = readCalibration(module);
    const bool optIn = opCtx.distributed_capability_available;

    SmallVector<Attribute> candidateEvidence;
    const DistributedCandidate *tp1 = nullptr;
    const DistributedCandidate *tp2 = nullptr;
    QwenDistributedLegalityResult tp1Legality, tp2Legality;
    DistributedCostEstimate tp1Cost, tp2Cost;
    DistributedProfitabilityEstimate tp1Profit, tp2Profit;

    for (const auto &c : candidates) {
      QwenDistributedLegalityResult legality;
      if (c.world_size <= 1) {
        // TP1 requires no operator partitioning and no distributed
        // capability -- it must remain selectable for any graph/profile
        // (D2 Part B requirement 7).
        legality.legal = true;
        legality.rule_results.push_back(
            {"tp1_requires_no_partitioning", "not_applicable",
             "world_size=1 candidates bypass operator/capability legality checks"});
      } else {
        legality = checkQwenCandidateLegality(c, opCtx);
      }
      DistributedCostEstimate cost = estimateDistributedCost(c, opCtx);
      // D6: only candidates the compiler is actually considering (opt_in
      // widens the space to include TP2; a legality failure never widens
      // it) get a real profitability estimate computed -- an excluded or
      // illegal TP2 still gets legality/cost evidence (transparency), but
      // never a throughput prediction that could be mistaken for having
      // influenced the decision.
      const bool consideredForProfitability =
          c.world_size <= 1 || (optIn && legality.legal);
      DistributedProfitabilityEstimate profit;
      if (consideredForProfitability) {
        profit = estimateDistributedProfitability(c, modelProfile, workloadProfile, calibration);
      } else {
        profit.computed = false;
        profit.infeasibility_reason =
            !optIn ? "excluded_opt_in_not_set" : "excluded_illegal_candidate";
      }
      if (c.candidate_id == "tp1") { tp1 = &c; tp1Legality = legality; tp1Cost = cost; tp1Profit = profit; }
      if (c.candidate_id == "tp2") { tp2 = &c; tp2Legality = legality; tp2Cost = cost; tp2Profit = profit; }
    }

    if (!tp1 || !tp2) {
      module.emitError("DistributedStrategyPlanningPass: expected exactly the "
                       "D1 {tp1, tp2} candidate set; internal inconsistency");
      signalPassFailure();
      return;
    }

    // ------------------------------------------------------------------
    // D6 profitability_contract_v1 selection.
    //
    // distributed.opt_in widens the *candidate space* (whether TP2 is
    // considered at all); it never itself picks the winner. Once TP2 is
    // both legal and under consideration, the candidate with the higher
    // calibrated predicted throughput wins, subject to a hard memory-
    // feasibility gate that takes priority over the throughput objective.
    // ------------------------------------------------------------------
    constexpr double kTieBreakEpsilonTokensPerSec = 1e-6;
    const DistributedCandidate *selected = tp1;
    const DistributedCandidate *preCommunicationSelected = tp1;
    std::string selectionReason;
    const std::string kPolicyId = kD9BreakEvenPolicyId;

    if (!optIn) {
      selected = tp1;
      selectionReason = tp2Legality.legal ? "tp2_excluded_opt_in_not_set"
                                          : "tp2_illegal_candidate_rejected";
    } else if (!tp2Legality.legal) {
      selected = tp1;
      const bool onlyCapabilityMissing =
          tp2Legality.rejection_reasons.size() == 1 &&
          tp2Legality.rejection_reasons[0].find("opt in") != std::string::npos;
      selectionReason = onlyCapabilityMissing ? "no_distributed_capability"
                                              : "tp2_illegal_candidate_rejected";
    } else if (!tp1Profit.computed || !tp2Profit.computed) {
      // Calibration or model/workload inputs unavailable -- conservative,
      // explicit fallback. Never a pipeline crash, never a silent TP2
      // pick: the reason is recorded and is distinguishable from every
      // profitability-based reason below.
      selected = tp1;
      selectionReason = "conservative_fallback_missing_or_invalid_calibration";
    } else if (!tp1Profit.feasible && !tp2Profit.feasible) {
      // Neither candidate fits the declared memory budget: fail closed
      // rather than silently emitting an unserviceable plan.
      module.emitError("DistributedStrategyPlanningPass: neither TP1 nor TP2 "
                       "fits the declared memory budget for this model/"
                       "workload; failing closed rather than emitting an "
                       "unserviceable plan");
      signalPassFailure();
      return;
    } else if (!tp1Profit.feasible && tp2Profit.feasible) {
      selected = tp2;
      selectionReason = "capacity_forced_tp1_infeasible";
    } else if (tp1Profit.feasible && !tp2Profit.feasible) {
      selected = tp1;
      selectionReason = "capacity_forced_tp2_infeasible";
    } else {
      const double tp1ComputeLatencyUs =
          tp1Profit.predicted_throughput_before_communication_tokens_per_s > 0.0
              ? 1'000'000.0 / tp1Profit.predicted_throughput_before_communication_tokens_per_s
              : INFINITY;
      const double tp2ComputeLatencyUs =
          tp2Profit.predicted_throughput_before_communication_tokens_per_s > 0.0
              ? 1'000'000.0 / tp2Profit.predicted_throughput_before_communication_tokens_per_s
              : INFINITY;
      double regressionComputeSavingsUs = tp1ComputeLatencyUs - tp2ComputeLatencyUs;
      std::string regressionComputeSavingsStatus = "finite";
      if (!std::isfinite(regressionComputeSavingsUs)) {
        regressionComputeSavingsUs = 0.0;
        regressionComputeSavingsStatus = "non_finite_regression_latency_delta_ignored";
      }
      const double structuralComputeSavingsAdjustmentUs =
          std::max(0.0, modelProfile.weight_footprint_mb -
                         calibration.d9_compute_reference_weight_mb) *
          calibration.d9_compute_savings_us_per_weight_mb_above_reference;
      const double computeSavingsUs = regressionComputeSavingsUs +
          structuralComputeSavingsAdjustmentUs;
      for (auto *profit : {&tp1Profit, &tp2Profit}) {
        profit->regression_compute_savings_us = regressionComputeSavingsUs;
        profit->regression_compute_savings_status = regressionComputeSavingsStatus;
        profit->structural_compute_savings_adjustment_us =
            structuralComputeSavingsAdjustmentUs;
        profit->compute_reference_weight_mb = calibration.d9_compute_reference_weight_mb;
        profit->compute_savings_us_per_weight_mb_above_reference =
            calibration.d9_compute_savings_us_per_weight_mb_above_reference;
        profit->estimated_compute_savings_us = computeSavingsUs;
      }
      tp2Profit.estimated_net_tp2_benefit_us = computeSavingsUs -
          tp2Profit.estimated_communication_penalty_us - tp2Profit.estimated_runtime_residual_us;
      tp1Profit.estimated_net_tp2_benefit_us = tp2Profit.estimated_net_tp2_benefit_us;

      preCommunicationSelected = computeSavingsUs > calibration.d9_decision_margin_us ? tp2 : tp1;
      if (tp2Profit.estimated_net_tp2_benefit_us > calibration.d9_decision_margin_us) {
        selected = tp2;
        selectionReason = "d9_profitable_tp2_selected_net_benefit_positive";
      } else {
        selected = tp1;
        selectionReason = "d9_profitable_tp1_selected_net_benefit_below_margin";
      }
    }

    const bool communicationChangedDecision =
        preCommunicationSelected->candidate_id != selected->candidate_id;
    candidateEvidence.clear();
    for (const auto &c : candidates) {
      const bool excluded = c.world_size > 1 && !optIn;
      const auto &legality = c.candidate_id == "tp1" ? tp1Legality : tp2Legality;
      const auto &cost = c.candidate_id == "tp1" ? tp1Cost : tp2Cost;
      const auto &profit = c.candidate_id == "tp1" ? tp1Profit : tp2Profit;
      candidateEvidence.push_back(
          encodeCandidateEvidence(ctx, c, legality, cost, opCtx, profit, excluded,
                                  communicationChangedDecision));
    }
    module->setAttr("distributed.candidates", ArrayAttr::get(ctx, candidateEvidence));

    module->setAttr("distributed.selected_candidate_id",
                    StringAttr::get(ctx, selected->candidate_id));
    module->setAttr("distributed.pre_communication_selected_candidate_id",
                    StringAttr::get(ctx, preCommunicationSelected->candidate_id));
    module->setAttr("distributed.communication_changed_decision",
                    BoolAttr::get(ctx, communicationChangedDecision));
    module->setAttr("distributed.selection_reason", StringAttr::get(ctx, selectionReason));
    module->setAttr("distributed.policy_id", StringAttr::get(ctx, kPolicyId));
    module->setAttr(
        "distributed.policy_truth_boundary",
        StringAttr::get(ctx,
                        "d9_qwen_pipeline_break_even_tp_selection_"
                        "calibrated_from_real_d5_measured_2x_rtx4090_throughput_"
                        "and_phase1_nccl_tests_measured_communication_cost_"
                        "for_2x_rtx4090_phb_p2p_unavailable_shm_not_a_runtime_"
                        "measured_guarantee_for_this_specific_compile"));

    if (selected->world_size > 1) {
      // Fail-closed consistency check: never emit a distributed plan for an
      // internally-inconsistent or illegal selection.
      if (!tp2Legality.legal) {
        module.emitError("DistributedStrategyPlanningPass: selected TP2 but "
                         "legality evidence marks it illegal; refusing to "
                         "emit an inconsistent distributed plan");
        signalPassFailure();
        return;
      }
      auto plan = buildDistributedPlan(*selected, opCtx.hidden_dim, opCtx.operator_id);
      if (!plan) {
        module.emitError("DistributedStrategyPlanningPass: buildDistributedPlan "
                         "failed for a candidate marked legal; internal "
                         "inconsistency, failing closed");
        signalPassFailure();
        return;
      }
      auto structural = validateDistributedPlan(*plan);
      if (!structural.legal) {
        module.emitError("DistributedStrategyPlanningPass: built plan failed "
                         "structural validation; failing closed");
        signalPassFailure();
        return;
      }
      encodeSelectedDistributedPlan(module, ctx, *plan);
    }
    // TP1 selection: no distributed.strategy/world_size/... attrs are set,
    // preserving D1's "absent entirely for legacy/TP1 plans" contract.
  }
};

} // namespace

std::unique_ptr<Pass> createDistributedStrategyPlanningPass() {
  return std::make_unique<DistributedStrategyPlanningPass>();
}

} // namespace mlir::hir
