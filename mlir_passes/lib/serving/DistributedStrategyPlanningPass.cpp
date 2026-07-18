// DistributedStrategyPlanningPass — D2: generate, evaluate, and select a
// TP1/TP2 distributed strategy candidate for one real Qwen operator
// instance, as part of the normal compile-for-target pipeline.
//
// Module-scoped (mirrors QuantizationPlanningPass's Pass<..., ModuleOp>
// precedent), because a distributed strategy is a single whole-model
// decision, not a per-function one. See include/FusionPasses.td for the
// full design description and mlir_passes/include/serving/
// DistributedPlanning.h for the reused D1 candidate/legality/build
// functions and the D2 Qwen-aware legality/cost extensions.

#include "FusionPasses.h"
#include "serving/DistributedPlanning.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

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

static DictionaryAttr
encodeCandidateEvidence(MLIRContext *ctx, const DistributedCandidate &c,
                        const QwenDistributedLegalityResult &legality,
                        const DistributedCostEstimate &cost,
                        const QwenOperatorContext &opCtx) {
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
  add("estimated_rank_local_compute",
      IntegerAttr::get(IntegerType::get(ctx, 64), cost.rank_local_compute_bytes));
  add("legality_status", StringAttr::get(ctx, legality.legal ? "legal" : "illegal"));
  add("rejection_reasons", stringArrayAttr(ctx, legality.rejection_reasons));
  add("selection_score", IntegerAttr::get(IntegerType::get(ctx, 64), cost.total_score));
  add("truth_boundary", StringAttr::get(ctx, cost.truth_boundary));

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
    SmallVector<Attribute> candidateEvidence;
    const DistributedCandidate *tp1 = nullptr;
    const DistributedCandidate *tp2 = nullptr;
    QwenDistributedLegalityResult tp1Legality, tp2Legality;
    DistributedCostEstimate tp1Cost, tp2Cost;

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
      candidateEvidence.push_back(encodeCandidateEvidence(ctx, c, legality, cost, opCtx));

      if (c.candidate_id == "tp1") { tp1 = &c; tp1Legality = legality; tp1Cost = cost; }
      if (c.candidate_id == "tp2") { tp2 = &c; tp2Legality = legality; tp2Cost = cost; }
    }
    module->setAttr("distributed.candidates", ArrayAttr::get(ctx, candidateEvidence));

    if (!tp1 || !tp2) {
      module.emitError("DistributedStrategyPlanningPass: expected exactly the "
                       "D1 {tp1, tp2} candidate set; internal inconsistency");
      signalPassFailure();
      return;
    }

    const DistributedCandidate *selected = tp1;
    std::string selectionReason;
    const std::string kPolicyId = "d2_explicit_opt_in_v1";

    if (tp2Legality.legal && opCtx.distributed_capability_available) {
      selected = tp2;
      selectionReason = "legal_tp2_explicit_opt_in_profile";
    } else if (tp2Legality.legal && !opCtx.distributed_capability_available) {
      selected = tp1;
      selectionReason = "tp2_legal_but_opt_in_not_set";
    } else {
      selected = tp1;
      const bool onlyCapabilityMissing =
          tp2Legality.rejection_reasons.size() == 1 &&
          tp2Legality.rejection_reasons[0].find("opt in") != std::string::npos;
      selectionReason = onlyCapabilityMissing ? "no_distributed_capability"
                                              : "tp2_illegal_candidate_rejected";
    }

    module->setAttr("distributed.selected_candidate_id",
                    StringAttr::get(ctx, selected->candidate_id));
    module->setAttr("distributed.selection_reason", StringAttr::get(ctx, selectionReason));
    module->setAttr("distributed.policy_id", StringAttr::get(ctx, kPolicyId));
    module->setAttr(
        "distributed.policy_truth_boundary",
        StringAttr::get(ctx,
                        "d2_qwen_pipeline_structural_planning_not_measured_gpu_"
                        "performance_not_nccl_calibrated_not_distributed_"
                        "profitability_claim"));

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
