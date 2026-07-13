// ServingCostModelPass — upgraded CandidateEvaluationPass.
//
// Applies ServingStaticCostModel_v1 to each compiler.candidates candidate dict
// and emits three layers of output per evaluated candidate:
//
//   Layer 1 (V0 backward-compat): evaluation.penalty_score, evaluation.status,
//     evaluation.reason, evaluation.truth_boundary.
//     Values and semantics are unchanged from CandidateEvaluationPass so that
//     PlanSelectionPass ranking and FileCheck tests remain green.
//
//   Layer 2 (V1 structured evidence): evaluation.cost.{compute,memory,dequant,
//     requant,layout_transform,cast,backend_switch,launch_overhead,kv_cache,
//     transfer,unsupported,total,model_id,truth_boundary}.
//     total == exact sum of all component fields (verifiable by the builder).
//
//   Layer 3 (V2 shape-aware evidence, shape_cost_model_v2): for supported op
//     kinds (matmul_like / normalization / elementwise) with fully static
//     shapes, evaluation.shape_cost.{flops_estimate, input_bytes_estimate,
//     output_bytes_estimate, weight_bytes_estimate,
//     total_memory_bytes_estimate, arithmetic_intensity_milli,
//     estimated_{compute,memory,boundary,total}_cost_nanos, status,
//     model_version, truth_boundary}. Bytes are dtype-aware (candidate dtype
//     -> quant.activation_dtype -> result element type ->
//     representation.effective_dtype; weight dtype from quant.weight_dtype).
//     Time estimates appear only when target.static_cost_profile.* module
//     attrs declare peak FLOPs / bandwidth — declared theoretical numbers,
//     never measured. Unknown op kinds and dynamic shapes fall back to the
//     V1 fixed model, honestly recorded in compiler.shape_profile.status.
//
//   Ranking: evaluation.penalty_score stays the V0 heuristic by default. When
//     the module opts in with serving.cost_model.mode = "shape_aware_v2" AND
//     every evaluated candidate of an op has a time estimate, penalty_score
//     becomes estimated_total_cost_nanos (V0 preserved as
//     evaluation.penalty_score_v0). PlanSelection tiering (fallback last,
//     unsupported never) is unaffected because tier precedes penalty.
//
// Pass registration keeps the name "candidate-evaluation" so that the MLIR
// pipeline string, FileCheck tests, and ServingPipeline.cpp are unaffected.
// The C++ factory is createServingCostModelPass() (new) with a backward-compat
// alias createCandidateEvaluationPass() declared in FusionPasses.h.

#include "serving/OpShapeFacts.h"
#include "serving/ImplementationCandidate.h"
#include "serving/ServingCostModel.h"
#include "serving/ShapeCostModel.h"
#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CANDIDATEEVALUATION
#include "FusionPasses.h.inc"

// ---------------------------------------------------------------------------
// V0 backward-compat penalty constants (kept for evaluation.penalty_score).
// These values are preserved to avoid changing PlanSelectionPass ranking
// and to keep FileCheck tests in candidate_evaluation.mlir green.
// ---------------------------------------------------------------------------
static constexpr int64_t kV0PenaltyDirectLower     = 0;
static constexpr int64_t kV0PenaltyDecomposition   = 5;
static constexpr int64_t kV0PenaltyReprConvBase    = 3;
static constexpr int64_t kV0PenaltyBoundaryDequant = 3;
static constexpr int64_t kV0PenaltyBoundaryCast    = 2;
static constexpr int64_t kV0PenaltyBoundaryLayout  = 2;
static constexpr int64_t kV0PenaltyLayoutConv      = 4;
static constexpr int64_t kV0PenaltyCastConv        = 2;
static constexpr int64_t kV0PenaltyBackendFallback = 20;
static constexpr int64_t kV0PenaltyUnsupported     = 100;
static constexpr int64_t kV0PenaltyUnknownCost     = 10;

// V0 truth_boundary (kept for evaluation.truth_boundary to minimize diff
// against tests that may check this field).
static constexpr StringLiteral kV0Truth =
    "candidate_evaluation_static_penalty_not_measured_latency";

// ---------------------------------------------------------------------------
// V2 shape-fact helpers (shape_cost_model_v2)
// ---------------------------------------------------------------------------
// Shape-fact derivation (classifyOpKind, computeShapeFacts, readProfileNums)
// lives in serving/OpShapeFacts.h, shared with TilePlanningPass.

// Resolve the activation/output dtype for one candidate.
// Priority: candidate "dtype" key -> cast target (effective dtype) ->
// op-level resolution (quant.activation_dtype -> result element type ->
// effective dtype).
static std::string resolveActivationDtype(DictionaryAttr dict,
                                          const std::string& candType,
                                          Operation& op,
                                          const std::string& effectiveDtype) {
  std::string d = getCandidateString(dict, "dtype");
  if (dtypeBits(d) > 0) return d;
  if (candType == "cast_conversion" && dtypeBits(effectiveDtype) > 0)
    return effectiveDtype;
  return resolveOpActivationDtype(op, effectiveDtype);
}

// ---------------------------------------------------------------------------
// V0 penalty evaluation — preserved for evaluation.penalty_score backward compat.
// Returns {penalty_score, status, reason}.
// ---------------------------------------------------------------------------
struct V0Eval {
  int64_t penaltyScore;
  std::string status;
  std::string reason;
};

static V0Eval evaluatePenaltyV0(const std::string& candType,
                                  const std::vector<std::string>& boundaryOps) {
  int64_t directLower = 0, decomposition = 0, reprConv = 0;
  int64_t layoutConv = 0, castConv = 0, backendFallback = 0;
  int64_t unsupported = 0, unknownCost = 0;
  std::string status = "evaluated";
  std::string reason;

  if (candType == "direct_lower") {
    directLower = kV0PenaltyDirectLower;
    reason = "exact_kernel_zero_penalty";
  } else if (candType == "algebraic_decomposition") {
    decomposition = kV0PenaltyDecomposition;
    reason = "decomposition_base_penalty";
  } else if (candType == "representation_conversion") {
    reprConv = kV0PenaltyReprConvBase;
    reason = "representation_conversion_base_penalty";
    for (const auto& b : boundaryOps) {
      if (b == "dequant_weight" || b == "dequant") {
        reprConv += kV0PenaltyBoundaryDequant;
        reason += "+dequant_boundary";
      } else if (b == "cast") {
        reprConv += kV0PenaltyBoundaryCast;
        reason += "+cast_boundary";
      } else if (b == "layout_transform") {
        reprConv += kV0PenaltyBoundaryLayout;
        reason += "+layout_boundary";
      } else {
        unknownCost += kV0PenaltyUnknownCost;
        status = "partially_evaluated";
        reason += "+unknown_boundary";
      }
    }
  } else if (candType == "layout_conversion") {
    layoutConv = kV0PenaltyLayoutConv;
    reason = "layout_conversion_penalty";
  } else if (candType == "cast_conversion") {
    castConv = kV0PenaltyCastConv;
    reason = "cast_conversion_penalty";
  } else if (candType == "backend_fallback") {
    backendFallback = kV0PenaltyBackendFallback;
    reason = "backend_fallback_high_penalty";
  } else if (candType == "unsupported") {
    unsupported = kV0PenaltyUnsupported;
    status = "rejected";
    reason = "no_viable_lowering_path";
  } else {
    unknownCost = kV0PenaltyUnknownCost;
    status = "partially_evaluated";
    reason = "unknown_candidate_type";
  }

  int64_t total = directLower + decomposition + reprConv + layoutConv
                + castConv + backendFallback + unsupported + unknownCost;
  return {total, status, reason};
}

// ---------------------------------------------------------------------------
// Evaluate one candidate dict (two-phase: evaluate, then assemble, so the
// shape_aware_v2 ranking mode can be decided per op across all candidates).
// ---------------------------------------------------------------------------
struct CandidateEval {
  ImplementationCandidate candidate;
  DictionaryAttr dict;      // original candidate fields
  V0Eval v0;
  DecisionCost v1;
  bool hasShapeCost = false;
  ShapeCostEstimateResult shape;
};

// Assemble the augmented DictionaryAttr with evaluation.penalty_score (V0 or
// V2 per useV2Penalty), evaluation.status/reason/truth_boundary,
// evaluation.cost.* (V1), and evaluation.shape_cost.* (V2, when computed).
static DictionaryAttr assembleCandidate(MLIRContext* ctx,
                                        const CandidateEval& ce,
                                        bool useV2Penalty) {
  const V0Eval& v0 = ce.v0;
  const DecisionCost& v1 = ce.v1;

  auto I64 = [&](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(ctx, 64), v);
  };
  auto S = [&](StringRef s) -> Attribute { return StringAttr::get(ctx, s); };

  // Copy original candidate fields, normalize canonical candidate fields, and
  // append evaluation.* fields.
  DictionaryAttr canonicalBase =
      encodeImplementationCandidate(ctx, ce.candidate, ce.dict);
  SmallVector<NamedAttribute> augmented(canonicalBase.begin(),
                                        canonicalBase.end());

  // Ranking score: V0 heuristic by default. In shape_aware_v2 mode (only
  // when every evaluated candidate of the op has a time estimate), evaluated
  // candidates are ranked by estimated_total_cost_nanos instead; the V0
  // score is preserved as evaluation.penalty_score_v0.
  int64_t penalty = v0.penaltyScore;
  if (useV2Penalty && v0.status == "evaluated" && ce.hasShapeCost &&
      ce.shape.has_time_estimates) {
    penalty = ce.shape.estimated_total_cost_nanos;
    augmented.push_back({StringAttr::get(ctx, "evaluation.penalty_score_v0"),
                         I64(v0.penaltyScore)});
  }

  // V0 backward-compat attrs.
  augmented.push_back({StringAttr::get(ctx, "evaluation.penalty_score"), I64(penalty)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.reason"),        S(v0.reason)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.status"),        S(v0.status)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.truth_boundary"),S(kV0Truth)});

  // V1 structured cost attrs.
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.backend_switch"),  I64(v1.backend_switch_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.cast"),            I64(v1.cast_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.compute"),         I64(v1.compute_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.dequant"),         I64(v1.dequant_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.kv_cache"),        I64(v1.kv_cache_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.launch_overhead"), I64(v1.launch_overhead_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.layout_transform"),I64(v1.layout_transform_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.memory"),          I64(v1.memory_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.model_id"),        S(v1.cost_model_id)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.requant"),         I64(v1.requant_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.total"),           I64(v1.total_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.transfer"),        I64(v1.transfer_cost)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.truth_boundary"),  S(v1.truth_boundary)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.cost.unsupported"),     I64(v1.unsupported_penalty)});

  // V2 shape-aware attrs (emitted only when shape facts were usable).
  if (ce.hasShapeCost) {
    const ShapeCostEstimateResult& sc = ce.shape;
    auto add = [&](StringRef key, Attribute a) {
      augmented.push_back({StringAttr::get(ctx, key), a});
    };
    add("evaluation.shape_cost.arithmetic_intensity_milli",
        I64(sc.arithmetic_intensity_milli));
    add("evaluation.shape_cost.flops_estimate", I64(sc.flops_estimate));
    add("evaluation.shape_cost.input_bytes_estimate", I64(sc.input_bytes));
    add("evaluation.shape_cost.model_version", S(kShapeCostModelVersion));
    add("evaluation.shape_cost.output_bytes_estimate", I64(sc.output_bytes));
    add("evaluation.shape_cost.status", S(sc.status));
    add("evaluation.shape_cost.total_memory_bytes_estimate",
        I64(sc.total_memory_bytes));
    add("evaluation.shape_cost.truth_boundary", S(kShapeCostTruthBoundary));
    add("evaluation.shape_cost.weight_bytes_estimate", I64(sc.weight_bytes));
    if (sc.has_time_estimates) {
      add("evaluation.shape_cost.estimated_boundary_cost_nanos",
          I64(sc.estimated_boundary_cost_nanos));
      add("evaluation.shape_cost.estimated_compute_cost_nanos",
          I64(sc.estimated_compute_cost_nanos));
      add("evaluation.shape_cost.estimated_memory_cost_nanos",
          I64(sc.estimated_memory_cost_nanos));
      add("evaluation.shape_cost.estimated_total_cost_nanos",
          I64(sc.estimated_total_cost_nanos));
    }
  }

  return DictionaryAttr::get(ctx, augmented);
}

// ---------------------------------------------------------------------------
// Pass struct — renamed to ServingCostModelPass.
// Still uses CandidateEvaluationBase to keep MLIR registration name
// "candidate-evaluation" and avoid breaking FileCheck tests / pipeline strings.
// ---------------------------------------------------------------------------
struct ServingCostModelPass
    : impl::CandidateEvaluationBase<ServingCostModelPass> {
  explicit ServingCostModelPass(StaticCostWeights w = StaticCostWeights{})
      : weights_(std::move(w)) {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = funcOp.getContext();
    ServingCostModel model(weights_);

    Operation* module = funcOp->getParentOp();
    StaticCostProfileNums profileNums = readProfileNums(module);

    // shape_aware_v2 ranking is opt-in via a module attr; default keeps the
    // V0 heuristic score untouched.
    bool v2ModeRequested = false;
    if (module)
      if (auto a = module->getAttrOfType<StringAttr>("serving.cost_model.mode"))
        v2ModeRequested = a.getValue() == "shape_aware_v2";

    std::string effectiveDtype;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.effective_dtype"))
      effectiveDtype = a.getValue().str();

    if (funcOp.getBody().empty()) return;
    for (Operation& op : funcOp.getBody().front().without_terminator()) {
      auto candArr = op.getAttrOfType<ArrayAttr>("compiler.candidates");
      if (!candArr) continue;

      // Dtype-independent shape facts, once per op.
      ShapeFacts facts = computeShapeFacts(op);

      std::string weightDtype;
      if (auto a = op.getAttrOfType<StringAttr>("quant.weight_dtype"))
        weightDtype = a.getValue().str();

      // Phase 1: evaluate every candidate (V0 + V1 + V2 when possible).
      SmallVector<CandidateEval, 4> evals;
      SmallVector<Attribute> passthrough; // non-dict elements, kept verbatim
      bool allEvaluatedHaveEstimates = true;
      for (auto elem : candArr) {
        auto dict = dyn_cast<DictionaryAttr>(elem);
        if (!dict) { passthrough.push_back(elem); continue; }

        CandidateEval ce;
        ce.dict = dict;
        ce.candidate =
            decodeImplementationCandidate(dict, "serving_cost_model_pass");
        std::string candType = ce.candidate.implementationKind;
        std::string fbBackend = ce.candidate.fallbackBackend;
        auto boundaryOps = ce.candidate.requiredBoundaryOps;

        ce.v0 = evaluatePenaltyV0(candType, boundaryOps);
        ce.v1 = model.compute(candType, boundaryOps, fbBackend);

        if (facts.usable()) {
          std::string actDtype =
              resolveActivationDtype(dict, candType, op, effectiveDtype);
          int64_t actBits = dtypeBits(actDtype);
          int64_t weightBits = dtypeBits(weightDtype);
          if (weightBits <= 0) weightBits = actBits;
          if (actBits > 0) {
            // Boundary traffic from the candidate's required boundary ops;
            // dequant reads the quantized weight and writes it at the float
            // activation width (modeled as 2x the float weight bytes).
            int64_t floatWeightBytes = facts.weight_elems * actBits / 8;
            int64_t outputBytes      = facts.output_elems * actBits / 8;
            int64_t boundaryBytes = 0;
            for (const auto& b : boundaryOps)
              boundaryBytes +=
                  boundaryBytesFor(b, outputBytes, floatWeightBytes);
            ce.shape = estimateShapeCost(facts, actBits, weightBits,
                                         boundaryBytes, profileNums);
            ce.hasShapeCost = true;
          }
        }
        if (ce.v0.status == "evaluated" &&
            (!ce.hasShapeCost || !ce.shape.has_time_estimates))
          allEvaluatedHaveEstimates = false;
        evals.push_back(std::move(ce));
      }

      // Phase 2: decide the ranking mode for this op, then assemble.
      bool useV2Penalty =
          v2ModeRequested && facts.usable() && allEvaluatedHaveEstimates;

      SmallVector<Attribute> evaluated;
      evaluated.reserve(candArr.size());
      for (const CandidateEval& ce : evals)
        evaluated.push_back(assembleCandidate(ctx, ce, useV2Penalty));
      for (Attribute a : passthrough)
        evaluated.push_back(a);

      op.setAttr("compiler.evaluated_candidates",
                 ArrayAttr::get(ctx, evaluated));
      op.setAttr("compiler.evaluated_candidates.truth_boundary",
                 StringAttr::get(ctx, kV0Truth));

      // Op-level shape profile record: op kind, whether shapes were usable,
      // and which ranking mode actually applied — the honest trail for
      // fallback cases.
      op.setAttr("compiler.shape_profile.op_kind",
                 StringAttr::get(ctx, facts.op_kind));
      op.setAttr("compiler.shape_profile.status",
                 StringAttr::get(ctx, facts.status));
      op.setAttr("compiler.shape_profile.truth_boundary",
                 StringAttr::get(ctx, kShapeCostTruthBoundary));
      StringRef rankingMode = "v0_heuristic";
      if (useV2Penalty)
        rankingMode = "shape_aware_v2_estimated_total_nanos";
      else if (v2ModeRequested)
        rankingMode = "v0_heuristic_shape_estimates_incomplete";
      op.setAttr("compiler.shape_profile.ranking_mode",
                 StringAttr::get(ctx, rankingMode));

      // Tile-plan integration (conservative): when TilePlanningPass found a
      // feasible tile, convert its reuse-limited global traffic into a time
      // annotation using the declared bandwidth. Reported alongside the
      // ideal each-byte-once estimate — it never replaces it and never
      // changes ranking.
      if (auto st = op.getAttrOfType<StringAttr>("tile.plan.status");
          st && st.getValue() == "planned" && profileNums.hasBandwidth()) {
        if (auto traffic = op.getAttrOfType<IntegerAttr>(
                "tile.plan.estimated_global_traffic_bytes")) {
          int64_t nanos = static_cast<int64_t>(
              static_cast<double>(traffic.getInt()) /
                  profileNums.mem_bandwidth_bytes_per_sec * 1e9 +
              0.5);
          op.setAttr("compiler.shape_profile.estimated_tiled_memory_cost_nanos",
                     IntegerAttr::get(IntegerType::get(ctx, 64), nanos));
        }
      }
    }
  }

private:
  StaticCostWeights weights_;
};

} // namespace
} // namespace mlir::hir

namespace mlir::hir {

std::unique_ptr<mlir::Pass>
createServingCostModelPass(const StaticCostWeights& weights) {
  return std::make_unique<ServingCostModelPass>(weights);
}

// createCandidateEvaluationPass() is declared inline in FusionPasses.h
// and calls createServingCostModelPass() — no definition needed here.

} // namespace mlir::hir
