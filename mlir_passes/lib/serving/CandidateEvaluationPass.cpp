#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_CANDIDATEEVALUATION
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "candidate_evaluation_static_penalty_not_measured_latency";

// ---------------------------------------------------------------------------
// Penalty constants (relative; no latency unit).
// ---------------------------------------------------------------------------

static constexpr int64_t kPenaltyDirectLower      = 0;
static constexpr int64_t kPenaltyDecomposition    = 5;
static constexpr int64_t kPenaltyReprConvBase     = 3;
static constexpr int64_t kPenaltyBoundaryDequant  = 3;
static constexpr int64_t kPenaltyBoundaryCast     = 2;
static constexpr int64_t kPenaltyBoundaryLayout   = 2;
static constexpr int64_t kPenaltyLayoutConv       = 4;
static constexpr int64_t kPenaltyCastConv         = 2;
static constexpr int64_t kPenaltyBackendFallback  = 20;
static constexpr int64_t kPenaltyUnsupported      = 100;
static constexpr int64_t kPenaltyUnknownCost      = 10;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string readStr(DictionaryAttr dict, StringRef key) {
  if (!dict) return {};
  if (auto a = dict.get(key))
    if (auto s = dyn_cast<StringAttr>(a)) return s.getValue().str();
  return {};
}

static std::vector<std::string> readStrs(DictionaryAttr dict, StringRef key) {
  std::vector<std::string> out;
  if (!dict) return out;
  if (auto a = dict.get(key))
    if (auto arr = dyn_cast<ArrayAttr>(a))
      for (auto e : arr)
        if (auto s = dyn_cast<StringAttr>(e)) out.push_back(s.getValue().str());
  return out;
}

// Evaluate one candidate dict; return an augmented DictionaryAttr with
// evaluation.* fields added.
static DictionaryAttr evaluateCandidate(MLIRContext *ctx, DictionaryAttr dict) {
  std::string candType  = readStr(dict, "candidate_type");
  auto boundaryOps      = readStrs(dict, "required_boundary_ops");

  int64_t directLowerPenalty     = 0;
  int64_t decompositionPenalty   = 0;
  int64_t reprConvPenalty        = 0;
  int64_t layoutConvPenalty      = 0;
  int64_t castConvPenalty        = 0;
  int64_t backendFallbackPenalty = 0;
  int64_t unsupportedPenalty     = 0;
  int64_t unknownCostPenalty     = 0;

  std::string status = "evaluated";
  std::string reason;

  if (candType == "direct_lower") {
    directLowerPenalty = kPenaltyDirectLower;
    reason = "exact_kernel_zero_penalty";
  } else if (candType == "algebraic_decomposition") {
    decompositionPenalty = kPenaltyDecomposition;
    reason = "decomposition_base_penalty";
  } else if (candType == "representation_conversion") {
    reprConvPenalty = kPenaltyReprConvBase;
    reason = "representation_conversion_base_penalty";
    for (const auto &b : boundaryOps) {
      if (b == "dequant_weight" || b == "dequant") {
        reprConvPenalty += kPenaltyBoundaryDequant;
        reason += "+dequant_boundary";
      } else if (b == "cast") {
        reprConvPenalty += kPenaltyBoundaryCast;
        reason += "+cast_boundary";
      } else if (b == "layout_transform") {
        reprConvPenalty += kPenaltyBoundaryLayout;
        reason += "+layout_boundary";
      } else {
        unknownCostPenalty += kPenaltyUnknownCost;
        status = "partially_evaluated";
        reason += "+unknown_boundary";
      }
    }
  } else if (candType == "layout_conversion") {
    layoutConvPenalty = kPenaltyLayoutConv;
    reason = "layout_conversion_penalty";
  } else if (candType == "cast_conversion") {
    castConvPenalty = kPenaltyCastConv;
    reason = "cast_conversion_penalty";
  } else if (candType == "backend_fallback") {
    backendFallbackPenalty = kPenaltyBackendFallback;
    reason = "backend_fallback_high_penalty";
  } else if (candType == "unsupported") {
    unsupportedPenalty = kPenaltyUnsupported;
    status = "rejected";
    reason = "no_viable_lowering_path";
  } else {
    unknownCostPenalty = kPenaltyUnknownCost;
    status = "partially_evaluated";
    reason = "unknown_candidate_type";
  }

  int64_t total = directLowerPenalty + decompositionPenalty + reprConvPenalty
                + layoutConvPenalty + castConvPenalty + backendFallbackPenalty
                + unsupportedPenalty + unknownCostPenalty;

  auto I64 = [&](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(ctx, 64), v);
  };
  auto S = [&](StringRef s) -> Attribute { return StringAttr::get(ctx, s); };

  // Build penalty_breakdown dict (keys sorted alphabetically by DictionaryAttr::get).
  SmallVector<NamedAttribute> breakdownEntries = {
    {StringAttr::get(ctx, "backend_fallback_penalty"),          I64(backendFallbackPenalty)},
    {StringAttr::get(ctx, "cast_conversion_penalty"),           I64(castConvPenalty)},
    {StringAttr::get(ctx, "decomposition_penalty"),             I64(decompositionPenalty)},
    {StringAttr::get(ctx, "direct_lower_penalty"),              I64(directLowerPenalty)},
    {StringAttr::get(ctx, "layout_conversion_penalty"),         I64(layoutConvPenalty)},
    {StringAttr::get(ctx, "representation_conversion_penalty"), I64(reprConvPenalty)},
    {StringAttr::get(ctx, "unknown_cost_penalty"),              I64(unknownCostPenalty)},
    {StringAttr::get(ctx, "unsupported_penalty"),               I64(unsupportedPenalty)},
  };
  DictionaryAttr breakdown = DictionaryAttr::get(ctx, breakdownEntries);

  // Copy all original candidate fields and append evaluation.* fields.
  // DictionaryAttr::get will re-sort all keys, so insertion order doesn't matter.
  SmallVector<NamedAttribute> augmented(dict.begin(), dict.end());
  augmented.push_back({StringAttr::get(ctx, "evaluation.penalty_breakdown"), breakdown});
  augmented.push_back({StringAttr::get(ctx, "evaluation.penalty_score"),     I64(total)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.reason"),            S(reason)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.status"),            S(status)});
  augmented.push_back({StringAttr::get(ctx, "evaluation.truth_boundary"),    S(kTruth)});

  return DictionaryAttr::get(ctx, augmented);
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct CandidateEvaluationPass
    : impl::CandidateEvaluationBase<CandidateEvaluationPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    if (funcOp.getBody().empty()) return;
    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      auto candArr = op.getAttrOfType<ArrayAttr>("compiler.candidates");
      if (!candArr) continue;

      SmallVector<Attribute> evaluated;
      evaluated.reserve(candArr.size());
      for (auto elem : candArr) {
        if (auto dict = dyn_cast<DictionaryAttr>(elem))
          evaluated.push_back(evaluateCandidate(ctx, dict));
        else
          evaluated.push_back(elem); // pass through non-dict entries unchanged
      }

      op.setAttr("compiler.evaluated_candidates",
                 ArrayAttr::get(ctx, evaluated));
      op.setAttr("compiler.evaluated_candidates.truth_boundary",
                 StringAttr::get(ctx, kTruth));
    }
  }
};

} // namespace
} // namespace mlir::hir

namespace mlir::hir {
std::unique_ptr<mlir::Pass> createCandidateEvaluationPass() {
  return std::make_unique<CandidateEvaluationPass>();
}
} // namespace mlir::hir
