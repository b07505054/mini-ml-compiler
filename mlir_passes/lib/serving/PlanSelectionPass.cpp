#include "FusionPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_PLANSELECTION
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "plan_selection_static_penalty_not_measured_runtime";

// ---------------------------------------------------------------------------
// Tiebreak priority (lower = preferred in a penalty tie).
// ---------------------------------------------------------------------------
static int tiebreakPriorityFor(const std::string &type) {
  if (type == "direct_lower")              return 0;
  if (type == "representation_conversion") return 1;
  if (type == "cast_conversion")           return 2;
  if (type == "layout_conversion")         return 3;
  if (type == "algebraic_decomposition")   return 4;
  if (type == "backend_fallback")          return 5;
  if (type == "unsupported")               return 6;
  return 7;
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------
static std::string readStrOp(Operation *op, StringRef key) {
  if (auto a = op->getAttrOfType<StringAttr>(key))
    return a.getValue().str();
  return {};
}

static std::string readStrDict(DictionaryAttr dict, StringRef key) {
  if (!dict) return {};
  if (auto a = dict.get(key))
    if (auto s = dyn_cast<StringAttr>(a)) return s.getValue().str();
  return {};
}

static int64_t readI64Dict(DictionaryAttr dict, StringRef key,
                           int64_t def = 0) {
  if (!dict) return def;
  if (auto a = dict.get(key))
    if (auto ia = dyn_cast<IntegerAttr>(a)) return ia.getInt();
  return def;
}

static std::vector<std::string> readStrsDict(DictionaryAttr dict,
                                             StringRef key) {
  std::vector<std::string> out;
  if (!dict) return out;
  if (auto a = dict.get(key))
    if (auto arr = dyn_cast<ArrayAttr>(a))
      for (auto e : arr)
        if (auto s = dyn_cast<StringAttr>(e)) out.push_back(s.getValue().str());
  return out;
}

// ---------------------------------------------------------------------------
// Candidate info for selection
// ---------------------------------------------------------------------------
struct CandidateInfo {
  DictionaryAttr dict;
  std::string type;
  std::string status;
  int64_t penaltyScore;
  int tiebreakPriority;
  bool isFallback;
  bool isUnsupported;

  // 0 = evaluated non-fallback  (best)
  // 1 = evaluated backend_fallback
  // 2 = partially_evaluated
  // 3 = rejected / unsupported  (worst)
  int tier() const {
    if (status == "evaluated" && !isFallback && !isUnsupported) return 0;
    if (status == "evaluated" && isFallback)                    return 1;
    if (status == "partially_evaluated")                        return 2;
    return 3;
  }
};

// Returns true if `a` is strictly preferred over `b`.
static bool preferCandidate(const CandidateInfo &a, const CandidateInfo &b) {
  int ta = a.tier(), tb = b.tier();
  if (ta != tb) return ta < tb;
  if (a.penaltyScore != b.penaltyScore) return a.penaltyScore < b.penaltyScore;
  return a.tiebreakPriority < b.tiebreakPriority;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------
struct PlanSelectionPass
    : impl::PlanSelectionBase<PlanSelectionPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    if (funcOp.getBody().empty()) return;
    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      auto evalArr = op.getAttrOfType<ArrayAttr>("compiler.evaluated_candidates");
      if (!evalArr) continue;

      // Parse all evaluated candidates.
      std::vector<CandidateInfo> candidates;
      for (auto elem : evalArr) {
        auto dict = dyn_cast<DictionaryAttr>(elem);
        if (!dict) continue;
        CandidateInfo ci;
        ci.dict             = dict;
        ci.type             = readStrDict(dict, "candidate_type");
        ci.status           = readStrDict(dict, "evaluation.status");
        ci.penaltyScore     = readI64Dict(dict, "evaluation.penalty_score");
        ci.tiebreakPriority = tiebreakPriorityFor(ci.type);
        ci.isFallback       = (ci.type == "backend_fallback");
        ci.isUnsupported    = (ci.type == "unsupported");
        candidates.push_back(std::move(ci));
      }

      if (candidates.empty()) continue;

      // Sort: best candidate first (stable for determinism).
      std::stable_sort(candidates.begin(), candidates.end(), preferCandidate);

      const CandidateInfo &best = candidates[0];

      // Derive selection reason from the winner's tier.
      std::string reason;
      switch (best.tier()) {
      case 0: reason = "lowest_penalty_evaluated";           break;
      case 1: reason = "backend_fallback_last_resort";        break;
      case 2: reason = "lowest_penalty_partially_evaluated";  break;
      default: reason = "no_valid_lowering_path";             break;
      }

      // Read op-level kernel.* attrs for provenance (empty string when absent).
      std::string backend   = readStrOp(&op, "kernel.backend");
      std::string kLibrary  = readStrOp(&op, "kernel.library");
      std::string kName     = readStrOp(&op, "kernel.name");

      // Boundary ops forwarded from the winning candidate.
      auto boundaryOps = readStrsDict(best.dict, "required_boundary_ops");

      auto S = [&](StringRef s) -> Attribute { return StringAttr::get(ctx, s); };
      auto I64 = [&](int64_t v) -> Attribute {
        return IntegerAttr::get(IntegerType::get(ctx, 64), v);
      };
      SmallVector<Attribute> bOpAttrs;
      for (const auto &b : boundaryOps) bOpAttrs.push_back(StringAttr::get(ctx, b));

      // Flat selected_plan.* attrs (keys sorted alphabetically by setAttr order
      // is irrelevant — MLIR sorts all op attr keys on output).
      op.setAttr("selected_plan.backend",             S(backend));
      op.setAttr("selected_plan.candidate_id",        S("plan_0"));
      op.setAttr("selected_plan.candidate_type",      S(best.type));
      op.setAttr("selected_plan.kernel_library",      S(kLibrary));
      op.setAttr("selected_plan.kernel_name",         S(kName));
      op.setAttr("selected_plan.penalty_score",       I64(best.penaltyScore));
      op.setAttr("selected_plan.reason",              S(reason));
      op.setAttr("selected_plan.required_boundary_ops",
                 ArrayAttr::get(ctx, bOpAttrs));
      op.setAttr("selected_plan.truth_boundary",      S(kTruth));

      // Selected candidate as a 1-element array for downstream consumers.
      op.setAttr("compiler.selected_candidates",
                 ArrayAttr::get(ctx, {best.dict}));

      // All non-selected evaluated candidates.
      SmallVector<Attribute> rejections;
      for (size_t i = 1; i < candidates.size(); ++i)
        rejections.push_back(candidates[i].dict);
      op.setAttr("compiler.selection_rejections",
                 ArrayAttr::get(ctx, rejections));
    }
  }
};

} // namespace
} // namespace mlir::hir

namespace mlir::hir {
std::unique_ptr<mlir::Pass> createPlanSelectionPass() {
  return std::make_unique<PlanSelectionPass>();
}
} // namespace mlir::hir
