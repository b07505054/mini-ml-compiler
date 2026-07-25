#pragma once

// Cost Model Slice 4: Deterministic Plan Selection.
//
// Continues from Slice 1 (candidate generation), Slice 2 (legality
// classification), and Slice 3 (analytical cost estimation) by converting
// their existing, ordered outputs into one deterministic compiler
// selection decision. This slice RANKS AND SELECTS ONLY: it introduces no
// new analytical cost formula (every field it compares was already
// produced by Slice 3), it never executes the selected plan, and it
// remains fully independent from Hailo, HEF, runtime execution, lowering,
// simulation, hardware benchmarking, empirical calibration, and
// autotuning.
//
// ---------------------------------------------------------------------
// Ranking-key contract (Section 2-3) -- a complete, documented total order
// ---------------------------------------------------------------------
// Primary key: totalEstimatedCycles ascending (Slice 3's own contract,
// consumed here, never redefined -- no weighted score of the form
// `0.5*compute + 0.3*dma + ...` is introduced anywhere in this module).
//
// Deterministic lexicographic tie-break chain, applied in this exact
// order whenever the preceding key(s) are equal:
//   1. totalEstimatedCycles      ascending
//   2. totalOffChipBytes         ascending
//   3. dmaCycles                 ascending
//   4. localMemoryCycles         ascending
//   5. computeCycles             ascending
//   6. physicalMacs              ascending
//   7. peUtilization             DESCENDING (higher utilization preferred)
//   8. originalCandidateIndex    ascending  (mandatory, always resolves
//                                             every remaining tie: indices
//                                             into the candidates array as
//                                             passed to this call are
//                                             always distinct)
//
// Key 7 compares `double` values with exact equality (`==`/`<`), never an
// epsilon tolerance -- this is safe (not "approximate") because
// peUtilization is a pure, deterministic function of the same integer
// inputs every time (see CandidateCostModel.h), so two calls over the
// same data always produce bit-identical doubles. Per the task's
// guidance, key 7 is deliberately NOT the final tie-break key precisely
// because it is a `double`: key 8 (an exact integer) always follows it,
// so no tie can ever be left unresolved by a floating-point comparison.
//
// This comparator uses no enum field (Precision, Dataflow, PEArrayShape
// are never compared) and therefore needs no enum-ordinal policy
// documentation; if a future extension ever adds one as a tie-break key,
// its ordinal semantics must be documented at that point.
//
// ---------------------------------------------------------------------
// Cross-stage integrity validation (Section 5) -- fails closed, never
// silently repairs, reorders, ignores, or mis-associates results
// ---------------------------------------------------------------------
//   - legalityResults.size() must equal candidates.size();
//   - legalityResults[i].candidateId must equal candidates[i].candidateId
//     for every i (this is the one place strict POSITIONAL correspondence
//     is required and enforced -- a shuffled legalityResults vector is
//     detected here and rejected);
//   - every legal candidate must have exactly one costEstimates entry
//     with a matching candidateId (looked up BY ID, not by position --
//     costEstimates may legitimately be a reordered subset per Slice 3's
//     own "legal candidates only, original relative order" contract, and
//     ID-based lookup makes this Slice robust to any such reordering);
//   - no illegal candidate may have a cost estimate;
//   - no duplicate cost estimate may exist for one candidate ID;
//   - no cost estimate may reference an unknown candidate ID.
//
// ---------------------------------------------------------------------
// Cost-estimate structural validation (Section 7) -- integrity check
// only, never a second cost computation
// ---------------------------------------------------------------------
//   - peUtilization must be finite (no NaN/Inf) and lie in [0, 1];
//   - totalTiles, numMTiles, numNTiles, computeCycles must be nonzero
//     (Slice 3 never produces zero here for a legal candidate; a zero
//     indicates a structurally broken estimate, not a valid corner case).
//
// ---------------------------------------------------------------------
// No-legal-candidate behavior (Section 6)
// ---------------------------------------------------------------------
// If every candidate is illegal, selectBestCandidate() returns an
// llvm::Error (SelectionFailureReason::NoLegalCandidates) whose message
// embeds totalCandidateCount, legalCandidateCount (0), and a rejection-
// reason histogram derived from `legalityResults` -- never a
// synthesized winner, never candidate zero, never an infinite synthetic
// cost.
//
// ---------------------------------------------------------------------
// Ranking output contract (Section 8)
// ---------------------------------------------------------------------
// `rankedLegalCandidates` contains each legal candidate exactly once,
// best-to-worst per the comparator above, sorted with a stable sort (so
// even candidates equal on every key up to but not including the
// mandatory index key are never reordered relative to what the
// comparator's total order dictates). `rank` is 0-based: the selected
// (best) candidate is rank 0.

#include "costmodel/CandidateCostModel.h"
#include "costmodel/CandidateLegality.h"
#include "costmodel/KernelCandidate.h"

#include "HIR/IR/HIROps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::costmodel {

enum class SelectionFailureReason {
  NoLegalCandidates,
  CandidateLegalityMismatch,
  MissingCostEstimate,
  DuplicateCostEstimate,
  UnknownCandidateId,
  CostForIllegalCandidate,
  InvalidCostEstimate,
};

llvm::StringRef toString(SelectionFailureReason reason);

// One legal candidate's position in the final deterministic ranking.
// Selection state (rank/isSelected) lives ONLY here -- KernelCandidate
// itself is never mutated or extended with rank/score/selected/winner/
// priority fields.
struct RankedCandidate {
  std::string candidateId;
  std::size_t originalCandidateIndex = 0;
  std::size_t rank = 0; // 0-based; rank 0 is the selected candidate.

  CandidateCostEstimate cost;

  bool isSelected = false;
};

// Structured selection explanation (Section 9) -- never only a bare
// "winner = candidate X". `hasRunnerUp` is false iff exactly one legal
// candidate exists; the other runner-up fields are then defined as 0 /
// 0.0 / false, and must not be read as a real comparison.
struct SelectionExplanation {
  std::uint64_t winningTotalCycles = 0;
  bool hasRunnerUp = false;
  std::uint64_t runnerUpTotalCycles = 0;
  std::uint64_t absoluteCycleAdvantage = 0;
  double relativeCycleAdvantage = 0.0;

  bool primaryCostWasTied = false;
  // Stable, enumerable identifier of the field that actually broke the
  // tie between the winner and the runner-up (e.g. "total_estimated_cycles"
  // when no tie occurred, "compute_cycles" when the primary cost tied and
  // the wave-based compute-cycle key decided it, or
  // "original_candidate_index" for a complete tie). This string is a
  // diagnostic label only -- selection logic itself never branches on it.
  std::string decisiveTieBreakField;
};

struct PlanSelectionResult {
  std::string selectedCandidateId;
  std::size_t selectedOriginalCandidateIndex = 0;

  KernelCandidate selectedCandidate;
  CandidateCostEstimate selectedCost;

  std::vector<RankedCandidate> rankedLegalCandidates;

  std::size_t totalCandidateCount = 0;
  std::size_t legalCandidateCount = 0;
  std::size_t illegalCandidateCount = 0;

  SelectionExplanation explanation;
};

// Core selection API: consumes the existing outputs of Slice 1
// (candidates), Slice 2 (legalityResults), and Slice 3 (costEstimates)
// exactly as produced -- introduces no new cost formula, executes
// nothing. Fails closed (returns an llvm::Error) on any cross-stage
// integrity violation (Section 5), any structurally invalid cost
// estimate (Section 7), or when no candidate is legal (Section 6).
llvm::Expected<PlanSelectionResult>
selectBestCandidate(llvm::ArrayRef<KernelCandidate> candidates,
                    llvm::ArrayRef<CandidateLegalityResult> legalityResults,
                    llvm::ArrayRef<CandidateCostEstimate> costEstimates);

// Convenience end-to-end orchestration: calls the existing public Slice
// 1-3 APIs (extractConv2DProblemShape, generateCandidates,
// checkCandidateLegality, estimateCandidateCosts) exactly as they already
// exist, then selectBestCandidate() above -- no candidate-generation,
// legality, or cost logic is duplicated here.
llvm::Expected<PlanSelectionResult> buildAndSelectPlan(mlir::hir::Conv2dOp op,
                                                        const NPUTargetConfig &target);

// Optional stable diagnostic serializer (Section 12): deterministic JSON
// text describing the selected candidate and its cost breakdown. Not
// required by, and never called from, the core selection API above; does
// not write to disk; not an executable plan; carries no runtime-specific
// field.
std::string serializeSelectedPlanToJson(const PlanSelectionResult &result);

} // namespace mlir::costmodel
