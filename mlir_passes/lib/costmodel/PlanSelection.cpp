#include "costmodel/PlanSelection.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <unordered_map>

namespace mlir::costmodel {

llvm::StringRef toString(SelectionFailureReason reason) {
  switch (reason) {
  case SelectionFailureReason::NoLegalCandidates:
    return "no_legal_candidates";
  case SelectionFailureReason::CandidateLegalityMismatch:
    return "candidate_legality_mismatch";
  case SelectionFailureReason::MissingCostEstimate:
    return "missing_cost_estimate";
  case SelectionFailureReason::DuplicateCostEstimate:
    return "duplicate_cost_estimate";
  case SelectionFailureReason::UnknownCandidateId:
    return "unknown_candidate_id";
  case SelectionFailureReason::CostForIllegalCandidate:
    return "cost_for_illegal_candidate";
  case SelectionFailureReason::InvalidCostEstimate:
    return "invalid_cost_estimate";
  }
  llvm_unreachable("unhandled SelectionFailureReason");
}

namespace {

llvm::Error makeError(SelectionFailureReason reason, const llvm::Twine &detail) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                  "[" + toString(reason).str() + "] " + detail.str());
}

// One legal candidate, matched by candidateId to its cost estimate, still
// carrying its index into the `candidates` array as originally passed to
// selectBestCandidate().
struct LegalEntry {
  std::size_t originalIndex;
  std::string candidateId;
  CandidateCostEstimate cost;
};

// Section 5 + Section 7: validates every cross-stage integrity rule and
// every structural cost-estimate invariant, then returns the matched
// legal entries in original candidate order (never reordered by this
// function -- ranking/sorting happens only in selectBestCandidate()).
llvm::Expected<std::vector<LegalEntry>>
buildLegalEntries(llvm::ArrayRef<KernelCandidate> candidates,
                  llvm::ArrayRef<CandidateLegalityResult> legalityResults,
                  llvm::ArrayRef<CandidateCostEstimate> costEstimates) {
  if (candidates.size() != legalityResults.size())
    return makeError(SelectionFailureReason::CandidateLegalityMismatch,
                      "legalityResults.size() (" + llvm::Twine(legalityResults.size()) +
                          ") != candidates.size() (" + llvm::Twine(candidates.size()) + ")");

  std::unordered_map<std::string, std::size_t> idToIndex;
  idToIndex.reserve(candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (legalityResults[i].candidateId != candidates[i].candidateId)
      return makeError(SelectionFailureReason::CandidateLegalityMismatch,
                        "legalityResults[" + llvm::Twine(i) + "].candidateId ('" +
                            legalityResults[i].candidateId + "') != candidates[" + llvm::Twine(i) +
                            "].candidateId ('" + candidates[i].candidateId + "')");
    idToIndex[candidates[i].candidateId] = i;
  }

  std::size_t legalCount = 0;
  for (const auto &l : legalityResults)
    if (l.isLegal)
      ++legalCount;

  if (legalCount == 0) {
    // Section 6: explicit failure, preserving a full rejection-reason
    // histogram -- never a synthesized winner, never candidate zero.
    std::map<LegalityReason, std::size_t> histogram;
    for (const auto &l : legalityResults)
      for (LegalityReason reason : l.reasons)
        ++histogram[reason];
    std::string detail = "total_candidates=" + std::to_string(candidates.size()) +
                          " legal_candidates=0 histogram={";
    bool first = true;
    for (const auto &[reason, count] : histogram) {
      if (!first)
        detail += ", ";
      first = false;
      detail += toString(reason).str() + "=" + std::to_string(count);
    }
    detail += "}";
    return makeError(SelectionFailureReason::NoLegalCandidates, detail);
  }

  std::vector<bool> matched(candidates.size(), false);
  std::vector<LegalEntry> entries;
  entries.reserve(legalCount);

  for (const CandidateCostEstimate &cost : costEstimates) {
    auto it = idToIndex.find(cost.candidateId);
    if (it == idToIndex.end())
      return makeError(SelectionFailureReason::UnknownCandidateId,
                        "cost estimate references unknown candidateId '" + cost.candidateId + "'");
    std::size_t idx = it->second;
    if (!legalityResults[idx].isLegal)
      return makeError(SelectionFailureReason::CostForIllegalCandidate,
                        "cost estimate exists for illegal candidate '" + cost.candidateId + "'");
    if (matched[idx])
      return makeError(SelectionFailureReason::DuplicateCostEstimate,
                        "duplicate cost estimate for candidate '" + cost.candidateId + "'");

    // Section 7: structural integrity only -- no cost is recomputed here.
    if (!std::isfinite(cost.peUtilization) || cost.peUtilization < 0.0 ||
        cost.peUtilization > 1.0)
      return makeError(SelectionFailureReason::InvalidCostEstimate,
                        "peUtilization for '" + cost.candidateId +
                            "' is not finite or not in [0, 1]");
    if (cost.totalTiles == 0 || cost.numMTiles == 0 || cost.numNTiles == 0 ||
        cost.computeCycles == 0)
      return makeError(SelectionFailureReason::InvalidCostEstimate,
                        "cost estimate for '" + cost.candidateId +
                            "' has an unexpected zero in a required count field");

    matched[idx] = true;
    entries.push_back(LegalEntry{idx, cost.candidateId, cost});
  }

  for (std::size_t i = 0; i < candidates.size(); ++i)
    if (legalityResults[i].isLegal && !matched[i])
      return makeError(SelectionFailureReason::MissingCostEstimate,
                        "legal candidate '" + candidates[i].candidateId +
                            "' has no corresponding cost estimate");

  // Original candidate ordering is preserved before ranking (Section 5) --
  // sorted here by originalIndex regardless of costEstimates' input order.
  std::sort(entries.begin(), entries.end(),
            [](const LegalEntry &a, const LegalEntry &b) { return a.originalIndex < b.originalIndex; });
  return entries;
}

// The complete, documented tie-break comparator (Section 2-3). Returns
// true iff `a` must sort strictly before `b`.
bool candidateIsBetter(const LegalEntry &a, const LegalEntry &b) {
  if (a.cost.totalEstimatedCycles != b.cost.totalEstimatedCycles)
    return a.cost.totalEstimatedCycles < b.cost.totalEstimatedCycles;
  if (a.cost.totalOffChipBytes != b.cost.totalOffChipBytes)
    return a.cost.totalOffChipBytes < b.cost.totalOffChipBytes;
  if (a.cost.dmaCycles != b.cost.dmaCycles)
    return a.cost.dmaCycles < b.cost.dmaCycles;
  if (a.cost.localMemoryCycles != b.cost.localMemoryCycles)
    return a.cost.localMemoryCycles < b.cost.localMemoryCycles;
  if (a.cost.computeCycles != b.cost.computeCycles)
    return a.cost.computeCycles < b.cost.computeCycles;
  if (a.cost.physicalMacs != b.cost.physicalMacs)
    return a.cost.physicalMacs < b.cost.physicalMacs;
  if (a.cost.peUtilization != b.cost.peUtilization)
    return a.cost.peUtilization > b.cost.peUtilization; // higher preferred
  return a.originalIndex < b.originalIndex;              // mandatory final key
}

// Identifies which key in the comparator chain above actually differs
// between two entries -- used only to populate the diagnostic
// `decisiveTieBreakField` string, never to make a selection decision.
std::string decisiveField(const LegalEntry &a, const LegalEntry &b) {
  if (a.cost.totalEstimatedCycles != b.cost.totalEstimatedCycles)
    return "total_estimated_cycles";
  if (a.cost.totalOffChipBytes != b.cost.totalOffChipBytes)
    return "total_off_chip_bytes";
  if (a.cost.dmaCycles != b.cost.dmaCycles)
    return "dma_cycles";
  if (a.cost.localMemoryCycles != b.cost.localMemoryCycles)
    return "local_memory_cycles";
  if (a.cost.computeCycles != b.cost.computeCycles)
    return "compute_cycles";
  if (a.cost.physicalMacs != b.cost.physicalMacs)
    return "physical_macs";
  if (a.cost.peUtilization != b.cost.peUtilization)
    return "pe_utilization";
  return "original_candidate_index";
}

} // namespace

llvm::Expected<PlanSelectionResult>
selectBestCandidate(llvm::ArrayRef<KernelCandidate> candidates,
                    llvm::ArrayRef<CandidateLegalityResult> legalityResults,
                    llvm::ArrayRef<CandidateCostEstimate> costEstimates) {
  llvm::Expected<std::vector<LegalEntry>> entriesOrErr =
      buildLegalEntries(candidates, legalityResults, costEstimates);
  if (!entriesOrErr)
    return entriesOrErr.takeError();
  std::vector<LegalEntry> entries = std::move(*entriesOrErr);

  // Section 8: stable sort over a complete total order (candidateIsBetter
  // never returns "equal" ambiguously -- the mandatory final key always
  // breaks every remaining tie), so llvm::stable_sort's stability is a
  // documented belt-and-suspenders guarantee, not something the ordering
  // actually depends on.
  llvm::stable_sort(entries, candidateIsBetter);

  PlanSelectionResult result;
  result.totalCandidateCount = candidates.size();
  result.legalCandidateCount = entries.size();
  result.illegalCandidateCount = candidates.size() - entries.size();

  result.rankedLegalCandidates.reserve(entries.size());
  for (std::size_t rank = 0; rank < entries.size(); ++rank) {
    RankedCandidate ranked;
    ranked.candidateId = entries[rank].candidateId;
    ranked.originalCandidateIndex = entries[rank].originalIndex;
    ranked.rank = rank;
    ranked.cost = entries[rank].cost;
    ranked.isSelected = (rank == 0);
    result.rankedLegalCandidates.push_back(std::move(ranked));
  }

  const LegalEntry &winner = entries.front();
  result.selectedCandidateId = winner.candidateId;
  result.selectedOriginalCandidateIndex = winner.originalIndex;
  result.selectedCandidate = candidates[winner.originalIndex];
  result.selectedCost = winner.cost;

  // Section 9: structured selection explanation.
  SelectionExplanation explanation;
  explanation.winningTotalCycles = winner.cost.totalEstimatedCycles;
  if (entries.size() >= 2) {
    const LegalEntry &runnerUp = entries[1];
    explanation.hasRunnerUp = true;
    explanation.runnerUpTotalCycles = runnerUp.cost.totalEstimatedCycles;
    explanation.absoluteCycleAdvantage =
        runnerUp.cost.totalEstimatedCycles >= winner.cost.totalEstimatedCycles
            ? runnerUp.cost.totalEstimatedCycles - winner.cost.totalEstimatedCycles
            : 0; // candidateIsBetter guarantees winner <= runnerUp; guarded anyway
    explanation.relativeCycleAdvantage =
        runnerUp.cost.totalEstimatedCycles > 0
            ? static_cast<double>(explanation.absoluteCycleAdvantage) /
                  static_cast<double>(runnerUp.cost.totalEstimatedCycles)
            : 0.0;
    explanation.primaryCostWasTied =
        (winner.cost.totalEstimatedCycles == runnerUp.cost.totalEstimatedCycles);
    explanation.decisiveTieBreakField = decisiveField(winner, runnerUp);
  } else {
    explanation.hasRunnerUp = false;
    explanation.runnerUpTotalCycles = 0;
    explanation.absoluteCycleAdvantage = 0;
    explanation.relativeCycleAdvantage = 0.0;
    explanation.primaryCostWasTied = false;
    explanation.decisiveTieBreakField = "no_runner_up";
  }
  result.explanation = explanation;

  return result;
}

llvm::Expected<PlanSelectionResult> buildAndSelectPlan(mlir::hir::Conv2dOp op,
                                                        const NPUTargetConfig &target) {
  Conv2DProblemShape problem = extractConv2DProblemShape(op);
  std::vector<KernelCandidate> candidates = generateCandidates(problem);
  std::vector<CandidateLegalityResult> legality = checkCandidateLegality(candidates, target);
  llvm::Expected<std::vector<CandidateCostEstimate>> costsOrErr =
      estimateCandidateCosts(candidates, legality, target);
  if (!costsOrErr)
    return costsOrErr.takeError();
  return selectBestCandidate(candidates, legality, *costsOrErr);
}

std::string serializeSelectedPlanToJson(const PlanSelectionResult &result) {
  const KernelCandidate &c = result.selectedCandidate;
  const CandidateCostEstimate &cost = result.selectedCost;
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "{\n";
  os << "  \"candidate_id\": \"" << c.candidateId << "\",\n";
  os << "  \"precision\": \"" << toString(c.precision) << "\",\n";
  os << "  \"dataflow\": \"" << toString(c.dataflow) << "\",\n";
  os << "  \"pe_array\": {\"rows\": " << c.peArray.rows << ", \"columns\": " << c.peArray.cols
     << "},\n";
  os << "  \"tile\": {\"height\": " << c.tile.height << ", \"width\": " << c.tile.width
     << ", \"output_channels\": " << c.tile.outputChannels
     << ", \"reduction_depth\": " << c.tile.reductionDepth << "},\n";
  os << "  \"cost\": {\n";
  os << "    \"total_estimated_cycles\": " << cost.totalEstimatedCycles << ",\n";
  os << "    \"compute_cycles\": " << cost.computeCycles << ",\n";
  os << "    \"dma_cycles\": " << cost.dmaCycles << ",\n";
  os << "    \"local_memory_cycles\": " << cost.localMemoryCycles << "\n";
  os << "  }\n";
  os << "}";
  os.flush();
  return out;
}

} // namespace mlir::costmodel
