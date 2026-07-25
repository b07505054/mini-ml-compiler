// Cost Model Slice 4: Deterministic Plan Selection unit tests.
//
// Uses the real Slice 1-3 pipeline for the end-to-end / reference-problem
// tests, and small hand-constructed (KernelCandidate, CandidateLegalityResult,
// CandidateCostEstimate) triples for the synthetic tie-break-ladder and
// fail-closed tests, where precise control over every cost field is
// needed. Numbered comments below correspond to the task's 35 required
// test items.

#include "costmodel/PlanSelection.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

using namespace mlir;
using namespace mlir::costmodel;

namespace {

mlir::hir::Conv2dOp buildConv2dOp(MLIRContext &ctx, Block &block) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();
  auto elementType = builder.getIntegerType(8);
  auto inputType = RankedTensorType::get({1, 64, 56, 56}, elementType);
  auto filterType = RankedTensorType::get({128, 64, 3, 3}, elementType);
  auto outputType = RankedTensorType::get({1, 128, 54, 54}, elementType);
  block.addArgument(inputType, loc);
  block.addArgument(filterType, loc);
  builder.setInsertionPointToStart(&block);
  return builder.create<mlir::hir::Conv2dOp>(loc, outputType, block.getArgument(0),
                                              block.getArgument(1));
}

const KernelCandidate &findCandidate(const std::vector<KernelCandidate> &candidates,
                                      Precision precision, Dataflow dataflow,
                                      const PEArrayShape &pe, const TileShape &tile) {
  for (const auto &c : candidates)
    if (c.precision == precision && c.dataflow == dataflow && c.peArray == pe && c.tile == tile)
      return c;
  std::fprintf(stderr, "findCandidate: no matching candidate found\n");
  std::abort();
}

const CandidateCostEstimate &findCost(const std::vector<CandidateCostEstimate> &costs,
                                       const std::string &id) {
  for (const auto &c : costs)
    if (c.candidateId == id)
      return c;
  std::fprintf(stderr, "findCost: no matching cost found\n");
  std::abort();
}

PlanSelectionResult mustSelect(llvm::ArrayRef<KernelCandidate> candidates,
                                llvm::ArrayRef<CandidateLegalityResult> legality,
                                llvm::ArrayRef<CandidateCostEstimate> costs) {
  llvm::Expected<PlanSelectionResult> r = selectBestCandidate(candidates, legality, costs);
  if (!r) {
    std::fprintf(stderr, "unexpected selection failure: %s\n",
                 llvm::toString(r.takeError()).c_str());
    std::abort();
  }
  return *r;
}

bool failed(llvm::Expected<PlanSelectionResult> &&r) {
  if (r) {
    (void)*r;
    return false;
  }
  llvm::consumeError(r.takeError());
  return true;
}

// ---- synthetic fixture builders (for the tie-break ladder / fail-closed
// tests, where precise control over every cost field is needed) ----

KernelCandidate makeSyntheticCandidate(const std::string &id) {
  KernelCandidate c;
  c.candidateId = id;
  c.problem.batch = 1;
  c.problem.inputChannels = 1;
  c.problem.inputHeight = 1;
  c.problem.inputWidth = 1;
  c.problem.outputChannels = 1;
  c.problem.kernelHeight = 1;
  c.problem.kernelWidth = 1;
  c.problem.outputHeight = 1;
  c.problem.outputWidth = 1;
  c.problem.outputShapeIsStaticAndSupported = true;
  c.precision = Precision::INT8;
  c.dataflow = Dataflow::WeightStationary;
  c.peArray = PEArrayShape{16, 16};
  c.tile = TileShape{1, 1, 1, 1};
  return c;
}

CandidateLegalityResult makeSyntheticLegal(const std::string &id) {
  CandidateLegalityResult r;
  r.candidateId = id;
  r.isLegal = true;
  r.inputTileBytes = 1;
  r.weightTileBytes = 1;
  r.outputTileBytes = 1;
  r.totalScratchpadBytes = 3;
  return r;
}

CandidateCostEstimate makeSyntheticCost(const std::string &id, std::uint64_t total,
                                         std::uint64_t offChip = 100, std::uint64_t dma = 100,
                                         std::uint64_t localMem = 100, std::uint64_t compute = 100,
                                         std::uint64_t physMacs = 100, double util = 0.5) {
  CandidateCostEstimate e;
  e.candidateId = id;
  e.logicalOutputSpatial = 1;
  e.logicalOutputChannels = 1;
  e.reductionK = 1;
  e.logicalMacs = 1;
  e.logicalOutputElements = 1;
  e.physicalReductionK = 1;
  e.numReductionTiles = 1;
  e.numMTiles = 1;
  e.numNTiles = 1;
  e.totalTiles = 1;
  e.physicalMacs = physMacs;
  e.physicalOutputElements = 1;
  e.paddingMacs = 0;
  e.spatialPaddingMacs = 0;
  e.reductionPaddingMacs = 0;
  e.activePELanes = 1;
  e.totalPELanes = 256;
  e.arrayWavesPerTile = 1;
  e.peUtilization = util;
  e.computeCycles = compute;
  e.inputLoadCount = 1;
  e.weightLoadCount = 1;
  e.outputWriteCount = 1;
  e.outputReadCount = 0;
  e.offChipInputBytes = offChip;
  e.offChipWeightBytes = 0;
  e.offChipOutputWriteBytes = 0;
  e.offChipOutputReadBytes = 0;
  e.totalOffChipBytes = offChip;
  e.dmaTransferCycles = dma;
  e.dmaSetupCycles = 0;
  e.dmaCycles = dma;
  e.localInputBytes = localMem;
  e.localWeightBytes = 0;
  e.localAccumReadBytes = 0;
  e.localAccumWriteBytes = 0;
  e.totalLocalBytes = localMem;
  e.localMemoryCycles = localMem;
  e.setupCycles = 1;
  e.synchronizationCycles = 1;
  e.computeDmaOverlapApplied = false;
  e.totalEstimatedCycles = total;
  return e;
}

} // namespace

int main() {
  std::puts("=== CandidatePlanSelectionTest ===");

  // Slice 6 policy: sentinel proving assert() is genuinely active under
  // this Release (-DNDEBUG) build -- see this target's -UNDEBUG option in
  // CMakeLists.txt. If that policy were ever lost, assert() below would
  // compile to nothing and this explicit (non-assert-gated) check would
  // abort with a clear message instead of a false PASS.
  {
    bool assertionsActive = false;
    assert((assertionsActive = true, true));
    if (!assertionsActive) {
      std::fprintf(stderr, "FATAL: assert() is compiled out in this test binary (NDEBUG?)\n");
      std::abort();
    }
  }

  MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::hir::HIRDialect>();

  Block block;
  mlir::hir::Conv2dOp conv = buildConv2dOp(ctx, block);
  Conv2DProblemShape problem = extractConv2DProblemShape(conv);
  std::vector<KernelCandidate> candidates = generateCandidates(problem);
  const NPUTargetConfig &target = genericNPUv1Target();
  std::vector<CandidateLegalityResult> legality = checkCandidateLegality(candidates, target);
  llvm::Expected<std::vector<CandidateCostEstimate>> costsOrErr =
      estimateCandidateCosts(candidates, legality, target);
  assert(bool(costsOrErr));
  std::vector<CandidateCostEstimate> costs = std::move(*costsOrErr);

  // ---- (1)+(2)+(3): Slice 1-3 pipeline (post-Slice-5 counts) ----
  assert(candidates.size() == 144);
  std::size_t legalCount = 0, illegalCount = 0;
  for (auto &l : legality)
    (l.isLegal ? legalCount : illegalCount)++;
  assert(legalCount == 78 && illegalCount == 66);
  assert(costs.size() == 78);
  std::puts("  [ok] Slice 1 (144), Slice 2 (78/66), Slice 3 (78 estimates) hold post-Slice-5");

  // ---- (4)+(5): Slice 4 ranks exactly 78, exactly one selected ----
  PlanSelectionResult result = mustSelect(candidates, legality, costs);
  assert(result.rankedLegalCandidates.size() == 78);
  assert(result.totalCandidateCount == 144);
  assert(result.legalCandidateCount == 78);
  assert(result.illegalCandidateCount == 66);
  std::size_t selectedCount = 0;
  for (auto &r : result.rankedLegalCandidates)
    if (r.isSelected)
      ++selectedCount;
  assert(selectedCount == 1);
  std::puts("  [ok] Slice 4 ranks exactly 78 candidates; exactly one selected");

  // ---- (6) selected candidate is the first ranked candidate ----
  assert(result.rankedLegalCandidates[0].isSelected);
  assert(result.rankedLegalCandidates[0].candidateId == result.selectedCandidateId);
  assert(result.rankedLegalCandidates[0].rank == 0);
  std::puts("  [ok] selected candidate is rank 0");

  // ---- (7)+(8) ranked candidates preserve IDs and original indices ----
  for (const auto &r : result.rankedLegalCandidates) {
    bool foundInCandidates = false;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].candidateId == r.candidateId) {
        foundInCandidates = true;
        assert(r.originalCandidateIndex == i);
      }
    }
    assert(foundInCandidates);
  }
  std::puts("  [ok] ranked candidates preserve original candidate IDs and generation indices");

  // ---- (9)+(10) ranking does not mutate inputs ----
  {
    std::vector<KernelCandidate> candidatesBefore = candidates;
    std::vector<CandidateLegalityResult> legalityBefore = legality;
    std::vector<CandidateCostEstimate> costsBefore = costs;
    PlanSelectionResult again = mustSelect(candidates, legality, costs);
    (void)again;
    assert(candidates == candidatesBefore);
    for (std::size_t i = 0; i < legality.size(); ++i)
      assert(legality[i] == legalityBefore[i]);
    for (std::size_t i = 0; i < costs.size(); ++i)
      assert(costs[i] == costsBefore[i]);
  }
  std::puts("  [ok] ranking does not mutate candidates, legality, or cost inputs");

  // ---- (11) repeated selection calls are field-identical ----
  {
    PlanSelectionResult r1 = mustSelect(candidates, legality, costs);
    PlanSelectionResult r2 = mustSelect(candidates, legality, costs);
    assert(r1.selectedCandidateId == r2.selectedCandidateId);
    assert(r1.rankedLegalCandidates.size() == r2.rankedLegalCandidates.size());
    for (std::size_t i = 0; i < r1.rankedLegalCandidates.size(); ++i) {
      assert(r1.rankedLegalCandidates[i].candidateId == r2.rankedLegalCandidates[i].candidateId);
      assert(r1.rankedLegalCandidates[i].rank == r2.rankedLegalCandidates[i].rank);
      assert(r1.rankedLegalCandidates[i].cost == r2.rankedLegalCandidates[i].cost);
    }
  }
  std::puts("  [ok] repeated selection calls are field-identical");

  // ---- (12) deterministic across independently built equivalent Conv2D ops ----
  {
    Block block2;
    mlir::hir::Conv2dOp conv2 = buildConv2dOp(ctx, block2);
    llvm::Expected<PlanSelectionResult> result2 = buildAndSelectPlan(conv2, target);
    assert(bool(result2));
    // candidateId strings are derived purely from problem shape + precision/
    // dataflow/pe/tile (Slice 1), so an independently-built equivalent op
    // selects the identically-named candidate.
    assert(result2->selectedCandidateId == result.selectedCandidateId);
  }
  std::puts("  [ok] selection is deterministic across independently built equivalent Conv2D ops");

  // ---- (13) primary ordering follows totalEstimatedCycles ----
  for (std::size_t i = 1; i < result.rankedLegalCandidates.size(); ++i) {
    assert(result.rankedLegalCandidates[i - 1].cost.totalEstimatedCycles <=
           result.rankedLegalCandidates[i].cost.totalEstimatedCycles);
  }
  std::puts("  [ok] primary ordering follows totalEstimatedCycles ascending");

  // ---- (14) WS/OS/IS appear in the expected cost groups ----
  // tileK=576 (full reduction, matching pre-Slice-5 behavior exactly --
  // KT=1 degenerates every Slice 5 formula to the original single-K-tile
  // numbers) is used for these spot-checks specifically because it is the
  // best-scoring K choice for this exact-divisible reference problem (see
  // the Slice 5 final-evidence report) -- it is not assumed to still be
  // the global winner among all 78 legal candidates without verification
  // below.
  const TileShape tile1{16, 8, 8, 576};
  const PEArrayShape pe16{16, 16};
  const PEArrayShape pe32{32, 32};
  const KernelCandidate &wsCand =
      findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
  const KernelCandidate &osCand =
      findCandidate(candidates, Precision::INT8, Dataflow::OutputStationary, pe16, tile1);
  const KernelCandidate &isCandPe16 =
      findCandidate(candidates, Precision::INT8, Dataflow::InputStationary, pe16, tile1);
  const KernelCandidate &isCandPe32 =
      findCandidate(candidates, Precision::INT8, Dataflow::InputStationary, pe32, tile1);
  const CandidateCostEstimate &wsCost = findCost(costs, wsCand.candidateId);
  const CandidateCostEstimate &osCost = findCost(costs, osCand.candidateId);
  const CandidateCostEstimate &isCostPe16 = findCost(costs, isCandPe16.candidateId);
  const CandidateCostEstimate &isCostPe32 = findCost(costs, isCandPe32.candidateId);
  assert(isCostPe16.totalEstimatedCycles < wsCost.totalEstimatedCycles);
  assert(wsCost.totalEstimatedCycles < osCost.totalEstimatedCycles);
  assert(isCostPe16.totalEstimatedCycles == 908516ull);
  assert(wsCost.totalEstimatedCycles == 2029125ull);
  assert(osCost.totalEstimatedCycles == 2482581ull);
  std::puts("  [ok] WS/OS/IS appear in the expected cost groups for the reference problem");

  // ---- (15) current reference winner is an Input Stationary candidate ----
  assert(result.selectedCandidate.dataflow == Dataflow::InputStationary);
  std::puts("  [ok] reference-problem winner is an Input Stationary candidate");

  // ---- (16)+(17): tied IS candidates resolved by the documented secondary
  // key, PE shape derived from the real comparator (not a hardcoded ID) ----
  assert(isCostPe16.totalEstimatedCycles == isCostPe32.totalEstimatedCycles); // (10) tie, restated
  assert(isCostPe16.totalOffChipBytes == isCostPe32.totalOffChipBytes);
  assert(isCostPe16.dmaCycles == isCostPe32.dmaCycles);
  assert(isCostPe16.localMemoryCycles == isCostPe32.localMemoryCycles);
  assert(isCostPe16.computeCycles != isCostPe32.computeCycles); // compute_cycles is decisive
  const KernelCandidate &expectedIsWinner =
      (isCostPe32.computeCycles < isCostPe16.computeCycles) ? isCandPe32 : isCandPe16;
  assert(result.selectedCandidateId == expectedIsWinner.candidateId);
  assert(result.selectedCandidate.peArray == expectedIsWinner.peArray);
  assert(result.explanation.primaryCostWasTied);
  assert(result.explanation.decisiveTieBreakField == "compute_cycles");
  assert(result.explanation.absoluteCycleAdvantage == 0);
  assert(result.explanation.hasRunnerUp);
  assert(result.explanation.runnerUpTotalCycles == result.explanation.winningTotalCycles);
  // Both candidates are DMA-bound (dmaTransferCycles >> computeCycles for
  // both), so the tie is a genuine tie in the analytical model -- it does
  // NOT mean the two PE arrays are predicted to run at identical latency
  // in general, only that this model currently predicts identical total
  // cycles for this specific problem+target; the documented policy then
  // deterministically prefers the lower-compute-demand candidate.
  std::puts("  [ok] tied IS candidates resolved by compute_cycles; PE shape derived from real data, not hardcoded");

  // ---- ranking-key sanity: rank 0/1 are exactly the tied IS/K576 pair
  // (already verified above); every other legal candidate -- including
  // the K576 WS/OS spot-checks -- must rank strictly worse, verified by
  // searching the actual ranked output rather than assuming fixed
  // positions (with 78 legal candidates now, WS/OS-K576 are not
  // necessarily at fixed indices 2-5). ----
  {
    assert(result.rankedLegalCandidates[0].rank == 0);
    assert(result.rankedLegalCandidates[1].rank == 1);
    auto rankOf = [&](const std::string &id) -> std::size_t {
      for (const auto &r : result.rankedLegalCandidates)
        if (r.candidateId == id)
          return r.rank;
      std::fprintf(stderr, "rankOf: candidate not found in ranking\n");
      std::abort();
    };
    assert(rankOf(wsCand.candidateId) > 1);
    assert(rankOf(osCand.candidateId) > rankOf(wsCand.candidateId));
  }
  std::puts("  [ok] IS/K576 pair occupies rank 0/1; WS/OS-K576 rank strictly worse, in cost order");

  // ---- (18) synthetic pair differing only in total cycles ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("A"), makeSyntheticCandidate("B")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("A"), makeSyntheticLegal("B")};
    std::vector<CandidateCostEstimate> e = {makeSyntheticCost("A", 100), makeSyntheticCost("B", 200)};
    PlanSelectionResult r = mustSelect(c, l, e);
    assert(r.selectedCandidateId == "A");
  }
  std::puts("  [ok] (18) lower total cycles wins");

  // ---- (19) tied total, differ in off-chip bytes ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("A"), makeSyntheticCandidate("B")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("A"), makeSyntheticLegal("B")};
    std::vector<CandidateCostEstimate> e = {makeSyntheticCost("A", 100, /*offChip=*/500),
                                             makeSyntheticCost("B", 100, /*offChip=*/300)};
    PlanSelectionResult r = mustSelect(c, l, e);
    assert(r.selectedCandidateId == "B");
  }
  std::puts("  [ok] (19) tied total cycles resolved by lower off-chip bytes");

  // ---- (20) tied through off-chip+dma, differ in local-memory cycles ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("A"), makeSyntheticCandidate("B")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("A"), makeSyntheticLegal("B")};
    std::vector<CandidateCostEstimate> e = {
        makeSyntheticCost("A", 100, 300, 50, /*localMem=*/900),
        makeSyntheticCost("B", 100, 300, 50, /*localMem=*/400)};
    PlanSelectionResult r = mustSelect(c, l, e);
    assert(r.selectedCandidateId == "B");
  }
  std::puts("  [ok] (20) tied through off-chip/dma resolved by lower local-memory cycles");

  // ---- (21) tied through local memory, differ in compute cycles ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("A"), makeSyntheticCandidate("B")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("A"), makeSyntheticLegal("B")};
    std::vector<CandidateCostEstimate> e = {
        makeSyntheticCost("A", 100, 300, 50, 400, /*compute=*/900),
        makeSyntheticCost("B", 100, 300, 50, 400, /*compute=*/250)};
    PlanSelectionResult r = mustSelect(c, l, e);
    assert(r.selectedCandidateId == "B");
  }
  std::puts("  [ok] (21) tied through local memory resolved by lower compute cycles");

  // ---- (22) complete tie resolved by original generation order ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("first"), makeSyntheticCandidate("second")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("first"), makeSyntheticLegal("second")};
    std::vector<CandidateCostEstimate> e = {makeSyntheticCost("first", 100), makeSyntheticCost("second", 100)};
    PlanSelectionResult r = mustSelect(c, l, e);
    // Every field is identical between "first" and "second" -- the
    // candidate appearing earlier in `candidates` (index 0, "first") must
    // win via the mandatory final tie-break key.
    assert(r.selectedCandidateId == "first");
    assert(r.explanation.decisiveTieBreakField == "original_candidate_index");
    assert(r.explanation.absoluteCycleAdvantage == 0);
  }
  std::puts("  [ok] (22) complete tie resolved by original candidate generation order");

  // ---- (23) permutation of legality relative to candidates is rejected ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("A"), makeSyntheticCandidate("B")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("B"), makeSyntheticLegal("A")}; // swapped
    std::vector<CandidateCostEstimate> e = {makeSyntheticCost("A", 100), makeSyntheticCost("B", 200)};
    assert(failed(selectBestCandidate(c, l, e)));
  }
  std::puts("  [ok] (23) legality permuted relative to candidates is rejected");

  // ---- (24) mismatched legality candidate ID fails closed ----
  {
    std::vector<CandidateLegalityResult> corrupted = legality;
    corrupted[0].candidateId = "not-a-real-id";
    assert(failed(selectBestCandidate(candidates, corrupted, costs)));
  }
  std::puts("  [ok] (24) mismatched legality candidate ID fails closed");

  // ---- (25) mismatched cost candidate ID fails closed ----
  {
    std::vector<CandidateCostEstimate> corrupted = costs;
    corrupted[0].candidateId = "not-a-real-id";
    assert(failed(selectBestCandidate(candidates, legality, corrupted)));
  }
  std::puts("  [ok] (25) mismatched cost candidate ID fails closed");

  // ---- (26) missing cost estimate for a legal candidate fails closed ----
  {
    std::vector<CandidateCostEstimate> missing = costs;
    missing.pop_back();
    assert(failed(selectBestCandidate(candidates, legality, missing)));
  }
  std::puts("  [ok] (26) missing cost estimate fails closed");

  // ---- (27) duplicate cost estimate fails closed ----
  {
    std::vector<CandidateCostEstimate> duped = costs;
    duped.push_back(costs.front());
    assert(failed(selectBestCandidate(candidates, legality, duped)));
  }
  std::puts("  [ok] (27) duplicate cost estimate fails closed");

  // ---- (28) cost estimate for an illegal candidate fails closed ----
  {
    std::size_t illegalIdx = 0;
    while (legality[illegalIdx].isLegal)
      ++illegalIdx;
    std::vector<CandidateCostEstimate> withIllegal = costs;
    withIllegal.push_back(
        makeSyntheticCost(candidates[illegalIdx].candidateId, 100));
    assert(failed(selectBestCandidate(candidates, legality, withIllegal)));
  }
  std::puts("  [ok] (28) cost estimate for an illegal candidate fails closed");

  // ---- (29) unknown candidate ID fails closed ----
  {
    std::vector<CandidateCostEstimate> withUnknown = costs;
    withUnknown.push_back(makeSyntheticCost("totally-unknown-id", 100));
    assert(failed(selectBestCandidate(candidates, legality, withUnknown)));
  }
  std::puts("  [ok] (29) unknown candidate ID fails closed");

  // ---- (30) no legal candidates returns explicit failure ----
  {
    NPUTargetConfig noSupport = target;
    noSupport.supportsInt8 = false;
    noSupport.supportsFp16 = false;
    std::vector<CandidateLegalityResult> allIllegal = checkCandidateLegality(candidates, noSupport);
    for (auto &r : allIllegal)
      assert(!r.isLegal);
    llvm::Expected<std::vector<CandidateCostEstimate>> emptyCostsOrErr =
        estimateCandidateCosts(candidates, allIllegal, noSupport);
    assert(bool(emptyCostsOrErr) && emptyCostsOrErr->empty());
    llvm::Expected<PlanSelectionResult> noneOrErr =
        selectBestCandidate(candidates, allIllegal, *emptyCostsOrErr);
    assert(!noneOrErr);
    std::string message = llvm::toString(noneOrErr.takeError());
    assert(message.find("no_legal_candidates") != std::string::npos);
    assert(message.find("total_candidates=144") != std::string::npos);
    assert(message.find("legal_candidates=0") != std::string::npos);
  }
  std::puts("  [ok] (30) no legal candidates returns explicit failure with full summary");

  // ---- (31) exactly one legal candidate: succeeds, explicitly no runner-up ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("only")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("only")};
    std::vector<CandidateCostEstimate> e = {makeSyntheticCost("only", 42)};
    PlanSelectionResult r = mustSelect(c, l, e);
    assert(r.selectedCandidateId == "only");
    assert(r.rankedLegalCandidates.size() == 1);
    assert(!r.explanation.hasRunnerUp);
    assert(r.explanation.runnerUpTotalCycles == 0);
    assert(r.explanation.absoluteCycleAdvantage == 0);
    assert(r.explanation.decisiveTieBreakField == "no_runner_up");
  }
  std::puts("  [ok] (31) single legal candidate succeeds with no runner-up");

  // ---- (32) invalid utilization fails closed ----
  {
    auto tryUtil = [&](double util) {
      std::vector<KernelCandidate> c = {makeSyntheticCandidate("A")};
      std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("A")};
      std::vector<CandidateCostEstimate> e = {
          makeSyntheticCost("A", 100, 100, 100, 100, 100, 100, util)};
      return failed(selectBestCandidate(c, l, e));
    };
    assert(tryUtil(std::numeric_limits<double>::quiet_NaN()));
    assert(tryUtil(std::numeric_limits<double>::infinity()));
    assert(tryUtil(-0.1));
    assert(tryUtil(1.1));
  }
  std::puts("  [ok] (32) invalid utilization (NaN/Inf/<0/>1) fails closed");

  // ---- (33) exact runner-up / cycle-advantage fields for a hand-computable case ----
  {
    std::vector<KernelCandidate> c = {makeSyntheticCandidate("A"), makeSyntheticCandidate("B")};
    std::vector<CandidateLegalityResult> l = {makeSyntheticLegal("A"), makeSyntheticLegal("B")};
    std::vector<CandidateCostEstimate> e = {makeSyntheticCost("A", 100), makeSyntheticCost("B", 150)};
    PlanSelectionResult r = mustSelect(c, l, e);
    assert(r.selectedCandidateId == "A");
    assert(r.explanation.hasRunnerUp);
    assert(r.explanation.winningTotalCycles == 100);
    assert(r.explanation.runnerUpTotalCycles == 150);
    assert(r.explanation.absoluteCycleAdvantage == 50);
    assert(r.explanation.relativeCycleAdvantage == 50.0 / 150.0);
    assert(!r.explanation.primaryCostWasTied);
    assert(r.explanation.decisiveTieBreakField == "total_estimated_cycles");
  }
  std::puts("  [ok] (33) runner-up and cycle-advantage fields exact for a hand-computable case");

  // ---- (34)+(35): structural facts, not runtime-testable directly ----
  // selectBestCandidate()'s signature (see PlanSelection.h) takes no
  // NPUTargetConfig at all -- it is structurally incapable of introducing
  // a new target-hardware-dependent cost formula or empirical coefficient;
  // every field it compares is copied verbatim from a Slice 3
  // CandidateCostEstimate. Neither PlanSelection.h nor PlanSelection.cpp
  // includes or references any Hailo, HEF, lowering, or simulator type.

  // ---- optional JSON serializer sanity (Section 12) ----
  {
    std::string json1 = serializeSelectedPlanToJson(result);
    std::string json2 = serializeSelectedPlanToJson(result);
    assert(json1 == json2); // deterministic
    assert(json1.find(result.selectedCandidateId) != std::string::npos);
    assert(json1.find("total_estimated_cycles") != std::string::npos);
  }
  std::puts("  [ok] optional JSON serializer is deterministic");

  std::puts("=== CandidatePlanSelectionTest: PASS ===");
  return 0;
}
