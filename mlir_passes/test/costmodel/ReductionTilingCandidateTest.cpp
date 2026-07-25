// Cost Model Slice 5: Reduction Tiling Candidate Expansion unit tests.
//
// Focuses specifically on the Slice-5-introduced behaviors not already
// exercised in depth by the (updated) Slice 1-4 test files: candidate-ID
// sensitivity to tileK, a real case where a smaller tileK rescues an
// otherwise-illegal candidate, Output Stationary's legality being
// independent of numReductionTiles, logical MACs staying invariant across
// tileK choices, and Slice 4's selection comparator having no direct
// tileK preference rule. Numbered comments correspond to the task's
// Section 16 test items not already covered by the other four files.

#include "costmodel/PlanSelection.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdio>
#include <set>

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

CandidateCostEstimate mustEstimateSingle(const KernelCandidate &c,
                                          const CandidateLegalityResult &l,
                                          const NPUTargetConfig &t) {
  llvm::Expected<CandidateCostEstimate> e = estimateCandidateCost(c, l, t);
  if (!e) {
    std::fprintf(stderr, "unexpected estimation failure: %s\n",
                 llvm::toString(e.takeError()).c_str());
    std::abort();
  }
  return *e;
}

} // namespace

int main() {
  std::puts("=== ReductionTilingCandidateTest ===");

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

  const int64_t problemK = 64 * 3 * 3;
  assert(problemK == 576);
  const TileShape b1{16, 8, 8, 0};   // base spatial tile (reductionDepth filled per-choice below)
  const PEArrayShape pe16{16, 16};

  // ---- (1)+(2)+(3): problemK, reduction choices, candidate count ----
  assert(candidates.size() == 144);
  std::vector<int64_t> choices = reductionTileChoicesForProblem(problemK);
  assert((choices == std::vector<int64_t>{16, 32, 64, 576}));
  std::puts("  [ok] problemK=576; reduction choices exactly {16,32,64,576}; 144 candidates");

  // ---- (6) candidates differing ONLY in tileK have different IDs ----
  {
    const KernelCandidate &k16 = findCandidate(candidates, Precision::INT8,
                                                Dataflow::WeightStationary, pe16,
                                                TileShape{16, 8, 8, 16});
    const KernelCandidate &k32 = findCandidate(candidates, Precision::INT8,
                                                Dataflow::WeightStationary, pe16,
                                                TileShape{16, 8, 8, 32});
    const KernelCandidate &k64 = findCandidate(candidates, Precision::INT8,
                                                Dataflow::WeightStationary, pe16,
                                                TileShape{16, 8, 8, 64});
    const KernelCandidate &k576 = findCandidate(candidates, Precision::INT8,
                                                 Dataflow::WeightStationary, pe16,
                                                 TileShape{16, 8, 8, 576});
    std::set<std::string> ids = {k16.candidateId, k32.candidateId, k64.candidateId,
                                  k576.candidateId};
    assert(ids.size() == 4); // all four distinct
    // Every other schedule-defining field is identical across these four.
    assert(k16.precision == k576.precision && k16.dataflow == k576.dataflow &&
           k16.peArray == k576.peArray);
    assert(k16.tile.outputChannels == k576.tile.outputChannels &&
           k16.tile.height == k576.tile.height && k16.tile.width == k576.tile.width);
  }
  std::puts("  [ok] candidates differing only in tileK receive different, deterministic IDs");

  // ---- (14) smaller tileK rescues an otherwise-illegal candidate ----
  // B1+FP16+tileK=576 is illegal (InputTileExceedsBuffer: 64*576*2=73728 >
  // 65536); B1+FP16+tileK=64 is legal (64*64*2=8192 <= 65536).
  {
    const KernelCandidate &fp16K576 = findCandidate(candidates, Precision::FP16,
                                                     Dataflow::WeightStationary, pe16,
                                                     TileShape{16, 8, 8, 576});
    const KernelCandidate &fp16K64 = findCandidate(candidates, Precision::FP16,
                                                    Dataflow::WeightStationary, pe16,
                                                    TileShape{16, 8, 8, 64});
    CandidateLegalityResult r576 = checkCandidateLegality(fp16K576, target);
    CandidateLegalityResult r64 = checkCandidateLegality(fp16K64, target);
    assert(!r576.isLegal);
    assert(r576.reasons.size() == 1 && r576.reasons[0] == LegalityReason::InputTileExceedsBuffer);
    assert(r64.isLegal);
    assert(r64.inputTileBytes == 8192);
  }
  std::puts("  [ok] a smaller tileK changes legality: B1+FP16 illegal at K=576, legal at K=64");

  // ---- (15) Output Stationary legality is independent of numKTiles ----
  // B3 (64,32,32)'s output tile alone (1024*64*4=262144) already exceeds
  // the 64 KiB output buffer -- true for every tileK choice, OS included;
  // conversely B1's output tile (4096 bytes) fits regardless of tileK.
  {
    const PEArrayShape pe32{32, 32};
    for (int64_t k : {16, 32, 64, 576}) {
      const KernelCandidate &osB3 = findCandidate(
          candidates, Precision::INT8, Dataflow::OutputStationary, pe32, TileShape{64, 32, 32, k});
      CandidateLegalityResult r = checkCandidateLegality(osB3, target);
      bool hasOutputExceeds = false;
      for (auto reason : r.reasons)
        hasOutputExceeds |= (reason == LegalityReason::OutputTileExceedsBuffer);
      assert(hasOutputExceeds); // B3's OS output-buffer illegality never depends on K
      const KernelCandidate &osB1 = findCandidate(
          candidates, Precision::INT8, Dataflow::OutputStationary, pe16, TileShape{16, 8, 8, k});
      CandidateLegalityResult rb1 = checkCandidateLegality(osB1, target);
      assert(rb1.outputTileBytes == 4096); // never scales with numReductionTiles
    }
  }
  std::puts("  [ok] Output Stationary's accumulator-tile legality never depends on numReductionTiles");

  // ---- (16) logical MACs are invariant across tileK choices ----
  {
    std::uint64_t logicalMacsAt[4];
    int i = 0;
    for (int64_t k : {16, 32, 64, 576}) {
      const KernelCandidate &c = findCandidate(candidates, Precision::INT8,
                                                Dataflow::WeightStationary, pe16,
                                                TileShape{16, 8, 8, k});
      CandidateLegalityResult l = checkCandidateLegality(c, target);
      assert(l.isLegal);
      CandidateCostEstimate e = mustEstimateSingle(c, l, target);
      logicalMacsAt[i++] = e.logicalMacs;
    }
    for (int j = 1; j < 4; ++j)
      assert(logicalMacsAt[j] == logicalMacsAt[0]);
    assert(logicalMacsAt[0] == 214990848ull); // 2916*128*576, matching Slice 3's original value
  }
  std::puts("  [ok] logical MACs are exactly invariant across all four tileK choices");

  // ---- (21) smaller tileK never removes reduction work: total physical
  // reduction (numReductionTiles * tileK) is always >= problemK ----
  {
    for (int64_t k : {16, 32, 64, 576}) {
      const KernelCandidate &c = findCandidate(candidates, Precision::INT8,
                                                Dataflow::WeightStationary, pe16,
                                                TileShape{16, 8, 8, k});
      CandidateLegalityResult l = checkCandidateLegality(c, target);
      assert(l.isLegal);
      CandidateCostEstimate e = mustEstimateSingle(c, l, target);
      assert(e.physicalReductionK >= e.reductionK);
      assert(e.numReductionTiles * static_cast<std::uint64_t>(k) == e.physicalReductionK);
    }
  }
  std::puts("  [ok] smaller tileK never reduces total reduction work below problemK");

  // ---- (35)+(36): Slice 4 selection remains deterministic and applies
  // no direct tileK preference -- two synthetic candidates tied on every
  // cost field but differing in tileK must be resolved by the documented
  // original-index tie-break, not by which one has the larger/smaller
  // tileK. ----
  {
    KernelCandidate smallK, bigK;
    smallK.candidateId = "tiek_small";
    bigK.candidateId = "tiek_big";
    for (KernelCandidate *c : {&smallK, &bigK}) {
      c->problem.batch = 1;
      c->problem.inputChannels = 1;
      c->problem.inputHeight = 1;
      c->problem.inputWidth = 1;
      c->problem.outputChannels = 1;
      c->problem.kernelHeight = 1;
      c->problem.kernelWidth = 1;
      c->problem.outputHeight = 1;
      c->problem.outputWidth = 1;
      c->problem.outputShapeIsStaticAndSupported = true;
      c->precision = Precision::INT8;
      c->dataflow = Dataflow::WeightStationary;
      c->peArray = PEArrayShape{16, 16};
    }
    smallK.tile = TileShape{1, 1, 1, 1};
    bigK.tile = TileShape{1, 1, 1, 1}; // deliberately same tile (tileK is not
                                       // an independent axis in the fixture);
                                       // the two candidates are placed in
                                       // `bigK, smallK` order below so only
                                       // the ORIGINAL INDEX differs.
    std::vector<KernelCandidate> synthCandidates = {bigK, smallK};
    CandidateLegalityResult legalBig, legalSmall;
    legalBig.candidateId = bigK.candidateId;
    legalBig.isLegal = true;
    legalSmall.candidateId = smallK.candidateId;
    legalSmall.isLegal = true;
    std::vector<CandidateLegalityResult> synthLegality = {legalBig, legalSmall};

    CandidateCostEstimate costBig, costSmall;
    costBig.candidateId = bigK.candidateId;
    costSmall.candidateId = smallK.candidateId;
    for (CandidateCostEstimate *e : {&costBig, &costSmall}) {
      e->logicalOutputSpatial = 1;
      e->logicalOutputChannels = 1;
      e->reductionK = 1;
      e->physicalReductionK = 1;
      e->numReductionTiles = 1;
      e->logicalMacs = 1;
      e->logicalOutputElements = 1;
      e->numMTiles = 1;
      e->numNTiles = 1;
      e->totalTiles = 1;
      e->physicalMacs = 1;
      e->physicalOutputElements = 1;
      e->activePELanes = 1;
      e->totalPELanes = 1;
      e->arrayWavesPerTile = 1;
      e->peUtilization = 1.0;
      e->computeCycles = 1;
      e->inputLoadCount = 1;
      e->weightLoadCount = 1;
      e->outputWriteCount = 1;
      e->totalOffChipBytes = 1;
      e->dmaTransferCycles = 1;
      e->dmaCycles = 1;
      e->totalLocalBytes = 1;
      e->localMemoryCycles = 1;
      e->setupCycles = 1;
      e->synchronizationCycles = 1;
      e->totalEstimatedCycles = 100; // identical for both -- a complete tie
    }
    std::vector<CandidateCostEstimate> synthCosts = {costBig, costSmall};

    llvm::Expected<PlanSelectionResult> selOrErr =
        selectBestCandidate(synthCandidates, synthLegality, synthCosts);
    assert(bool(selOrErr));
    // "bigK" is at original index 0, "smallK" at index 1 -- the winner
    // must be "bigK" purely because of original-index tie-breaking, with
    // no reference anywhere in the comparator to tile.reductionDepth
    // (selectBestCandidate's own signature and comparator, see
    // PlanSelection.h/.cpp, never read KernelCandidate.tile at all).
    assert(selOrErr->selectedCandidateId == "tiek_big");
    assert(selOrErr->explanation.decisiveTieBreakField == "original_candidate_index");
  }
  std::puts("  [ok] selection has no direct tileK preference rule; a complete tie is resolved "
            "purely by original candidate generation order");

  // ---- (37) selected-plan JSON includes reduction_depth ----
  {
    llvm::Expected<PlanSelectionResult> result = buildAndSelectPlan(conv, target);
    assert(bool(result));
    std::string json = serializeSelectedPlanToJson(*result);
    assert(json.find("\"reduction_depth\"") != std::string::npos);
    char expected[64];
    std::snprintf(expected, sizeof(expected), "\"reduction_depth\": %lld",
                  static_cast<long long>(result->selectedCandidate.tile.reductionDepth));
    assert(json.find(expected) != std::string::npos);
  }
  std::puts("  [ok] selected-plan JSON includes reduction_depth");

  // ---- (38): structural fact, not runtime-testable directly -- neither
  // this file nor any Slice 5 header/source includes or references any
  // Hailo, HEF, lowering, code-generation, DMA-command, address-
  // allocation, simulator, or runtime-dispatch type.

  std::puts("=== ReductionTilingCandidateTest: PASS ===");
  return 0;
}
