// Cost Model Slice 3: Analytical Cost Estimation unit tests.
//
// Uses the real Slice 1 generator and Slice 2 legality checker, then
// exercises estimateCandidateCost()/estimateCandidateCosts() against the
// synthetic generic_npu_v1 reference target. See CandidateCostModel.h for
// the documented formulas this test pins down with exact hand-computed
// assertions (numbered comments below correspond to the task's 27
// required test items).

#include "costmodel/CandidateCostModel.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdio>
#include <optional>

using namespace mlir;
using namespace mlir::costmodel;

namespace {

mlir::hir::Conv2dOp buildConv2dOp(MLIRContext &ctx, Block &block,
                                   bool dynamicOutputHeight = false) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();
  auto elementType = builder.getIntegerType(8);
  auto inputType = RankedTensorType::get({1, 64, 56, 56}, elementType);
  auto filterType = RankedTensorType::get({128, 64, 3, 3}, elementType);
  int64_t outH = dynamicOutputHeight ? ShapedType::kDynamic : 54;
  auto outputType = RankedTensorType::get({1, 128, outH, 54}, elementType);

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

const CandidateLegalityResult &findResultFor(const std::vector<KernelCandidate> &candidates,
                                              const std::vector<CandidateLegalityResult> &results,
                                              const KernelCandidate &target) {
  for (std::size_t i = 0; i < candidates.size(); ++i)
    if (candidates[i].candidateId == target.candidateId)
      return results[i];
  std::fprintf(stderr, "findResultFor: no matching result found\n");
  std::abort();
}

CandidateCostEstimate mustEstimate(const KernelCandidate &candidate,
                                    const CandidateLegalityResult &legality,
                                    const NPUTargetConfig &target) {
  llvm::Expected<CandidateCostEstimate> est = estimateCandidateCost(candidate, legality, target);
  if (!est) {
    std::fprintf(stderr, "unexpected estimation failure: %s\n",
                 llvm::toString(est.takeError()).c_str());
    std::abort();
  }
  return *est;
}

bool failed(llvm::Expected<CandidateCostEstimate> &&est) {
  if (est) {
    (void)*est;
    return false;
  }
  llvm::consumeError(est.takeError());
  return true;
}

} // namespace

int main() {
  std::puts("=== CandidateCostModelTest ===");

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

  // ---- (1) real Conv2D now generates 144 candidates (Slice 5) ----
  std::vector<KernelCandidate> candidates = generateCandidates(problem);
  assert(candidates.size() == 144);

  const NPUTargetConfig &target = genericNPUv1Target();

  // ---- (2) Slice 2 now classifies 78 legal / 66 illegal (Slice 5) ----
  std::vector<CandidateLegalityResult> legality = checkCandidateLegality(candidates, target);
  std::size_t legalCount = 0, illegalCount = 0;
  for (const auto &r : legality)
    (r.isLegal ? legalCount : illegalCount)++;
  assert(legalCount == 78);
  assert(illegalCount == 66);
  std::puts("  [ok] Slice 1 (144 candidates) and Slice 2 (78 legal / 66 illegal) hold post-Slice-5");

  // ---- (3) exactly the legal candidates receive valid cost estimates ----
  llvm::Expected<std::vector<CandidateCostEstimate>> batchOrErr =
      estimateCandidateCosts(candidates, legality, target);
  assert(bool(batchOrErr));
  std::vector<CandidateCostEstimate> estimates = std::move(*batchOrErr);
  assert(estimates.size() == 78);
  std::puts("  [ok] exactly the 78 legal candidates received cost estimates");

  // ---- (4) illegal candidates fail closed if directly submitted ----
  {
    std::size_t illegalIdx = 0;
    while (legality[illegalIdx].isLegal)
      ++illegalIdx;
    assert(failed(estimateCandidateCost(candidates[illegalIdx], legality[illegalIdx], target)));
  }
  std::puts("  [ok] illegal candidate fails closed when directly submitted for estimation");

  // ---- (5) repeated estimation calls are field-identical ----
  {
    const TileShape tile1{16, 8, 8, 576};
    const PEArrayShape pe16{16, 16};
    const KernelCandidate &c =
        findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
    const CandidateLegalityResult &r = findResultFor(candidates, legality, c);
    CandidateCostEstimate e1 = mustEstimate(c, r, target);
    CandidateCostEstimate e2 = mustEstimate(c, r, target);
    assert(e1 == e2);
  }
  std::puts("  [ok] repeated estimateCandidateCost() calls are field-identical");

  // ---- (6) batch output ordering is deterministic ----
  {
    llvm::Expected<std::vector<CandidateCostEstimate>> again =
        estimateCandidateCosts(candidates, legality, target);
    assert(bool(again));
    assert(again->size() == estimates.size());
    for (std::size_t i = 0; i < estimates.size(); ++i)
      assert((*again)[i] == estimates[i]);
  }
  std::puts("  [ok] batch estimation ordering and content are deterministic");

  // ---- (7) candidate IDs remain unchanged ----
  {
    const TileShape tile1{16, 8, 8, 576};
    const PEArrayShape pe16{16, 16};
    const KernelCandidate &c =
        findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
    KernelCandidate before = c;
    const CandidateLegalityResult &r = findResultFor(candidates, legality, c);
    CandidateCostEstimate e = mustEstimate(c, r, target);
    assert(c == before);
    assert(e.candidateId == c.candidateId);
  }
  std::puts("  [ok] candidate IDs unchanged; no mutation");

  // ---- exact hand-computed values for the reference problem ----
  // logicalOutputSpatial=1*54*54=2916, logicalOutputChannels=128,
  // reductionK=64*3*3=576, logicalMacs=2916*128*576=214,990,848
  const std::uint64_t kLogicalOutputSpatial = 2916;
  const std::uint64_t kReductionK = 576;
  const std::uint64_t kLogicalMacs = 214990848ull;
  const std::uint64_t kLogicalOutputElements = 373248ull;

  const TileShape tile1{16, 8, 8, 576}; // tileM=64, tileN=16
  const PEArrayShape pe16{16, 16};
  const PEArrayShape pe32{32, 32};

  const KernelCandidate &wsCand =
      findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
  CandidateCostEstimate wsPe16 =
      mustEstimate(wsCand, findResultFor(candidates, legality, wsCand), target);

  // ---- (8) logical MAC count matches exact hand-calculated reference ----
  assert(wsPe16.logicalOutputSpatial == kLogicalOutputSpatial);
  assert(wsPe16.reductionK == kReductionK);
  assert(wsPe16.logicalMacs == kLogicalMacs);
  assert(wsPe16.logicalOutputElements == kLogicalOutputElements);
  std::puts("  [ok] logical MAC count matches exact hand-calculated reference (214,990,848)");

  // ---- (9)+(10)+(11): physical MACs, padding presence/absence ----
  // numMTiles = ceil(2916/64) = 46 (2916/64=45.5625, not exact -> padding)
  // numNTiles = ceil(128/16) = 8 (exact -> no N-boundary padding)
  // physicalM=2944, physicalN=128, physicalMacs=2944*128*576=217,055,232
  // paddingMacs=217,055,232-214,990,848=2,064,384 (> 0: non-divisible M tile)
  assert(wsPe16.numMTiles == 46);
  assert(wsPe16.numNTiles == 8);
  assert(wsPe16.totalTiles == 368);
  assert(wsPe16.physicalMacs == 217055232ull);
  assert(wsPe16.physicalOutputElements == 376832ull);
  assert(wsPe16.paddingMacs == 2064384ull);
  assert(wsPe16.paddingMacs > 0); // (11) non-divisible tile has positive padding
  std::puts("  [ok] physical MACs include tile-boundary padding exactly as hand-computed (positive case)");

  // (10) an exactly-divisible tile has zero padding MACs: pick a synthetic
  // problem whose logical extents are exact multiples of tileM/tileN.
  {
    KernelCandidate exact = wsCand;
    exact.problem.batch = 1;
    exact.problem.outputHeight = 8; // outputSpatial = 1*8*8=64 = tileM exactly
    exact.problem.outputWidth = 8;
    exact.problem.outputChannels = 16; // = tileN exactly
    exact.problem.outputShapeIsStaticAndSupported = true;
    CandidateLegalityResult exactLegality = checkCandidateLegality(exact, target);
    assert(exactLegality.isLegal);
    CandidateCostEstimate exactEst = mustEstimate(exact, exactLegality, target);
    assert(exactEst.numMTiles == 1 && exactEst.numNTiles == 1);
    assert(exactEst.numReductionTiles == 1); // tileK=576 == problemK exactly
    assert(exactEst.paddingMacs == 0);
    assert(exactEst.spatialPaddingMacs == 0);
    assert(exactEst.reductionPaddingMacs == 0);
    assert(exactEst.physicalMacs == exactEst.logicalMacs);
  }
  std::puts("  [ok] exactly-divisible tile (M, N, and K all exact) has zero padding MACs");

  // (18)+(19) K-padding split: pick a tileK that does NOT evenly divide
  // problemK=576 (e.g. tileK=100) to get a positive reductionPaddingMacs
  // while holding the base tile's M/N exactly divisible (isolating the K
  // padding term from the spatial term).
  {
    KernelCandidate kPadded = wsCand;
    kPadded.problem.outputHeight = 8;
    kPadded.problem.outputWidth = 8;
    kPadded.problem.outputChannels = 16;
    kPadded.problem.outputShapeIsStaticAndSupported = true;
    // tileK=160 (divisible by INT8's channel/reduction alignment of 16,
    // so this stays legal on generic_npu_v1): numReductionTiles =
    // ceil(576/160) = 4 (576/160=3.6, not exact); physicalK=640.
    kPadded.tile.reductionDepth = 160;
    CandidateLegalityResult kPaddedLegality = checkCandidateLegality(kPadded, target);
    assert(kPaddedLegality.isLegal);
    CandidateCostEstimate kPaddedEst = mustEstimate(kPadded, kPaddedLegality, target);
    assert(kPaddedEst.numReductionTiles == 4);
    assert(kPaddedEst.physicalReductionK == 640); // 4*160, > reductionK=576
    assert(kPaddedEst.reductionK == 576);
    assert(kPaddedEst.spatialPaddingMacs == 0); // M/N still exact
    assert(kPaddedEst.reductionPaddingMacs > 0); // K padding alone is positive
    // physicalMacs = physicalM(64)*physicalN(16)*physicalK(640) = 655360
    assert(kPaddedEst.physicalMacs == 64ull * 16ull * 640ull);
    assert(kPaddedEst.paddingMacs == kPaddedEst.spatialPaddingMacs + kPaddedEst.reductionPaddingMacs);
  }
  std::puts("  [ok] a non-divisible tileK produces positive, separately-observable K-padding MACs");

  // ---- (12) 16x16 vs 32x32 PE array: different utilization / compute cycles ----
  const KernelCandidate &wsCand32 =
      findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe32, tile1);
  CandidateCostEstimate wsPe32 =
      mustEstimate(wsCand32, findResultFor(candidates, legality, wsCand32), target);
  assert(wsPe16.computeCycles == 211968ull);
  assert(wsPe32.computeCycles == 105984ull);
  assert(wsPe16.computeCycles != wsPe32.computeCycles);
  assert(wsPe16.peUtilization != wsPe32.peUtilization);
  assert(wsPe16.peUtilization > wsPe32.peUtilization); // smaller N=16 tile wastes half of 32 cols
  // peCapacityMacs = computeCycles(211968) * totalPELanes(256) *
  // macsPerPEPerCycle(4) = 217,055,232 -- this ALREADY equals physicalMacs
  // exactly (no PE-internal wave waste for the 16x16 array here), so no
  // additional *4 belongs in this comparison.
  double expectedUtil16 = static_cast<double>(kLogicalMacs) / static_cast<double>(217055232ull);
  assert(wsPe16.peUtilization == expectedUtil16);
  std::puts("  [ok] 16x16 vs 32x32 PE array produce different utilization and compute cycles");

  // ---- (23) utilization remains in [0,1] ----
  for (const auto &e : estimates) {
    assert(e.peUtilization > 0.0);
    assert(e.peUtilization <= 1.0);
  }
  std::puts("  [ok] utilization remains in [0,1] for every legal candidate");

  // ---- (13) INT8 vs FP16 precision/throughput/byte differences ----
  {
    // FP16 tile1 is illegal on generic_npu_v1 (InputTileExceedsBuffer, per
    // Slice 2) so build a permissive target here just to compare the
    // precision-dependent cost fields in isolation.
    NPUTargetConfig permissive = target;
    permissive.inputBufferBytes = 1ull << 40;
    permissive.weightBufferBytes = 1ull << 40;
    permissive.outputBufferBytes = 1ull << 40;
    permissive.scratchpadBytes = 1ull << 40;
    permissive.dmaAlignmentBytes = 1;
    permissive.channelAlignmentInt8 = 1;
    permissive.channelAlignmentFp16 = 1;

    const KernelCandidate &int8Cand =
        findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
    const KernelCandidate &fp16Cand =
        findCandidate(candidates, Precision::FP16, Dataflow::WeightStationary, pe16, tile1);
    CandidateLegalityResult int8Legality = checkCandidateLegality(int8Cand, permissive);
    CandidateLegalityResult fp16Legality = checkCandidateLegality(fp16Cand, permissive);
    assert(int8Legality.isLegal && fp16Legality.isLegal);
    CandidateCostEstimate int8Est = mustEstimate(int8Cand, int8Legality, permissive);
    CandidateCostEstimate fp16Est = mustEstimate(fp16Cand, fp16Legality, permissive);

    // operand bytes: FP16 = 2x INT8 (Slice 2 formula, reused here)
    assert(fp16Est.offChipInputBytes == 2 * int8Est.offChipInputBytes);
    assert(fp16Est.offChipWeightBytes == 2 * int8Est.offChipWeightBytes);
    assert(fp16Est.offChipOutputWriteBytes ==
           int8Est.offChipOutputWriteBytes); // accumulator unchanged
    // throughput: fp16MacsPerPEPerCycle(1) < int8MacsPerPEPerCycle(4)
    // -> FP16 needs 4x the reduction cycles per wave, hence 4x compute cycles
    assert(fp16Est.computeCycles == 4 * int8Est.computeCycles);
  }
  std::puts("  [ok] INT8 vs FP16 use different precision throughput and operand byte sizes");

  // ---- exact per-dataflow off-chip traffic for tile1 (WS/OS/IS) ----
  const KernelCandidate &osCand =
      findCandidate(candidates, Precision::INT8, Dataflow::OutputStationary, pe16, tile1);
  const KernelCandidate &isCand =
      findCandidate(candidates, Precision::INT8, Dataflow::InputStationary, pe16, tile1);
  CandidateCostEstimate osPe16 =
      mustEstimate(osCand, findResultFor(candidates, legality, osCand), target);
  CandidateCostEstimate isPe16 =
      mustEstimate(isCand, findResultFor(candidates, legality, isCand), target);

  // ---- (17) exact expected transfer counts per dataflow ----
  // At tileK=576=problemK, numReductionTiles(KT)=1 for every dataflow, so
  // every formula in CandidateCostModel.h degenerates EXACTLY to Slice
  // 3/4's original single-K-tile numbers: outputReadCount = MT*NT*(KT-1)
  // = 0 for all three dataflows, and outputWriteCount = MT*NT*KT = MT*NT
  // (WS/IS) or MT*NT (OS) -- i.e. all three equal 368 here, unchanged.
  assert(wsPe16.weightLoadCount == 8 && wsPe16.inputLoadCount == 368 &&
         wsPe16.outputWriteCount == 368 && wsPe16.outputReadCount == 0);
  assert(osPe16.weightLoadCount == 368 && osPe16.inputLoadCount == 368 &&
         osPe16.outputWriteCount == 368 && osPe16.outputReadCount == 0);
  assert(isPe16.inputLoadCount == 46 && isPe16.weightLoadCount == 368 &&
         isPe16.outputWriteCount == 368 && isPe16.outputReadCount == 0);
  assert(wsPe16.offChipWeightBytes == 73728ull);
  assert(wsPe16.offChipInputBytes == 13565952ull);
  assert(wsPe16.offChipOutputWriteBytes == 1507328ull);
  assert(wsPe16.offChipOutputReadBytes == 0ull);
  assert(wsPe16.totalOffChipBytes == 15147008ull);
  assert(osPe16.offChipWeightBytes == 3391488ull);
  assert(osPe16.totalOffChipBytes == 18464768ull);
  assert(isPe16.offChipInputBytes == 1695744ull);
  assert(isPe16.totalOffChipBytes == 6594560ull);
  std::puts("  [ok] each dataflow produces exact expected transfer counts and byte totals");

  // ---- (14) WS produces lower weight off-chip traffic than OS ----
  assert(wsPe16.offChipWeightBytes < osPe16.offChipWeightBytes);
  // ---- (15) IS produces lower input off-chip traffic than OS ----
  assert(isPe16.offChipInputBytes < osPe16.offChipInputBytes);
  // ---- (16) OS gets no invented K-reduction output-traffic benefit:
  // output traffic is identical across all three dataflows AT KT=1 ----
  assert(wsPe16.offChipOutputWriteBytes == osPe16.offChipOutputWriteBytes);
  assert(osPe16.offChipOutputWriteBytes == isPe16.offChipOutputWriteBytes);
  assert(wsPe16.offChipOutputReadBytes == 0 && osPe16.offChipOutputReadBytes == 0 &&
         isPe16.offChipOutputReadBytes == 0);
  std::puts("  [ok] WS weight reuse, IS input reuse, and OS's non-invented output-traffic parity all confirmed");

  // ---- Slice 5: OS's real modeled advantage only appears once KT > 1.
  // Re-run WS/OS/IS at tileK=64 (KT=9) on a permissive target (K=64 is
  // legal for tile1 on generic_npu_v1 too, but use a permissive target so
  // this comparison isolates the dataflow effect from buffer capacity). ----
  {
    NPUTargetConfig permissive = target;
    permissive.inputBufferBytes = 1ull << 40;
    permissive.weightBufferBytes = 1ull << 40;
    permissive.outputBufferBytes = 1ull << 40;
    permissive.scratchpadBytes = 1ull << 40;

    KernelCandidate wsK64 = wsCand, osK64 = osCand, isK64 = isCand;
    wsK64.tile.reductionDepth = 64;
    osK64.tile.reductionDepth = 64;
    isK64.tile.reductionDepth = 64;
    CandidateLegalityResult wsK64Legality = checkCandidateLegality(wsK64, permissive);
    CandidateLegalityResult osK64Legality = checkCandidateLegality(osK64, permissive);
    CandidateLegalityResult isK64Legality = checkCandidateLegality(isK64, permissive);
    assert(wsK64Legality.isLegal && osK64Legality.isLegal && isK64Legality.isLegal);
    CandidateCostEstimate wsK64Est = mustEstimate(wsK64, wsK64Legality, permissive);
    CandidateCostEstimate osK64Est = mustEstimate(osK64, osK64Legality, permissive);
    CandidateCostEstimate isK64Est = mustEstimate(isK64, isK64Legality, permissive);

    assert(wsK64Est.numReductionTiles == 9 && osK64Est.numReductionTiles == 9 &&
           isK64Est.numReductionTiles == 9); // ceil(576/64)=9, KT>1 now

    // (24)+(25) exact WS/IS partial-output spill/reload counts.
    assert(wsK64Est.outputWriteCount == wsK64Est.totalTiles * 9);
    assert(wsK64Est.outputReadCount == wsK64Est.totalTiles * 8); // MT*NT*(KT-1)
    assert(isK64Est.outputWriteCount == isK64Est.totalTiles * 9);
    assert(isK64Est.outputReadCount == isK64Est.totalTiles * 8);
    // (24) OS output write count = MT*NT exactly (no KT factor).
    assert(osK64Est.outputWriteCount == osK64Est.totalTiles);
    // (25) OS output read count is always zero.
    assert(osK64Est.outputReadCount == 0);

    // (28) OS has strictly lower total off-chip output traffic than
    // WS/IS once KT > 1 (they pay partial-sum spill traffic; OS does not).
    std::uint64_t wsOutputTraffic = wsK64Est.offChipOutputReadBytes + wsK64Est.offChipOutputWriteBytes;
    std::uint64_t osOutputTraffic = osK64Est.offChipOutputReadBytes + osK64Est.offChipOutputWriteBytes;
    std::uint64_t isOutputTraffic = isK64Est.offChipOutputReadBytes + isK64Est.offChipOutputWriteBytes;
    assert(osOutputTraffic < wsOutputTraffic);
    assert(osOutputTraffic < isOutputTraffic);
    assert(wsK64Est.offChipOutputReadBytes > 0); // real, nonzero spill traffic for WS
    assert(isK64Est.offChipOutputReadBytes > 0); // real, nonzero spill traffic for IS

    // (32) local accumulator traffic differs between OS and non-OS.
    assert(osK64Est.localAccumReadBytes == 0);
    assert(wsK64Est.localAccumReadBytes > 0);
    assert(isK64Est.localAccumReadBytes > 0);
  }
  std::puts("  [ok] OS's accumulator-residency advantage is real once numReductionTiles > 1; "
            "absent (no artificial benefit) at KT=1");

  // ---- (18) DMA-cycle ceiling division does not undercount fractional cycles ----
  // WS: 15,147,008 * 1e9 / 1e10 = 1,514,700.8 -> must ceil to 1,514,701, not truncate to 1,514,700
  assert(wsPe16.dmaTransferCycles == 1514701ull);
  assert(isPe16.dmaTransferCycles == 659456ull); // exact case: 6,594,560/10 = 659,456.0
  std::puts("  [ok] DMA-cycle ceiling division does not undercount a fractional cycle");

  // ---- (19) DMA setup overhead scales with number of modeled transfers
  // (dmaOperationCount = inputLoad + weightLoad + outputRead + outputWrite;
  // outputRead=0 at KT=1, so this reduces to Slice 3/4's original sum) ----
  assert(wsPe16.dmaSetupCycles == (8ull + 368ull + 0ull + 368ull) * 50ull);
  assert(osPe16.dmaSetupCycles == (368ull + 368ull + 0ull + 368ull) * 50ull);
  assert(isPe16.dmaSetupCycles == (46ull + 368ull + 0ull + 368ull) * 50ull);
  assert(wsPe16.dmaSetupCycles < osPe16.dmaSetupCycles); // fewer WS transfers than OS
  assert(wsPe16.dmaCycles == wsPe16.dmaTransferCycles + wsPe16.dmaSetupCycles);
  std::puts("  [ok] DMA setup overhead scales exactly with the number of modeled transfers");

  // ---- (30)+(31): smaller tileK (larger KT) strictly increases DMA setup
  // and synchronization cycles (more, smaller transfers/events) ----
  {
    NPUTargetConfig permissive = target;
    permissive.inputBufferBytes = 1ull << 40;
    permissive.weightBufferBytes = 1ull << 40;
    permissive.outputBufferBytes = 1ull << 40;
    permissive.scratchpadBytes = 1ull << 40;
    KernelCandidate wsK64 = wsCand;
    wsK64.tile.reductionDepth = 64; // KT=9, vs wsPe16's KT=1
    CandidateLegalityResult wsK64Legality = checkCandidateLegality(wsK64, permissive);
    assert(wsK64Legality.isLegal);
    CandidateCostEstimate wsK64Est = mustEstimate(wsK64, wsK64Legality, permissive);
    assert(wsK64Est.dmaSetupCycles > wsPe16.dmaSetupCycles);
    assert(wsK64Est.synchronizationCycles > wsPe16.synchronizationCycles);
    assert(wsK64Est.synchronizationCycles == wsK64Est.totalTiles * 9 * 10);
  }
  std::puts("  [ok] smaller tileK (larger KT) strictly increases DMA setup and synchronization cycles");

  // ---- (21) local-memory traffic separately observable from off-chip ----
  assert(wsPe16.totalLocalBytes == 15147008ull); // computed independently, happens to match here
  assert(wsPe16.localAccumReadBytes == 0 && wsPe16.localAccumWriteBytes == 1507328ull);
  assert(wsPe16.localMemoryCycles == 473344ull);
  assert(osPe16.localMemoryCycles == 577024ull);
  assert(isPe16.localMemoryCycles == 206080ull);
  assert(wsPe16.localMemoryCycles != wsPe16.dmaCycles); // distinct components, not aliased
  std::puts("  [ok] local-memory traffic is separately observable from off-chip traffic");

  // ---- (22) kernel-launch and synchronization overhead separately observable ----
  assert(wsPe16.setupCycles == 200ull);
  assert(wsPe16.synchronizationCycles == 368ull * 1ull * 10ull); // totalTiles*KT*syncCycles, KT=1 here
  assert(wsPe16.setupCycles == osPe16.setupCycles); // target-level constant, dataflow-invariant
  std::puts("  [ok] kernel-launch and synchronization overhead are separately observable");

  // ---- exact total estimated cycles (overlap enabled on generic_npu_v1) ----
  assert(wsPe16.computeDmaOverlapApplied);
  assert(wsPe16.totalEstimatedCycles == 2029125ull);
  assert(osPe16.totalEstimatedCycles == 2482581ull);
  assert(isPe16.totalEstimatedCycles == 908516ull);
  std::puts("  [ok] total estimated cycles (with overlap) match exact hand-computed values");

  // ---- (20) disabling overlap increases or preserves total cycles ----
  {
    NPUTargetConfig noOverlap = target;
    noOverlap.supportsComputeDmaOverlap = false;
    CandidateLegalityResult noOverlapLegality = checkCandidateLegality(wsCand, noOverlap);
    assert(noOverlapLegality.isLegal);
    CandidateCostEstimate noOverlapEst = mustEstimate(wsCand, noOverlapLegality, noOverlap);
    assert(!noOverlapEst.computeDmaOverlapApplied);
    assert(noOverlapEst.totalEstimatedCycles >= wsPe16.totalEstimatedCycles);
    // exact: without overlap, compute + dmaTransfer are both added (not maxed)
    assert(noOverlapEst.totalEstimatedCycles ==
           wsPe16.computeCycles + wsPe16.dmaTransferCycles + wsPe16.dmaSetupCycles +
               wsPe16.localMemoryCycles + wsPe16.setupCycles + wsPe16.synchronizationCycles);
  }
  std::puts("  [ok] disabling compute/DMA overlap increases or preserves total estimated cycles");

  // ---- (24) invalid target bandwidth/clock/throughput/PE-shape fails closed ----
  {
    NPUTargetConfig badClock = target;
    badClock.clockHz = 0;
    assert(failed(estimateCandidateCost(wsCand, findResultFor(candidates, legality, wsCand), badClock)));

    NPUTargetConfig badBandwidth = target;
    badBandwidth.offChipBandwidthBytesPerSecond = 0;
    assert(failed(
        estimateCandidateCost(wsCand, findResultFor(candidates, legality, wsCand), badBandwidth)));

    NPUTargetConfig badThroughput = target;
    badThroughput.int8MacsPerPEPerCycle = 0;
    assert(failed(estimateCandidateCost(wsCand, findResultFor(candidates, legality, wsCand),
                                         badThroughput)));

    NPUTargetConfig badLocalBw = target;
    badLocalBw.localMemoryBandwidthBytesPerCycle = 0;
    assert(failed(
        estimateCandidateCost(wsCand, findResultFor(candidates, legality, wsCand), badLocalBw)));

    KernelCandidate badPE = wsCand;
    badPE.peArray = PEArrayShape{0, 16};
    CandidateLegalityResult badPELegality = findResultFor(candidates, legality, wsCand);
    badPELegality.candidateId = badPE.candidateId; // keep IDs aligned for this synthetic probe
    assert(failed(estimateCandidateCost(badPE, badPELegality, target)));
  }
  std::puts("  [ok] invalid target bandwidth/clock/throughput and invalid PE shape fail closed");

  // ---- (25) arithmetic overflow never wraps ----
  {
    KernelCandidate huge = wsCand;
    huge.problem.inputChannels = int64_t(1) << 40;
    huge.problem.kernelHeight = int64_t(1) << 40; // reductionK overflows int64/uint64
    huge.problem.kernelWidth = 1;
    CandidateLegalityResult hugeLegality = findResultFor(candidates, legality, wsCand);
    hugeLegality.candidateId = huge.candidateId;
    assert(failed(estimateCandidateCost(huge, hugeLegality, target)));
  }
  std::puts("  [ok] arithmetic overflow fails closed (never wraps)");

  // ---- (26) dynamic/unsupported Conv2D shape fails closed ----
  {
    Block dynBlock;
    mlir::hir::Conv2dOp dynConv = buildConv2dOp(ctx, dynBlock, /*dynamicOutputHeight=*/true);
    Conv2DProblemShape dynProblem = extractConv2DProblemShape(dynConv);
    assert(!dynProblem.outputShapeIsStaticAndSupported);
    std::vector<KernelCandidate> dynCandidates = generateCandidates(dynProblem);
    const KernelCandidate &dynCand =
        findCandidate(dynCandidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
    // Adversarial: hand-construct a legality result that WRONGLY claims
    // legal, to prove estimateCandidateCost independently fails closed on
    // the dynamic shape rather than blindly trusting the passed-in flag.
    CandidateLegalityResult wronglyLegal;
    wronglyLegal.candidateId = dynCand.candidateId;
    wronglyLegal.isLegal = true;
    wronglyLegal.inputTileBytes = 36864;
    wronglyLegal.weightTileBytes = 9216;
    wronglyLegal.outputTileBytes = 4096;
    assert(failed(estimateCandidateCost(dynCand, wronglyLegal, target)));
  }
  std::puts("  [ok] dynamic/unsupported Conv2D shape fails closed even against a wrongly-legal input");

  // ---- boundary-padding overhead example (for evidence) ----
  assert(wsPe16.paddingMacs == 2064384ull);
  std::puts("  [ok] boundary-padding overhead example confirmed (2,064,384 padding MACs)");

  // ---- (27) no ranking/winner/selected/priority/score field exists ----
  // Structural fact, not a runtime-testable one: CandidateCostEstimate
  // (see CandidateCostModel.h) has no such field -- verified by
  // inspection of the header, consistent with every assertion above
  // treating each cost component as independently observable rather than
  // collapsed into one opaque score.

  std::puts("=== CandidateCostModelTest: PASS ===");
  return 0;
}
