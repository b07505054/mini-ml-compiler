// Cost Model Slice 6: Selected Schedule Materialization unit tests.
//
// Uses the real Slice 1-5 pipeline for the reference-problem tests, plus
// small hand-constructed (KernelCandidate, CandidateLegalityResult,
// CandidateCostEstimate) triples -- and synthetic Conv2D problems built
// via real mlir::hir::Conv2dOp instances -- for the boundary-tile and
// KT>1 dataflow-specific tests, where precise control is needed.
// Numbered comments correspond to the task's Section 15 test items.

#include "costmodel/SchedulePlan.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdio>

using namespace mlir;
using namespace mlir::costmodel;

namespace {

mlir::hir::Conv2dOp buildConv2dOp(MLIRContext &ctx, Block &block, int64_t n, int64_t c,
                                   int64_t h, int64_t w, int64_t k, int64_t r, int64_t s,
                                   int64_t outH, int64_t outW) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();
  auto elementType = builder.getIntegerType(8);
  auto inputType = RankedTensorType::get({n, c, h, w}, elementType);
  auto filterType = RankedTensorType::get({k, c, r, s}, elementType);
  auto outputType = RankedTensorType::get({n, k, outH, outW}, elementType);
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

SchedulePlan mustMaterialize(const KernelCandidate &c, const CandidateLegalityResult &l,
                             const CandidateCostEstimate &cost, const Conv2DProblemShape &problem,
                             const NPUTargetConfig &target) {
  llvm::Expected<SchedulePlan> s = materializeCandidateSchedule(c, l, cost, problem, target);
  if (!s) {
    std::fprintf(stderr, "unexpected materialization failure: %s\n",
                 llvm::toString(s.takeError()).c_str());
    std::abort();
  }
  return *s;
}

bool failed(llvm::Expected<SchedulePlan> &&s) {
  if (s) {
    (void)*s;
    return false;
  }
  llvm::consumeError(s.takeError());
  return true;
}

std::uint64_t countKind(const SchedulePlan &plan, ScheduleOperationKind kind) {
  std::uint64_t n = 0;
  for (const auto &op : plan.operations)
    if (op.kind == kind)
      ++n;
  return n;
}

} // namespace

int main() {
  std::puts("=== SelectedScheduleMaterializationTest ===");

  // ---- (1) sentinel: prove assertions are genuinely active under this
  // Release (-DNDEBUG) build, per the task's explicit requirement. If
  // -UNDEBUG were ever removed from this target, `assert()` below would
  // compile to nothing, `assertionsActive` would stay false, and this
  // explicit (non-assert-gated) check would abort with a clear message
  // instead of silently proceeding. ----
  {
    bool assertionsActive = false;
    assert((assertionsActive = true, true));
    if (!assertionsActive) {
      std::fprintf(stderr,
                    "FATAL: assert() is compiled out in this test binary (NDEBUG?) -- the "
                    "-UNDEBUG policy has been lost; every assertion below would be a silent "
                    "no-op. Aborting explicitly rather than reporting a false PASS.\n");
      std::abort();
    }
  }
  std::puts("  [ok] sentinel confirms assert() is genuinely active (not compiled out by NDEBUG)");

  MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::hir::HIRDialect>();

  Block block;
  mlir::hir::Conv2dOp conv = buildConv2dOp(ctx, block, 1, 64, 56, 56, 128, 3, 3, 54, 54);
  Conv2DProblemShape problem = extractConv2DProblemShape(conv);
  std::vector<KernelCandidate> candidates = generateCandidates(problem);
  const NPUTargetConfig &target = genericNPUv1Target();
  std::vector<CandidateLegalityResult> legality = checkCandidateLegality(candidates, target);
  llvm::Expected<std::vector<CandidateCostEstimate>> costsOrErr =
      estimateCandidateCosts(candidates, legality, target);
  assert(bool(costsOrErr));
  std::vector<CandidateCostEstimate> costs = std::move(*costsOrErr);

  // ---- (2)+(3): reference pipeline unchanged (144 / 78 / 66) ----
  assert(candidates.size() == 144);
  std::size_t legalCount = 0, illegalCount = 0;
  for (auto &l : legality)
    (l.isLegal ? legalCount : illegalCount)++;
  assert(legalCount == 78 && illegalCount == 66);
  std::puts("  [ok] reference pipeline: 144 candidates, 78 legal, 66 illegal");

  // ---- (4) selection still chooses the actual Slice 5 winner ----
  llvm::Expected<PlanSelectionResult> selectionOrErr =
      selectBestCandidate(candidates, legality, costs);
  assert(bool(selectionOrErr));
  PlanSelectionResult selection = std::move(*selectionOrErr);
  assert(selection.selectedCandidate.dataflow == Dataflow::InputStationary);
  assert(selection.selectedCost.totalEstimatedCycles == 908516ull);
  std::puts("  [ok] selection still chooses the actual Slice 5 winner (IS, total=908,516)");

  // ---- (5) selected schedule's candidate ID matches the selected candidate ----
  llvm::Expected<SchedulePlan> scheduleOrErr =
      materializeSelectedSchedule(selection, problem, target);
  assert(bool(scheduleOrErr));
  SchedulePlan schedule = std::move(*scheduleOrErr);
  assert(schedule.candidateId == selection.selectedCandidateId);
  std::puts("  [ok] selected schedule's candidate ID matches the selected candidate");

  // ---- (6) IS loop order M -> K -> N ----
  assert((schedule.loopOrder == std::vector<ScheduleLoopDimension>{
              ScheduleLoopDimension::M, ScheduleLoopDimension::K, ScheduleLoopDimension::N}));
  std::puts("  [ok] selected (IS) schedule has loop order M -> K -> N");

  // ---- (9) IS marks input residency across N ----
  assert(schedule.residency.inputResidentAcrossN);
  assert(!schedule.residency.weightResidentAcrossM);
  assert(!schedule.residency.outputResidentAcrossK);
  std::puts("  [ok] IS marks input residency across N (only)");

  // ---- (12)-(17): reference MT/NT/KT and operation counts ----
  assert(schedule.numMTiles == 46);
  assert(schedule.numNTiles == 8);
  assert(schedule.numKTiles == 1);
  assert(schedule.inputLoadCount == 46);
  assert(schedule.weightLoadCount == 368);
  assert(schedule.computeCount == 368);
  assert(schedule.finalOutputStoreCount == 368);
  // ---- (17) partial reload/store counts are zero because KT=1 ----
  assert(schedule.partialOutputReloadCount == 0);
  assert(schedule.partialOutputStoreCount == 0);
  assert(schedule.synchronizationCount == 368);
  std::printf("  [ok] reference MT=%llu NT=%llu KT=%llu; inputLoads=%llu weightLoads=%llu "
              "compute=%llu finalStores=%llu partialReload=%llu partialStore=%llu sync=%llu\n",
              (unsigned long long)schedule.numMTiles, (unsigned long long)schedule.numNTiles,
              (unsigned long long)schedule.numKTiles, (unsigned long long)schedule.inputLoadCount,
              (unsigned long long)schedule.weightLoadCount,
              (unsigned long long)schedule.computeCount,
              (unsigned long long)schedule.finalOutputStoreCount,
              (unsigned long long)schedule.partialOutputReloadCount,
              (unsigned long long)schedule.partialOutputStoreCount,
              (unsigned long long)schedule.synchronizationCount);

  // ---- (18) final M tile: validM=36, physicalM=64, marked boundary ----
  {
    bool foundFinalMTileCompute = false;
    for (const auto &op : schedule.operations) {
      if (op.kind == ScheduleOperationKind::ComputeTile && op.coordinate.mTile == 45) {
        assert(op.extent.physicalM == 64);
        assert(op.extent.validM == 36); // 2916 - 45*64 = 36
        assert(op.isBoundaryTile);
        foundFinalMTileCompute = true;
      }
    }
    assert(foundFinalMTileCompute);
  }
  std::puts("  [ok] final M tile has validM=36, physicalM=64, marked as a boundary tile");

  // ---- (19)+(41) boundary marking and coordinate range sanity across
  // every operation in the reference schedule ----
  for (const auto &op : schedule.operations) {
    assert(op.coordinate.mTile < schedule.numMTiles);
    assert(op.coordinate.nTile < schedule.numNTiles);
    assert(op.coordinate.kTile < schedule.numKTiles);
    assert(op.extent.validM <= op.extent.physicalM);
    assert(op.extent.validN <= op.extent.physicalN);
    assert(op.extent.validK <= op.extent.physicalK);
    bool expectedBoundary = op.extent.validM < op.extent.physicalM ||
                            op.extent.validN < op.extent.physicalN ||
                            op.extent.validK < op.extent.physicalK;
    assert(op.isBoundaryTile == expectedBoundary);
  }
  std::puts("  [ok] every operation has an in-range tile coordinate and correct boundary marking");

  // ---- (20) operation indices dense and monotonically increasing ----
  for (std::size_t i = 0; i < schedule.operations.size(); ++i)
    assert(schedule.operations[i].operationIndex == i);
  std::puts("  [ok] operation indices are dense and monotonically increasing");

  // ---- (21) repeated materialization is field-identical ----
  {
    SchedulePlan again = mustMaterialize(selection.selectedCandidate,
                                          checkCandidateLegality(selection.selectedCandidate, target),
                                          selection.selectedCost, problem, target);
    assert(again == schedule);
  }
  std::puts("  [ok] repeated materialization is field-identical");

  // ---- (22) equivalent independently built Conv2D ops produce equivalent schedules ----
  {
    Block block2;
    mlir::hir::Conv2dOp conv2 = buildConv2dOp(ctx, block2, 1, 64, 56, 56, 128, 3, 3, 54, 54);
    Conv2DProblemShape problem2 = extractConv2DProblemShape(conv2);
    assert(problem2 == problem);
    std::vector<KernelCandidate> candidates2 = generateCandidates(problem2);
    std::vector<CandidateLegalityResult> legality2 = checkCandidateLegality(candidates2, target);
    llvm::Expected<std::vector<CandidateCostEstimate>> costs2OrErr =
        estimateCandidateCosts(candidates2, legality2, target);
    assert(bool(costs2OrErr));
    llvm::Expected<PlanSelectionResult> selection2OrErr =
        selectBestCandidate(candidates2, legality2, *costs2OrErr);
    assert(bool(selection2OrErr));
    llvm::Expected<SchedulePlan> schedule2OrErr =
        materializeSelectedSchedule(*selection2OrErr, problem2, target);
    assert(bool(schedule2OrErr));
    assert(schedule2OrErr->candidateId == schedule.candidateId);
    assert(schedule2OrErr->operations.size() == schedule.operations.size());
    assert(*schedule2OrErr == schedule);
  }
  std::puts("  [ok] equivalent independently built Conv2D ops produce equivalent schedules");

  // ---- (11) validateScheduleAgainstCost reconciles exactly for the
  // reference schedule (also exercised implicitly inside
  // materializeCandidateSchedule already, called again here directly) ----
  assert(!validateScheduleAgainstCost(schedule, selection.selectedCost));
  std::puts("  [ok] schedule reconciles exactly against the Slice 5 cost estimate");

  // ---- (23) candidate/legality/cost/selection ID mismatch fails closed ----
  {
    KernelCandidate badCandidate = selection.selectedCandidate;
    badCandidate.candidateId = "not-the-real-id";
    assert(failed(materializeCandidateSchedule(
        badCandidate, checkCandidateLegality(selection.selectedCandidate, target),
        selection.selectedCost, problem, target)));

    CandidateLegalityResult badLegality = checkCandidateLegality(selection.selectedCandidate, target);
    badLegality.candidateId = "not-the-real-id";
    assert(failed(materializeCandidateSchedule(selection.selectedCandidate, badLegality,
                                                selection.selectedCost, problem, target)));

    CandidateCostEstimate badCost = selection.selectedCost;
    badCost.candidateId = "not-the-real-id";
    assert(failed(materializeCandidateSchedule(
        selection.selectedCandidate, checkCandidateLegality(selection.selectedCandidate, target),
        badCost, problem, target)));

    PlanSelectionResult badSelection = selection;
    badSelection.selectedCandidateId = "not-the-real-id";
    assert(failed(materializeSelectedSchedule(badSelection, problem, target)));
  }
  std::puts("  [ok] candidate/legality/cost/selection ID mismatches all fail closed");

  // ---- (24) illegal candidate fails closed ----
  {
    std::size_t illegalIdx = 0;
    while (legality[illegalIdx].isLegal)
      ++illegalIdx;
    assert(failed(materializeCandidateSchedule(candidates[illegalIdx], legality[illegalIdx],
                                                CandidateCostEstimate{candidates[illegalIdx].candidateId},
                                                problem, target)));
  }
  std::puts("  [ok] illegal candidate fails closed");

  // ---- (25) invalid/zero tile dimensions fail closed ----
  {
    KernelCandidate zeroK = selection.selectedCandidate;
    zeroK.candidateId = "zero_k_probe";
    zeroK.tile.reductionDepth = 0;
    CandidateLegalityResult l;
    l.candidateId = zeroK.candidateId;
    l.isLegal = true; // deliberately wrong, to prove this slice checks independently too
    CandidateCostEstimate c;
    c.candidateId = zeroK.candidateId;
    assert(failed(materializeCandidateSchedule(zeroK, l, c, problem, target)));

    KernelCandidate zeroPE = selection.selectedCandidate;
    zeroPE.candidateId = "zero_pe_probe";
    zeroPE.peArray = PEArrayShape{0, 16};
    CandidateLegalityResult l2;
    l2.candidateId = zeroPE.candidateId;
    l2.isLegal = true;
    CandidateCostEstimate c2;
    c2.candidateId = zeroPE.candidateId;
    assert(failed(materializeCandidateSchedule(zeroPE, l2, c2, problem, target)));
  }
  std::puts("  [ok] invalid/zero tile or PE dimensions fail closed");

  // ---- KT > 1 dataflow-specific tests: use a synthetic small Conv2D
  // problem sized so tile1's tileK=64 gives a genuine KT=9 (matches
  // Slice 5's own K=64 examples), on a permissive target so WS/OS/IS are
  // all legal simultaneously. ----
  {
    NPUTargetConfig permissive = target;
    permissive.inputBufferBytes = 1ull << 40;
    permissive.weightBufferBytes = 1ull << 40;
    permissive.outputBufferBytes = 1ull << 40;
    permissive.scratchpadBytes = 1ull << 40;

    const TileShape tileK64{16, 8, 8, 64};
    const PEArrayShape pe16{16, 16};
    const KernelCandidate &wsK64 =
        findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tileK64);
    const KernelCandidate &isK64 =
        findCandidate(candidates, Precision::INT8, Dataflow::InputStationary, pe16, tileK64);
    const KernelCandidate &osK64 =
        findCandidate(candidates, Precision::INT8, Dataflow::OutputStationary, pe16, tileK64);

    CandidateLegalityResult wsL = checkCandidateLegality(wsK64, permissive);
    CandidateLegalityResult isL = checkCandidateLegality(isK64, permissive);
    CandidateLegalityResult osL = checkCandidateLegality(osK64, permissive);
    assert(wsL.isLegal && isL.isLegal && osL.isLegal);

    llvm::Expected<CandidateCostEstimate> wsCostOrErr = estimateCandidateCost(wsK64, wsL, permissive);
    llvm::Expected<CandidateCostEstimate> isCostOrErr = estimateCandidateCost(isK64, isL, permissive);
    llvm::Expected<CandidateCostEstimate> osCostOrErr = estimateCandidateCost(osK64, osL, permissive);
    assert(bool(wsCostOrErr) && bool(isCostOrErr) && bool(osCostOrErr));

    SchedulePlan wsSchedule = mustMaterialize(wsK64, wsL, *wsCostOrErr, problem, permissive);
    SchedulePlan isSchedule = mustMaterialize(isK64, isL, *isCostOrErr, problem, permissive);
    SchedulePlan osSchedule = mustMaterialize(osK64, osL, *osCostOrErr, problem, permissive);

    assert(wsSchedule.numKTiles == 9 && isSchedule.numKTiles == 9 && osSchedule.numKTiles == 9);

    // ---- (7) WS loop order N -> K -> M ----
    assert((wsSchedule.loopOrder == std::vector<ScheduleLoopDimension>{
                ScheduleLoopDimension::N, ScheduleLoopDimension::K, ScheduleLoopDimension::M}));
    // ---- (8) OS loop order M -> N -> K ----
    assert((osSchedule.loopOrder == std::vector<ScheduleLoopDimension>{
                ScheduleLoopDimension::M, ScheduleLoopDimension::N, ScheduleLoopDimension::K}));
    // ---- (10) WS marks weight residency across M ----
    assert(wsSchedule.residency.weightResidentAcrossM && !wsSchedule.residency.inputResidentAcrossN &&
           !wsSchedule.residency.outputResidentAcrossK);
    // ---- (11) OS marks output residency across K ----
    assert(osSchedule.residency.outputResidentAcrossK && !osSchedule.residency.inputResidentAcrossN &&
           !osSchedule.residency.weightResidentAcrossM);
    std::puts("  [ok] WS (N->K->M, weight-across-M) and OS (M->N->K, output-across-K) loop "
              "order and residency confirmed at KT>1");

    // ---- (26) WS at KT>1 emits partial reloads and stores ----
    assert(wsSchedule.partialOutputReloadCount == wsSchedule.numMTiles * wsSchedule.numNTiles * 8);
    assert(wsSchedule.partialOutputStoreCount == wsSchedule.numMTiles * wsSchedule.numNTiles * 8);
    assert(wsSchedule.partialOutputReloadCount > 0 && wsSchedule.partialOutputStoreCount > 0);
    // ---- (27) IS at KT>1 emits partial reloads and stores ----
    assert(isSchedule.partialOutputReloadCount == isSchedule.numMTiles * isSchedule.numNTiles * 8);
    assert(isSchedule.partialOutputStoreCount == isSchedule.numMTiles * isSchedule.numNTiles * 8);
    assert(isSchedule.partialOutputReloadCount > 0 && isSchedule.partialOutputStoreCount > 0);
    // ---- (28) OS at KT>1 emits NO partial reload or partial store ----
    assert(osSchedule.partialOutputReloadCount == 0);
    assert(osSchedule.partialOutputStoreCount == 0);
    std::puts("  [ok] WS/IS emit real partial reload+store traffic at KT>1; OS emits none");

    // ---- (29) WS weight-load count is NT*KT ----
    assert(wsSchedule.weightLoadCount == wsSchedule.numNTiles * wsSchedule.numKTiles);
    // ---- (30) IS input-load count is MT*KT ----
    assert(isSchedule.inputLoadCount == isSchedule.numMTiles * isSchedule.numKTiles);
    // ---- (31) OS final-output-store count is MT*NT ----
    assert(osSchedule.finalOutputStoreCount == osSchedule.numMTiles * osSchedule.numNTiles);
    // ---- (32) compute count is always MT*NT*KT ----
    for (const SchedulePlan *p : {&wsSchedule, &isSchedule, &osSchedule})
      assert(p->computeCount == p->numMTiles * p->numNTiles * p->numKTiles);
    // ---- (33) synchronization count equals compute count ----
    for (const SchedulePlan *p : {&wsSchedule, &isSchedule, &osSchedule})
      assert(p->synchronizationCount == p->computeCount);
    std::puts("  [ok] WS weightLoads=NT*KT, IS inputLoads=MT*KT, OS finalStores=MT*NT, "
              "compute=sync=MT*NT*KT for all three");

    // ---- (34)+(35)+(36) per-operation bytes match I/W/O and reconcile
    // exactly with Slice 5 cost fields ----
    for (auto pair : {std::make_pair(&wsSchedule, &*wsCostOrErr),
                      std::make_pair(&isSchedule, &*isCostOrErr),
                      std::make_pair(&osSchedule, &*osCostOrErr)}) {
      const SchedulePlan &sched = *pair.first;
      const CandidateCostEstimate &c = *pair.second;
      for (const auto &op : sched.operations) {
        if (op.kind == ScheduleOperationKind::LoadInputTile)
          assert(op.bytes == c.offChipInputBytes / c.inputLoadCount);
        if (op.kind == ScheduleOperationKind::LoadWeightTile)
          assert(op.bytes == c.offChipWeightBytes / c.weightLoadCount);
        if (op.kind == ScheduleOperationKind::ComputeTile || op.kind == ScheduleOperationKind::Synchronize)
          assert(op.bytes == 0);
      }
      assert(!validateScheduleAgainstCost(sched, c));
    }
    std::puts("  [ok] per-operation bytes match I/W/O; schedules reconcile exactly with cost "
              "for WS/IS/OS at KT>1");
  }

  // ---- (37)-(40) synthetic boundary-tile problems: M, N, and K padding ----
  {
    // M-boundary: outputHeight*outputWidth not divisible by tile1's tileM=64.
    Block bM;
    mlir::hir::Conv2dOp convM = buildConv2dOp(ctx, bM, 1, 64, 56, 56, 128, 3, 3, 10, 10); // logicalM=100
    Conv2DProblemShape probM = extractConv2DProblemShape(convM);
    std::vector<KernelCandidate> candM = generateCandidates(probM);
    const TileShape tile576{16, 8, 8, 576};
    const PEArrayShape pe16{16, 16};
    const KernelCandidate &cM =
        findCandidate(candM, Precision::INT8, Dataflow::WeightStationary, pe16, tile576);
    CandidateLegalityResult lM = checkCandidateLegality(cM, target);
    assert(lM.isLegal);
    llvm::Expected<CandidateCostEstimate> costM = estimateCandidateCost(cM, lM, target);
    assert(bool(costM));
    SchedulePlan schedM = mustMaterialize(cM, lM, *costM, probM, target);
    // logicalM=100, tileM=64 -> MT=2 (ceil(100/64)=2); final tile validM=100-64=36
    assert(schedM.logicalM == 100 && schedM.numMTiles == 2);
    bool foundMBoundary = false;
    for (const auto &op : schedM.operations)
      if (op.kind == ScheduleOperationKind::ComputeTile && op.coordinate.mTile == 1) {
        assert(op.extent.validM == 36 && op.extent.physicalM == 64);
        assert(op.isBoundaryTile);
        foundMBoundary = true;
      }
    assert(foundMBoundary);
    std::puts("  [ok] (37) synthetic problem demonstrates M-boundary padding "
              "(validM=36 < physicalM=64)");

    // N-boundary: outputChannels not divisible by tile1's tileN=16.
    Block bN;
    mlir::hir::Conv2dOp convN = buildConv2dOp(ctx, bN, 1, 64, 56, 56, 20, 3, 3, 54, 54); // logicalN=20
    Conv2DProblemShape probN = extractConv2DProblemShape(convN);
    std::vector<KernelCandidate> candN = generateCandidates(probN);
    const KernelCandidate &cN =
        findCandidate(candN, Precision::INT8, Dataflow::WeightStationary, pe16, tile576);
    CandidateLegalityResult lN = checkCandidateLegality(cN, target);
    assert(lN.isLegal);
    llvm::Expected<CandidateCostEstimate> costN = estimateCandidateCost(cN, lN, target);
    assert(bool(costN));
    SchedulePlan schedN = mustMaterialize(cN, lN, *costN, probN, target);
    // logicalN=20, tileN=16 -> NT=2; final tile validN=20-16=4
    assert(schedN.logicalN == 20 && schedN.numNTiles == 2);
    bool foundNBoundary = false;
    for (const auto &op : schedN.operations)
      if (op.kind == ScheduleOperationKind::ComputeTile && op.coordinate.nTile == 1) {
        assert(op.extent.validN == 4 && op.extent.physicalN == 16);
        assert(op.isBoundaryTile);
        foundNBoundary = true;
      }
    assert(foundNBoundary);
    std::puts("  [ok] (38) synthetic problem demonstrates N-boundary padding "
              "(validN=4 < physicalN=16)");

    // K-boundary: inputChannels*kernelH*kernelW not divisible by tileK=64.
    Block bK;
    mlir::hir::Conv2dOp convK = buildConv2dOp(ctx, bK, 1, 100, 56, 56, 128, 1, 1, 54, 54); // logicalK=100
    Conv2DProblemShape probK = extractConv2DProblemShape(convK);
    std::vector<KernelCandidate> candK = generateCandidates(probK);
    const TileShape tileK64{16, 8, 8, 64};
    const KernelCandidate &cK =
        findCandidate(candK, Precision::INT8, Dataflow::WeightStationary, pe16, tileK64);
    CandidateLegalityResult lK = checkCandidateLegality(cK, target);
    assert(lK.isLegal);
    llvm::Expected<CandidateCostEstimate> costK = estimateCandidateCost(cK, lK, target);
    assert(bool(costK));
    SchedulePlan schedK = mustMaterialize(cK, lK, *costK, probK, target);
    // logicalK=100, tileK=64 -> KT=2; final k tile validK=100-64=36
    assert(schedK.logicalK == 100 && schedK.numKTiles == 2);
    bool foundKBoundary = false;
    for (const auto &op : schedK.operations)
      if (op.kind == ScheduleOperationKind::ComputeTile && op.coordinate.kTile == 1) {
        assert(op.extent.validK == 36 && op.extent.physicalK == 64);
        assert(op.isBoundaryTile);
        assert(op.isFinalReductionTile);
        foundKBoundary = true;
      }
    assert(foundKBoundary);
    // DMA byte counts must still use the PHYSICAL (padded) extent, never
    // shrunk to the valid boundary bytes (Section 12).
    for (const auto &op : schedK.operations)
      if (op.kind == ScheduleOperationKind::LoadInputTile)
        assert(op.bytes == static_cast<std::uint64_t>(lK.inputTileBytes)); // physical I, unchanged by boundary
    std::puts("  [ok] (39)+(12) synthetic problem demonstrates K-boundary padding "
              "(validK=36 < physicalK=64); DMA bytes stay at the physical (padded) extent");

    // ---- (40) valid extents never exceed physical extents, anywhere ----
    for (const SchedulePlan *s : {&schedM, &schedN, &schedK})
      for (const auto &op : s->operations) {
        assert(op.extent.validM <= op.extent.physicalM);
        assert(op.extent.validN <= op.extent.physicalN);
        assert(op.extent.validK <= op.extent.physicalK);
      }
    std::puts("  [ok] (40) valid extents never exceed physical extents in any boundary schedule");
  }

  // ---- (42)+(43) JSON serialization determinism and content ----
  {
    std::string json1 = serializeSchedulePlanToJson(schedule);
    std::string json2 = serializeSchedulePlanToJson(schedule);
    assert(json1 == json2);
    assert(json1.find(schedule.candidateId) != std::string::npos);
    assert(json1.find("\"loop_order\": [\"M\", \"K\", \"N\"]") != std::string::npos);
    assert(json1.find("\"k\": 576") != std::string::npos); // tileK
    assert(json1.find("\"input_across_n\": true") != std::string::npos);
    assert(json1.find("\"input_loads\": 46") != std::string::npos);
    assert(json1.find("\"weight_loads\": 368") != std::string::npos);
    assert(json1.find("\"synchronizations\": 368") != std::string::npos);
  }
  std::puts("  [ok] JSON serialization is deterministic and contains loop order, tileK, "
            "residency, and operation counts");

  // ---- (44)+(45): structural facts, not runtime-testable directly --
  // TileExtent/ScheduleOperation/SchedulePlan (see SchedulePlan.h) carry
  // no address/pointer/offset-into-memory field of any kind, and neither
  // SchedulePlan.h nor SchedulePlan.cpp includes or references any MLIR
  // lowering pass, simulator, runtime, or Hailo type.

  std::puts("=== SelectedScheduleMaterializationTest: PASS ===");
  return 0;
}
