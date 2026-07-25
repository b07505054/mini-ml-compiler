// Cost Model Slice 7: Buffer Lifetime and Scratchpad Allocation unit
// tests. Uses the real Slice 1-6 pipeline for the reference-problem
// tests, small synthetic Conv2D problems (via real mlir::hir::Conv2dOp
// instances) for the KT>1 dataflow-residency examples, and
// hand-constructed BufferLifetime lists (via allocateScratchpadFromLifetimes)
// for precise, hand-computable allocator/interference/alignment tests.
// Numbered comments correspond to the task's Section 20 test items.

#include "costmodel/ScratchpadAllocation.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdio>
#include <limits>

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

SchedulePlan mustSchedule(const KernelCandidate &c, const CandidateLegalityResult &l,
                          const CandidateCostEstimate &cost, const Conv2DProblemShape &problem,
                          const NPUTargetConfig &target) {
  llvm::Expected<SchedulePlan> s = materializeCandidateSchedule(c, l, cost, problem, target);
  if (!s) {
    std::fprintf(stderr, "unexpected schedule failure: %s\n",
                 llvm::toString(s.takeError()).c_str());
    std::abort();
  }
  return *s;
}

ScratchpadAllocationPlan mustAllocate(const SchedulePlan &schedule, const NPUTargetConfig &target) {
  llvm::Expected<ScratchpadAllocationPlan> p = allocateScratchpad(schedule, target);
  if (!p) {
    std::fprintf(stderr, "unexpected allocation failure: %s\n",
                 llvm::toString(p.takeError()).c_str());
    std::abort();
  }
  return *p;
}

template <typename T> bool failedExpected(llvm::Expected<T> &&e) {
  if (e) {
    (void)*e;
    return false;
  }
  llvm::consumeError(e.takeError());
  return true;
}

bool failedError(llvm::Error &&e) {
  if (!e)
    return false;
  llvm::consumeError(std::move(e));
  return true;
}

const BufferAllocation *findAllocFor(const ScratchpadAllocationPlan &plan, std::uint64_t bufferId) {
  for (const auto &a : plan.allocations)
    if (a.bufferId.value == bufferId)
      return &a;
  return nullptr;
}

BufferLifetime makeLifetime(std::uint64_t id, BufferRole role, MemoryRegionKind region,
                            std::uint64_t size, std::uint64_t align, std::uint64_t first,
                            std::uint64_t last, TileCoordinate coord = {}) {
  BufferLifetime bl;
  bl.bufferId = LogicalBufferId{id};
  bl.role = role;
  bl.coordinate = coord;
  bl.sizeBytes = size;
  bl.alignmentBytes = align;
  bl.firstOperationIndex = first;
  bl.lastOperationIndex = last;
  bl.region = region;
  return bl;
}

} // namespace

int main() {
  std::puts("=== ScratchpadAllocationTest ===");

  // ---- (1) sentinel: prove assertions are genuinely active ----
  {
    bool assertionsActive = false;
    assert((assertionsActive = true, true));
    if (!assertionsActive) {
      std::fprintf(stderr, "FATAL: assert() is compiled out in this test binary (NDEBUG?)\n");
      std::abort();
    }
  }
  std::puts("  [ok] sentinel confirms assert() is genuinely active");

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

  // ---- (2) Slices 1-6 still produce the reference selected schedule ----
  assert(candidates.size() == 144);
  std::size_t legalCount = 0, illegalCount = 0;
  for (auto &l : legality)
    (l.isLegal ? legalCount : illegalCount)++;
  assert(legalCount == 78 && illegalCount == 66);
  llvm::Expected<PlanSelectionResult> selectionOrErr =
      selectBestCandidate(candidates, legality, costs);
  assert(bool(selectionOrErr));
  PlanSelectionResult selection = std::move(*selectionOrErr);
  assert(selection.selectedCandidate.dataflow == Dataflow::InputStationary);
  assert(selection.selectedCost.totalEstimatedCycles == 908516ull);
  llvm::Expected<SchedulePlan> scheduleOrErr =
      materializeSelectedSchedule(selection, problem, target);
  assert(bool(scheduleOrErr));
  SchedulePlan schedule = std::move(*scheduleOrErr);
  CandidateLegalityResult selectedLegality =
      checkCandidateLegality(selection.selectedCandidate, target);
  assert(selectedLegality.isLegal);
  std::puts("  [ok] Slices 1-6 still produce the reference selected schedule (144/78/66, IS winner)");

  // ---- (3)+(4)+(5) reference tile sizes exact ----
  assert(selectedLegality.inputTileBytes == 36864);
  assert(selectedLegality.weightTileBytes == 9216);
  assert(selectedLegality.outputTileBytes == 4096);
  std::puts("  [ok] reference tile sizes: input=36864 weight=9216 output=4096");

  // ---- allocate ----
  ScratchpadAllocationPlan alloc = mustAllocate(schedule, target);

  // ---- (6)-(9) reference peak bytes exact ----
  assert(alloc.peakInputBufferBytes == 36864);
  assert(alloc.peakWeightBufferBytes == 9216);
  assert(alloc.peakOutputBufferBytes == 4096);
  assert(alloc.peakAggregateScratchpadBytes == 50176);
  std::printf("  [ok] reference peaks: input=%llu weight=%llu output=%llu aggregate=%llu\n",
              (unsigned long long)alloc.peakInputBufferBytes,
              (unsigned long long)alloc.peakWeightBufferBytes,
              (unsigned long long)alloc.peakOutputBufferBytes,
              (unsigned long long)alloc.peakAggregateScratchpadBytes);

  // ---- (10) all reference allocations fit the target ----
  assert(!validateAllocationAgainstTarget(alloc, target));
  std::puts("  [ok] all reference allocations fit the target");

  // ---- (11) every allocation offset satisfies alignment ----
  for (const auto &a : alloc.allocations)
    assert(a.offsetBytes % a.alignmentBytes == 0);
  std::puts("  [ok] every allocation offset satisfies alignment");

  // ---- (12) every lifetime has exactly one allocation ----
  assert(alloc.allocations.size() == alloc.lifetimes.size());
  for (const auto &lt : alloc.lifetimes)
    assert(findAllocFor(alloc, lt.bufferId.value) != nullptr);
  std::puts("  [ok] every lifetime has exactly one allocation");

  // ---- (13) no interfering same-region allocations overlap ----
  for (std::size_t i = 0; i < alloc.allocations.size(); ++i)
    for (std::size_t j = i + 1; j < alloc.allocations.size(); ++j) {
      const auto &a = alloc.allocations[i];
      const auto &b = alloc.allocations[j];
      if (a.region != b.region)
        continue;
      bool timeOverlap =
          !(a.lastOperationIndex < b.firstOperationIndex || b.lastOperationIndex < a.firstOperationIndex);
      if (!timeOverlap)
        continue;
      bool addrOverlap =
          a.offsetBytes < b.offsetBytes + b.sizeBytes && b.offsetBytes < a.offsetBytes + a.sizeBytes;
      assert(!addrOverlap);
    }
  std::puts("  [ok] no interfering same-region allocations overlap in address range");

  // ---- (14) disjoint same-region lifetimes reuse offsets: every input
  // allocation in the reference schedule reuses offset 0 (all 46 input
  // buffers are strictly sequential in this single-threaded schedule) ----
  {
    std::uint64_t inputCount = 0;
    for (const auto &a : alloc.allocations)
      if (a.region == MemoryRegionKind::InputBuffer) {
        assert(a.offsetBytes == 0);
        ++inputCount;
      }
    assert(inputCount == 46);
  }
  std::puts("  [ok] disjoint same-region (input) lifetimes reuse offset 0 (46 buffers, 1 offset)");

  // ---- (15) different regions independently use offset zero ----
  {
    bool inputAtZero = false, weightAtZero = false, outputAtZero = false;
    for (const auto &a : alloc.allocations) {
      if (a.region == MemoryRegionKind::InputBuffer && a.offsetBytes == 0)
        inputAtZero = true;
      if (a.region == MemoryRegionKind::WeightBuffer && a.offsetBytes == 0)
        weightAtZero = true;
      if (a.region == MemoryRegionKind::OutputBuffer && a.offsetBytes == 0)
        outputAtZero = true;
    }
    assert(inputAtZero && weightAtZero && outputAtZero);
  }
  std::puts("  [ok] different regions independently use offset zero");

  // ---- (27) allocation-versus-legality reconciliation succeeds for the
  // reference candidate ----
  assert(!validateAllocationAgainstLegality(alloc, selectedLegality));
  std::puts("  [ok] allocation-versus-legality reconciliation succeeds for the reference candidate");

  // ---- (28) a synthetic inconsistent legality result fails reconciliation ----
  {
    CandidateLegalityResult tooSmall = selectedLegality;
    tooSmall.inputTileBytes = 100; // far below the real 36864 requirement
    assert(failedError(validateAllocationAgainstLegality(alloc, tooSmall)));
  }
  std::puts("  [ok] a synthetic inconsistent legality result fails reconciliation");

  // ---- (24) candidate ID mismatch fails closed ----
  {
    ScratchpadAllocationPlan bad = alloc;
    bad.candidateId = "not-the-real-id";
    assert(failedError(validateAllocationAgainstLegality(bad, selectedLegality)));
  }
  std::puts("  [ok] candidate ID mismatch fails closed");

  // ---- (25) capacity overflow fails closed (per region) ----
  {
    NPUTargetConfig tooSmallInput = target;
    tooSmallInput.inputBufferBytes = 100;
    assert(failedError(validateAllocationAgainstTarget(alloc, tooSmallInput)));

    NPUTargetConfig tooSmallWeight = target;
    tooSmallWeight.weightBufferBytes = 100;
    assert(failedError(validateAllocationAgainstTarget(alloc, tooSmallWeight)));

    NPUTargetConfig tooSmallOutput = target;
    tooSmallOutput.outputBufferBytes = 100;
    assert(failedError(validateAllocationAgainstTarget(alloc, tooSmallOutput)));
  }
  std::puts("  [ok] per-region capacity overflow fails closed");

  // ---- (26) aggregate scratchpad overflow fails closed even if
  // individual regions fit ----
  {
    NPUTargetConfig tightAggregate = target;
    tightAggregate.scratchpadBytes = 50000; // < 50176, but each individual region still fits
    assert(failedError(validateAllocationAgainstTarget(alloc, tightAggregate)));
  }
  std::puts("  [ok] aggregate scratchpad overflow fails closed even when every region fits");

  // ---- (22)+(23) deterministic repeated + equivalent independently
  // built Conv2D operations ----
  {
    ScratchpadAllocationPlan again = mustAllocate(schedule, target);
    assert(again == alloc);

    Block block2;
    mlir::hir::Conv2dOp conv2 = buildConv2dOp(ctx, block2, 1, 64, 56, 56, 128, 3, 3, 54, 54);
    Conv2DProblemShape problem2 = extractConv2DProblemShape(conv2);
    std::vector<KernelCandidate> candidates2 = generateCandidates(problem2);
    std::vector<CandidateLegalityResult> legality2 = checkCandidateLegality(candidates2, target);
    llvm::Expected<std::vector<CandidateCostEstimate>> costs2 =
        estimateCandidateCosts(candidates2, legality2, target);
    assert(bool(costs2));
    llvm::Expected<PlanSelectionResult> selection2 =
        selectBestCandidate(candidates2, legality2, *costs2);
    assert(bool(selection2));
    llvm::Expected<SchedulePlan> schedule2 =
        materializeSelectedSchedule(*selection2, problem2, target);
    assert(bool(schedule2));
    ScratchpadAllocationPlan alloc2 = mustAllocate(*schedule2, target);
    assert(alloc2 == alloc);
  }
  std::puts("  [ok] allocation is deterministic across repeated calls and independently built ops");

  // ---- (31)+(32) JSON serialization ----
  {
    std::string json1 = serializeScratchpadAllocationPlanToJson(alloc);
    std::string json2 = serializeScratchpadAllocationPlanToJson(alloc);
    assert(json1 == json2);
    assert(json1.find(alloc.candidateId) != std::string::npos);
    assert(json1.find("\"input\": 36864") != std::string::npos);
    assert(json1.find("\"weight\": 9216") != std::string::npos);
    assert(json1.find("\"output\": 4096") != std::string::npos);
    assert(json1.find("\"aggregate\": 50176") != std::string::npos);
    assert(json1.find("\"region\"") != std::string::npos);
    assert(json1.find("\"offset\"") != std::string::npos);
    assert(json1.find("\"first_op\"") != std::string::npos);
    assert(json1.find("\"last_op\"") != std::string::npos);
  }
  std::puts("  [ok] JSON serialization is deterministic and contains regions/offsets/sizes/lifetimes/peaks");

  // ---- KT>1 dataflow-residency examples (WS, IS, OS) -- reuse Slice 6's
  // small tileK=64 (KT=9) permissive-target setup ----
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

    llvm::Expected<CandidateCostEstimate> wsCost = estimateCandidateCost(wsK64, wsL, permissive);
    llvm::Expected<CandidateCostEstimate> isCost = estimateCandidateCost(isK64, isL, permissive);
    llvm::Expected<CandidateCostEstimate> osCost = estimateCandidateCost(osK64, osL, permissive);
    assert(bool(wsCost) && bool(isCost) && bool(osCost));

    SchedulePlan wsSchedule = mustSchedule(wsK64, wsL, *wsCost, problem, permissive);
    SchedulePlan isSchedule = mustSchedule(isK64, isL, *isCost, problem, permissive);
    SchedulePlan osSchedule = mustSchedule(osK64, osL, *osCost, problem, permissive);
    assert(wsSchedule.numKTiles == 9 && isSchedule.numKTiles == 9 && osSchedule.numKTiles == 9);

    ScratchpadAllocationPlan wsAlloc = mustAllocate(wsSchedule, permissive);
    ScratchpadAllocationPlan isAlloc = mustAllocate(isSchedule, permissive);
    ScratchpadAllocationPlan osAlloc = mustAllocate(osSchedule, permissive);

    // ---- (16) WS weight lifetime spans multiple M computes ----
    for (const auto &lt : wsAlloc.lifetimes)
      if (lt.role == BufferRole::WeightTile) {
        std::uint64_t computeSpan = 0;
        for (const auto &op : wsSchedule.operations)
          if (op.kind == ScheduleOperationKind::ComputeTile &&
              op.operationIndex >= lt.firstOperationIndex && op.operationIndex <= lt.lastOperationIndex)
            ++computeSpan;
        assert(computeSpan == wsSchedule.numMTiles); // resident across all M
      }
    std::puts("  [ok] (16) WS weight lifetime spans multiple ComputeTile operations (all M)");

    // ---- (17) IS input lifetime spans multiple N computes ----
    for (const auto &lt : isAlloc.lifetimes)
      if (lt.role == BufferRole::InputTile) {
        std::uint64_t computeSpan = 0;
        for (const auto &op : isSchedule.operations)
          if (op.kind == ScheduleOperationKind::ComputeTile &&
              op.operationIndex >= lt.firstOperationIndex && op.operationIndex <= lt.lastOperationIndex)
            ++computeSpan;
        assert(computeSpan == isSchedule.numNTiles); // resident across all N
      }
    std::puts("  [ok] (17) IS input lifetime spans multiple ComputeTile operations (all N)");

    // ---- (18)+(19) OS output lifetime spans multiple K computes; one
    // resident accumulator per (m,n), not KT accumulators ----
    std::uint64_t osAccumCount = 0;
    for (const auto &lt : osAlloc.lifetimes)
      if (lt.role == BufferRole::OutputAccumulator) {
        ++osAccumCount;
        std::uint64_t computeSpan = 0;
        for (const auto &op : osSchedule.operations)
          if (op.kind == ScheduleOperationKind::ComputeTile &&
              op.operationIndex >= lt.firstOperationIndex && op.operationIndex <= lt.lastOperationIndex)
            ++computeSpan;
        assert(computeSpan == osSchedule.numKTiles); // resident across all K
      }
    assert(osAccumCount == osSchedule.numMTiles * osSchedule.numNTiles); // one per (m,n), not KT each
    std::puts("  [ok] (18)+(19) OS output lifetime spans all K computes; exactly one resident "
              "accumulator per (m,n), not KT accumulators");

    // ---- (20) WS/IS partial-spill schedules create separate accumulator
    // residency intervals across K tiles: numAccumulators == MT*NT*KT for
    // both WS and IS (one short-lived interval per k, not one long one) ----
    std::uint64_t wsAccumCount = 0, isAccumCount = 0;
    for (const auto &lt : wsAlloc.lifetimes)
      if (lt.role == BufferRole::OutputAccumulator)
        ++wsAccumCount;
    for (const auto &lt : isAlloc.lifetimes)
      if (lt.role == BufferRole::OutputAccumulator)
        ++isAccumCount;
    assert(wsAccumCount == wsSchedule.numMTiles * wsSchedule.numNTiles * wsSchedule.numKTiles);
    assert(isAccumCount == isSchedule.numMTiles * isSchedule.numNTiles * isSchedule.numKTiles);
    std::puts("  [ok] (20) WS/IS create MT*NT*KT separate accumulator residency intervals "
              "(one per k tile), not MT*NT resident ones");

    // Sanity: WS/IS/OS all still reconcile against their own cost estimates.
    assert(!validateAllocationAgainstTarget(wsAlloc, permissive));
    assert(!validateAllocationAgainstTarget(isAlloc, permissive));
    assert(!validateAllocationAgainstTarget(osAlloc, permissive));
  }

  // ---- (21) boundary tiles use physical padded allocation sizes ----
  {
    Block bM;
    mlir::hir::Conv2dOp convM = buildConv2dOp(ctx, bM, 1, 64, 56, 56, 128, 3, 3, 10, 10);
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
    SchedulePlan schedM = mustSchedule(cM, lM, *costM, probM, target);
    assert(schedM.numMTiles == 2); // logicalM=100, tileM=64 -> boundary on final M tile
    ScratchpadAllocationPlan allocM = mustAllocate(schedM, target);
    // Every InputTile lifetime (boundary or not) uses the full physical
    // 64*576*1=36864-byte footprint -- never shrunk to the boundary's
    // valid 36-row extent.
    for (const auto &lt : allocM.lifetimes)
      if (lt.role == BufferRole::InputTile)
        assert(lt.sizeBytes == 36864);
  }
  std::puts("  [ok] boundary tiles use the physical padded allocation size, never a compact "
            "boundary-shrunk size");

  // ---- (29)+(30)+(2 from Section 17): hand-computable first-fit and
  // alignment-padding cases via allocateScratchpadFromLifetimes ----
  {
    // Two disjoint input lifetimes (sequential) reuse offset 0.
    std::vector<BufferLifetime> disjoint = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 100, 16, 0, 5),
        makeLifetime(1, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 100, 16, 6, 10),
    };
    llvm::Expected<ScratchpadAllocationPlan> p =
        allocateScratchpadFromLifetimes("synthetic:disjoint", disjoint);
    assert(bool(p));
    assert(findAllocFor(*p, 0)->offsetBytes == 0);
    assert(findAllocFor(*p, 1)->offsetBytes == 0); // reused, no overlap
    assert(p->peakInputBufferBytes == 100);
  }
  std::puts("  [ok] (17.1) two disjoint input lifetimes reuse the same offset");

  {
    // Two OVERLAPPING input lifetimes cannot share an offset.
    std::vector<BufferLifetime> overlapping = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 100, 16, 0, 10),
        makeLifetime(1, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 100, 16, 5, 15),
    };
    llvm::Expected<ScratchpadAllocationPlan> p =
        allocateScratchpadFromLifetimes("synthetic:overlap", overlapping);
    assert(bool(p));
    const BufferAllocation *a0 = findAllocFor(*p, 0);
    const BufferAllocation *a1 = findAllocFor(*p, 1);
    assert(a0->offsetBytes != a1->offsetBytes);
    // (30) exact first-fit + alignment padding: size 100 rounds the next
    // offset up to the next multiple of 16 -- ceil(100/16)*16 = 112.
    assert(a0->offsetBytes == 0);
    assert(a1->offsetBytes == 112);
    // Peak bytes are a timeline sweep of concurrently-LIVE sizes (Section
    // 11), independent of the allocator's chosen offsets: both 100-byte
    // buffers are live simultaneously during operation indices 5-10, so
    // peak = 100+100 = 200 (NOT the allocator's offset-based footprint of
    // 112+100=212, which reflects alignment padding rather than true
    // concurrent memory need).
    assert(p->peakInputBufferBytes == 200);
  }
  std::puts("  [ok] (17.2)+(29)+(30) two overlapping input lifetimes cannot share an offset; "
            "first-fit picks the lowest aligned offset (112) with exact alignment padding");

  {
    // Two disjoint WEIGHT lifetimes reuse the same offset.
    std::vector<BufferLifetime> disjointWeight = {
        makeLifetime(0, BufferRole::WeightTile, MemoryRegionKind::WeightBuffer, 200, 16, 0, 3),
        makeLifetime(1, BufferRole::WeightTile, MemoryRegionKind::WeightBuffer, 200, 16, 4, 7),
    };
    llvm::Expected<ScratchpadAllocationPlan> p =
        allocateScratchpadFromLifetimes("synthetic:disjoint_weight", disjointWeight);
    assert(bool(p));
    assert(findAllocFor(*p, 0)->offsetBytes == 0);
    assert(findAllocFor(*p, 1)->offsetBytes == 0);
  }
  std::puts("  [ok] (17.3) two disjoint weight lifetimes reuse the same offset");

  {
    // Two overlapping OUTPUT lifetimes do not overlap addresses.
    std::vector<BufferLifetime> overlappingOutput = {
        makeLifetime(0, BufferRole::OutputAccumulator, MemoryRegionKind::OutputBuffer, 64, 16, 0, 10),
        makeLifetime(1, BufferRole::OutputAccumulator, MemoryRegionKind::OutputBuffer, 64, 16, 3, 12),
    };
    llvm::Expected<ScratchpadAllocationPlan> p =
        allocateScratchpadFromLifetimes("synthetic:overlap_output", overlappingOutput);
    assert(bool(p));
    const BufferAllocation *a0 = findAllocFor(*p, 0);
    const BufferAllocation *a1 = findAllocFor(*p, 1);
    bool overlap = a0->offsetBytes < a1->offsetBytes + a1->sizeBytes &&
                  a1->offsetBytes < a0->offsetBytes + a0->sizeBytes;
    assert(!overlap);
  }
  std::puts("  [ok] (17.4) two overlapping output lifetimes do not overlap in address range");

  {
    // (17.5) Buffers in different regions may have the same relative offset.
    std::vector<BufferLifetime> mixed = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 64, 16, 0, 5),
        makeLifetime(1, BufferRole::WeightTile, MemoryRegionKind::WeightBuffer, 64, 16, 0, 5),
        makeLifetime(2, BufferRole::OutputAccumulator, MemoryRegionKind::OutputBuffer, 64, 16, 0, 5),
    };
    llvm::Expected<ScratchpadAllocationPlan> p =
        allocateScratchpadFromLifetimes("synthetic:mixed_regions", mixed);
    assert(bool(p));
    assert(findAllocFor(*p, 0)->offsetBytes == 0);
    assert(findAllocFor(*p, 1)->offsetBytes == 0);
    assert(findAllocFor(*p, 2)->offsetBytes == 0);
  }
  std::puts("  [ok] (17.5) buffers in different regions independently use the same relative offset");

  // ---- Section 18 failure tests ----
  {
    // duplicate buffer ID
    std::vector<BufferLifetime> dup = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 64, 16, 0, 5),
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 64, 16, 10, 15),
    };
    llvm::Expected<ScratchpadAllocationPlan> p =
        allocateScratchpadFromLifetimes("synthetic:dup", dup);
    // allocateScratchpadFromLifetimes itself does not dedupe-check at the
    // lifetime level, but validateAllocationAgainstTarget must catch the
    // resulting duplicate allocation deterministically.
    assert(bool(p));
    assert(failedError(validateAllocationAgainstTarget(*p, target)));
  }
  std::puts("  [ok] duplicate buffer ID fails closed (caught by target validation)");

  {
    // zero buffer size
    std::vector<BufferLifetime> zeroSize = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 0, 16, 0, 5),
    };
    assert(failedExpected(allocateScratchpadFromLifetimes("synthetic:zero_size", zeroSize)));
  }
  std::puts("  [ok] zero buffer size fails closed");

  {
    // zero alignment
    std::vector<BufferLifetime> zeroAlign = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 64, 0, 0, 5),
    };
    assert(failedExpected(allocateScratchpadFromLifetimes("synthetic:zero_align", zeroAlign)));
  }
  std::puts("  [ok] zero alignment fails closed");

  {
    // malformed operation ordering (non-dense indices) -> analyzeBufferLifetimes fails
    SchedulePlan bad = schedule;
    bad.operations[1].operationIndex = 999;
    assert(failedExpected(analyzeBufferLifetimes(bad, target)));
  }
  std::puts("  [ok] malformed operation ordering fails closed");

  {
    // missing load before compute: drop the first LoadInputTile op
    SchedulePlan bad = schedule;
    // find and remove the very first LoadInputTile
    for (auto it = bad.operations.begin(); it != bad.operations.end(); ++it)
      if (it->kind == ScheduleOperationKind::LoadInputTile) {
        bad.operations.erase(it);
        break;
      }
    for (std::size_t i = 0; i < bad.operations.size(); ++i)
      bad.operations[i].operationIndex = i;
    assert(failedExpected(analyzeBufferLifetimes(bad, target)));
  }
  std::puts("  [ok] missing load before compute fails closed");

  {
    // missing accumulator before compute: remove... construct directly via
    // a minimal hand-built 2-op schedule: a ComputeTile at kTile>0 with no
    // preceding load/reload at all.
    SchedulePlan tiny;
    tiny.candidateId = "synthetic:missing_accum";
    tiny.precision = Precision::INT8;
    tiny.dataflow = Dataflow::WeightStationary;
    tiny.peArray = PEArrayShape{16, 16};
    tiny.tile = TileShape{16, 8, 8, 64};
    tiny.numMTiles = 1;
    tiny.numNTiles = 1;
    tiny.numKTiles = 2;
    ScheduleOperation loadIn;
    loadIn.operationIndex = 0;
    loadIn.kind = ScheduleOperationKind::LoadInputTile;
    loadIn.coordinate = {0, 0, 1};
    loadIn.bytes = 64;
    ScheduleOperation loadW;
    loadW.operationIndex = 1;
    loadW.kind = ScheduleOperationKind::LoadWeightTile;
    loadW.coordinate = {0, 0, 1};
    loadW.bytes = 64;
    ScheduleOperation compute;
    compute.operationIndex = 2;
    compute.kind = ScheduleOperationKind::ComputeTile;
    compute.coordinate = {0, 0, 1}; // kTile=1 > 0, but no ReloadPartialOutput preceded it
    tiny.operations = {loadIn, loadW, compute};
    assert(failedExpected(analyzeBufferLifetimes(tiny, target)));
  }
  std::puts("  [ok] missing accumulator before compute (kTile>0, no reload) fails closed");

  {
    // store without a live accumulator
    SchedulePlan tiny;
    tiny.candidateId = "synthetic:store_no_accum";
    tiny.dataflow = Dataflow::WeightStationary;
    ScheduleOperation store;
    store.operationIndex = 0;
    store.kind = ScheduleOperationKind::StoreFinalOutput;
    store.coordinate = {0, 0, 0};
    store.bytes = 64;
    tiny.operations = {store};
    assert(failedExpected(analyzeBufferLifetimes(tiny, target)));
  }
  std::puts("  [ok] store without a live accumulator fails closed");

  {
    // address arithmetic overflow
    std::vector<BufferLifetime> overflowing = {
        makeLifetime(0, BufferRole::InputTile, MemoryRegionKind::InputBuffer,
                    std::numeric_limits<std::uint64_t>::max() - 10, 16, 0, 5),
        makeLifetime(1, BufferRole::InputTile, MemoryRegionKind::InputBuffer, 100, 16, 3, 8),
    };
    assert(failedExpected(allocateScratchpadFromLifetimes("synthetic:overflow", overflowing)));
  }
  std::puts("  [ok] address arithmetic overflow fails closed");

  // ---- (33)-(35): structural facts, not runtime-testable directly --
  // BufferLifetime/BufferAllocation/ScratchpadAllocationPlan (see
  // ScratchpadAllocation.h) carry no double-buffering flag, no
  // asynchronous-DMA-dependency field, and no physical hardware address
  // of any kind (offsetBytes is always symbolic/region-relative); neither
  // ScratchpadAllocation.h nor .cpp includes or references any MLIR
  // lowering pass, simulator, runtime, or Hailo type.

  std::puts("=== ScratchpadAllocationTest: PASS ===");
  return 0;
}
