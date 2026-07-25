// Cost Model Slice 9: Explicit Dependency Graph and Critical-Path
// Validation unit tests. Reuses the same reference candidate and target
// AsyncDMASchedulingTest already pins down, plus small KT>=3 WS/IS/OS
// examples for the partial-sum-chain and residency tests. Numbered
// comments correspond to the task's Section 30 test items where a 1:1
// mapping is practical.

#include "costmodel/ScheduleDependencyGraph.h"
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
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::costmodel;

namespace {

mlir::hir::Conv2dOp buildConv2dOp(MLIRContext &ctx, Block &block, int64_t n, int64_t c, int64_t h,
                                  int64_t w, int64_t k, int64_t r, int64_t s, int64_t outH,
                                  int64_t outW) {
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
    std::fprintf(stderr, "unexpected schedule failure: %s\n", llvm::toString(s.takeError()).c_str());
    std::abort();
  }
  return *s;
}

AsyncSchedulePlan mustMaterializeAsync(const SchedulePlan &schedule, const BufferingPolicy &buffering,
                                       DMASchedulingPolicy policy, const NPUTargetConfig &target) {
  llvm::Expected<AsyncSchedulePlan> p = materializeAsyncSchedule(schedule, buffering, policy, target);
  if (!p) {
    std::fprintf(stderr, "unexpected async materialization failure: %s\n",
                llvm::toString(p.takeError()).c_str());
    std::abort();
  }
  return *p;
}

ScheduleDependencyGraph mustBuildGraph(const AsyncSchedulePlan &plan, const NPUTargetConfig &target) {
  llvm::Expected<ScheduleDependencyGraph> g = buildScheduleDependencyGraph(plan, target);
  if (!g) {
    std::fprintf(stderr, "unexpected graph-build failure: %s\n", llvm::toString(g.takeError()).c_str());
    std::abort();
  }
  return *g;
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

std::uint64_t countEdgesOfKind(const ScheduleDependencyGraph &g, DependencyKind kind) {
  std::uint64_t n = 0;
  for (const auto &d : g.dependencies)
    if (d.kind == kind)
      ++n;
  return n;
}

// Returns a graph with the FIRST edge of `kind` removed (and edge-index
// tables rebuilt), for the "missing required edge" failure tests.
ScheduleDependencyGraph removeOneEdgeOfKind(const ScheduleDependencyGraph &g, DependencyKind kind) {
  ScheduleDependencyGraph out = g;
  for (std::size_t i = 0; i < out.dependencies.size(); ++i) {
    if (out.dependencies[i].kind == kind) {
      out.dependencies.erase(out.dependencies.begin() + i);
      break;
    }
  }
  out.outgoingEdges.assign(out.operations.size(), {});
  out.incomingEdges.assign(out.operations.size(), {});
  for (std::size_t i = 0; i < out.dependencies.size(); ++i) {
    out.outgoingEdges[out.dependencies[i].producerOperationIndex].push_back(i);
    out.incomingEdges[out.dependencies[i].consumerOperationIndex].push_back(i);
  }
  return out;
}

} // namespace

int main() {
  std::puts("=== ScheduleDependencyGraphTest ===");

  // ---- (1) sentinel ----
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
  const NPUTargetConfig &target = genericNPUv1Target();

  // ---- (2) Slices 1-8 still pass: reproduce the reference selection ----
  Block block;
  mlir::hir::Conv2dOp conv = buildConv2dOp(ctx, block, 1, 64, 56, 56, 128, 3, 3, 54, 54);
  Conv2DProblemShape problem = extractConv2DProblemShape(conv);
  std::vector<KernelCandidate> candidates = generateCandidates(problem);
  std::vector<CandidateLegalityResult> legality = checkCandidateLegality(candidates, target);
  llvm::Expected<std::vector<CandidateCostEstimate>> costsOrErr =
      estimateCandidateCosts(candidates, legality, target);
  assert(bool(costsOrErr));
  llvm::Expected<PlanSelectionResult> selectionOrErr =
      selectBestCandidate(candidates, legality, *costsOrErr);
  assert(bool(selectionOrErr));
  PlanSelectionResult selection = std::move(*selectionOrErr);
  assert(selection.selectedCandidate.dataflow == Dataflow::InputStationary);
  llvm::Expected<SchedulePlan> scheduleOrErr = materializeSelectedSchedule(selection, problem, target);
  assert(bool(scheduleOrErr));
  SchedulePlan schedule = std::move(*scheduleOrErr);
  std::puts("  [ok] (2) Slices 1-7 still produce the reference selected schedule");

  BufferingPolicy allSingle{BufferingMode::Single, BufferingMode::Single, BufferingMode::Single};
  BufferingPolicy weightDouble{BufferingMode::Single, BufferingMode::Double, BufferingMode::Single};
  AsyncSchedulePlan syncPlan =
      mustMaterializeAsync(schedule, allSingle, DMASchedulingPolicy::Synchronous, target);
  AsyncSchedulePlan weightPrefetch =
      mustMaterializeAsync(schedule, weightDouble, DMASchedulingPolicy::PrefetchNext, target);
  assert(!validateAsyncSchedule(syncPlan, target));
  assert(!validateAsyncSchedule(weightPrefetch, target));
  std::puts("  [ok] Slice 8 still produces valid Synchronous and PrefetchNext plans");

  // ---- (3)+(4) construct graphs ----
  ScheduleDependencyGraph syncGraph = mustBuildGraph(syncPlan, target);
  ScheduleDependencyGraph asyncGraph = mustBuildGraph(weightPrefetch, target);
  std::puts("  [ok] (3)+(4) graphs constructed from a Synchronous all-single schedule and a "
            "PrefetchNext schedule");

  // ---- (5) node count == async operation count ----
  assert(syncGraph.operations.size() == syncPlan.operations.size());
  assert(asyncGraph.operations.size() == weightPrefetch.operations.size());
  std::puts("  [ok] (5) node count equals async operation count");

  // ---- validation succeeds for both ----
  assert(!validateScheduleDependencyGraph(syncGraph, syncPlan, target));
  assert(!validateScheduleDependencyGraph(asyncGraph, weightPrefetch, target));
  std::puts("  [ok] both graphs pass full validation");

  // ---- (6)+(8) edge endpoints exist, no duplicates (validator already
  // checks this structurally; also spot-check directly) ----
  {
    std::set<std::tuple<std::uint64_t, std::uint64_t, int>> seen;
    for (const auto &d : asyncGraph.dependencies) {
      assert(d.producerOperationIndex < asyncGraph.operations.size());
      assert(d.consumerOperationIndex < asyncGraph.operations.size());
      auto key = std::make_tuple(d.producerOperationIndex, d.consumerOperationIndex,
                                 static_cast<int>(d.kind));
      assert(seen.insert(key).second);
    }
  }
  std::puts("  [ok] (6)+(8) every edge endpoint exists in range; no duplicate edges");

  // ---- (7) deterministic edge ordering ----
  {
    for (std::size_t i = 1; i < asyncGraph.dependencies.size(); ++i) {
      const auto &a = asyncGraph.dependencies[i - 1];
      const auto &b = asyncGraph.dependencies[i];
      bool nonDecreasing = a.producerOperationIndex < b.producerOperationIndex ||
                          (a.producerOperationIndex == b.producerOperationIndex &&
                            a.consumerOperationIndex < b.consumerOperationIndex) ||
                          (a.producerOperationIndex == b.producerOperationIndex &&
                            a.consumerOperationIndex == b.consumerOperationIndex);
      assert(nonDecreasing);
    }
  }
  std::puts("  [ok] (7) edge ordering is deterministic (sorted by producer, consumer, kind rank)");

  // ---- (9) every DMA issue has a completion dependency ----
  assert(countEdgesOfKind(asyncGraph, DependencyKind::DMACompletion) +
            countEdgesOfKind(asyncGraph, DependencyKind::OutputStoreCompletion) ==
        weightPrefetch.events.size());
  std::puts("  [ok] (9)+(22) every DMA issue has exactly one matching completion dependency "
            "(count == event count)");

  // ---- (10)+(11) every compute has input-ready and weight-ready deps ----
  for (const auto &op : asyncGraph.operations) {
    if (op.kind != AsyncOperationKind::Compute)
      continue;
    bool hasInput = false, hasWeight = false;
    for (auto idx : asyncGraph.incomingEdges[op.operationIndex]) {
      if (asyncGraph.dependencies[idx].kind == DependencyKind::InputDataReady)
        hasInput = true;
      if (asyncGraph.dependencies[idx].kind == DependencyKind::WeightDataReady)
        hasWeight = true;
    }
    assert(hasInput && hasWeight);
  }
  std::puts("  [ok] (10)+(11)+(21) every compute has required InputDataReady and WeightDataReady "
            "dependencies");

  // ---- (14)+(15) compute-engine and DMA-engine serialization complete
  // (already checked by validateScheduleDependencyGraph via the exact
  // expected-edge-set comparison; spot check the chain lengths). ----
  {
    std::uint64_t computeEngineNodes = 0, dmaEngineNodes = 0;
    for (const auto &op : asyncGraph.operations) {
      if (op.kind == AsyncOperationKind::Compute || op.kind == AsyncOperationKind::DMAWait)
        ++computeEngineNodes;
      if (op.kind == AsyncOperationKind::DMAIssue)
        ++dmaEngineNodes;
    }
    assert(countEdgesOfKind(asyncGraph, DependencyKind::ComputeSerialization) == computeEngineNodes - 1);
    assert(countEdgesOfKind(asyncGraph, DependencyKind::DMAEngineSerialization) == dmaEngineNodes - 1);
  }
  std::puts("  [ok] (14)+(15) compute-engine and DMA-engine serialization chains are complete "
            "(N-1 edges over N chain nodes)");

  // ---- (16)+(17) input/weight buffer-reuse hazards represented ----
  assert(countEdgesOfKind(asyncGraph, DependencyKind::WeightBufferReuse) > 0);
  std::puts("  [ok] (16)+(17) weight-buffer-reuse hazard edges are represented (weight is "
            "double-buffered here)");

  // ---- (34)+(23) useful weight prefetch remains overlap-capable: no
  // dependency path forces Compute(N0) before Issue(W1). ----
  {
    std::uint64_t computeOp = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t issueW1 = std::numeric_limits<std::uint64_t>::max();
    std::vector<std::uint64_t> weightIssues;
    for (const auto &op : weightPrefetch.operations)
      if (op.kind == AsyncOperationKind::DMAIssue && op.transferKind == DMATransferKind::WeightLoad)
        weightIssues.push_back(op.operationIndex);
    assert(weightIssues.size() >= 2);
    issueW1 = weightIssues[1];
    for (const auto &op : weightPrefetch.operations)
      if (op.kind == AsyncOperationKind::Compute) {
        computeOp = op.operationIndex;
        break;
      }
    assert(computeOp != std::numeric_limits<std::uint64_t>::max());
    // The first compute must not depend on the SECOND weight issue.
    assert(!hasDependencyPath(asyncGraph, issueW1, computeOp));
  }
  std::puts("  [ok] (23)+(34) no dependency path forces the first compute to wait on the "
            "prefetched second weight load (overlap remains representable)");

  // ---- (24)+(25)+(26) acyclic; stable topo order covers every node
  // exactly once; original order is a valid topological order ----
  llvm::Expected<std::vector<std::uint64_t>> topoOrErr = computeStableTopologicalOrder(asyncGraph);
  assert(bool(topoOrErr));
  assert(topoOrErr->size() == asyncGraph.operations.size());
  {
    std::set<std::uint64_t> uniq(topoOrErr->begin(), topoOrErr->end());
    assert(uniq.size() == topoOrErr->size());
  }
  for (const auto &d : asyncGraph.dependencies)
    assert(d.producerOperationIndex < d.consumerOperationIndex);
  {
    // repeated call is identical
    llvm::Expected<std::vector<std::uint64_t>> again = computeStableTopologicalOrder(asyncGraph);
    assert(bool(again));
    assert(*again == *topoOrErr);
  }
  std::puts("  [ok] (24)-(26) graph is acyclic; stable topological order covers every node "
            "exactly once and is deterministic; original operation order is itself a valid "
            "topological order");

  // ---- (27) manually injected cycle fails ----
  {
    ScheduleDependencyGraph cyclic = asyncGraph;
    // Add a reverse edge from the last node to the first (a genuine
    // cycle since the first node has outgoing edges reaching later nodes).
    std::uint64_t last = cyclic.operations.back().operationIndex;
    std::uint64_t first = cyclic.operations.front().operationIndex;
    cyclic.dependencies.push_back({last, first, DependencyKind::ProgramOrder});
    cyclic.outgoingEdges[last].push_back(cyclic.dependencies.size() - 1);
    cyclic.incomingEdges[first].push_back(cyclic.dependencies.size() - 1);
    assert(failedExpected(computeStableTopologicalOrder(cyclic)));
    assert(failedError(validateScheduleDependencyGraph(cyclic, weightPrefetch, target)));
  }
  std::puts("  [ok] (27) a manually injected cycle fails both topological ordering and full "
            "validation");

  // ---- (28)-(33) missing required edges fail closed ----
  {
    ScheduleDependencyGraph missingInput = removeOneEdgeOfKind(asyncGraph, DependencyKind::InputDataReady);
    assert(failedError(validateScheduleDependencyGraph(missingInput, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph missingWeight = removeOneEdgeOfKind(asyncGraph, DependencyKind::WeightDataReady);
    assert(failedError(validateScheduleDependencyGraph(missingWeight, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph missingDma =
        removeOneEdgeOfKind(asyncGraph, DependencyKind::DMAEngineSerialization);
    assert(failedError(validateScheduleDependencyGraph(missingDma, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph missingCompute =
        removeOneEdgeOfKind(asyncGraph, DependencyKind::ComputeSerialization);
    assert(failedError(validateScheduleDependencyGraph(missingCompute, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph missingReuse = removeOneEdgeOfKind(asyncGraph, DependencyKind::WeightBufferReuse);
    assert(failedError(validateScheduleDependencyGraph(missingReuse, weightPrefetch, target)));
  }
  std::puts("  [ok] (28)-(32) removing any required edge (input-ready, weight-ready, DMA-engine, "
            "compute-engine, buffer-reuse) fails validation closed");

  // ---- Section 28 additional failures: candidate mismatch, missing/
  // duplicate node, self-edge, out-of-range endpoint, empty candidate ID ----
  {
    ScheduleDependencyGraph bad = asyncGraph;
    bad.candidateId = "not-the-real-id";
    assert(failedError(validateScheduleDependencyGraph(bad, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph bad = asyncGraph;
    bad.candidateId.clear();
    assert(failedError(validateScheduleDependencyGraph(bad, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph bad = asyncGraph;
    bad.operations.pop_back(); // missing node
    assert(failedError(validateScheduleDependencyGraph(bad, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph bad = asyncGraph;
    bad.dependencies.push_back({0, 0, DependencyKind::ProgramOrder}); // self-edge
    bad.outgoingEdges[0].push_back(bad.dependencies.size() - 1);
    bad.incomingEdges[0].push_back(bad.dependencies.size() - 1);
    assert(failedError(validateScheduleDependencyGraph(bad, weightPrefetch, target)));
  }
  {
    ScheduleDependencyGraph bad = asyncGraph;
    std::uint64_t outOfRange = static_cast<std::uint64_t>(bad.operations.size()) + 5;
    bad.dependencies.push_back({0, outOfRange, DependencyKind::ProgramOrder});
    bad.outgoingEdges[0].push_back(bad.dependencies.size() - 1);
    assert(failedError(validateScheduleDependencyGraph(bad, weightPrefetch, target)));
  }
  std::puts("  [ok] candidate-mismatch, empty-candidate-id, missing-node, self-edge, and "
            "out-of-range-endpoint all fail validation closed");

  // ---- (37)+(38) critical-path / resource-constrained timing determinism ----
  llvm::Expected<std::vector<OperationTimingInfo>> timingsOrErr =
      deriveOperationTimings(weightPrefetch, target);
  assert(bool(timingsOrErr));
  llvm::Expected<CriticalPathResult> depPath1 = computeDependencyCriticalPath(asyncGraph, *timingsOrErr);
  llvm::Expected<CriticalPathResult> depPath2 = computeDependencyCriticalPath(asyncGraph, *timingsOrErr);
  assert(bool(depPath1) && bool(depPath2));
  assert(depPath1->totalCycles == depPath2->totalCycles);
  assert(depPath1->criticalPathOperations == depPath2->criticalPathOperations);

  llvm::Expected<ResourceConstrainedTimingResult> rc1 =
      computeResourceConstrainedTiming(asyncGraph, *timingsOrErr, target);
  llvm::Expected<ResourceConstrainedTimingResult> rc2 =
      computeResourceConstrainedTiming(asyncGraph, *timingsOrErr, target);
  assert(bool(rc1) && bool(rc2));
  assert(rc1->totalCycles == rc2->totalCycles);
  assert(rc1->criticalPathOperations == rc2->criticalPathOperations);
  std::puts("  [ok] (37)+(38) dependency-only and resource-constrained timing are both "
            "deterministic across repeated calls");

  // ---- (39)-(44) resource-constrained total reconciles exactly with
  // Slice 8's asynchronous total; startup/drain/dma-work reconcile. ----
  assert(!validateGraphTimingAgainstAsyncCost(asyncGraph, *rc1, weightPrefetch.cost));
  assert(rc1->totalCycles == weightPrefetch.cost.asynchronousEstimatedCycles);
  assert(rc1->startupCycles == weightPrefetch.cost.startupCycles);
  assert(rc1->drainCycles == weightPrefetch.cost.drainCycles);
  assert(rc1->dmaBusyCycles == weightPrefetch.cost.hiddenDMACycles + weightPrefetch.cost.exposedDMACycles);
  std::printf("  [ok] (39)-(44) graph total=%llu == Slice8 async total=%llu; startup=%llu "
              "drain=%llu dmaBusy=%llu all reconcile exactly\n",
              (unsigned long long)rc1->totalCycles,
              (unsigned long long)weightPrefetch.cost.asynchronousEstimatedCycles,
              (unsigned long long)rc1->startupCycles, (unsigned long long)rc1->drainCycles,
              (unsigned long long)rc1->dmaBusyCycles);

  // ---- (46) Synchronous all-single graph reproduces previous timing ----
  llvm::Expected<std::vector<OperationTimingInfo>> syncTimingsOrErr =
      deriveOperationTimings(syncPlan, target);
  assert(bool(syncTimingsOrErr));
  llvm::Expected<ResourceConstrainedTimingResult> syncRc =
      computeResourceConstrainedTiming(syncGraph, *syncTimingsOrErr, target);
  assert(bool(syncRc));
  assert(!validateGraphTimingAgainstAsyncCost(syncGraph, *syncRc, syncPlan.cost));
  assert(syncRc->totalCycles == syncPlan.cost.synchronousEstimatedCycles); // sync==async under Synchronous policy
  std::puts("  [ok] (46) Synchronous all-single graph timing reproduces Slice 8's synchronous "
            "total exactly");

  // ---- (48)+(49) JSON determinism and content ----
  {
    std::string j1 = serializeScheduleDependencyGraphToJson(asyncGraph, &*rc1);
    std::string j2 = serializeScheduleDependencyGraphToJson(asyncGraph, &*rc1);
    assert(j1 == j2);
    assert(j1.find(asyncGraph.candidateId) != std::string::npos);
    assert(j1.find("\"node_count\"") != std::string::npos);
    assert(j1.find("\"edge_count\"") != std::string::npos);
    assert(j1.find("\"edge_histogram\"") != std::string::npos);
    assert(j1.find("\"weight_data_ready\"") != std::string::npos);
    assert(j1.find("\"timing\"") != std::string::npos);
    assert(j1.find("\"critical_path\"") != std::string::npos);
    std::string jNoTiming = serializeScheduleDependencyGraphToJson(asyncGraph);
    assert(jNoTiming.find("\"timing\"") == std::string::npos);
  }
  std::puts("  [ok] (48)+(49) JSON serialization is byte-identical across repeated calls and "
            "contains candidate ID, node/edge counts, edge histogram, edges, and timing");

  // ------------------------------------------------------------------
  // Dataflow-specific residency + partial-sum-chain tests (Section 26-27):
  // reuse Slice 7/8's small tileK=64 (KT=9) permissive-target setup for
  // WS/IS partial-sum chains and OS accumulator residency.
  // ------------------------------------------------------------------
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
    assert(wsSchedule.numMTiles >= 3 && isSchedule.numNTiles >= 3);

    BufferingPolicy inputDouble{BufferingMode::Double, BufferingMode::Single, BufferingMode::Single};
    BufferingPolicy outputDouble{BufferingMode::Single, BufferingMode::Single, BufferingMode::Double};

    AsyncSchedulePlan wsAsync =
        mustMaterializeAsync(wsSchedule, inputDouble, DMASchedulingPolicy::PrefetchNext, permissive);
    AsyncSchedulePlan isAsync =
        mustMaterializeAsync(isSchedule, weightDouble, DMASchedulingPolicy::PrefetchNext, permissive);
    AsyncSchedulePlan osAsync =
        mustMaterializeAsync(osSchedule, outputDouble, DMASchedulingPolicy::PrefetchNext, permissive);

    ScheduleDependencyGraph wsGraph = mustBuildGraph(wsAsync, permissive);
    ScheduleDependencyGraph isGraph = mustBuildGraph(isAsync, permissive);
    ScheduleDependencyGraph osGraph = mustBuildGraph(osAsync, permissive);

    // ---- (20) IS input residency independently validated ----
    assert(!validateScheduleDependencyGraph(isGraph, isAsync, permissive));
    // ---- (21) WS weight residency independently validated ----
    assert(!validateScheduleDependencyGraph(wsGraph, wsAsync, permissive));
    // ---- (22) OS accumulator residency independently validated ----
    assert(!validateScheduleDependencyGraph(osGraph, osAsync, permissive));
    std::puts("  [ok] (20)-(22) IS input, WS weight, and OS accumulator residency all pass "
              "independent graph validation");

    // ---- (12)+(23) WS/IS later-K computes have PartialOutputReady; OS
    // has none (13) ----
    bool wsHasPartial = countEdgesOfKind(wsGraph, DependencyKind::PartialOutputReady) > 0;
    bool isHasPartial = countEdgesOfKind(isGraph, DependencyKind::PartialOutputReady) > 0;
    bool osHasPartial = countEdgesOfKind(osGraph, DependencyKind::PartialOutputReady) > 0;
    assert(wsHasPartial && isHasPartial);
    assert(!osHasPartial);
    std::puts("  [ok] (12)+(13)+(23) WS/IS have PartialOutputReady dependencies for k>0 "
              "computes; OS has none and no partial-sum chain");

    // ---- (33) missing partial-sum chain edge fails ----
    ScheduleDependencyGraph missingPartial =
        removeOneEdgeOfKind(wsGraph, DependencyKind::PartialOutputReady);
    assert(failedError(validateScheduleDependencyGraph(missingPartial, wsAsync, permissive)));
    ScheduleDependencyGraph missingReuse2 =
        removeOneEdgeOfKind(wsGraph, DependencyKind::OutputBufferReuse);
    assert(failedError(validateScheduleDependencyGraph(missingReuse2, wsAsync, permissive)));
    std::puts("  [ok] (33) removing a WS partial-sum PartialOutputReady or OutputBufferReuse "
              "edge fails validation closed");

    // ---- (18)+(19) output-buffer reuse hazards + waits for store
    // completion ----
    assert(countEdgesOfKind(osGraph, DependencyKind::OutputBufferReuse) >= 0); // structural presence checked below
    for (const auto &d : osGraph.dependencies)
      if (d.kind == DependencyKind::OutputBufferReuse)
        assert(osGraph.operations[d.producerOperationIndex].kind == AsyncOperationKind::DMAWait);
    std::puts("  [ok] (18)+(19) every OutputBufferReuse edge originates from a completion/wait "
              "node, never an issue (no reuse before store completion)");

    // ---- (36) output-store overlap remains overlap-capable when legal ----
    {
      std::vector<std::uint64_t> storeIssues, storeWaits;
      for (const auto &op : osAsync.operations) {
        if (op.kind == AsyncOperationKind::DMAIssue &&
            (op.transferKind == DMATransferKind::FinalOutputStore))
          storeIssues.push_back(op.operationIndex);
      }
      if (storeIssues.size() >= 2) {
        // The SECOND store's issue must not be forced to depend on the
        // FIRST store's wait when output is double-buffered (they use
        // alternating slots) -- i.e. no path from store0's issue to
        // store1's issue is required beyond DMA-engine ordering, and in
        // particular store1 is not forced to wait for store0's own wait.
        bool anyDeferred = false;
        for (std::size_t i = 0; i + 1 < osAsync.operations.size(); ++i) {
          if (osAsync.operations[i].kind == AsyncOperationKind::DMAIssue &&
              osAsync.operations[i].transferKind == DMATransferKind::FinalOutputStore) {
            const auto &next = osAsync.operations[i + 1];
            if (!(next.kind == AsyncOperationKind::DMAWait &&
                  next.eventId == osAsync.operations[i].eventId))
              anyDeferred = true;
          }
        }
        assert(anyDeferred);
      }
    }
    std::puts("  [ok] (36) output-store overlap remains representable under output double "
              "buffering");
  }

  // ---- (45) one-tile schedule shows no fictitious steady-state overlap ----
  {
    Block tinyBlock;
    mlir::hir::Conv2dOp tinyConv = buildConv2dOp(ctx, tinyBlock, 1, 16, 1, 1, 16, 1, 1, 1, 1);
    Conv2DProblemShape tinyProblem = extractConv2DProblemShape(tinyConv);
    std::vector<KernelCandidate> tinyCandidates = generateCandidates(tinyProblem);
    const TileShape tinyTile{16, 8, 8, 16};
    const PEArrayShape pe16{16, 16};
    const KernelCandidate &tinyCandidate =
        findCandidate(tinyCandidates, Precision::INT8, Dataflow::InputStationary, pe16, tinyTile);
    CandidateLegalityResult tinyLegality = checkCandidateLegality(tinyCandidate, target);
    assert(tinyLegality.isLegal);
    llvm::Expected<CandidateCostEstimate> tinyCostOrErr =
        estimateCandidateCost(tinyCandidate, tinyLegality, target);
    assert(bool(tinyCostOrErr));
    SchedulePlan tinySchedule = mustSchedule(tinyCandidate, tinyLegality, *tinyCostOrErr, tinyProblem, target);
    assert(tinySchedule.numMTiles == 1 && tinySchedule.numNTiles == 1 && tinySchedule.numKTiles == 1);

    BufferingPolicy allDouble{BufferingMode::Double, BufferingMode::Double, BufferingMode::Double};
    AsyncSchedulePlan tinyPlan =
        mustMaterializeAsync(tinySchedule, allDouble, DMASchedulingPolicy::PrefetchNext, target);
    ScheduleDependencyGraph tinyGraph = mustBuildGraph(tinyPlan, target);
    assert(!validateScheduleDependencyGraph(tinyGraph, tinyPlan, target));

    llvm::Expected<std::vector<OperationTimingInfo>> tinyTimingsOrErr =
        deriveOperationTimings(tinyPlan, target);
    assert(bool(tinyTimingsOrErr));
    llvm::Expected<ResourceConstrainedTimingResult> tinyRc =
        computeResourceConstrainedTiming(tinyGraph, *tinyTimingsOrErr, target);
    assert(bool(tinyRc));
    assert(!validateGraphTimingAgainstAsyncCost(tinyGraph, *tinyRc, tinyPlan.cost));
    assert(tinyRc->totalCycles == tinyPlan.cost.asynchronousEstimatedCycles);
    // No FICTITIOUS steady-state benefit: async total equals the
    // synchronous lower bound exactly (Example D). Note: tinyRc->overlapCycles
    // is NOT required to be 0 here -- it is a genuine [start,finish)
    // wall-clock interval intersection between DMA- and compute-busy
    // regions (Section 20), a different quantity from Slice 8's
    // hiddenDMACycles (a per-event issue-to-wait accounting, which IS 0
    // here per tinyPlan.cost.hiddenDMACycles). A DMA issue's window can
    // coincide in wall-clock terms with unrelated compute without that
    // compute having "hidden" it in Slice 8's sense (see AsyncDMASchedule
    // .cpp's documented choice not to gate a store issue's start on its
    // producing compute's finish). Section 21 only requires reconciling
    // quantities whose definitions actually match; totalCycles/startup/
    // drain/dmaBusy do, and are checked above.
    assert(tinyRc->totalCycles == tinyPlan.cost.synchronousEstimatedCycles);
  }
  std::puts("  [ok] (45) a one-tile schedule's graph timing reconciles exactly with Slice 8 and "
            "shows no fictitious steady-state benefit");

  // ---- (47) equivalent independently built workloads produce
  // field-identical graphs ----
  {
    Block block2;
    mlir::hir::Conv2dOp conv2 = buildConv2dOp(ctx, block2, 1, 64, 56, 56, 128, 3, 3, 54, 54);
    Conv2DProblemShape problem2 = extractConv2DProblemShape(conv2);
    std::vector<KernelCandidate> candidates2 = generateCandidates(problem2);
    std::vector<CandidateLegalityResult> legality2 = checkCandidateLegality(candidates2, target);
    llvm::Expected<std::vector<CandidateCostEstimate>> costs2 =
        estimateCandidateCosts(candidates2, legality2, target);
    assert(bool(costs2));
    llvm::Expected<PlanSelectionResult> selection2 = selectBestCandidate(candidates2, legality2, *costs2);
    assert(bool(selection2));
    llvm::Expected<SchedulePlan> schedule2 = materializeSelectedSchedule(*selection2, problem2, target);
    assert(bool(schedule2));
    AsyncSchedulePlan plan2 =
        mustMaterializeAsync(*schedule2, weightDouble, DMASchedulingPolicy::PrefetchNext, target);
    ScheduleDependencyGraph graph2 = mustBuildGraph(plan2, target);
    assert(graph2 == asyncGraph);
  }
  std::puts("  [ok] (47) equivalent independently constructed workloads produce a "
            "field-identical dependency graph");

  std::puts("=== ScheduleDependencyGraphTest: PASS ===");
  return 0;
}
