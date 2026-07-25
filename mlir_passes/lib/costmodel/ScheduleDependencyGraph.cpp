#include "costmodel/ScheduleDependencyGraph.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <map>
#include <set>

namespace mlir::costmodel {

llvm::StringRef toString(DependencyKind kind) {
  switch (kind) {
  case DependencyKind::DMACompletion:
    return "dma_completion";
  case DependencyKind::InputDataReady:
    return "input_data_ready";
  case DependencyKind::WeightDataReady:
    return "weight_data_ready";
  case DependencyKind::PartialOutputReady:
    return "partial_output_ready";
  case DependencyKind::ComputeSerialization:
    return "compute_serialization";
  case DependencyKind::DMAEngineSerialization:
    return "dma_engine_serialization";
  case DependencyKind::InputBufferReuse:
    return "input_buffer_reuse";
  case DependencyKind::WeightBufferReuse:
    return "weight_buffer_reuse";
  case DependencyKind::OutputBufferReuse:
    return "output_buffer_reuse";
  case DependencyKind::OutputStoreCompletion:
    return "output_store_completion";
  case DependencyKind::ResidencyConstraint:
    return "residency_constraint";
  case DependencyKind::ProgramOrder:
    return "program_order";
  }
  llvm_unreachable("unhandled DependencyKind");
}

llvm::StringRef toString(ScheduleResourceKind kind) {
  switch (kind) {
  case ScheduleResourceKind::None:
    return "none";
  case ScheduleResourceKind::DMAEngine:
    return "dma_engine";
  case ScheduleResourceKind::ComputeEngine:
    return "compute_engine";
  }
  llvm_unreachable("unhandled ScheduleResourceKind");
}

bool ScheduleDependencyGraph::operator==(const ScheduleDependencyGraph &other) const {
  if (candidateId != other.candidateId)
    return false;
  if (operations.size() != other.operations.size() || dependencies.size() != other.dependencies.size())
    return false;
  for (std::size_t i = 0; i < operations.size(); ++i)
    if (!(operations[i] == other.operations[i]))
      return false;
  for (std::size_t i = 0; i < dependencies.size(); ++i)
    if (!(dependencies[i] == other.dependencies[i]))
      return false;
  return true;
}

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

struct CheckedArith {
  bool overflowed = false;
  std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    if (__builtin_add_overflow(a, b, &r))
      overflowed = true;
    return r;
  }
  std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    std::uint64_t r = 0;
    if (__builtin_mul_overflow(a, b, &r))
      overflowed = true;
    return r;
  }
  std::uint64_t ceilDiv(std::uint64_t a, std::uint64_t b) {
    if (b == 0) {
      overflowed = true;
      return 0;
    }
    return a / b + (a % b != 0 ? 1 : 0);
  }
};

bool transferCyclesChecked(std::uint64_t bytes, std::uint64_t clockHz, std::uint64_t bandwidth,
                           std::uint64_t &result) {
  if (bandwidth == 0)
    return false;
  __uint128_t num = static_cast<__uint128_t>(bytes) * static_cast<__uint128_t>(clockHz);
  __uint128_t cycles = num / bandwidth + (num % bandwidth != 0 ? 1 : 0);
  if (cycles > std::numeric_limits<std::uint64_t>::max())
    return false;
  result = static_cast<std::uint64_t>(cycles);
  return true;
}

// Explicit, documented tie-break order for edge sorting (Section 14) --
// deliberately NOT `static_cast<int>(kind)` used implicitly; this
// function IS the one place the order is decided.
int kindRank(DependencyKind kind) {
  switch (kind) {
  case DependencyKind::DMACompletion:
    return 0;
  case DependencyKind::OutputStoreCompletion:
    return 1;
  case DependencyKind::InputDataReady:
    return 2;
  case DependencyKind::WeightDataReady:
    return 3;
  case DependencyKind::PartialOutputReady:
    return 4;
  case DependencyKind::ComputeSerialization:
    return 5;
  case DependencyKind::DMAEngineSerialization:
    return 6;
  case DependencyKind::InputBufferReuse:
    return 7;
  case DependencyKind::WeightBufferReuse:
    return 8;
  case DependencyKind::OutputBufferReuse:
    return 9;
  case DependencyKind::ResidencyConstraint:
    return 10;
  case DependencyKind::ProgramOrder:
    return 11;
  }
  llvm_unreachable("unhandled DependencyKind");
}

std::pair<int, std::uint32_t> slotKey(const BufferedTileRef &ref) {
  std::uint32_t effective = (ref.slot == BufferSlot::Single) ? 0u : ref.slotIndex;
  return {static_cast<int>(ref.role), effective};
}

bool sameCoordinate(const TileCoordinate &a, const TileCoordinate &b) {
  return a.mTile == b.mTile && a.nTile == b.nTile && a.kTile == b.kTile;
}

void sortAndDedupe(std::vector<ScheduleDependency> &deps) {
  std::sort(deps.begin(), deps.end(), [](const ScheduleDependency &a, const ScheduleDependency &b) {
    if (a.producerOperationIndex != b.producerOperationIndex)
      return a.producerOperationIndex < b.producerOperationIndex;
    if (a.consumerOperationIndex != b.consumerOperationIndex)
      return a.consumerOperationIndex < b.consumerOperationIndex;
    return kindRank(a.kind) < kindRank(b.kind);
  });
  deps.erase(std::unique(deps.begin(), deps.end(),
                        [](const ScheduleDependency &a, const ScheduleDependency &b) {
                          return a.producerOperationIndex == b.producerOperationIndex &&
                                a.consumerOperationIndex == b.consumerOperationIndex &&
                                a.kind == b.kind;
                        }),
            deps.end());
}

// The single, shared dependency-derivation routine used by BOTH
// buildScheduleDependencyGraph() (to populate the graph) and
// validateScheduleDependencyGraph() (to independently recompute the
// expected edge set and compare it against whatever graph it was
// handed -- see the header's "direct-edge policy" note). Producing this
// from `schedule.operations` + `schedule.events` alone (never any Slice
// 8-internal state) is what makes the validator genuinely independent of
// the builder's own bookkeeping, even though both call this function.
std::vector<ScheduleDependency> deriveExpectedDependencies(const AsyncSchedulePlan &schedule) {
  std::vector<ScheduleDependency> deps;
  auto addDep = [&](std::uint64_t producer, std::uint64_t consumer, DependencyKind kind) {
    if (producer == consumer)
      return;
    deps.push_back({producer, consumer, kind});
  };

  // ---- 1. DMACompletion / OutputStoreCompletion: issue -> wait, one per event ----
  for (const DMAEvent &ev : schedule.events) {
    bool isStore = ev.transferKind == DMATransferKind::PartialOutputStore ||
                  ev.transferKind == DMATransferKind::FinalOutputStore;
    addDep(ev.issueOperationIndex, ev.waitOperationIndex,
          isStore ? DependencyKind::OutputStoreCompletion : DependencyKind::DMACompletion);
  }

  // ---- 2. single forward walk: data-ready, engine serialization, buffer reuse ----
  std::optional<std::uint64_t> lastInputWait, lastWeightWait, lastPartialReloadWait;
  std::optional<std::uint64_t> lastComputeEngineNode; // last {Compute, DMAWait} node
  std::optional<std::uint64_t> lastDmaIssueNode;

  std::map<std::pair<int, std::uint32_t>, std::uint64_t> lastComputeUseOfSlot; // input/weight
  std::map<std::pair<int, std::uint32_t>, std::uint64_t> lastWaitTouchOfSlot;  // output

  for (const AsyncScheduleOperation &op : schedule.operations) {
    switch (op.kind) {
    case AsyncOperationKind::DMAIssue: {
      if (lastDmaIssueNode)
        addDep(*lastDmaIssueNode, op.operationIndex, DependencyKind::DMAEngineSerialization);
      lastDmaIssueNode = op.operationIndex;

      if (op.transferKind == DMATransferKind::InputLoad && op.input) {
        auto it = lastComputeUseOfSlot.find(slotKey(*op.input));
        if (it != lastComputeUseOfSlot.end())
          addDep(it->second, op.operationIndex, DependencyKind::InputBufferReuse);
      } else if (op.transferKind == DMATransferKind::WeightLoad && op.weight) {
        auto it = lastComputeUseOfSlot.find(slotKey(*op.weight));
        if (it != lastComputeUseOfSlot.end())
          addDep(it->second, op.operationIndex, DependencyKind::WeightBufferReuse);
      } else if (op.transferKind == DMATransferKind::PartialOutputReload && op.output) {
        auto it = lastWaitTouchOfSlot.find(slotKey(*op.output));
        if (it != lastWaitTouchOfSlot.end())
          addDep(it->second, op.operationIndex, DependencyKind::OutputBufferReuse);
      }
      // Store issues never overwrite a slot at issue time (they read the
      // existing accumulator) -- no reuse edge targets a store issue.
      break;
    }
    case AsyncOperationKind::DMAWait: {
      if (lastComputeEngineNode)
        addDep(*lastComputeEngineNode, op.operationIndex, DependencyKind::ComputeSerialization);
      lastComputeEngineNode = op.operationIndex;

      if (op.transferKind == DMATransferKind::InputLoad)
        lastInputWait = op.operationIndex;
      else if (op.transferKind == DMATransferKind::WeightLoad)
        lastWeightWait = op.operationIndex;
      else if (op.transferKind == DMATransferKind::PartialOutputReload) {
        lastPartialReloadWait = op.operationIndex;
        if (op.output)
          lastWaitTouchOfSlot[slotKey(*op.output)] = op.operationIndex;
      } else if ((op.transferKind == DMATransferKind::PartialOutputStore ||
                 op.transferKind == DMATransferKind::FinalOutputStore) &&
                op.output) {
        lastWaitTouchOfSlot[slotKey(*op.output)] = op.operationIndex;
      }
      break;
    }
    case AsyncOperationKind::Compute: {
      if (lastComputeEngineNode)
        addDep(*lastComputeEngineNode, op.operationIndex, DependencyKind::ComputeSerialization);
      lastComputeEngineNode = op.operationIndex;

      if (lastInputWait)
        addDep(*lastInputWait, op.operationIndex, DependencyKind::InputDataReady);
      if (lastWeightWait)
        addDep(*lastWeightWait, op.operationIndex, DependencyKind::WeightDataReady);
      if (op.coordinate.kTile > 0 && lastPartialReloadWait)
        addDep(*lastPartialReloadWait, op.operationIndex, DependencyKind::PartialOutputReady);

      // A new output-tile group (kTile == 0) reusing a slot from an
      // earlier group needs a reuse edge from that slot's last touch.
      if (op.coordinate.kTile == 0 && op.output) {
        auto it = lastWaitTouchOfSlot.find(slotKey(*op.output));
        if (it != lastWaitTouchOfSlot.end())
          addDep(it->second, op.operationIndex, DependencyKind::OutputBufferReuse);
      }

      if (op.input)
        lastComputeUseOfSlot[slotKey(*op.input)] = op.operationIndex;
      if (op.weight)
        lastComputeUseOfSlot[slotKey(*op.weight)] = op.operationIndex;
      break;
    }
    case AsyncOperationKind::ComputeBarrier:
      break;
    }
  }

  // ---- 3. diagnostic-only ResidencyConstraint edges (Section 11) ----
  {
    std::vector<std::pair<std::uint64_t, TileCoordinate>> runKeys;
    for (const AsyncScheduleOperation &op : schedule.operations) {
      if (op.kind != AsyncOperationKind::Compute)
        continue;
      std::optional<TileCoordinate> key;
      switch (schedule.dataflow) {
      case Dataflow::InputStationary:
        if (op.input)
          key = op.input->coordinate;
        break;
      case Dataflow::WeightStationary:
        if (op.weight)
          key = op.weight->coordinate;
        break;
      case Dataflow::OutputStationary:
        if (op.output)
          key = op.output->coordinate;
        break;
      }
      if (key)
        runKeys.push_back({op.operationIndex, *key});
    }
    std::size_t i = 0;
    while (i < runKeys.size()) {
      std::size_t j = i;
      while (j + 1 < runKeys.size() && sameCoordinate(runKeys[j + 1].second, runKeys[i].second))
        ++j;
      if (j > i)
        addDep(runKeys[i].first, runKeys[j].first, DependencyKind::ResidencyConstraint);
      i = j + 1;
    }
  }

  sortAndDedupe(deps);
  return deps;
}

} // namespace

llvm::Expected<ScheduleDependencyGraph> buildScheduleDependencyGraph(const AsyncSchedulePlan &schedule,
                                                                      const NPUTargetConfig &target) {
  if (llvm::Error err = validateAsyncSchedule(schedule, target))
    return std::move(err);
  if (schedule.candidateId.empty())
    return makeError("buildScheduleDependencyGraph: schedule.candidateId is empty");

  ScheduleDependencyGraph graph;
  graph.candidateId = schedule.candidateId;
  graph.operations = schedule.operations;
  graph.dependencies = deriveExpectedDependencies(schedule);

  std::size_t n = graph.operations.size();
  graph.outgoingEdges.assign(n, {});
  graph.incomingEdges.assign(n, {});
  for (std::size_t i = 0; i < graph.dependencies.size(); ++i) {
    graph.outgoingEdges[graph.dependencies[i].producerOperationIndex].push_back(i);
    graph.incomingEdges[graph.dependencies[i].consumerOperationIndex].push_back(i);
  }
  return graph;
}

llvm::Expected<std::vector<std::uint64_t>>
computeStableTopologicalOrder(const ScheduleDependencyGraph &graph) {
  std::size_t n = graph.operations.size();
  std::vector<std::uint64_t> inDegree(n, 0);
  for (const ScheduleDependency &d : graph.dependencies)
    ++inDegree[d.consumerOperationIndex];

  std::set<std::uint64_t> ready; // sorted ascending -> lowest-index-first tie-break
  for (std::uint64_t i = 0; i < n; ++i)
    if (inDegree[i] == 0)
      ready.insert(i);

  std::vector<std::uint64_t> order;
  order.reserve(n);
  while (!ready.empty()) {
    std::uint64_t node = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(node);
    for (std::uint64_t edgeIdx : graph.outgoingEdges[node]) {
      std::uint64_t consumer = graph.dependencies[edgeIdx].consumerOperationIndex;
      if (--inDegree[consumer] == 0)
        ready.insert(consumer);
    }
  }

  if (order.size() != n) {
    std::vector<std::uint64_t> remaining;
    for (std::uint64_t i = 0; i < n; ++i)
      if (inDegree[i] != 0)
        remaining.push_back(i);
    std::string witness;
    for (const ScheduleDependency &d : graph.dependencies) {
      if (inDegree[d.consumerOperationIndex] != 0 && inDegree[d.producerOperationIndex] != 0) {
        witness = ("witness edge: producer=" + llvm::Twine(d.producerOperationIndex) +
                  " consumer=" + llvm::Twine(d.consumerOperationIndex) +
                  " kind=" + toString(d.kind))
                     .str();
        break;
      }
    }
    return makeError("computeStableTopologicalOrder: cycle detected -- " +
                     llvm::Twine(remaining.size()) + " node(s) never reached zero in-degree; " +
                     witness);
  }
  return order;
}

bool hasDependencyPath(const ScheduleDependencyGraph &graph, std::uint64_t producer,
                       std::uint64_t consumer) {
  if (producer >= graph.operations.size() || consumer >= graph.operations.size())
    return false;
  std::vector<bool> visited(graph.operations.size(), false);
  std::vector<std::uint64_t> stack = {producer};
  visited[producer] = true;
  while (!stack.empty()) {
    std::uint64_t node = stack.back();
    stack.pop_back();
    if (node == consumer)
      return true;
    for (std::uint64_t edgeIdx : graph.outgoingEdges[node]) {
      std::uint64_t next = graph.dependencies[edgeIdx].consumerOperationIndex;
      if (!visited[next]) {
        visited[next] = true;
        stack.push_back(next);
      }
    }
  }
  return false;
}

llvm::Expected<std::vector<OperationTimingInfo>>
deriveOperationTimings(const AsyncSchedulePlan &schedule, const NPUTargetConfig &target) {
  if (schedule.peArray.rows <= 0 || schedule.peArray.cols <= 0)
    return makeError("deriveOperationTimings: non-positive PE array dimension");
  std::uint64_t macsPerPEPerCycle = (schedule.precision == Precision::INT8)
                                        ? target.int8MacsPerPEPerCycle
                                        : target.fp16MacsPerPEPerCycle;
  if (macsPerPEPerCycle == 0)
    return makeError("deriveOperationTimings: target's precision throughput is zero");
  if (target.clockHz == 0 || target.offChipBandwidthBytesPerSecond == 0)
    return makeError("deriveOperationTimings: target clock or off-chip bandwidth is zero");

  CheckedArith ck;
  std::uint64_t tileM = ck.mul(static_cast<std::uint64_t>(schedule.tile.height),
                               static_cast<std::uint64_t>(schedule.tile.width));
  std::uint64_t tileN = static_cast<std::uint64_t>(schedule.tile.outputChannels);
  std::uint64_t tileK = static_cast<std::uint64_t>(schedule.tile.reductionDepth);
  std::uint64_t mWaves = ck.ceilDiv(tileM, static_cast<std::uint64_t>(schedule.peArray.rows));
  std::uint64_t nWaves = ck.ceilDiv(tileN, static_cast<std::uint64_t>(schedule.peArray.cols));
  std::uint64_t arrayWavesPerTile = ck.mul(mWaves, nWaves);
  std::uint64_t reductionCyclesPerKTile = ck.ceilDiv(tileK, macsPerPEPerCycle);
  std::uint64_t perComputeCycles = ck.mul(arrayWavesPerTile, reductionCyclesPerKTile);
  if (ck.overflowed)
    return makeError("deriveOperationTimings: integer overflow deriving per-compute cycles");

  std::vector<OperationTimingInfo> timings;
  timings.reserve(schedule.operations.size());
  for (const AsyncScheduleOperation &op : schedule.operations) {
    OperationTimingInfo t;
    t.operationIndex = op.operationIndex;
    switch (op.kind) {
    case AsyncOperationKind::DMAIssue: {
      std::uint64_t transfer = 0;
      if (!transferCyclesChecked(op.bytes, target.clockHz, target.offChipBandwidthBytesPerSecond,
                                 transfer))
        return makeError("deriveOperationTimings: overflow computing DMA transfer cycles");
      t.durationCycles = ck.add(static_cast<std::uint64_t>(target.dmaSetupCycles), transfer);
      t.resource = ScheduleResourceKind::DMAEngine;
      break;
    }
    case AsyncOperationKind::Compute:
      t.durationCycles = perComputeCycles;
      t.resource = ScheduleResourceKind::ComputeEngine;
      break;
    case AsyncOperationKind::DMAWait:
      t.durationCycles = 0; // see ComputeSerialization doc comment: resolved dynamically
      t.resource = ScheduleResourceKind::ComputeEngine;
      break;
    case AsyncOperationKind::ComputeBarrier:
      t.durationCycles = 0;
      t.resource = ScheduleResourceKind::None;
      break;
    }
    timings.push_back(t);
  }
  if (ck.overflowed)
    return makeError("deriveOperationTimings: integer overflow");
  return timings;
}

llvm::Expected<CriticalPathResult>
computeDependencyCriticalPath(const ScheduleDependencyGraph &graph,
                              llvm::ArrayRef<OperationTimingInfo> timings) {
  std::size_t n = graph.operations.size();
  if (timings.size() != n)
    return makeError("computeDependencyCriticalPath: timings size does not match node count");

  llvm::Expected<std::vector<std::uint64_t>> topoOrErr = computeStableTopologicalOrder(graph);
  if (!topoOrErr)
    return topoOrErr.takeError();

  CheckedArith ck;
  std::vector<std::uint64_t> start(n, 0), finish(n, 0);
  for (std::uint64_t node : *topoOrErr) {
    std::uint64_t maxPred = 0;
    for (std::uint64_t edgeIdx : graph.incomingEdges[node])
      maxPred = std::max(maxPred, finish[graph.dependencies[edgeIdx].producerOperationIndex]);
    start[node] = maxPred;
    finish[node] = ck.add(start[node], timings[node].durationCycles);
  }
  if (ck.overflowed)
    return makeError("computeDependencyCriticalPath: overflow computing earliest finish");

  CriticalPathResult result;
  result.earliestStartCycles = start;
  result.earliestFinishCycles = finish;
  std::uint64_t total = 0;
  std::uint64_t endNode = 0;
  for (std::uint64_t i = 0; i < n; ++i)
    if (finish[i] >= total) {
      total = finish[i];
      endNode = i;
    }
  result.totalCycles = total;

  // Deterministic backtrace: at each step, pick the lowest-index
  // predecessor whose finish exactly equals this node's start.
  std::vector<std::uint64_t> path = {endNode};
  std::uint64_t cur = endNode;
  while (start[cur] > 0 || !graph.incomingEdges[cur].empty()) {
    std::optional<std::uint64_t> chosen;
    for (std::uint64_t edgeIdx : graph.incomingEdges[cur]) {
      std::uint64_t pred = graph.dependencies[edgeIdx].producerOperationIndex;
      if (finish[pred] == start[cur]) {
        if (!chosen || pred < *chosen)
          chosen = pred;
      }
    }
    if (!chosen)
      break;
    path.push_back(*chosen);
    cur = *chosen;
  }
  std::reverse(path.begin(), path.end());
  result.criticalPathOperations = std::move(path);
  return result;
}

llvm::Expected<ResourceConstrainedTimingResult>
computeResourceConstrainedTiming(const ScheduleDependencyGraph &graph,
                                  llvm::ArrayRef<OperationTimingInfo> timings,
                                  const NPUTargetConfig &target) {
  std::size_t n = graph.operations.size();
  if (timings.size() != n)
    return makeError("computeResourceConstrainedTiming: timings size does not match node count");
  if (target.dmaEngineCount == 0)
    return makeError("computeResourceConstrainedTiming: target.dmaEngineCount is zero");

  llvm::Expected<std::vector<std::uint64_t>> topoOrErr = computeStableTopologicalOrder(graph);
  if (!topoOrErr)
    return topoOrErr.takeError();

  auto edgeFinishOfKind = [&](std::uint64_t node, DependencyKind kind,
                              const std::vector<std::uint64_t> &finish) -> std::optional<std::uint64_t> {
    std::optional<std::uint64_t> best;
    for (std::uint64_t edgeIdx : graph.incomingEdges[node]) {
      const ScheduleDependency &d = graph.dependencies[edgeIdx];
      if (d.kind != kind)
        continue;
      std::uint64_t f = finish[d.producerOperationIndex];
      if (!best || f > *best)
        best = f;
    }
    return best;
  };

  CheckedArith ck;
  std::vector<std::uint64_t> start(n, 0), finish(n, 0);
  std::uint64_t startup = target.kernelLaunchCycles;

  for (std::uint64_t node : *topoOrErr) {
    const AsyncScheduleOperation &op = graph.operations[node];
    if (op.kind == AsyncOperationKind::DMAWait) {
      std::optional<std::uint64_t> chainFinish =
          edgeFinishOfKind(node, DependencyKind::ComputeSerialization, finish);
      std::optional<std::uint64_t> completionFinish =
          edgeFinishOfKind(node, DependencyKind::DMACompletion, finish);
      if (!completionFinish)
        completionFinish = edgeFinishOfKind(node, DependencyKind::OutputStoreCompletion, finish);
      std::uint64_t chain = chainFinish.value_or(startup);
      std::uint64_t completion = completionFinish.value_or(startup);
      start[node] = chain;
      if (chain < completion)
        finish[node] = ck.add(completion, static_cast<std::uint64_t>(target.tileSynchronizationCycles));
      else
        finish[node] = chain;
      continue;
    }

    // Every other node kind: generic max(predecessor finish) + duration,
    // restricted to the four resource-timeline edge kinds (Section 20).
    std::uint64_t maxPred = startup;
    for (std::uint64_t edgeIdx : graph.incomingEdges[node]) {
      DependencyKind k = graph.dependencies[edgeIdx].kind;
      if (k != DependencyKind::DMAEngineSerialization && k != DependencyKind::ComputeSerialization &&
          k != DependencyKind::DMACompletion && k != DependencyKind::OutputStoreCompletion)
        continue;
      maxPred = std::max(maxPred, finish[graph.dependencies[edgeIdx].producerOperationIndex]);
    }
    start[node] = maxPred;
    finish[node] = ck.add(start[node], timings[node].durationCycles);
  }
  if (ck.overflowed)
    return makeError("computeResourceConstrainedTiming: overflow computing earliest finish");

  ResourceConstrainedTimingResult result;
  result.startCycles = start;
  result.finishCycles = finish;
  result.startupCycles = startup;

  // A "drain" wait is one appended purely to close out an event with no
  // natural consumer position left in the schedule (Slice 8's
  // materializeAsyncSchedule: an OutputStoreCompletion event whose
  // owning output-tile group has no later group to defer its wait to --
  // see AsyncDMASchedule.cpp's `Placement::DeferredToDrain`). Such a
  // wait is observable from the graph alone by two properties every
  // ordinary (non-drain) wait lacks simultaneously: it occurs strictly
  // after the LAST Compute in the whole schedule (nothing computational
  // ever follows it), AND it is NOT immediately adjacent to its own
  // issue (an ordinary trailing wait -- e.g. the final store of a
  // single-buffered output -- is always issued-then-waited back to
  // back). This mirrors Slice 8's own coreOpCount boundary without
  // requiring any Slice-8-internal state on AsyncSchedulePlan.
  std::uint64_t lastComputeIndex = 0;
  bool anyCompute = false;
  for (std::uint64_t i = 0; i < n; ++i)
    if (graph.operations[i].kind == AsyncOperationKind::Compute) {
      lastComputeIndex = i;
      anyCompute = true;
    }
  std::vector<bool> isDrainWait(n, false);
  for (std::uint64_t i = 0; i < n; ++i) {
    if (graph.operations[i].kind != AsyncOperationKind::DMAWait)
      continue;
    if (anyCompute && i <= lastComputeIndex)
      continue;
    std::optional<std::uint64_t> issueOp;
    for (std::uint64_t edgeIdx : graph.incomingEdges[i]) {
      DependencyKind k = graph.dependencies[edgeIdx].kind;
      if (k == DependencyKind::DMACompletion || k == DependencyKind::OutputStoreCompletion)
        issueOp = graph.dependencies[edgeIdx].producerOperationIndex;
    }
    if (issueOp && *issueOp + 1 != i)
      isDrainWait[i] = true;
  }

  std::uint64_t total = 0;
  std::uint64_t lastCoreFinish = startup;
  std::uint64_t dmaBusy = 0, computeBusy = 0;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> dmaIntervals, computeIntervals;
  for (std::uint64_t i = 0; i < n; ++i) {
    total = std::max(total, finish[i]);
    AsyncOperationKind kind = graph.operations[i].kind;
    if (kind == AsyncOperationKind::DMAIssue) {
      lastCoreFinish = std::max(lastCoreFinish, finish[i]);
      dmaBusy = ck.add(dmaBusy, timings[i].durationCycles);
      if (finish[i] > start[i])
        dmaIntervals.push_back({start[i], finish[i]});
    } else if (kind == AsyncOperationKind::Compute) {
      lastCoreFinish = std::max(lastCoreFinish, finish[i]);
      computeBusy = ck.add(computeBusy, timings[i].durationCycles);
      if (finish[i] > start[i])
        computeIntervals.push_back({start[i], finish[i]});
    } else if (kind == AsyncOperationKind::DMAWait && !isDrainWait[i]) {
      lastCoreFinish = std::max(lastCoreFinish, finish[i]);
    }
  }
  if (ck.overflowed)
    return makeError("computeResourceConstrainedTiming: overflow accumulating busy cycles");

  // Genuine interval-intersection sweep (never `dmaBusy + computeBusy -
  // total`) between the DMA-engine and compute-engine occupied intervals.
  std::sort(dmaIntervals.begin(), dmaIntervals.end());
  std::sort(computeIntervals.begin(), computeIntervals.end());
  std::uint64_t overlap = 0;
  std::size_t di = 0, ci = 0;
  while (di < dmaIntervals.size() && ci < computeIntervals.size()) {
    std::uint64_t lo = std::max(dmaIntervals[di].first, computeIntervals[ci].first);
    std::uint64_t hi = std::min(dmaIntervals[di].second, computeIntervals[ci].second);
    if (hi > lo)
      overlap = ck.add(overlap, hi - lo);
    if (dmaIntervals[di].second < computeIntervals[ci].second)
      ++di;
    else
      ++ci;
  }
  if (ck.overflowed)
    return makeError("computeResourceConstrainedTiming: overflow computing overlap");

  result.totalCycles = total;
  result.dmaBusyCycles = dmaBusy;
  result.computeBusyCycles = computeBusy;
  result.overlapCycles = overlap;
  result.drainCycles = (total >= lastCoreFinish) ? (total - lastCoreFinish) : 0;

  // Critical path: reuse the same backtrace idea as
  // computeDependencyCriticalPath, but only over the resource-timeline
  // edges actually used above.
  std::uint64_t endNode = 0;
  for (std::uint64_t i = 0; i < n; ++i)
    if (finish[i] >= finish[endNode])
      endNode = i;
  std::vector<std::uint64_t> path = {endNode};
  std::uint64_t cur = endNode;
  bool progressed = true;
  while (progressed) {
    progressed = false;
    std::optional<std::uint64_t> chosen;
    for (std::uint64_t edgeIdx : graph.incomingEdges[cur]) {
      DependencyKind k = graph.dependencies[edgeIdx].kind;
      if (k != DependencyKind::DMAEngineSerialization && k != DependencyKind::ComputeSerialization &&
          k != DependencyKind::DMACompletion && k != DependencyKind::OutputStoreCompletion)
        continue;
      std::uint64_t pred = graph.dependencies[edgeIdx].producerOperationIndex;
      if (finish[pred] == start[cur]) {
        if (!chosen || pred < *chosen)
          chosen = pred;
      }
    }
    if (chosen) {
      path.push_back(*chosen);
      cur = *chosen;
      progressed = true;
    }
  }
  std::reverse(path.begin(), path.end());
  result.criticalPathOperations = std::move(path);

  return result;
}

llvm::Error validateGraphTimingAgainstAsyncCost(const ScheduleDependencyGraph &graph,
                                                const ResourceConstrainedTimingResult &graphTiming,
                                                const AsyncCandidateCost &slice8Cost) {
  (void)graph;
  // Exact mapping (Section 21): Slice 8 has no dedicated dmaBusyCycles/
  // computeBusyCycles field, but `hiddenDMACycles + exposedDMACycles` is
  // exactly the sum, over every DMA issue, of that issue's own
  // (setup+transfer) cost -- precisely this module's dmaBusyCycles
  // definition (Section 20's doc comment) -- so that sum is the common
  // quantity compared here in place of a nonexistent direct field.
  CheckedArith ck;
  std::uint64_t slice8DmaWork = ck.add(slice8Cost.hiddenDMACycles, slice8Cost.exposedDMACycles);
  if (ck.overflowed)
    return makeError("validateGraphTimingAgainstAsyncCost: overflow summing Slice 8 DMA work");
  if (graphTiming.totalCycles != slice8Cost.asynchronousEstimatedCycles)
    return makeError("validateGraphTimingAgainstAsyncCost: totalCycles (" +
                     llvm::Twine(graphTiming.totalCycles) +
                     ") != Slice 8 asynchronousEstimatedCycles (" +
                     llvm::Twine(slice8Cost.asynchronousEstimatedCycles) + ")");
  if (graphTiming.startupCycles != slice8Cost.startupCycles)
    return makeError("validateGraphTimingAgainstAsyncCost: startupCycles mismatch");
  if (graphTiming.drainCycles != slice8Cost.drainCycles)
    return makeError("validateGraphTimingAgainstAsyncCost: drainCycles mismatch");
  if (graphTiming.dmaBusyCycles != slice8DmaWork)
    return makeError("validateGraphTimingAgainstAsyncCost: dmaBusyCycles (" +
                     llvm::Twine(graphTiming.dmaBusyCycles) +
                     ") != Slice 8 hiddenDMACycles+exposedDMACycles (" + llvm::Twine(slice8DmaWork) +
                     ")");
  return llvm::Error::success();
}

// ===========================================================================
// Section 15: validation.
// ===========================================================================
llvm::Error validateScheduleDependencyGraph(const ScheduleDependencyGraph &graph,
                                            const AsyncSchedulePlan &schedule,
                                            const NPUTargetConfig &target) {
  if (graph.candidateId != schedule.candidateId)
    return makeError("validateScheduleDependencyGraph: candidateId mismatch");
  if (graph.candidateId.empty())
    return makeError("validateScheduleDependencyGraph: empty candidateId");

  std::size_t n = graph.operations.size();
  if (n != schedule.operations.size())
    return makeError("validateScheduleDependencyGraph: node count does not match schedule "
                      "operation count");
  for (std::size_t i = 0; i < n; ++i) {
    if (graph.operations[i].operationIndex != i)
      return makeError("validateScheduleDependencyGraph: node " + llvm::Twine(i) +
                        " has a non-dense operationIndex");
    if (!(graph.operations[i] == schedule.operations[i]))
      return makeError("validateScheduleDependencyGraph: node " + llvm::Twine(i) +
                        " does not match the schedule's operation");
  }
  if (graph.outgoingEdges.size() != n || graph.incomingEdges.size() != n)
    return makeError("validateScheduleDependencyGraph: edge-index tables are not node-aligned");

  // Endpoint range / no self-edge / no duplicate.
  std::set<std::tuple<std::uint64_t, std::uint64_t, int>> seenEdges;
  for (const ScheduleDependency &d : graph.dependencies) {
    if (d.producerOperationIndex >= n || d.consumerOperationIndex >= n)
      return makeError("validateScheduleDependencyGraph: edge endpoint out of range");
    if (d.producerOperationIndex == d.consumerOperationIndex)
      return makeError("validateScheduleDependencyGraph: self-edge at node " +
                        llvm::Twine(d.producerOperationIndex));
    auto key = std::make_tuple(d.producerOperationIndex, d.consumerOperationIndex,
                               static_cast<int>(d.kind));
    if (!seenEdges.insert(key).second)
      return makeError("validateScheduleDependencyGraph: duplicate edge (producer=" +
                        llvm::Twine(d.producerOperationIndex) +
                        ", consumer=" + llvm::Twine(d.consumerOperationIndex) +
                        ", kind=" + toString(d.kind) + ")");
    if (!(d.producerOperationIndex < d.consumerOperationIndex))
      return makeError("validateScheduleDependencyGraph: dependency is reversed relative to "
                        "original operation order (producer must precede consumer)");
  }
  // outgoingEdges/incomingEdges must exactly reflect `dependencies`.
  {
    std::vector<std::vector<std::uint64_t>> expectedOut(n), expectedIn(n);
    for (std::uint64_t i = 0; i < graph.dependencies.size(); ++i) {
      expectedOut[graph.dependencies[i].producerOperationIndex].push_back(i);
      expectedIn[graph.dependencies[i].consumerOperationIndex].push_back(i);
    }
    for (std::size_t i = 0; i < n; ++i) {
      auto a = graph.outgoingEdges[i], b = expectedOut[i];
      auto c = graph.incomingEdges[i], d = expectedIn[i];
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      std::sort(c.begin(), c.end());
      std::sort(d.begin(), d.end());
      if (a != b || c != d)
        return makeError("validateScheduleDependencyGraph: outgoing/incoming edge index tables "
                          "do not match the dependency list at node " +
                          llvm::Twine(i));
    }
  }

  // Independently recompute the expected edge set and require exact
  // agreement -- this is what makes "missing edge" / "extra edge" tests
  // (Section 22, 28) fail closed without hand-writing one check per kind.
  {
    std::vector<ScheduleDependency> expected = deriveExpectedDependencies(schedule);
    std::vector<ScheduleDependency> actual = graph.dependencies;
    sortAndDedupe(actual);
    if (expected.size() != actual.size())
      return makeError("validateScheduleDependencyGraph: edge count (" +
                       llvm::Twine(actual.size()) + ") does not match the independently "
                       "recomputed expected edge count (" + llvm::Twine(expected.size()) + ")");
    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (!(expected[i] == actual[i]))
        return makeError("validateScheduleDependencyGraph: edge #" + llvm::Twine(i) +
                         " (producer=" + llvm::Twine(actual[i].producerOperationIndex) +
                         ", consumer=" + llvm::Twine(actual[i].consumerOperationIndex) +
                         ", kind=" + toString(actual[i].kind) +
                         ") does not match the independently recomputed expected edge "
                         "(producer=" + llvm::Twine(expected[i].producerOperationIndex) +
                         ", consumer=" + llvm::Twine(expected[i].consumerOperationIndex) +
                         ", kind=" + toString(expected[i].kind) + ")");
    }
  }

  // OutputBufferReuse must originate from a completion/wait node, never
  // an issue (Section 12: "cannot be reused after store issue but before
  // completion"); Input/WeightBufferReuse must originate from a Compute.
  for (const ScheduleDependency &d : graph.dependencies) {
    AsyncOperationKind producerKind = graph.operations[d.producerOperationIndex].kind;
    if (d.kind == DependencyKind::OutputBufferReuse && producerKind != AsyncOperationKind::DMAWait)
      return makeError("validateScheduleDependencyGraph: OutputBufferReuse producer must be a "
                        "completion/wait node");
    if ((d.kind == DependencyKind::InputBufferReuse || d.kind == DependencyKind::WeightBufferReuse) &&
        producerKind != AsyncOperationKind::Compute)
      return makeError("validateScheduleDependencyGraph: Input/WeightBufferReuse producer must "
                        "be a Compute node");
  }

  // Residency contracts: the resident role's slot must be stable across
  // every compute sharing the same group coordinate (Section 11/26),
  // scanned directly from operation metadata.
  {
    std::optional<TileCoordinate> curKey;
    std::optional<BufferSlot> curSlot;
    for (const AsyncScheduleOperation &op : graph.operations) {
      if (op.kind != AsyncOperationKind::Compute)
        continue;
      std::optional<BufferedTileRef> ref;
      switch (schedule.dataflow) {
      case Dataflow::InputStationary:
        ref = op.input;
        break;
      case Dataflow::WeightStationary:
        ref = op.weight;
        break;
      case Dataflow::OutputStationary:
        ref = op.output;
        break;
      }
      if (!ref)
        return makeError("validateScheduleDependencyGraph: compute missing the resident role's "
                          "buffer reference");
      if (curKey && sameCoordinate(*curKey, ref->coordinate)) {
        if (ref->slot != *curSlot)
          return makeError("validateScheduleDependencyGraph: resident buffer changed slot "
                            "inside its own residency group");
      } else {
        curKey = ref->coordinate;
      }
      curSlot = ref->slot;
    }
  }

  // WS/IS partial-sum chains; OS must contain none.
  {
    bool isSpillDataflow =
        schedule.dataflow == Dataflow::WeightStationary || schedule.dataflow == Dataflow::InputStationary;
    if (!isSpillDataflow) {
      for (const AsyncScheduleOperation &op : graph.operations)
        if (op.transferKind == DMATransferKind::PartialOutputReload ||
            op.transferKind == DMATransferKind::PartialOutputStore)
          return makeError("validateScheduleDependencyGraph: Output Stationary schedule "
                            "unexpectedly contains a partial-sum spill operation");
    } else {
      // Group compute ops by (mTile, nTile) -- NOT by adjacency: WS/IS's
      // loop nesting (k-loop outside the m-loop/n-loop) means one
      // accumulator's k=0..KT-1 touches are interleaved with every OTHER
      // output tile's own touches in program order, so a
      // consecutive-run scan would see nothing but groups of size 1.
      // Every (m,n) with more than one compute must show exactly
      // (size-1) reload/store transitions and exactly one
      // FinalOutputStore closing it.
      std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> computeCountByCoord;
      std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> reloadCountByCoord;
      std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> partialStoreCountByCoord;
      std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> finalStoreCountByCoord;
      for (const AsyncScheduleOperation &op : graph.operations) {
        if (op.kind == AsyncOperationKind::Compute && op.output) {
          ++computeCountByCoord[{op.output->coordinate.mTile, op.output->coordinate.nTile}];
        } else if (op.kind == AsyncOperationKind::DMAIssue && op.output) {
          auto key = std::make_pair(op.output->coordinate.mTile, op.output->coordinate.nTile);
          if (op.transferKind == DMATransferKind::PartialOutputReload)
            ++reloadCountByCoord[key];
          else if (op.transferKind == DMATransferKind::PartialOutputStore)
            ++partialStoreCountByCoord[key];
          else if (op.transferKind == DMATransferKind::FinalOutputStore)
            ++finalStoreCountByCoord[key];
        }
      }
      for (const auto &[key, groupSize] : computeCountByCoord) {
        if (groupSize > 1) {
          std::uint64_t reloadCount = reloadCountByCoord[key];
          std::uint64_t partialStoreCount = partialStoreCountByCoord[key];
          std::uint64_t finalStoreCount = finalStoreCountByCoord[key];
          if (reloadCount != groupSize - 1 || partialStoreCount != groupSize - 1 || finalStoreCount != 1)
            return makeError("validateScheduleDependencyGraph: WS/IS partial-sum chain has the "
                              "wrong reload/store transition count for a group of size " +
                             llvm::Twine(groupSize));
        }
      }
    }
  }

  // Acyclicity + full topological coverage.
  llvm::Expected<std::vector<std::uint64_t>> topoOrErr = computeStableTopologicalOrder(graph);
  if (!topoOrErr)
    return topoOrErr.takeError();
  if (topoOrErr->size() != n)
    return makeError("validateScheduleDependencyGraph: topological order does not cover every node");
  {
    std::set<std::uint64_t> uniq(topoOrErr->begin(), topoOrErr->end());
    if (uniq.size() != topoOrErr->size())
      return makeError("validateScheduleDependencyGraph: topological order contains a duplicate node");
  }

  return llvm::Error::success();
}

// ===========================================================================
// Section 29: stable, deterministic JSON serialization.
// ===========================================================================
std::string
serializeScheduleDependencyGraphToJson(const ScheduleDependencyGraph &graph,
                                       const ResourceConstrainedTimingResult *timing) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "{\n";
  os << "  \"candidate_id\": \"" << graph.candidateId << "\",\n";
  os << "  \"node_count\": " << graph.operations.size() << ",\n";
  os << "  \"edge_count\": " << graph.dependencies.size() << ",\n";

  std::map<DependencyKind, std::uint64_t> histogram;
  for (const ScheduleDependency &d : graph.dependencies)
    ++histogram[d.kind];
  static const DependencyKind kAllKinds[] = {
      DependencyKind::DMACompletion,     DependencyKind::InputDataReady,
      DependencyKind::WeightDataReady,   DependencyKind::PartialOutputReady,
      DependencyKind::ComputeSerialization, DependencyKind::DMAEngineSerialization,
      DependencyKind::InputBufferReuse,  DependencyKind::WeightBufferReuse,
      DependencyKind::OutputBufferReuse, DependencyKind::OutputStoreCompletion,
      DependencyKind::ResidencyConstraint, DependencyKind::ProgramOrder,
  };
  os << "  \"edge_histogram\": {";
  for (std::size_t i = 0; i < std::size(kAllKinds); ++i) {
    os << "\"" << toString(kAllKinds[i]) << "\": " << histogram[kAllKinds[i]];
    if (i + 1 < std::size(kAllKinds))
      os << ", ";
  }
  os << "},\n";

  os << "  \"edges\": [\n";
  for (std::size_t i = 0; i < graph.dependencies.size(); ++i) {
    const ScheduleDependency &d = graph.dependencies[i];
    os << "    {\"producer\": " << d.producerOperationIndex
       << ", \"consumer\": " << d.consumerOperationIndex << ", \"kind\": \"" << toString(d.kind)
       << "\"}";
    if (i + 1 < graph.dependencies.size())
      os << ",";
    os << "\n";
  }
  os << "  ]";

  if (timing) {
    os << ",\n  \"timing\": {\"total_cycles\": " << timing->totalCycles << ", \"critical_path\": [";
    for (std::size_t i = 0; i < timing->criticalPathOperations.size(); ++i) {
      os << timing->criticalPathOperations[i];
      if (i + 1 < timing->criticalPathOperations.size())
        os << ", ";
    }
    os << "]}\n";
  } else {
    os << "\n";
  }
  os << "}";
  os.flush();
  return out;
}

} // namespace mlir::costmodel
