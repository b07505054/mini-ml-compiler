#pragma once

// Cost Model Slice 9: Explicit Dependency Graph and Critical-Path
// Validation.
//
// Converts a Slice 8 AsyncSchedulePlan into an explicit dependency DAG
// and independently re-derives/validates everything Slice 8's builder
// only guaranteed by construction: data dependencies, DMA-completion
// dependencies, buffer-overwrite hazards, output-store-completion
// hazards, compute/DMA-engine serialization, dataflow-specific residency
// contracts, WS/IS partial-sum chains, acyclicity, and a
// resource-constrained timing model that reconciles EXACTLY with Slice
// 8's two-clock event simulator. This slice remains a static
// compiler-planning model: no MLIR/SCF/Affine lowering, no code
// generation, no simulation, no hardware execution, no empirical
// calibration, no Hailo integration, no multiple DMA channels, no
// prefetch distance beyond one, no schedule rewriting/reordering.
//
// ---------------------------------------------------------------------
// Node identity (Section 2)
// ---------------------------------------------------------------------
// A node IS an AsyncScheduleOperation's operationIndex -- graph.operations
// is a direct copy of schedule.operations (same size, same index
// alignment); no host pointer or hash-table order is ever used as node
// identity.
//
// ---------------------------------------------------------------------
// Dependency-kind schema and this module's exact edge policy
// ---------------------------------------------------------------------
// All twelve kinds in DependencyKind below are declared; this Slice's
// concrete schedules (produced by Slice 8's materializeAsyncSchedule)
// use eleven of them structurally -- see the doc comment on each kind
// for its precise producer/consumer meaning. `ProgramOrder` is declared
// for forward compatibility (Section 13 sanctions adding it "only where
// operation semantics require ordering not otherwise represented") but
// is NEVER emitted by buildScheduleDependencyGraph() for a schedule
// produced by the current Slice 8 builder: every real ordering
// requirement in such a schedule is already covered by one of the other
// eleven kinds (see "Why no compute-to-store-issue edge" below). This
// mirrors Slice 7's own documented precedent of an intentionally-unused
// enumerator (MemoryRegionKind::Scratchpad).
//
//   DMACompletion:          IssueX -> WaitX, for InputLoad/WeightLoad/
//                           PartialOutputReload transfers.
//   OutputStoreCompletion:  IssueX -> WaitX, for PartialOutputStore/
//                           FinalOutputStore transfers ONLY -- a
//                           deliberately distinct kind from
//                           DMACompletion (not a subtype folded in) so
//                           the edge histogram/evidence can show output-
//                           store completions independently, per
//                           Section 25's required evidence item.
//   InputDataReady:         the most recently completed InputLoad Wait
//                           -> every ComputeTile it feeds (one producer,
//                           possibly many consumers when the input is
//                           IS-resident across N).
//   WeightDataReady:        ditto for WeightLoad Waits (WS-resident
//                           across M).
//   PartialOutputReady:     the most recently completed
//                           PartialOutputReload Wait -> every ComputeTile
//                           it feeds. Never emitted for Output
//                           Stationary (OS never reloads).
//   ComputeSerialization:   the single compute-ENGINE timeline, in
//                           program order. This chain includes BOTH
//                           Compute nodes AND DMAWait nodes -- see "Why
//                           DMAWait shares the compute-engine resource"
//                           below; this is what lets
//                           computeResourceConstrainedTiming reconcile
//                           exactly with Slice 8's single `computeClock`
//                           variable, which likewise advances via both
//                           Compute steps and (conditionally) DMAWait
//                           steps in one sequential pass.
//   DMAEngineSerialization: consecutive DMAIssue nodes (any transfer
//                           kind), in program (issue) order -- the
//                           single DMA-engine timeline (Section 17:
//                           dmaEngineCount == 1 for this Slice).
//   InputBufferReuse:       the last ComputeTile that read a given
//                           (InputTile, slot) -> the next DMAIssue
//                           (InputLoad) that overwrites that same
//                           physical slot.
//   WeightBufferReuse:      ditto for (WeightTile, slot) / WeightLoad.
//   OutputBufferReuse:      the Wait of the most recent operation that
//                           WROTE (store) or READ-BACK (reload) a given
//                           (OutputAccumulator, slot) -> the next
//                           operation that touches that same physical
//                           slot (either a PartialOutputReload issue
//                           reading the just-stored data back within the
//                           SAME output-tile group, or a ComputeTile
//                           that opens a NEW output-tile group reusing
//                           the slot two groups later). Both cases are
//                           the same underlying "physical slot became
//                           free/valid, here is its next accessor"
//                           relationship -- see the .cpp for why they are
//                           not split into two kinds.
//   ResidencyConstraint:    diagnostic-only (Section 11: "only if useful
//                           for diagnostics") edge from a resident
//                           group's first ComputeTile to its last
//                           ComputeTile, one per IS input-resident group,
//                           WS weight-resident group, and OS
//                           accumulator-resident group. Never consulted
//                           by cycle detection, topological order, or
//                           timing -- purely an inspectable record of
//                           which computes the graph-level residency
//                           validator (Section 11/26) checked.
//   ProgramOrder:           declared, unused by this Slice's builder
//                           (see above).
//
// Why no compute-to-store-issue edge (and why ProgramOrder stays
// unused): Slice 8's reference simulator (AsyncDMASchedule.cpp's
// simulateCycles) advances `dmaClock` unconditionally at every DMAIssue
// -- including output-store issues -- with NO check against the
// producing compute's own completion; the DMA engine is modeled as an
// always-busy queue (documented in AsyncDMASchedule.h's Section 16-18
// doc comment). Adding a compute -> store-issue timing edge here would
// be a MORE accurate model but would make computeResourceConstrainedTiming
// diverge from Slice 8's numbers, breaking the exact reconciliation this
// Slice's core purpose requires (Section 21: "Existing event/timeline
// simulator = Independent explicit-DAG timing model"). A future slice
// that revises Slice 8's DMA-engine-starvation assumption would add this
// edge (as `ProgramOrder`) and revise Slice 8's simulator to match, in
// lockstep. Nothing in Section 15's validation list requires this edge
// to exist, so omitting it does not weaken any required check.
//
// Why DMAWait shares the compute-engine resource: Slice 8's
// simulateCycles has exactly two clocks, `computeClock` and `dmaClock`.
// `computeClock` advances at every Compute step (+= perComputeCycles)
// AND, conditionally, at every DMAWait step (jumps to
// max(computeClock, eventFinish) + syncCycles-if-stalled). `dmaClock`
// advances ONLY at DMAIssue steps. There is no third clock. A DMAWait is
// therefore, in Slice 8's own model, an action the COMPUTE side takes to
// synchronize -- not a DMA-engine action -- so this Slice types DMAWait
// nodes as ScheduleResourceKind::ComputeEngine and threads them into the
// same ComputeSerialization chain as Compute nodes, in exact program
// order. See computeResourceConstrainedTiming's doc comment for the
// resulting per-node timing rule.
//
// ---------------------------------------------------------------------
// Direct-edge policy (Section 24)
// ---------------------------------------------------------------------
// The stored graph contains DIRECT edges only -- no transitive closure.
// Reachability (hasDependencyPath) is computed on demand via graph
// traversal, never precomputed into the edge list.

#include "costmodel/AsyncDMASchedule.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::costmodel {

enum class DependencyKind {
  DMACompletion,
  InputDataReady,
  WeightDataReady,
  PartialOutputReady,
  ComputeSerialization,
  DMAEngineSerialization,
  InputBufferReuse,
  WeightBufferReuse,
  OutputBufferReuse,
  OutputStoreCompletion,
  ResidencyConstraint,
  ProgramOrder,
};

llvm::StringRef toString(DependencyKind kind);

struct ScheduleDependency {
  std::uint64_t producerOperationIndex = 0;
  std::uint64_t consumerOperationIndex = 0;
  DependencyKind kind = DependencyKind::ProgramOrder;

  bool operator==(const ScheduleDependency &other) const {
    return producerOperationIndex == other.producerOperationIndex &&
           consumerOperationIndex == other.consumerOperationIndex && kind == other.kind;
  }
};

struct ScheduleDependencyGraph {
  std::string candidateId;

  std::vector<AsyncScheduleOperation> operations;
  std::vector<ScheduleDependency> dependencies;

  // outgoingEdges[i] / incomingEdges[i]: indices into `dependencies`
  // whose producer/consumer (respectively) is operation i. Populated by
  // buildScheduleDependencyGraph(); always index-aligned with
  // `operations`.
  std::vector<std::vector<std::uint64_t>> outgoingEdges;
  std::vector<std::vector<std::uint64_t>> incomingEdges;

  bool operator==(const ScheduleDependencyGraph &other) const;
};

// Section 3-4: builds the graph from an already-Slice-8-valid schedule
// (Section 4, item 10: existing validateAsyncSchedule must succeed
// first -- checked internally; fails closed otherwise). Consumes the
// actual operation list and symbolic events, never aggregate counts.
llvm::Expected<ScheduleDependencyGraph>
buildScheduleDependencyGraph(const AsyncSchedulePlan &schedule, const NPUTargetConfig &target);

// Section 15: independent structural/dependency/residency validation.
// Fails closed; returns the first violation found in a fixed, documented
// order.
llvm::Error validateScheduleDependencyGraph(const ScheduleDependencyGraph &graph,
                                            const AsyncSchedulePlan &schedule,
                                            const NPUTargetConfig &target);

// Section 16-17: deterministic cycle-safe topological order (Kahn's
// algorithm; ties broken by lowest operationIndex, via a sorted ready
// set). Returns an error (with cycle evidence in the message: a witness
// producer/consumer/kind and the set of nodes never scheduled) if the
// graph is not a DAG.
llvm::Expected<std::vector<std::uint64_t>>
computeStableTopologicalOrder(const ScheduleDependencyGraph &graph);

// Section 23: on-demand reachability (no transitive closure stored).
bool hasDependencyPath(const ScheduleDependencyGraph &graph, std::uint64_t producer,
                       std::uint64_t consumer);

// ===========================================================================
// Section 18: node timing metadata.
// ===========================================================================
enum class ScheduleResourceKind { None, DMAEngine, ComputeEngine };

llvm::StringRef toString(ScheduleResourceKind kind);

struct OperationTimingInfo {
  std::uint64_t operationIndex = 0;
  std::uint64_t durationCycles = 0;
  ScheduleResourceKind resource = ScheduleResourceKind::None;
};

// Derives one OperationTimingInfo per graph node, in operationIndex
// order. DMAIssue: target.dmaSetupCycles + ceil(bytes*clockHz/bandwidth)
// (identical formula to AsyncDMASchedule.cpp's simulateCycles), resource
// DMAEngine. Compute: the same per-(m,n,k)-tile compute-cycle formula
// Slice 8 reuses from Slice 3 (arrayWavesPerTile * reductionCyclesPerKTile),
// resource ComputeEngine. DMAWait / ComputeBarrier: duration 0, resource
// ComputeEngine for DMAWait (see the header doc comment on
// ComputeSerialization) and None for ComputeBarrier (unused kind).
// Fails closed on a zero-throughput target or arithmetic overflow.
llvm::Expected<std::vector<OperationTimingInfo>>
deriveOperationTimings(const AsyncSchedulePlan &schedule, const NPUTargetConfig &target);

// ===========================================================================
// Section 19: dependency-only critical path (uses EVERY edge in the
// graph, including buffer-reuse/residency edges -- the fullest,
// most-conservative dependency closure). This is deliberately a
// DIFFERENT, generally-larger-or-equal number than the resource-
// constrained result below; see that function's doc comment for why the
// two are not the same computation.
// ===========================================================================
struct CriticalPathResult {
  std::uint64_t totalCycles = 0;
  std::vector<std::uint64_t> criticalPathOperations;
  std::vector<std::uint64_t> earliestStartCycles;  // index-aligned with graph.operations
  std::vector<std::uint64_t> earliestFinishCycles; // index-aligned with graph.operations
};

llvm::Expected<CriticalPathResult>
computeDependencyCriticalPath(const ScheduleDependencyGraph &graph,
                              llvm::ArrayRef<OperationTimingInfo> timings);

// ===========================================================================
// Section 20: resource-constrained timing -- the model that reconciles
// EXACTLY with Slice 8's simulateCycles. Uses only the four "resource
// timeline" edge kinds (DMACompletion, OutputStoreCompletion,
// ComputeSerialization, DMAEngineSerialization) for its forward pass --
// NOT the buffer-reuse/residency edges (see the header doc comment on
// "Why no compute-to-store-issue edge" for why the two APIs
// deliberately use different edge subsets). Every node's earliest
// start/finish is generic max(predecessor finishes) + duration EXCEPT
// DMAWait nodes, which use Slice 8's exact conditional rule:
//   chainFinish = finish of the incoming ComputeSerialization predecessor
//                 (or target.kernelLaunchCycles if none)
//   completionFinish = finish of the incoming DMACompletion /
//                       OutputStoreCompletion predecessor (this event's
//                       issue)
//   if chainFinish < completionFinish:
//       finish(wait) = completionFinish + target.tileSynchronizationCycles  (stalled)
//   else:
//       finish(wait) = chainFinish                                          (not stalled)
// This is Slice 8's own per-event hidden/exposed accounting, re-derived
// independently from the graph rather than the original operation walk.
// ===========================================================================
struct ResourceConstrainedTimingResult {
  std::uint64_t totalCycles = 0;
  std::uint64_t dmaBusyCycles = 0;     // sum of DMAEngine-resource node durations
  std::uint64_t computeBusyCycles = 0; // sum of ComputeEngine (Compute-kind only) durations
  std::uint64_t overlapCycles = 0;     // genuine [start,finish) interval intersection, DMA vs Compute
  std::uint64_t startupCycles = 0;     // target.kernelLaunchCycles
  std::uint64_t drainCycles = 0;       // totalCycles - max(finish over DMAIssue/Compute nodes only)

  std::vector<std::uint64_t> startCycles;  // index-aligned with graph.operations
  std::vector<std::uint64_t> finishCycles; // index-aligned with graph.operations
  std::vector<std::uint64_t> criticalPathOperations;
};

llvm::Expected<ResourceConstrainedTimingResult>
computeResourceConstrainedTiming(const ScheduleDependencyGraph &graph,
                                  llvm::ArrayRef<OperationTimingInfo> timings,
                                  const NPUTargetConfig &target);

// Section 21: fails closed on the FIRST mismatching field (fixed order:
// totalCycles, dmaBusyCycles vs cost.dmaOperationCount-derived bytes is
// NOT compared -- only cycle-domain fields are; see the .cpp for the
// exact field-by-field mapping), comparing common quantities with exact
// integer equality (never a tolerance).
llvm::Error validateGraphTimingAgainstAsyncCost(const ScheduleDependencyGraph &graph,
                                                const ResourceConstrainedTimingResult &graphTiming,
                                                const AsyncCandidateCost &slice8Cost);

// ===========================================================================
// Section 29: stable, deterministic JSON serialization. `timing` is
// optional; when supplied, adds a "timing" object with total_cycles and
// critical_path.
// ===========================================================================
std::string
serializeScheduleDependencyGraphToJson(const ScheduleDependencyGraph &graph,
                                       const ResourceConstrainedTimingResult *timing = nullptr);

} // namespace mlir::costmodel
