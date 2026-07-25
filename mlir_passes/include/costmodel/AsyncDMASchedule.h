#pragma once

// Cost Model Slice 8: Double Buffering and Asynchronous DMA Scheduling.
//
// Extends Slice 6's synchronous SchedulePlan (load/compute/store, one
// implicit Synchronize after every ComputeTile) into a statically
// planned ASYNCHRONOUS operation schedule: selectable per-role ping/pong
// buffering, symbolic DMA issue/wait events, and a single deterministic
// prefetch-next-tile scheduling policy. This slice remains a STATIC
// compiler model: no hardware DMA descriptors, no runtime command
// queues, no real threads, no actual asynchronous execution, no MLIR/
// SCF/Affine lowering, no code generation, no simulation, no hardware
// execution, no empirical calibration, no bank-conflict modeling, no NoC
// routing, no Hailo integration.
//
// ---------------------------------------------------------------------
// Extended candidate identity (Sections 2-3, 15)
// ---------------------------------------------------------------------
// A Slice 8 candidate is conceptually Slice 1's KernelCandidate x
// BufferingPolicy (BufferingPolicy.h) x DMASchedulingPolicy. This module
// never mutates KernelCandidate itself; ExtendedKernelCandidate pairs an
// existing (unchanged) KernelCandidate with the two new axes and a
// derived candidateId (base id + a stable suffix -- see
// extendedCandidateIdSuffix). generateExtendedCandidates() enumerates
// the full Cartesian product, in exactly candidates-outer, then
// allBufferingPolicies()'s fixed 8-entry order, then
// allDMASchedulingPolicies()'s fixed 2-entry order -- for N input
// candidates this constructs exactly N*8*2 ExtendedKernelCandidates (no
// pruning); e.g. 144 base compute candidates -> 2304 extended
// candidates, all actually constructed (not merely claimed).
//
// ---------------------------------------------------------------------
// Buffering-policy legality (Section 14) -- additive to Slice 2
// ---------------------------------------------------------------------
// checkBufferingLegality() classifies ONE (already Slice-2-legal)
// candidate's buffering policy against the target: for each role,
// requiredBytes = slotCount(mode) * <role tile bytes from Slice 2's
// CandidateLegalityResult>; a role whose mode is Double and whose
// requiredBytes exceeds its own dedicated buffer capacity is illegal
// (InputDoubleBufferExceedsBuffer / WeightDoubleBufferExceedsBuffer /
// OutputDoubleBufferExceedsBuffer -- LegalityReason, extended in
// CandidateLegality.h). aggregateRequiredBytes is the conservative sum
// across all three roles' requiredBytes (slot count x tile bytes,
// Section 14's explicitly-sanctioned conservative pre-check, NOT the
// concrete asynchronous-lifetime peak computed later by
// allocateAsyncScratchpad) checked against target.scratchpadBytes
// (AsyncAggregateScratchpadExceeded). target.dmaEngineCount == 0 is
// illegal (UnsupportedAsyncSchedule) -- Section 17 requires at least one
// logical DMA engine. A candidate whose base Slice 2 legality already
// failed is illegal here too (UnsupportedBufferingPolicy) -- buffering
// legality is never evaluated independently of base legality.
//
// ---------------------------------------------------------------------
// Ping/pong slot assignment (Section 5, 10) -- deterministic, derived
// from the schedule, never from pointers or hash-table order
// ---------------------------------------------------------------------
// materializeAsyncSchedule() calls the existing Slice 7
// analyzeBufferLifetimes() on the input SchedulePlan (consuming the
// REAL Slice 6 operation sequence, per Section 8 -- never reconstructing
// only aggregate counts) to obtain the ordered logical-buffer residency
// chain per role. This is deliberate reuse, not reinvention: Slice 6
// already places a dataflow's stationary operand's Load exactly once
// per residency group (e.g. Weight Stationary's LoadWeightTile precedes
// its whole M loop), so Slice 7's per-role lifetime list IS ALREADY the
// correct "logical iteration order" a role's slot must alternate over --
// no separate per-dataflow residency special-casing is needed for INPUT
// or WEIGHT roles in this module.
//
//   inputSlot(i)  = (i-th InputTile lifetime's index)  % 2  (if double)
//   weightSlot(i) = (i-th WeightTile lifetime's index) % 2  (if double)
//
// OUTPUT is the one role needing an explicit grouping step (Section 10's
// WS/IS partial-sum exception): consecutive OutputAccumulator lifetimes
// sharing the same (mTile, nTile) are one logical "output tile" group
// (Weight/Input Stationary's KT>1 spill model opens/closes a SEPARATE
// short-lived accumulator lifetime per k -- Slice 7 already models this
// structurally, see ScratchpadAllocation.h -- but they all belong to one
// output tile and must never alternate slots against each other). The
// group's slot is assigned once, from the group's own sequence index
// among groups (first-appearance order); every lifetime in the group
// uses that one slot. Output Stationary's groups are always exactly one
// lifetime each (already resident across all K), so this degenerates to
// per-lifetime alternation automatically for OS.
//
// ---------------------------------------------------------------------
// Async operation / event schema (Section 6-7)
// ---------------------------------------------------------------------
// Uses the unified generic form the task explicitly sanctions:
// AsyncOperationKind{DMAIssue, DMAWait, Compute, ComputeBarrier} +
// DMATransferKind{InputLoad, WeightLoad, PartialOutputReload,
// PartialOutputStore, FinalOutputStore}. Every DMAIssue produces exactly
// one DMAEvent with a unique, stable, dense event ID; every DMAWait
// consumes exactly one not-yet-consumed event; every issued event is
// waited (or drained, Section 25's end-of-schedule requirement) before
// the schedule ends. Dependency tracking is entirely via these integer
// event IDs -- never strings.
//
// ---------------------------------------------------------------------
// Scheduling policy (Section 3, 9-13) -- this module's one documented,
// deterministic list-scheduling rule
// ---------------------------------------------------------------------
// DMASchedulingPolicy::Synchronous: every DMAIssue is immediately
// followed by its own DMAWait, in exactly Slice 6's original operation
// order -- structurally identical to the prior synchronous model (no
// prefetched future transfer can ever appear; see
// validateAsyncSchedule's check of the same name). This is the required
// Section 20 compatibility path: Synchronous + all-single buffering
// reproduces the same operation/byte counts Slices 3-7 already computed
// (see the "Section 25 reconciliation" note below) -- it does not,
// however, byte-for-byte reproduce a SchedulePlan (Synchronize
// operations are represented by adjacent DMAWait/Compute pairs instead
// of a separate barrier op; ScheduleOperation vs AsyncScheduleOperation
// are deliberately distinct schemas per Section 6).
//
// DMASchedulingPolicy::PrefetchNext (prefetch distance == 1, at most one
// outstanding future tile per role -- Section 3): for a DOUBLE-buffered
// role with a next logical instance, that instance's Issue is placed
// right after the previous instance's Wait (i.e. logically "during" the
// compute that consumes the previous instance -- the actual overlap
// amount is determined by materializeAsyncCost's timestamp simulation,
// not by literal operation adjacency), and its Wait is deferred to
// immediately before the first operation that needs it. A
// SINGLE-buffered role, or a role with no next instance (KT==1 / only
// one tile / final instance), never prefetches -- its Issue/Wait stay
// adjacent, reconciling to the Synchronous behavior for that instance
// (Section 19: no fictitious overlap benefit is ever invented; Example
// D). Partial-output reload/store pairs WITHIN one (m,n) accumulator
// group (Weight/Input Stationary's KT>1 spill path) are never
// prefetched -- a reload always reads back exactly the data its own
// immediately preceding store wrote to the SAME physical slot (Section
// 10: slots never alternate within one output-tile group), so there is
// no independent work to overlap them with; this is a structural
// consequence of the model, not a missing feature. Only DISTINCT output
// tiles' FINAL/PARTIAL stores participate in cross-tile store/compute
// overlap (Section 12): a double-buffered output group's store may
// remain outstanding while the NEXT group (the alternate slot) computes
// and stores; its Wait is deferred until the slot is needed again (the
// group two positions later, which reuses the same slot parity) or
// schedule-end drain, whichever comes first. A single-buffered output
// group's store must complete (Wait immediately follows Issue) before
// the next group's first operation may reuse the one physical slot.
//
// ---------------------------------------------------------------------
// Async-aware cost model (Section 16-18) -- two-timeline simulation
// ---------------------------------------------------------------------
// One compute-engine timeline and one DMA-engine timeline (Section 17:
// target.dmaEngineCount must be >= 1; this Slice always models exactly
// one logical DMA engine regardless of its value being > 1 -- a future
// slice may model concurrent channels). DMA issues serialize on the DMA
// timeline in program (issue) order: dmaClock += target.dmaSetupCycles +
// ceil(bytes * target.clockHz / target.offChipBandwidthBytesPerSecond)
// per issue (checked arithmetic; reuses Slice 3's own per-byte bandwidth
// formula, never a new one). Each event's total cost is split into
// hidden/exposed at its Wait: hidden = min(eventCost, computeCyclesElapsedSinceIssue),
// exposed = eventCost - hidden -- so hiddenDMACycles + exposedDMACycles
// always reconciles EXACTLY to the sum of all modeled per-event DMA
// costs (Section 18's required invariant). A Wait only pays
// target.tileSynchronizationCycles when the compute engine actually
// reaches it before the event finishes (a real stall) -- never
// unconditionally. startupCycles == target.kernelLaunchCycles (charged
// once). drainCycles is the extra time the mandatory end-of-schedule
// drain (Section 25) adds beyond what compute alone would have taken.
// asynchronousEstimatedCycles is the compute timeline's final time
// after that drain. synchronousEstimatedCycles is the SAME simulation
// run over this candidate's Synchronous-policy operation sequence (not
// a separately-maintained formula) -- this is what lets the Section 20
// compatibility path and Section 25's "async <= sync only when overlap
// actually occurred" invariant both hold by construction rather than by
// coincidence between two independently written formulas.
//
// overlapCycles is defined equal to hiddenDMACycles (the DMA work
// actually hidden behind compute) -- a simpler two-engine model is
// explicitly sanctioned by Section 16, and this is that model's one
// chosen "benefit" metric, documented here rather than left ambiguous.
//
// ---------------------------------------------------------------------
// Ping/pong scratchpad allocation (Section 21) -- fixed offsets, not
// generic first-fit
// ---------------------------------------------------------------------
// allocateAsyncScratchpad() reserves, per region, exactly
// slotCount(mode) FIXED physical slots sized to that region's maximum
// observed tile byte footprint: Ping at offset 0, Pong at
// alignUp(tileBytes, target.dmaAlignmentBytes) (Section 21's own
// recommended formula) -- never a generic first-fit placement that could
// map two simultaneously-live ping/pong intervals to the same offset.
// The concrete peak per region is therefore exactly slotCount(mode) *
// tileBytes -- by construction never more than, and (because the slot
// reservation is fixed rather than demand-driven) exactly equal to, the
// Section 14 conservative capacity estimate for that role.
//
// ---------------------------------------------------------------------
// Reconciliation (Section 25-26) -- fails closed, never silently repairs
// ---------------------------------------------------------------------
// validateAsyncCostAgainstSchedule() reconstructs DMA operation/byte
// counts, compute count, partial-reload/store/final-store counts, and
// issued/waited event counts DIRECTLY from `schedule.operations` /
// `schedule.events` and requires exact equality against
// `schedule.cost`'s own copies of those same fields -- never a second
// independent cost formula. validateAsyncSchedule() (Section 24) is the
// structural/dependency validator (dense operation indices, unique
// event IDs, well-formed wait-after-issue, no double-wait, no
// unconsumed event, no premature buffer use, no live-slot overwrite,
// resident-role slot stability within its group, single-buffer roles
// never having two simultaneously live physical slots, double-buffer
// roles never using a slot index outside {0,1}, Synchronous schedules
// containing no prefetched future transfer, and candidate/schedule
// identity agreement).

#include "costmodel/BufferingPolicy.h"
#include "costmodel/CandidateCostModel.h"
#include "costmodel/CandidateLegality.h"
#include "costmodel/KernelCandidate.h"
#include "costmodel/PlanSelection.h"
#include "costmodel/SchedulePlan.h"
#include "costmodel/ScratchpadAllocation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::costmodel {

// ===========================================================================
// Section 3: asynchronous scheduling policy axis.
// ===========================================================================
enum class DMASchedulingPolicy { Synchronous, PrefetchNext };

llvm::StringRef toString(DMASchedulingPolicy policy);

// Fixed 2-entry order: {Synchronous, PrefetchNext}. Part of Slice 8's
// determinism contract (see generateExtendedCandidates).
const std::vector<DMASchedulingPolicy> &allDMASchedulingPolicies();

// ===========================================================================
// Sections 2-3, 15: extended candidate identity.
// ===========================================================================

// ":buf_i=single:buf_w=double:buf_o=single:dma=prefetch_next" -- appended
// to a base KernelCandidate's candidateId to form an extended candidate's
// ID (Section 15). Stable and deterministic; never includes a pointer,
// address, or timestamp.
std::string extendedCandidateIdSuffix(const BufferingPolicy &buffering,
                                      DMASchedulingPolicy dmaPolicy);

struct ExtendedKernelCandidate {
  std::string candidateId; // candidate.candidateId + the suffix above
  KernelCandidate candidate;
  BufferingPolicy buffering;
  DMASchedulingPolicy dmaPolicy = DMASchedulingPolicy::Synchronous;
};

// Full Cartesian product candidates x allBufferingPolicies() x
// allDMASchedulingPolicies(), fully constructed (no pruning) in exactly
// that nested order. For `candidates.size() == N` this always returns
// exactly N*8*2 entries -- callers wanting only a legal/profitable
// subset must filter the result themselves (e.g. via
// checkBufferingLegality); this function enumerates only.
std::vector<ExtendedKernelCandidate>
generateExtendedCandidates(llvm::ArrayRef<KernelCandidate> candidates);

// ===========================================================================
// Section 14: buffering-policy legality (additive to Slice 2's
// CandidateLegalityResult -- see the new LegalityReason values declared
// in CandidateLegality.h).
// ===========================================================================
struct BufferingLegalityResult {
  std::string candidateId; // the EXTENDED candidate id

  bool isLegal = false;
  std::vector<LegalityReason> reasons;

  std::uint64_t inputRequiredBytes = 0;     // slotCount(input)  * legality.inputTileBytes
  std::uint64_t weightRequiredBytes = 0;    // slotCount(weight) * legality.weightTileBytes
  std::uint64_t outputRequiredBytes = 0;    // slotCount(output) * legality.outputTileBytes
  std::uint64_t aggregateRequiredBytes = 0; // conservative sum of the three above
};

// Pure classification, mirroring Slice 2's checkCandidateLegality style:
// never mutates its inputs, never ranks/selects/schedules. `baseLegality`
// must correspond to `candidate` (candidateId checked); if
// !baseLegality.isLegal, the result is illegal via
// LegalityReason::UnsupportedBufferingPolicy without evaluating anything
// else (there is no sensible tile-byte footprint to reason about for an
// already-illegal base candidate).
BufferingLegalityResult checkBufferingLegality(const KernelCandidate &candidate,
                                                const CandidateLegalityResult &baseLegality,
                                                const BufferingPolicy &buffering,
                                                DMASchedulingPolicy dmaPolicy,
                                                const NPUTargetConfig &target);

// ===========================================================================
// Sections 5-7: ping/pong slot identity and the async operation/event
// schema.
// ===========================================================================
enum class BufferSlot { Single, Ping, Pong };

llvm::StringRef toString(BufferSlot slot);

struct BufferedTileRef {
  BufferRole role = BufferRole::InputTile;
  TileCoordinate coordinate;
  BufferSlot slot = BufferSlot::Single;
  std::uint32_t slotIndex = 0; // 0 for Single/Ping, 1 for Pong

  bool operator==(const BufferedTileRef &other) const {
    return role == other.role && coordinate.mTile == other.coordinate.mTile &&
           coordinate.nTile == other.coordinate.nTile &&
           coordinate.kTile == other.coordinate.kTile && slot == other.slot &&
           slotIndex == other.slotIndex;
  }
};

enum class AsyncOperationKind { DMAIssue, DMAWait, Compute, ComputeBarrier };

llvm::StringRef toString(AsyncOperationKind kind);

enum class DMATransferKind {
  InputLoad,
  WeightLoad,
  PartialOutputReload,
  PartialOutputStore,
  FinalOutputStore
};

llvm::StringRef toString(DMATransferKind kind);

struct AsyncScheduleOperation {
  std::uint64_t operationIndex = 0;
  AsyncOperationKind kind = AsyncOperationKind::Compute;

  std::optional<DMATransferKind> transferKind;
  std::optional<std::uint64_t> eventId;

  TileCoordinate coordinate;

  std::optional<BufferedTileRef> input;
  std::optional<BufferedTileRef> weight;
  std::optional<BufferedTileRef> output;

  std::uint64_t bytes = 0;

  bool operator==(const AsyncScheduleOperation &other) const;
};

struct DMAEvent {
  std::uint64_t eventId = 0;
  DMATransferKind transferKind = DMATransferKind::InputLoad;
  std::uint64_t issueOperationIndex = 0;
  std::uint64_t waitOperationIndex = 0; // valid once the event is drained/waited
  BufferedTileRef buffer;

  bool operator==(const DMAEvent &other) const;
};

// ===========================================================================
// Section 18: async-aware cycle breakdown.
// ===========================================================================
struct AsyncCandidateCost {
  std::string candidateId;

  std::uint64_t synchronousEstimatedCycles = 0;
  std::uint64_t asynchronousEstimatedCycles = 0;
  std::uint64_t exposedDMACycles = 0;
  std::uint64_t hiddenDMACycles = 0;
  std::uint64_t overlapCycles = 0; // == hiddenDMACycles, see header doc comment
  std::uint64_t startupCycles = 0;
  std::uint64_t drainCycles = 0;

  // ---- Section 25 reconciliation counts (reconstructed FROM the
  // AsyncSchedulePlan's own operations/events, never a second formula) ----
  std::uint64_t dmaOperationCount = 0;
  std::uint64_t dmaByteCount = 0;
  std::uint64_t computeCount = 0;
  std::uint64_t partialReloadCount = 0;
  std::uint64_t partialStoreCount = 0;
  std::uint64_t finalStoreCount = 0;
  std::uint64_t issuedEventCount = 0;
  std::uint64_t waitedEventCount = 0;

  bool operator==(const AsyncCandidateCost &other) const;
};

// ===========================================================================
// Section 8: the materialized async schedule.
// ===========================================================================
struct AsyncSlotCounts {
  std::uint32_t input = 1;
  std::uint32_t weight = 1;
  std::uint32_t output = 1;
};

struct AsyncSchedulePlan {
  std::string candidateId;     // base candidateId + extendedCandidateIdSuffix(...)
  std::string baseCandidateId; // the Slice 1-7 candidate this was built from

  Precision precision = Precision::INT8;
  Dataflow dataflow = Dataflow::WeightStationary;
  PEArrayShape peArray;
  TileShape tile;

  BufferingPolicy buffering;
  DMASchedulingPolicy dmaPolicy = DMASchedulingPolicy::Synchronous;
  std::uint32_t dmaEngineCount = 1;

  AsyncSlotCounts slots;

  std::vector<AsyncScheduleOperation> operations;
  std::vector<DMAEvent> events;

  AsyncCandidateCost cost;

  bool operator==(const AsyncSchedulePlan &other) const;
};

// Section 8's core transformation. Consumes the REAL Slice 6 operation
// sequence (via analyzeBufferLifetimes -- see the header doc comment
// above); never reconstructs only aggregate operation counts. Fails
// closed (llvm::Error) on: target.dmaEngineCount == 0; target's
// clock/bandwidth fields <= 0; a malformed/inconsistent
// `synchronousSchedule` (whatever analyzeBufferLifetimes itself would
// reject); integer overflow anywhere in the cycle simulation.
llvm::Expected<AsyncSchedulePlan>
materializeAsyncSchedule(const SchedulePlan &synchronousSchedule,
                          const BufferingPolicy &buffering,
                          DMASchedulingPolicy schedulingPolicy,
                          const NPUTargetConfig &target);

// ===========================================================================
// Section 24: structural/dependency validation. Fails closed.
// ===========================================================================
llvm::Error validateAsyncSchedule(const AsyncSchedulePlan &plan, const NPUTargetConfig &target);

// ===========================================================================
// Section 25: cost/schedule reconciliation. Fails closed.
// ===========================================================================
llvm::Error validateAsyncCostAgainstSchedule(const AsyncSchedulePlan &schedule,
                                              const AsyncCandidateCost &cost);

// ===========================================================================
// Section 21: ping/pong scratchpad allocation.
// ===========================================================================
struct AsyncBufferAllocation {
  BufferRole role = BufferRole::InputTile;
  MemoryRegionKind region = MemoryRegionKind::InputBuffer;
  BufferSlot slot = BufferSlot::Single;
  std::uint64_t offsetBytes = 0;
  std::uint64_t sizeBytes = 0;

  bool operator==(const AsyncBufferAllocation &other) const {
    return role == other.role && region == other.region && slot == other.slot &&
           offsetBytes == other.offsetBytes && sizeBytes == other.sizeBytes;
  }
};

struct AsyncScratchpadAllocationPlan {
  std::string candidateId;

  std::vector<AsyncBufferAllocation> allocations; // <= 2 per region (Ping/Pong or Single)

  std::uint64_t peakInputBufferBytes = 0;
  std::uint64_t peakWeightBufferBytes = 0;
  std::uint64_t peakOutputBufferBytes = 0;
  std::uint64_t peakAggregateScratchpadBytes = 0;
};

// Fixed-offset ping/pong allocation (Section 21) -- Ping at 0, Pong at
// alignUp(tileBytes, target.dmaAlignmentBytes), per region. Fails closed
// on zero alignment or address overflow.
llvm::Expected<AsyncScratchpadAllocationPlan>
allocateAsyncScratchpad(const AsyncSchedulePlan &plan, const NPUTargetConfig &target);

// Section 26: the concrete allocation peak must never exceed the
// Section 14 conservative buffering-legality capacity estimate, and
// must fit the target's own per-region/aggregate capacities. Fails
// closed on any mismatch or overflow; never silently modifies a policy
// after allocation.
llvm::Error validateAsyncAllocationAgainstBufferingLegality(const AsyncScratchpadAllocationPlan &allocation,
                                                             const BufferingLegalityResult &bufferingLegality,
                                                             const NPUTargetConfig &target);

// ===========================================================================
// Section 27: stable, deterministic JSON serialization.
// ===========================================================================
std::string serializeAsyncSchedulePlanToJson(const AsyncSchedulePlan &plan);

// ===========================================================================
// Section 20: deterministic ranking helper over extended candidates that
// already have a materialized AsyncSchedulePlan. Ranking key, in order:
//   1. asynchronousEstimatedCycles ascending
//   2. exposedDMACycles            ascending
//   3. peakAggregateScratchpadBytes ascending
//   4. number of double-buffered roles ascending (fewer buffers wins a tie)
//   5. originalIndex ascending (mandatory final tie-break)
// Never favors maximum buffering in a tie (Section 20); never modifies
// the previously selected synchronous result when async scheduling is
// disabled (this helper only compares/orders -- it does not select or
// mutate anything).
// ===========================================================================
struct RankedAsyncCandidate {
  std::string candidateId;
  std::size_t originalIndex = 0;
  std::size_t rank = 0; // 0-based; rank 0 is best.
};

std::vector<RankedAsyncCandidate>
rankAsyncCandidates(llvm::ArrayRef<AsyncSchedulePlan> plans,
                    llvm::ArrayRef<AsyncScratchpadAllocationPlan> allocations);

} // namespace mlir::costmodel
