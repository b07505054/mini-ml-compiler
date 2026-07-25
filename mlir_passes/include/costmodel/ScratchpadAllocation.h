#pragma once

// Cost Model Slice 7: Buffer Lifetime and Scratchpad Allocation.
//
// Converts a Slice 6 SchedulePlan into an explicit, deterministic
// SYMBOLIC scratchpad allocation: which logical buffers exist, when each
// is live, which interfere, which may reuse memory, and their concrete
// (region, offset) placement. Static planning only -- no double
// buffering, no asynchronous DMA, no concrete runtime DMA descriptors,
// no MLIR lowering, no executable code, no simulation, no hardware
// execution, no Hailo integration.
//
// ---------------------------------------------------------------------
// Memory-region semantics (Section 2)
// ---------------------------------------------------------------------
// InputBuffer, WeightBuffer, and OutputBuffer are DISTINCT logical
// regions, each with its own dedicated capacity
// (target.inputBufferBytes / weightBufferBytes / outputBufferBytes) --
// this preserves Slice 2's existing legality semantics exactly (Slice 2
// already checks each tile's footprint against its own eponymous
// buffer). `scratchpadBytes` is an AGGREGATE total-capacity ceiling
// covering all three regions combined, not a fourth physically
// independent region -- MemoryRegionKind::Scratchpad exists in the enum
// only as the (unused by role-to-region mapping, see Section 11)
// conceptual "aggregate" designation for documentation/diagnostics; no
// BufferAllocation is ever placed directly "in" it.
//
// ---------------------------------------------------------------------
// Buffer-role identity and lifetime rules (Sections 4-5)
// ---------------------------------------------------------------------
// A logical buffer is a RESIDENCY INTERVAL, not one DMA operation. Rules
// (derived generically from the operation stream, not hardcoded per
// dataflow -- see analyzeBufferLifetimes):
//
//   InputTile / WeightTile: a buffer opens at its Load*Tile operation and
//   closes at the LAST ComputeTile that uses it BEFORE the next Load of
//   the same role appears (or the schedule's end). This alone produces
//   the correct dataflow-specific residency automatically: Weight
//   Stationary's LoadWeightTile precedes its whole mTile loop, so its
//   buffer's lifetime naturally spans every compute in that loop (MT
//   computes) before the next (n,k) pair's LoadWeightTile appears;
//   ordinary per-compute loads (WS's LoadInputTile, IS's LoadWeightTile,
//   OS's both) are immediately followed by their own ComputeTile and then
//   immediately re-loaded, so their lifetime naturally spans exactly one
//   compute.
//
//   OutputAccumulator: a buffer opens at either a ReloadPartialOutput
//   operation OR (if none precedes it yet for this (m,n) chain) the
//   first ComputeTile of the chain, and closes at the next
//   StorePartialOutput or StoreFinalOutput. For Output Stationary
//   (which never emits Reload/StorePartial at all -- see SchedulePlan.h)
//   this alone produces exactly ONE accumulator buffer per (m,n),
//   resident across all KT computes through StoreFinalOutput. For Weight
//   Stationary / Input Stationary's conservative partial-spill policy,
//   this produces a SEPARATE accumulator residency interval per k tile
//   (the local accumulator dies at each StorePartialOutput and a NEW one
//   is born at the next ReloadPartialOutput) -- this slice does NOT
//   pretend the accumulator stays resident across K when the schedule
//   explicitly spills it; the distinction is structural, not a comment.
//
// ---------------------------------------------------------------------
// Sizes, alignment (Sections 7-8)
// ---------------------------------------------------------------------
// Every BufferLifetime's `sizeBytes` is read directly from the
// ScheduleOperation.bytes field that opened it (LoadInputTile.bytes = I,
// LoadWeightTile.bytes = W, the accumulator-opening operation's bytes
// for OutputAccumulator = O) -- physical padded tile bytes, exactly
// Slice 2/5/6's existing formulas, never independently reinvented.
// Boundary tiles therefore occupy the same allocation size as full tiles
// (no compact boundary allocation in this slice). `alignmentBytes` is
// `target.dmaAlignmentBytes` for every role (no role-specific byte-
// alignment field exists on NPUTargetConfig beyond this).
//
// ---------------------------------------------------------------------
// Interference (Section 9) -- inclusive endpoints
// ---------------------------------------------------------------------
// Two buffers in the SAME region interfere iff their [firstOperationIndex,
// lastOperationIndex] intervals overlap INCLUSIVELY:
//   interferes(A, B) = !(A.last < B.first || B.last < A.first)
// Touching intervals (A.last == B.first) DO interfere under this policy
// (the shared operation index conservatively counts as simultaneous use
// by both) -- this never arises naturally from Slice 6's own emission
// order (consecutive same-role buffers are always separated by at least
// one intervening operation index), but is honored exactly as specified
// for hand-constructed inputs. Buffers in DIFFERENT regions never
// interfere regardless of their intervals (they do not share an offset
// space).
//
// ---------------------------------------------------------------------
// Allocator (Section 10) -- deterministic aligned first-fit, per region
// ---------------------------------------------------------------------
// Buffers within one region are sorted by (firstOperationIndex,
// coordinate.mTile, coordinate.nTile, coordinate.kTile, bufferId.value)
// -- a total, deterministic order with no dependence on pointers, hash-
// table iteration, or allocation order. Already-placed allocations whose
// lastOperationIndex < the new buffer's firstOperationIndex are expired
// (no longer considered live); the lowest aligned offset that does not
// overlap any still-live allocation in the region is chosen. This is a
// simple, deterministic first-fit -- not NP-hard optimal packing.
//
// ---------------------------------------------------------------------
// Peak bytes (Section 11)
// ---------------------------------------------------------------------
// Computed from BufferLifetime data (sizes + intervals), independent of
// the allocator's chosen offsets: peakInputBufferBytes/
// peakWeightBufferBytes/peakOutputBufferBytes are each the maximum, over
// every operation index, of the SUM of sizeBytes of that region's
// buffers whose interval contains that index (i.e. the true concurrent-
// residency requirement, not an allocator-fragmentation-inflated
// footprint). peakAggregateScratchpadBytes is the maximum, over every
// operation index, of the SUM across all three regions simultaneously --
// a genuine timeline sweep, not simply the sum of the three independent
// per-region peaks (which would only be a safe upper bound, not
// necessarily tight).

#include "costmodel/CandidateLegality.h"
#include "costmodel/SchedulePlan.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::costmodel {

enum class MemoryRegionKind { InputBuffer, WeightBuffer, OutputBuffer, Scratchpad };
llvm::StringRef toString(MemoryRegionKind kind);

enum class BufferRole { InputTile, WeightTile, OutputAccumulator };
llvm::StringRef toString(BufferRole role);

struct LogicalBufferId {
  std::uint64_t value = 0;
  bool operator==(const LogicalBufferId &other) const { return value == other.value; }
  bool operator<(const LogicalBufferId &other) const { return value < other.value; }
};

struct BufferLifetime {
  LogicalBufferId bufferId;
  BufferRole role = BufferRole::InputTile;

  TileCoordinate coordinate;

  std::uint64_t sizeBytes = 0;
  std::uint64_t alignmentBytes = 0;

  std::uint64_t firstOperationIndex = 0;
  std::uint64_t lastOperationIndex = 0;

  MemoryRegionKind region = MemoryRegionKind::InputBuffer;

  bool operator==(const BufferLifetime &other) const;
};

struct BufferAllocation {
  LogicalBufferId bufferId;
  MemoryRegionKind region = MemoryRegionKind::InputBuffer;

  std::uint64_t offsetBytes = 0;
  std::uint64_t sizeBytes = 0;
  std::uint64_t alignmentBytes = 0;

  std::uint64_t firstOperationIndex = 0;
  std::uint64_t lastOperationIndex = 0;

  bool operator==(const BufferAllocation &other) const;
};

struct ScratchpadAllocationPlan {
  std::string candidateId;

  std::vector<BufferLifetime> lifetimes;
  std::vector<BufferAllocation> allocations;

  std::uint64_t peakInputBufferBytes = 0;
  std::uint64_t peakWeightBufferBytes = 0;
  std::uint64_t peakOutputBufferBytes = 0;
  std::uint64_t peakAggregateScratchpadBytes = 0;

  bool operator==(const ScratchpadAllocationPlan &other) const;
};

// Optional addressed-operation view (Section 14): symbolic
// region-relative addresses only, never a hardware DMA descriptor.
struct AddressedScheduleOperation {
  ScheduleOperation operation;
  std::optional<BufferAllocation> input;
  std::optional<BufferAllocation> weight;
  std::optional<BufferAllocation> output;
};

// Derives the ordered list of logical buffer lifetimes directly from
// `schedule.operations` (Section 6) -- fails closed (llvm::Error) on:
// non-dense/non-monotonic operation indices; a ComputeTile with no live
// input, weight, or accumulator buffer open; a Store*/Reload with no
// live accumulator; any buffer still open after the last operation; or
// an observed lifetime span inconsistent with `schedule.residency`
// (Section 6's cross-check). `target` supplies only alignmentBytes
// (target.dmaAlignmentBytes) for each lifetime.
llvm::Expected<std::vector<BufferLifetime>> analyzeBufferLifetimes(const SchedulePlan &schedule,
                                                                    const NPUTargetConfig &target);

// End-to-end: analyzeBufferLifetimes() + deterministic per-region
// first-fit allocation + peak-byte computation. candidateId is copied
// from `schedule.candidateId`.
llvm::Expected<ScratchpadAllocationPlan> allocateScratchpad(const SchedulePlan &schedule,
                                                            const NPUTargetConfig &target);

// Lower-level entry point: runs the identical per-region deterministic
// first-fit allocation + peak-byte computation directly on an already-
// built (possibly hand-constructed) lifetime list, skipping
// analyzeBufferLifetimes(). Useful for constructing precise,
// hand-computable allocator test scenarios (e.g. deliberately
// overlapping same-role lifetimes) without needing a full valid
// SchedulePlan's operation stream. Fails closed on the same structural
// problems allocateScratchpad() would (zero size/alignment, address
// overflow) -- it does NOT re-validate load/use/store ordering, since
// there is no operation stream here to validate against.
llvm::Expected<ScratchpadAllocationPlan>
allocateScratchpadFromLifetimes(const std::string &candidateId,
                                std::vector<BufferLifetime> lifetimes);

// Section 12: fails closed unless peakInput/Weight/Output <=
// target's corresponding buffer capacity AND peakAggregateScratchpadBytes
// <= target.scratchpadBytes, every allocation's offset+size stays inside
// its region's declared capacity, every offset is aligned, no two
// interfering same-region allocations overlap, every lifetime has
// exactly one allocation (no missing, no duplicate, no unknown
// allocation), and plan.candidateId is non-empty (identity is checked
// against a specific candidate by the caller via
// validateAllocationAgainstLegality below, which additionally requires
// an exact ID match).
llvm::Error validateAllocationAgainstTarget(const ScratchpadAllocationPlan &plan,
                                            const NPUTargetConfig &target);

// Section 13: fails closed if plan.candidateId != legality.candidateId,
// or if any concrete peak requirement exceeds what Slice 2 predicted
// for that role (legality.inputTileBytes/weightTileBytes/outputTileBytes/
// totalScratchpadBytes) -- surfacing, not silently repairing, any
// discrepancy between the static legality footprint model and the
// concrete materialized-schedule allocation.
llvm::Error validateAllocationAgainstLegality(const ScratchpadAllocationPlan &allocation,
                                              const CandidateLegalityResult &legality);

// Builds the per-operation addressed view (Section 14): for each
// operation, looks up whichever input/weight/output BufferAllocation (if
// any) is live at that operation's index. Symbolic region-relative
// addresses only -- never a hardware DMA descriptor, never written here
// to disk.
std::vector<AddressedScheduleOperation>
buildAddressedSchedule(const SchedulePlan &schedule, const ScratchpadAllocationPlan &allocation);

// Deterministic diagnostic JSON (Section 19): candidate_id, peak_bytes,
// and the allocation table (buffer_id/role/region/offset/size/first_op/
// last_op per allocation). Never writes to disk; never includes a host
// pointer.
std::string serializeScratchpadAllocationPlanToJson(const ScratchpadAllocationPlan &plan);

} // namespace mlir::costmodel
