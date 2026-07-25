#pragma once

// Cost Model Slice 6: Selected Schedule Materialization.
//
// Converts the already-selected abstract candidate (Slice 4's
// PlanSelectionResult, built from Slices 1-3/5's KernelCandidate,
// CandidateLegalityResult, CandidateCostEstimate) into an explicit,
// inspectable, deterministic COMPILER SCHEDULE DESCRIPTION: iteration
// space, loop order, dataflow-specific buffer residency, and one
// ScheduleOperation per (load/reload/compute/store/synchronize) event in
// exact emitted order. This slice materializes a schedule DESCRIPTION
// only -- it generates no executable code: no MLIR (SCF/Affine/Linalg),
// no LLVM IR, no real DMA commands, no physical scratchpad addresses, no
// runtime dispatch, no simulation, no hardware execution, no empirical
// calibration, and no Hailo integration of any kind.
//
// ---------------------------------------------------------------------
// Dataflow-specific loop order and residency (Section 3) -- explicit
// schedule semantics, never derived from enum ordinal values
// ---------------------------------------------------------------------
//   Weight Stationary:  loop order N -> K -> M
//                        weightResidentAcrossM = true (only)
//   Input Stationary:   loop order M -> K -> N
//                        inputResidentAcrossN = true (only)
//   Output Stationary:  loop order M -> N -> K
//                        outputResidentAcrossK = true (only)
//
// ---------------------------------------------------------------------
// Operation-emission rules (Sections 5-9)
// ---------------------------------------------------------------------
// Let MT/NT/KT be numMTiles/numNTiles/numReductionTiles (recomputed here
// from `problem` + `candidate.tile` and cross-checked against the
// supplied CandidateCostEstimate -- see materializeCandidateSchedule).
//
//   Weight Stationary (for nTile, for kTile, { LoadWeightTile; for mTile
//     { LoadInputTile; [kTile>0: ReloadPartialOutput]; ComputeTile;
//     [kTile+1<KT: StorePartialOutput else StoreFinalOutput];
//     Synchronize } }):
//       weightLoadCount          = NT * KT
//       inputLoadCount           = MT * NT * KT
//       partialOutputReloadCount = MT * NT * (KT - 1)
//       partialOutputStoreCount  = MT * NT * (KT - 1)
//       finalOutputStoreCount    = MT * NT
//       computeCount = synchronizationCount = MT * NT * KT
//
//   Input Stationary (for mTile, for kTile, { LoadInputTile; for nTile
//     { LoadWeightTile; [kTile>0: ReloadPartialOutput]; ComputeTile;
//     [kTile+1<KT: StorePartialOutput else StoreFinalOutput];
//     Synchronize } }):
//       inputLoadCount           = MT * KT
//       weightLoadCount          = MT * NT * KT
//       partialOutputReloadCount = MT * NT * (KT - 1)
//       partialOutputStoreCount  = MT * NT * (KT - 1)
//       finalOutputStoreCount    = MT * NT
//       computeCount = synchronizationCount = MT * NT * KT
//
//   Output Stationary (for mTile, for nTile, { for kTile { LoadInputTile;
//     LoadWeightTile; ComputeTile; Synchronize }; StoreFinalOutput }):
//       inputLoadCount = weightLoadCount = MT * NT * KT
//       partialOutputReloadCount = partialOutputStoreCount = 0
//       finalOutputStoreCount = MT * NT
//       computeCount = synchronizationCount = MT * NT * KT
//
// Every dataflow therefore emits exactly one Synchronize after every
// ComputeTile (Section 8) -- no additional barriers for DMA loads/stores
// are introduced; concrete asynchronous DMA dependencies and double
// buffering remain explicitly out of scope for this slice.
//
// ---------------------------------------------------------------------
// Placeholder tile-coordinate convention for loop-hoisted operations
// ---------------------------------------------------------------------
// Some operations are emitted OUTSIDE the loop over one of the three
// axes (e.g. Weight Stationary's LoadWeightTile precedes its mTile loop;
// Input Stationary's LoadInputTile precedes its nTile loop), yet
// ScheduleOperation's TileCoordinate always carries all three indices.
// For these operations, the axis they do not yet range over is recorded
// as 0 by convention (WS's LoadWeightTile: mTile=0; IS's LoadInputTile:
// nTile=0) -- this reflects "emitted before/shared across that axis's
// iteration", not a claim that the operation pertains to index 0
// specifically; consumers needing the real residency semantics should
// read BufferResidencyPlan, not this coordinate's placeholder axis.
// Output Stationary's StoreFinalOutput (emitted after its kTile loop) is
// tagged kTile = KT-1 -- this IS semantically meaningful (the final
// reduction tile), not a placeholder.
//
// ---------------------------------------------------------------------
// Valid vs. physical extents (Section 4, 12) -- conservative padded model
// ---------------------------------------------------------------------
//   physical extent: the allocated and TRANSFERRED padded tile (always
//                     tileM / tileN / tileK, per candidate)
//   valid extent:     the correctness mask / logical work actually
//                     needed at this tile coordinate =
//                       min(tileDim, logicalDim - checked(tileIndex * tileDim))
//   isBoundaryTile:   validM < physicalM || validN < physicalN ||
//                     validK < physicalK
// Byte counts (Section 7) always use the PHYSICAL extent -- this slice
// does NOT shrink DMA byte counts to valid boundary bytes, staying
// consistent with Slices 2/3/5's conservative full-physical-tile
// accounting.
//
// ---------------------------------------------------------------------
// Per-operation byte counts (Section 7) -- reused, never reinvented
// ---------------------------------------------------------------------
//   I = tileM * tileK * input element bytes   (== legality.inputTileBytes)
//   W = tileK * tileN * weight element bytes  (== legality.weightTileBytes)
//   O = tileM * tileN * accumulator bytes     (== legality.outputTileBytes)
//   LoadInputTile.bytes = I;  LoadWeightTile.bytes = W
//   ReloadPartialOutput.bytes = StorePartialOutput.bytes
//     = StoreFinalOutput.bytes = O
//   ComputeTile.bytes = Synchronize.bytes = 0
//
// ---------------------------------------------------------------------
// Determinism (Section 10)
// ---------------------------------------------------------------------
// operations[i].operationIndex == i for every i, in exact emission
// order (nested nested for-loops over plain integer indices, matching
// the dataflow-specific loop order above) -- no unordered_map/hash-table
// iteration or pointer-derived ordering is used anywhere in this module.

#include "costmodel/CandidateCostModel.h"
#include "costmodel/CandidateLegality.h"
#include "costmodel/KernelCandidate.h"
#include "costmodel/PlanSelection.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::costmodel {

enum class ScheduleLoopDimension { M, N, K };

llvm::StringRef toString(ScheduleLoopDimension dim);

enum class ScheduleOperationKind {
  LoadInputTile,
  LoadWeightTile,
  ReloadPartialOutput,
  ComputeTile,
  StorePartialOutput,
  StoreFinalOutput,
  Synchronize,
};

llvm::StringRef toString(ScheduleOperationKind kind);

struct TileCoordinate {
  std::uint64_t mTile = 0;
  std::uint64_t nTile = 0;
  std::uint64_t kTile = 0;
};

struct TileExtent {
  std::uint64_t physicalM = 0;
  std::uint64_t physicalN = 0;
  std::uint64_t physicalK = 0;

  std::uint64_t validM = 0;
  std::uint64_t validN = 0;
  std::uint64_t validK = 0;
};

struct ScheduleOperation {
  std::uint64_t operationIndex = 0;
  ScheduleOperationKind kind = ScheduleOperationKind::ComputeTile;

  TileCoordinate coordinate;
  TileExtent extent;

  std::uint64_t bytes = 0;
  bool isBoundaryTile = false;
  bool isFinalReductionTile = false;

  bool operator==(const ScheduleOperation &other) const {
    return operationIndex == other.operationIndex && kind == other.kind &&
           coordinate.mTile == other.coordinate.mTile &&
           coordinate.nTile == other.coordinate.nTile &&
           coordinate.kTile == other.coordinate.kTile &&
           extent.physicalM == other.extent.physicalM &&
           extent.physicalN == other.extent.physicalN &&
           extent.physicalK == other.extent.physicalK &&
           extent.validM == other.extent.validM && extent.validN == other.extent.validN &&
           extent.validK == other.extent.validK && bytes == other.bytes &&
           isBoundaryTile == other.isBoundaryTile &&
           isFinalReductionTile == other.isFinalReductionTile;
  }
};

struct BufferResidencyPlan {
  bool inputResidentAcrossN = false;
  bool weightResidentAcrossM = false;
  bool outputResidentAcrossK = false;

  bool operator==(const BufferResidencyPlan &other) const {
    return inputResidentAcrossN == other.inputResidentAcrossN &&
           weightResidentAcrossM == other.weightResidentAcrossM &&
           outputResidentAcrossK == other.outputResidentAcrossK;
  }
};

struct SchedulePlan {
  std::string candidateId;

  Precision precision = Precision::INT8;
  Dataflow dataflow = Dataflow::WeightStationary;
  PEArrayShape peArray;
  TileShape tile;

  std::uint64_t logicalM = 0;
  std::uint64_t logicalN = 0;
  std::uint64_t logicalK = 0;

  std::uint64_t numMTiles = 0;
  std::uint64_t numNTiles = 0;
  std::uint64_t numKTiles = 0;

  std::vector<ScheduleLoopDimension> loopOrder;
  BufferResidencyPlan residency;

  std::vector<ScheduleOperation> operations;

  std::uint64_t inputLoadCount = 0;
  std::uint64_t weightLoadCount = 0;
  std::uint64_t partialOutputReloadCount = 0;
  std::uint64_t computeCount = 0;
  std::uint64_t partialOutputStoreCount = 0;
  std::uint64_t finalOutputStoreCount = 0;
  std::uint64_t synchronizationCount = 0;

  bool operator==(const SchedulePlan &other) const;
};

// Lower-level API: materializes the schedule for exactly one
// already-classified-legal candidate. Validates (fails closed via
// llvm::Error, never silently repairs) that: candidate/legality/cost
// candidateIds all match; legality.isLegal is true; problem's output
// shape is statically supported; candidate.tile.reductionDepth (and
// every other tile dimension) is explicitly positive; MT/NT/KT
// recomputed here from `problem` + `candidate.tile` exactly match
// cost.numMTiles/numNTiles/numReductionTiles. Internally reconciles the
// materialized operation counts/bytes against `cost` before returning
// (see validateScheduleAgainstCost) -- a schedule is never returned that
// fails its own reconciliation.
llvm::Expected<SchedulePlan>
materializeCandidateSchedule(const KernelCandidate &candidate,
                            const CandidateLegalityResult &legality,
                            const CandidateCostEstimate &cost,
                            const Conv2DProblemShape &problem,
                            const NPUTargetConfig &target);

// Higher-level API: materializes the schedule for a Slice 4
// PlanSelectionResult's selected candidate. Validates that
// `selection.selectedCandidate.candidateId == selection.selectedCandidateId`
// (the selected candidate matches the selection result itself), then
// recomputes the selected candidate's legality via the existing Slice 2
// API (checkCandidateLegality is a pure, deterministic function of
// (candidate, target), so this reproduces byte-identical results to
// whatever legality was used to build `selection` in the first place --
// no new legality logic is introduced), and delegates to
// materializeCandidateSchedule() above.
llvm::Expected<SchedulePlan> materializeSelectedSchedule(const PlanSelectionResult &selection,
                                                          const Conv2DProblemShape &problem,
                                                          const NPUTargetConfig &target);

// Structural reconciliation (Section 11) ONLY -- never a second cost
// model, never a cycle recomputation. Returns an llvm::Error describing
// the first mismatch found (checked in a fixed, documented order) if any
// of the following are not EXACTLY equal between `schedule` and `cost`:
// input/weight load counts, partial-output reload count, total output-
// write count (partialOutputStoreCount + finalOutputStoreCount vs
// cost.outputWriteCount), total DMA operation count, synchronization
// count (vs cost.numMTiles*numNTiles*numReductionTiles), off-chip
// input/weight/output-read/output-write/total bytes, and compute
// operation count (vs numMTiles*numNTiles*numReductionTiles).
llvm::Error validateScheduleAgainstCost(const SchedulePlan &schedule,
                                        const CandidateCostEstimate &cost);

// Optional stable diagnostic serializer (Section 13): deterministic JSON
// text with candidate_id, dataflow, loop_order, tile, tile_counts,
// residency, and operation_counts. Individual operations are NOT
// included (schedules may be large) -- this is a deterministic compact
// summary, not an executable plan; it never writes to disk.
std::string serializeSchedulePlanToJson(const SchedulePlan &plan);

} // namespace mlir::costmodel
