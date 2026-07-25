#pragma once

// Cost Model Slice 3: Analytical Cost Estimation.
// (Slice 5 update: K-tile-aware cost estimation -- tileK is now a real,
// candidate-controlled physical reduction-tile depth, not always the
// problem's full reduction. This is a documented, intentional formula
// change from Slice 3/4's original always-full-K model.)
//
// Continues from Slice 1 (KernelCandidate.h, candidate generation) and
// Slice 2 (CandidateLegality.h, legality classification against
// NPUTargetConfig) by adding a deterministic, fully explainable
// analytical cost breakdown for every LEGAL KernelCandidate. This slice
// performs COST ESTIMATION ONLY: no ranking, no sorting, no winner
// selection, no `selected` field, no autotuning, no kernel execution, no
// real-hardware calibration, and it never modifies a legality decision.
// It remains fully independent from Hailo, HEF, runtime execution,
// lowering, simulator execution, and real-hardware benchmarking --
// NPUTargetConfig's Slice 3 fields are synthetic compiler-model
// parameters, not a claim about Hailo or any real commercial NPU.
//
// ---------------------------------------------------------------------
// M / N / K (Slice 5: tileK is explicit and candidate-owned)
// ---------------------------------------------------------------------
//   tileM = tile.height * tile.width      (output spatial positions)
//   tileN = tile.outputChannels           (output channels)
//   tileK = tile.reductionDepth           (physical reduction-tile depth,
//                                          candidate-controlled, Slice 5)
//   problemK = inputChannels * kernelHeight * kernelWidth (the problem's
//              FULL logical reduction -- Conv2D reduction semantics are
//              not redefined by this slice)
//
// ---------------------------------------------------------------------
// Logical vs. physical work (Section 4, 5, 7)
// ---------------------------------------------------------------------
//   logicalOutputSpatial  = batch * outputHeight * outputWidth
//   logicalOutputChannels = outputChannels
//   reductionK (logical K) = problemK
//   logicalMacs           = logicalOutputSpatial * logicalOutputChannels
//                            * reductionK              (UNCHANGED formula:
//                            logical MACs never include K padding, since
//                            they are defined against the LOGICAL,
//                            untiled reduction, regardless of tileK)
//   logicalOutputElements = logicalOutputSpatial * logicalOutputChannels
//
//   numMTiles  = ceilDiv(logicalOutputSpatial, tileM)
//   numNTiles  = ceilDiv(logicalOutputChannels, tileN)
//   numReductionTiles (KT) = ceilDiv(reductionK, tileK)
//   totalTiles = numMTiles * numNTiles
//   physicalM  = numMTiles * tileM
//   physicalN  = numNTiles * tileN
//   physicalReductionK = numReductionTiles * tileK   (>= reductionK)
//   physicalMacs = physicalM * physicalN * physicalReductionK
//   physicalOutputElements = physicalM * physicalN
//
//   Padding is exposed as three fields, split by source (no subjective
//   padding-penalty coefficient is applied anywhere -- the cost
//   difference emerges purely from the extra physical work itself):
//     spatialPaddingMacs   = (physicalM * physicalN * reductionK) - logicalMacs
//                            (the M/N-tile-boundary padding alone, holding
//                            K at its logical value)
//     reductionPaddingMacs = physicalMacs - (physicalM * physicalN * reductionK)
//                            (the K-tile-boundary padding alone, on top of
//                            the already-padded M/N extents)
//     paddingMacs          = spatialPaddingMacs + reductionPaddingMacs
//                            (== physicalMacs - logicalMacs exactly)
//
// ---------------------------------------------------------------------
// PE-array utilization and wave model (Section 6) -- UNCHANGED by K
// ---------------------------------------------------------------------
// K-tiling does not change the PE array's spatial (M/N) mapping at all:
//   activeRows = min(tileM, peRows);  activeCols = min(tileN, peCols)
//   activePELanes = activeRows * activeCols
//   totalPELanes  = peRows * peCols
//   mWaves = ceilDiv(tileM, peRows);  nWaves = ceilDiv(tileN, peCols)
//   arrayWavesPerTile = mWaves * nWaves
//
//   peCapacityMacs = computeCycles * totalPELanes * macsPerPEPerCycle
//   peUtilization  = logicalMacs / peCapacityMacs
//
//   This remains guaranteed to land in (0, 1] with the Section 7 compute-
//   cycle formula below: arrayWavesPerTile * totalPELanes >= tileM *
//   tileN (unchanged), and numReductionTiles * reductionCyclesPerKTile *
//   macsPerPEPerCycle >= numReductionTiles * tileK = physicalReductionK
//   (by the ceiling definition of reductionCyclesPerKTile), so
//   peCapacityMacs is always >= physicalMacs, which is itself always >=
//   logicalMacs. No silent clamping is ever applied.
//
// ---------------------------------------------------------------------
// Compute cycles (Section 8) -- now accounts for all K tiles
// ---------------------------------------------------------------------
//   macsPerPEPerCycle = (precision == INT8) ? target.int8MacsPerPEPerCycle
//                                            : target.fp16MacsPerPEPerCycle
//   reductionCyclesPerKTile = ceilDiv(tileK, macsPerPEPerCycle)
//   computeCyclesPerMNTile  = arrayWavesPerTile * numReductionTiles
//                             * reductionCyclesPerKTile
//   computeCycles           = totalTiles * computeCyclesPerMNTile
//
//   A smaller tileK never reduces total MAC/compute work below what the
//   full reduction requires (numReductionTiles grows inversely with
//   tileK, so numReductionTiles * reductionCyclesPerKTile only grows or
//   holds steady as tileK shrinks -- see peUtilization's invariant
//   above); it can only add K-boundary padding (when tileK does not
//   evenly divide reductionK) on top. No unexplained coefficient is
//   introduced anywhere in this formula.
//
// ---------------------------------------------------------------------
// Dataflow-specific off-chip traffic (Section 9) -- exact per-dataflow
// tables, now K-tile-aware
// ---------------------------------------------------------------------
// Per-tile byte sizes (I, W, O below) are reused directly from the
// CandidateLegalityResult already computed by Slice 2
// (inputTileBytes/weightTileBytes/outputTileBytes, all now tileK-driven)
// -- this module does not recompute them. Let MT=numMTiles, NT=numNTiles,
// KT=numReductionTiles. Boundary tiles transfer their full PHYSICAL
// padded tile bytes, consistent with Slice 2's conservative accounting.
//
//   Weight Stationary:  inputLoadCount = weightLoadCount(*) = MT*NT*KT ; NT*KT
//   Output Stationary:  inputLoadCount = weightLoadCount    = MT*NT*KT
//   Input Stationary:   inputLoadCount = MT*KT ;  weightLoadCount = MT*NT*KT
//   (*) i.e. WS: inputLoadCount = MT*NT*KT, weightLoadCount = NT*KT
//
//   offChipInputBytes  = inputLoadCount  * legality.inputTileBytes
//   offChipWeightBytes = weightLoadCount * legality.weightTileBytes
//
// ---------------------------------------------------------------------
// Non-output-stationary partial-sum spill/reload (Section 10) -- this
// slice's chosen, documented policy for output traffic
// ---------------------------------------------------------------------
// With KT possibly > 1, WS and IS do NOT keep an accumulator resident
// across reduction tiles in this model (only Output Stationary's whole
// legality/cost definition is built around accumulator residency -- see
// CandidateLegality.h). For WS and IS, every intermediate reduction tile
// spills its partial output and reloads it for the next reduction tile;
// the final reduction tile reloads and writes the final result:
//
//   WS / IS:  outputWriteCount = MT*NT*KT     outputReadCount = MT*NT*(KT-1)
//   OS:       outputWriteCount = MT*NT        outputReadCount = 0
//
//   offChipOutputWriteBytes = outputWriteCount * legality.outputTileBytes
//   offChipOutputReadBytes  = outputReadCount  * legality.outputTileBytes
//   totalOffChipBytes = offChipInputBytes + offChipWeightBytes
//                        + offChipOutputReadBytes + offChipOutputWriteBytes
//
// This models the spill/reload as full OFF-CHIP DMA traffic -- the most
// conservative assumption, and the one that gives Output Stationary its
// full modeled advantage when K is tiled (a later slice could instead
// model an intermediate on-chip-only spill tier). When KT == 1, every
// dataflow's outputReadCount is exactly 0 and outputWriteCount is exactly
// MT*NT -- i.e. this formula degenerates EXACTLY to Slice 3/4's original
// single-K-tile behavior; OS receives no artificial advantage in that
// case, because there is nothing for it to avoid yet.
//
// ---------------------------------------------------------------------
// DMA-cycle model (Section 9 continued) -- unchanged shape, new counts
// ---------------------------------------------------------------------
//   dmaTransferCycles = ceil(totalOffChipBytes * clockHz
//                             / offChipBandwidthBytesPerSecond)
//     (128-bit intermediate, range-checked back into uint64_t)
//   dmaOperationCount = inputLoadCount + weightLoadCount + outputReadCount
//                        + outputWriteCount
//   dmaSetupCycles    = dmaOperationCount * target.dmaSetupCycles
//   dmaCycles         = dmaTransferCycles + dmaSetupCycles
//
//   Smaller tileK increases KT, which increases dmaOperationCount (more,
//   smaller transfers) -- this naturally increases dmaSetupCycles with no
//   subjective "small K penalty" coefficient.
//
// ---------------------------------------------------------------------
// Local-memory traffic (Section 11) -- hierarchy model, explicit
// ---------------------------------------------------------------------
// This model treats local memory as a MANDATORY intermediate hierarchy
// level between off-chip DRAM and the PE array (DRAM <-> local buffer <->
// PE) for every traffic category -- not just accumulator spill. Every
// byte that crosses off-chip DMA is therefore counted AGAIN as local
// buffer<->PE traffic: this is an intentional, documented modeling
// choice (not double-counting by oversight), because these represent two
// distinct physical hierarchy levels, each with its own bandwidth/cycle
// cost in this model (this is the same philosophy Slice 3 originally
// established for WS's weight reuse and IS's input reuse; Slice 5 simply
// carries it through the new K-tiled counts and the new accumulator
// read/write split):
//
//   localInputBytes  = inputLoadCount  * legality.inputTileBytes  (same
//                       count as the off-chip model above, for every
//                       dataflow)
//   localWeightBytes = weightLoadCount * legality.weightTileBytes (ditto)
//   localAccumReadBytes  = outputReadCount  * legality.outputTileBytes
//   localAccumWriteBytes = outputWriteCount * legality.outputTileBytes
//   totalLocalBytes = localInputBytes + localWeightBytes
//                      + localAccumReadBytes + localAccumWriteBytes
//   localMemoryCycles = ceilDiv(totalLocalBytes,
//                                 target.localMemoryBandwidthBytesPerCycle)
//
// This is not claimed to be cycle-accurate.
//
// ---------------------------------------------------------------------
// Setup / synchronization overhead (Section 12) -- K-tile progression
// ---------------------------------------------------------------------
//   setupCycles           = target.kernelLaunchCycles          (unchanged)
//   synchronizationCycles = totalTiles * numReductionTiles
//                            * target.tileSynchronizationCycles
//     (MT*NT*KT synchronization events -- Slice 5 update: previously
//     totalTiles*tileSynchronizationCycles when KT was implicitly 1)
//   (no invented branch/cache/pipeline-stall penalties of any kind; a
//   smaller tileK naturally increases this via a larger KT, never via a
//   subjective "small K penalty")
//
// ---------------------------------------------------------------------
// Compute/DMA overlap policy (Section 13) -- unchanged
// ---------------------------------------------------------------------
// Overlap remains a TARGET-level analytical assumption -- there is still
// no per-candidate double-buffering flag; K-tiling changes traffic and
// event COUNTS, but no concrete load/compute/store pipeline schedule has
// been materialized by this or any prior slice:
//
//   if !target.supportsComputeDmaOverlap:
//     totalEstimatedCycles = computeCycles + dmaTransferCycles
//                             + dmaSetupCycles + localMemoryCycles
//                             + setupCycles + synchronizationCycles
//   else (conservative overlap: only the compute/DMA-transfer core
//         pipeline is assumed to overlap):
//     corePipelineCycles = max(computeCycles, dmaTransferCycles)
//     totalEstimatedCycles = corePipelineCycles + dmaSetupCycles
//                             + localMemoryCycles + setupCycles
//                             + synchronizationCycles
//
//   `computeDmaOverlapApplied` on the result records which branch ran.
//
// ---------------------------------------------------------------------
// Overflow and validation (fail closed, never silent)
// ---------------------------------------------------------------------
// All integer arithmetic is checked; any overflow anywhere produces an
// llvm::Error instead of a wrapped value. estimateCandidateCost() fails
// closed (returns an llvm::Error, does not estimate) when: the supplied
// legality result is not legal; the supplied legality's candidateId does
// not match the candidate; the conv2d problem's output shape is not
// statically supported (checked independently of what `legality` claims,
// as defense in depth); the target's clock/bandwidth/relevant-precision-
// throughput/local-memory-bandwidth is zero; the candidate's PE array or
// tile (including tileK) has a non-positive dimension.

#include "costmodel/CandidateLegality.h"
#include "costmodel/KernelCandidate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::costmodel {

struct CandidateCostEstimate {
  std::string candidateId;

  // ---- logical vs. physical work (Section 4-5, 7) ----
  std::uint64_t logicalOutputSpatial = 0;
  std::uint64_t logicalOutputChannels = 0;
  std::uint64_t reductionK = 0;         // logical K (= problemK)
  std::uint64_t physicalReductionK = 0; // numReductionTiles * tileK
  std::uint64_t numReductionTiles = 0;
  std::uint64_t logicalMacs = 0;
  std::uint64_t logicalOutputElements = 0;

  std::uint64_t numMTiles = 0;
  std::uint64_t numNTiles = 0;
  std::uint64_t totalTiles = 0;
  std::uint64_t physicalMacs = 0;
  std::uint64_t physicalOutputElements = 0;
  std::uint64_t paddingMacs = 0;          // total = spatial + reduction
  std::uint64_t spatialPaddingMacs = 0;
  std::uint64_t reductionPaddingMacs = 0;

  // ---- PE-array utilization (Section 6, unchanged by K) ----
  std::uint64_t activePELanes = 0;
  std::uint64_t totalPELanes = 0;
  std::uint64_t arrayWavesPerTile = 0;
  double peUtilization = 0.0;

  // ---- compute cycles (Section 8) ----
  std::uint64_t computeCycles = 0;

  // ---- off-chip traffic (Section 9-10) ----
  std::uint64_t inputLoadCount = 0;
  std::uint64_t weightLoadCount = 0;
  std::uint64_t outputWriteCount = 0;
  std::uint64_t outputReadCount = 0; // partial-sum reload count (0 for OS)
  std::uint64_t offChipInputBytes = 0;
  std::uint64_t offChipWeightBytes = 0;
  std::uint64_t offChipOutputWriteBytes = 0;
  std::uint64_t offChipOutputReadBytes = 0;
  std::uint64_t totalOffChipBytes = 0;

  // ---- DMA cycles (Section 9 continued) ----
  std::uint64_t dmaTransferCycles = 0;
  std::uint64_t dmaSetupCycles = 0;
  std::uint64_t dmaCycles = 0; // == dmaTransferCycles + dmaSetupCycles

  // ---- local-memory traffic (Section 11) ----
  std::uint64_t localInputBytes = 0;
  std::uint64_t localWeightBytes = 0;
  std::uint64_t localAccumReadBytes = 0;
  std::uint64_t localAccumWriteBytes = 0;
  std::uint64_t totalLocalBytes = 0;
  std::uint64_t localMemoryCycles = 0;

  // ---- setup / synchronization (Section 12) ----
  std::uint64_t setupCycles = 0;
  std::uint64_t synchronizationCycles = 0;

  // ---- overlap policy + final total (Section 13) ----
  bool computeDmaOverlapApplied = false;
  std::uint64_t totalEstimatedCycles = 0;

  bool operator==(const CandidateCostEstimate &other) const;
};

// Estimates the analytical cost of exactly one candidate, already
// classified legal by Slice 2. Pure function of its three inputs:
// deterministic, never mutates `candidate`, never ranks/sorts/selects.
// Fails closed (returns an llvm::Error) rather than estimating when the
// candidate is not legal, `legality` does not correspond to `candidate`,
// the conv2d output shape is not statically supported, the target is
// invalid (zero clock/bandwidth/throughput/local-memory-bandwidth), or
// the candidate's PE array or tile shape (including tileK) has a
// non-positive dimension. Never wraps on overflow.
llvm::Expected<CandidateCostEstimate>
estimateCandidateCost(const KernelCandidate &candidate,
                      const CandidateLegalityResult &legality,
                      const NPUTargetConfig &target);

// Batch form. Preserves the original candidate order and emits an
// estimate ONLY for candidates whose corresponding legality result is
// legal -- illegal candidates are skipped (not estimated, not present in
// the output vector; this is a documented consequence of the
// already-established Slice 2 legality classification, not a new
// ranking/filtering decision made by this slice). Returns an error
// (instead of a partial vector) if `candidates.size() !=
// legalityResults.size()`, or if any `legalityResults[i].candidateId !=
// candidates[i].candidateId` -- a legality result is never silently
// associated with the wrong candidate.
llvm::Expected<std::vector<CandidateCostEstimate>>
estimateCandidateCosts(llvm::ArrayRef<KernelCandidate> candidates,
                        llvm::ArrayRef<CandidateLegalityResult> legalityResults,
                        const NPUTargetConfig &target);

} // namespace mlir::costmodel
