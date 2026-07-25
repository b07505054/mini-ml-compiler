#pragma once

// Cost Model Slice 2: Candidate Legality Checking.
// (Slice 5 update: K-tile-aware legality -- TileShape.reductionDepth is
// now a real, candidate-controlled physical reduction-tile depth
// (tileK), not always the problem's full reduction. This is a documented,
// intentional formula change from Slices 2-4's original always-full-K
// model; see "M / N / K semantics" below.)
//
// Continues from Slice 1 (costmodel/KernelCandidate.h) by adding an
// explicit target hardware contract (NPUTargetConfig) and a deterministic
// legality classifier for every KernelCandidate Slice 1 enumerates. This
// slice performs LEGALITY CHECKING ONLY: no estimated cycles, no ranking,
// no scoring, no sorting, no winner selection, no autotuning, no
// benchmarking, no execution. It remains fully independent from Hailo,
// HEF, runtime execution, lowering, and simulation -- NPUTargetConfig is a
// synthetic compiler-model target, not a real commercial NPU and not a
// model of Hailo hardware.
//
// ---------------------------------------------------------------------
// M / N / K semantics (Slice 5: tileK is now explicit and candidate-owned)
// ---------------------------------------------------------------------
//   N (output channels in the tile)     = tile.outputChannels
//   M (output spatial positions in the tile)
//                                       = tile.height * tile.width
//   tileK (physical reduction-tile depth) = tile.reductionDepth
//                                            (candidate-controlled, Slice 5)
//   problemK (the problem's FULL logical reduction) =
//       inputChannels * kernelHeight * kernelWidth   (unchanged formula;
//       Conv2D reduction semantics are not redefined by this slice)
//
// tileK is never 0 and never silently reinterpreted as "the full
// reduction" -- every candidate carries an explicit, positive tileK (see
// the TileShape doc comment in KernelCandidate.h). tileK may be smaller
// than, or exactly equal to, problemK; per this slice's policy (Section
// 2 of the task) it may never exceed problemK.
//
//   numReductionTiles = ceilDiv(problemK, tileK)
//   physicalReductionK = numReductionTiles * tileK   (>= problemK; the
//     final reduction tile may be physically padded -- exposed on
//     CandidateLegalityResult as `reductionK` (logical, = problemK),
//     `physicalReductionK`, and `numReductionTiles`)
//
// Storage bytes per element, by precision (documented, not implicit):
//   INT8 input/weight element  = 1 byte
//   FP16 input/weight element  = 2 bytes
//   INT8 accumulator element   = 4 bytes (INT32 accumulation)
//   FP16 accumulator element   = 4 bytes (FP32 accumulation)
// (the accumulator is 4 bytes either way, but the underlying accumulator
// *type* differs by precision -- INT32 vs FP32 -- which is why this is
// spelled out explicitly here rather than left as an implicit constant.)
//
// Tile footprint formulas (Slice 5: now driven by tileK, not problemK --
// still NOT cycle-accurate, NOT a layout-exact physical memory model,
// and NOT double-buffered):
//
//   input_tile_bytes  = M * tileK * input_element_bytes
//   weight_tile_bytes = tileK * N * weight_element_bytes
//   output_tile_bytes = M * N * accumulator_element_bytes  (UNCHANGED --
//     the output accumulator tile's legality footprint never depends on
//     tileK or numReductionTiles: it is one resident accumulator tile
//     when supported by the dataflow, not one per reduction tile. This
//     is the key modeling choice that gives Output Stationary its
//     defining legality property -- see "Dataflow-specific legality"
//     below.)
//   total_scratchpad_bytes = input_tile_bytes + weight_tile_bytes
//                            + output_tile_bytes
//
// ---------------------------------------------------------------------
// Boundary-tile policy (Slice 2's chosen, documented policy; unchanged in
// spirit for M/N, now also governs tileK per Section 5)
// ---------------------------------------------------------------------
// A tile's M/N dimensions MAY exceed the logical conv2d problem's
// remaining output extent (output_spatial, output_channels) -- boundary
// tiles are allowed and are NOT a legality violation on their own. The
// only hard M/N-shape-validity requirement is that every tile dimension
// is strictly positive (InvalidTileShape otherwise). Symmetrically for
// the reduction dimension (Slice 5): tileK MAY be smaller than problemK
// and need not divide it evenly -- the final reduction tile may be
// physically padded up to physicalReductionK -- but tileK must be
// strictly positive AND must not exceed problemK (InvalidReductionTile
// otherwise; see Section 5's policy). Legality is evaluated using the
// DECLARED PHYSICAL tile shape (M, N, and tileK), including whatever
// padding that implies -- so a boundary/padded tile consumes its full
// declared scratchpad footprint even though this slice models no
// performance/utilization penalty for that padding. Alignment rules are
// therefore applied to the PHYSICAL tile's N dimension and the PHYSICAL
// tileK, never to the logical problem's raw output_channels/problemK -- a
// logical extent that happens not to divide evenly is exactly what
// padding tiles exist to absorb, and is not, by itself, a rejection
// reason.
//
// ---------------------------------------------------------------------
// Reduction alignment (Slice 5, Section 5)
// ---------------------------------------------------------------------
// tileK is aligned using the SAME per-precision alignment value already
// declared on NPUTargetConfig for N (channelAlignmentInt8 /
// channelAlignmentFp16) -- this slice does not add a second target field
// for reduction alignment, since the task's own example values (INT8: 16,
// FP16: 8) are identical to the existing channel-alignment values, and
// "the requirements must come from NPUTargetConfig" is already satisfied
// by reusing them. The two checks are nonetheless reported as two
// DISTINCT reasons (ChannelAlignmentViolation for N,
// ReductionAlignmentViolation for tileK) rather than folded into one, per
// the task's explicit request for a precise, non-overloaded reason.
//
// ---------------------------------------------------------------------
// Dataflow-specific legality (Slice 2, Section 6 K-tiling update)
// ---------------------------------------------------------------------
// Section 6's "stationary operand" rule (Weight Stationary requires the
// complete tileK x N weight tile to fit weightBufferBytes, and weight
// residency is per N/K tile pair; Output Stationary requires the
// complete M x N accumulator tile to remain resident and fit
// outputBufferBytes ACROSS ALL numReductionTiles -- its capacity
// requirement is never multiplied by numReductionTiles, which is exactly
// what makes Output Stationary's legality independent of the reduction
// tile count; Input Stationary requires the complete M x tileK input
// tile to fit inputBufferBytes, and input residency is per M/K tile
// pair) is realized by the SAME per-buffer capacity checks this slice
// already runs unconditionally for every candidate (see "General
// per-buffer constraints" below) -- each dataflow's defining property
// (which operand must be fully resident) maps 1:1 onto exactly one of
// those three checks, and the output-tile-bytes formula above already
// guarantees the accumulator's footprint never scales with
// numReductionTiles for any dataflow. This slice does not add a second,
// redundant computation for this rule; a future slice that models
// double buffering or streaming residency for the *non*-stationary
// operands is the point where dataflow choice would start producing
// genuinely divergent legality outcomes for an identical tile shape --
// this slice does not invent that distinction (explicitly out of scope:
// "do not model double buffering", "do not invent implicit overlap or
// buffering assumptions"). These are analytical schedule abstractions,
// not a claim about any commercial NPU.
//
// ---------------------------------------------------------------------
// General per-buffer / total-scratchpad / alignment constraints
// ---------------------------------------------------------------------
// Applied identically to every candidate, independent of dataflow:
//   - input_tile_bytes  > target.inputBufferBytes   -> InputTileExceedsBuffer
//   - weight_tile_bytes > target.weightBufferBytes  -> WeightTileExceedsBuffer
//   - output_tile_bytes > target.outputBufferBytes  -> OutputTileExceedsBuffer
//   - total_scratchpad_bytes > target.scratchpadBytes
//                                                    -> TotalScratchpadExceeded
//   - channel (N) alignment: let A = (precision == INT8) ?
//     channelAlignmentInt8 : channelAlignmentFp16; violated if N % A != 0
//                                                    -> ChannelAlignmentViolation
//   - reduction (tileK) alignment (Slice 5): using the SAME A as above,
//     violated if tileK % A != 0             -> ReductionAlignmentViolation
//   - DMA alignment (this slice's explicit, documented rule -- the task
//     names the reason but does not fix an exact formula): each of the
//     three tile byte footprints (input/weight/output) must be a
//     multiple of target.dmaAlignmentBytes, modeling the requirement that
//     a DMA-transferred tile buffer starts/ends on an aligned boundary
//                                                    -> DMAAlignmentViolation
//
// ---------------------------------------------------------------------
// Overflow, invalid-reduction-tile, and dynamic-shape handling (fail
// closed, never silent)
// ---------------------------------------------------------------------
// All byte-footprint arithmetic uses llvm::MulOverflow/AddOverflow over
// int64_t. Any overflow in the byte-sum arithmetic makes the candidate
// illegal via the explicit ByteFootprintOverflow reason, with every
// byte-footprint field in the result reported as 0 (never a silently
// wrapped/garbage value). problemK itself is also computed via checked
// arithmetic; if THAT overflows, or if tileK <= 0, or if tileK >
// problemK, the candidate is illegal via InvalidReductionTile (evaluated
// before the byte-footprint block, which is skipped entirely in that
// case -- there is no sensible tileK to compute byte footprints from). A
// conv2d problem whose output tensor is not a statically-shaped rank-4
// tensor (see costmodel::extractConv2DProblemShape) is recorded on
// Conv2DProblemShape as outputShapeIsStaticAndSupported = false; every
// candidate for such a problem is illegal via the explicit
// DynamicOrUnsupportedShape reason, in addition to (not instead of)
// whatever the other, shape-independent checks find.

#include "costmodel/KernelCandidate.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::costmodel {

// Explicit target hardware contract. This is a synthetic compiler-model
// target for legality checking -- it makes no claim about Hailo or any
// real commercial NPU.
struct NPUTargetConfig {
  std::string targetId;

  std::size_t scratchpadBytes = 0;
  std::size_t weightBufferBytes = 0;
  std::size_t inputBufferBytes = 0;
  std::size_t outputBufferBytes = 0;

  int dmaAlignmentBytes = 1;
  int channelAlignmentInt8 = 1;
  int channelAlignmentFp16 = 1;

  bool supportsInt8 = false;
  bool supportsFp16 = false;

  std::vector<Dataflow> supportedDataflows;
  std::vector<PEArrayShape> supportedPEArrays;

  // -----------------------------------------------------------------
  // Slice 3 addition (additive): analytical cost-model parameters.
  // Extending NPUTargetConfig in place (rather than introducing a
  // second target type) keeps one target object flowing through both
  // checkCandidateLegality() and estimateCandidateCost() -- see
  // CandidateCostModel.h. Synthetic compiler-model values only; no
  // claim about Hailo or any real commercial NPU.
  // -----------------------------------------------------------------
  std::uint64_t clockHz = 0;

  std::uint64_t offChipBandwidthBytesPerSecond = 0;
  std::uint64_t localMemoryBandwidthBytesPerCycle = 0;

  std::uint64_t dmaSetupCycles = 0;
  std::uint64_t kernelLaunchCycles = 0;
  std::uint64_t tileSynchronizationCycles = 0;

  std::uint64_t int8MacsPerPEPerCycle = 0;
  std::uint64_t fp16MacsPerPEPerCycle = 0;

  bool supportsComputeDmaOverlap = false;

  // -----------------------------------------------------------------
  // Slice 8 addition (additive): number of logical DMA engines this
  // target exposes. Defaults to 1 so every pre-Slice-8 target/test is
  // unaffected. Slice 8's async scheduler always models exactly one
  // logical DMA engine regardless of a value > 1 (Section 17) -- a
  // future slice may add concurrent-channel modeling; 0 is illegal
  // (LegalityReason::UnsupportedAsyncSchedule / a hard error from
  // materializeAsyncSchedule).
  // -----------------------------------------------------------------
  std::uint32_t dmaEngineCount = 1;
};

// Synthetic reference target used by this slice's tests:
//   target_id: generic_npu_v1
//   scratchpad: 256 KiB, weight buffer: 128 KiB,
//   input buffer: 64 KiB, output buffer: 64 KiB
//   DMA alignment: 16 bytes; INT8 channel alignment: 16; FP16: 8
//   supported dataflows: WS, OS, IS; supported PE arrays: 16x16, 32x32
//   clock: 1 GHz; off-chip bandwidth: 10 GB/s; local memory: 32 bytes/cycle
//   DMA setup: 50 cycles; kernel launch: 200 cycles; tile sync: 10 cycles
//   INT8: 4 MACs/PE/cycle; FP16: 1 MAC/PE/cycle; compute/DMA overlap: true
// A compiler-model target only -- not Hailo, not a real commercial NPU.
const NPUTargetConfig &genericNPUv1Target();

// Deterministic rejection reasons. Declaration order below is also the
// fixed, documented order in which checkCandidateLegality() evaluates and
// appends reasons -- see CandidateLegalityResult::reasons.
enum class LegalityReason {
  UnsupportedPrecision,
  UnsupportedDataflow,
  UnsupportedPEArray,
  InvalidTileShape,
  InvalidReductionTile, // Slice 5: tileK <= 0, tileK > problemK, or problemK overflowed
  DynamicOrUnsupportedShape,
  ByteFootprintOverflow,
  InputTileExceedsBuffer,
  WeightTileExceedsBuffer,
  OutputTileExceedsBuffer,
  TotalScratchpadExceeded,
  ChannelAlignmentViolation,
  ReductionAlignmentViolation, // Slice 5: tileK violates precision alignment
  DMAAlignmentViolation,

  // -----------------------------------------------------------------
  // Slice 8 additions (additive): buffering-policy legality reasons.
  // See AsyncDMASchedule.h's checkBufferingLegality -- these are never
  // produced by Slice 2's checkCandidateLegality() above, only by
  // Slice 8's own classifier.
  // -----------------------------------------------------------------
  InputDoubleBufferExceedsBuffer,
  WeightDoubleBufferExceedsBuffer,
  OutputDoubleBufferExceedsBuffer,
  AsyncAggregateScratchpadExceeded,
  UnsupportedBufferingPolicy,
  UnsupportedAsyncSchedule,
};

// Stable string form for diagnostics/tests only -- not the primary
// internal representation (that is the enum above).
llvm::StringRef toString(LegalityReason reason);

struct CandidateLegalityResult {
  // Slice 3 addition (additive): mirrors the candidate this result was
  // computed for, so a downstream consumer (Slice 3's cost estimator)
  // can detect a mismatched/misaligned (candidate, legality) pairing
  // instead of silently trusting positional correspondence alone.
  std::string candidateId;

  bool isLegal = false;
  std::vector<LegalityReason> reasons;

  std::size_t inputTileBytes = 0;
  std::size_t weightTileBytes = 0;
  std::size_t outputTileBytes = 0;
  std::size_t totalScratchpadBytes = 0;

  // Slice 5 additions: only populated (nonzero) when tileK was valid
  // (InvalidReductionTile not among `reasons`) -- see the doc comment
  // above for the exact formulas.
  std::uint64_t reductionK = 0;         // logical, = problemK
  std::uint64_t physicalReductionK = 0; // numReductionTiles * tileK
  std::uint64_t numReductionTiles = 0;

  bool operator==(const CandidateLegalityResult &other) const {
    return candidateId == other.candidateId && isLegal == other.isLegal &&
           reasons == other.reasons &&
           inputTileBytes == other.inputTileBytes &&
           weightTileBytes == other.weightTileBytes &&
           outputTileBytes == other.outputTileBytes &&
           totalScratchpadBytes == other.totalScratchpadBytes &&
           reductionK == other.reductionK &&
           physicalReductionK == other.physicalReductionK &&
           numReductionTiles == other.numReductionTiles;
  }
};

// Classifies exactly one candidate against exactly one target. Pure
// function of its two inputs: deterministic, never mutates `candidate`,
// never scores/ranks/selects. Returns every applicable rejection reason
// (not merely the first), in the fixed order documented on
// LegalityReason above.
CandidateLegalityResult checkCandidateLegality(const KernelCandidate &candidate,
                                                const NPUTargetConfig &target);

// Convenience batch form. Output[i] corresponds to candidates[i] --
// original order is always preserved; this is not a filter.
std::vector<CandidateLegalityResult>
checkCandidateLegality(llvm::ArrayRef<KernelCandidate> candidates,
                        const NPUTargetConfig &target);

// Convenience partition into legal/illegal candidate indices, preserving
// each subsequence's original relative order. Provided so callers/tests
// can observe both groups without losing the ability to look at every
// original candidate and its result (see checkCandidateLegality above).
struct LegalityPartition {
  std::vector<std::size_t> legalIndices;
  std::vector<std::size_t> illegalIndices;
};
LegalityPartition partitionByLegality(llvm::ArrayRef<CandidateLegalityResult> results);

} // namespace mlir::costmodel
