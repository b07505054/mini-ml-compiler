#include "costmodel/CandidateLegality.h"

#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::costmodel {

const NPUTargetConfig &genericNPUv1Target() {
  static const NPUTargetConfig kTarget = [] {
    NPUTargetConfig target;
    target.targetId = "generic_npu_v1";
    target.scratchpadBytes = 256 * 1024;
    target.weightBufferBytes = 128 * 1024;
    target.inputBufferBytes = 64 * 1024;
    target.outputBufferBytes = 64 * 1024;
    target.dmaAlignmentBytes = 16;
    target.channelAlignmentInt8 = 16;
    target.channelAlignmentFp16 = 8;
    target.supportsInt8 = true;
    target.supportsFp16 = true;
    target.supportedDataflows = {Dataflow::WeightStationary,
                                  Dataflow::OutputStationary,
                                  Dataflow::InputStationary};
    target.supportedPEArrays = {PEArrayShape{16, 16}, PEArrayShape{32, 32}};

    // Slice 3 analytical cost-model parameters (synthetic; see the doc
    // comment on genericNPUv1Target() in CandidateLegality.h).
    target.clockHz = 1'000'000'000ull;                        // 1 GHz
    target.offChipBandwidthBytesPerSecond = 10'000'000'000ull; // 10 GB/s
    target.localMemoryBandwidthBytesPerCycle = 32;
    target.dmaSetupCycles = 50;
    target.kernelLaunchCycles = 200;
    target.tileSynchronizationCycles = 10;
    target.int8MacsPerPEPerCycle = 4;
    target.fp16MacsPerPEPerCycle = 1;
    target.supportsComputeDmaOverlap = true;
    return target;
  }();
  return kTarget;
}

llvm::StringRef toString(LegalityReason reason) {
  switch (reason) {
  case LegalityReason::UnsupportedPrecision:
    return "unsupported_precision";
  case LegalityReason::UnsupportedDataflow:
    return "unsupported_dataflow";
  case LegalityReason::UnsupportedPEArray:
    return "unsupported_pe_array";
  case LegalityReason::InvalidTileShape:
    return "invalid_tile_shape";
  case LegalityReason::InvalidReductionTile:
    return "invalid_reduction_tile";
  case LegalityReason::DynamicOrUnsupportedShape:
    return "dynamic_or_unsupported_shape";
  case LegalityReason::ByteFootprintOverflow:
    return "byte_footprint_overflow";
  case LegalityReason::InputTileExceedsBuffer:
    return "input_tile_exceeds_buffer";
  case LegalityReason::WeightTileExceedsBuffer:
    return "weight_tile_exceeds_buffer";
  case LegalityReason::OutputTileExceedsBuffer:
    return "output_tile_exceeds_buffer";
  case LegalityReason::TotalScratchpadExceeded:
    return "total_scratchpad_exceeded";
  case LegalityReason::ChannelAlignmentViolation:
    return "channel_alignment_violation";
  case LegalityReason::ReductionAlignmentViolation:
    return "reduction_alignment_violation";
  case LegalityReason::DMAAlignmentViolation:
    return "dma_alignment_violation";
  case LegalityReason::InputDoubleBufferExceedsBuffer:
    return "input_double_buffer_exceeds_buffer";
  case LegalityReason::WeightDoubleBufferExceedsBuffer:
    return "weight_double_buffer_exceeds_buffer";
  case LegalityReason::OutputDoubleBufferExceedsBuffer:
    return "output_double_buffer_exceeds_buffer";
  case LegalityReason::AsyncAggregateScratchpadExceeded:
    return "async_aggregate_scratchpad_exceeded";
  case LegalityReason::UnsupportedBufferingPolicy:
    return "unsupported_buffering_policy";
  case LegalityReason::UnsupportedAsyncSchedule:
    return "unsupported_async_schedule";
  }
  llvm_unreachable("unhandled LegalityReason");
}

static bool supportsDataflow(const NPUTargetConfig &target, Dataflow dataflow) {
  return std::find(target.supportedDataflows.begin(),
                    target.supportedDataflows.end(),
                    dataflow) != target.supportedDataflows.end();
}

static bool supportsPEArray(const NPUTargetConfig &target,
                             const PEArrayShape &peArray) {
  return std::find(target.supportedPEArrays.begin(),
                    target.supportedPEArrays.end(),
                    peArray) != target.supportedPEArrays.end();
}

// Returns true (and sets `bytes`) on success; returns false on overflow,
// leaving `bytes` at 0.
static bool computeCheckedProduct3(int64_t a, int64_t b, int64_t c,
                                    int64_t &bytes) {
  int64_t ab = 0;
  if (llvm::MulOverflow(a, b, ab))
    return false;
  int64_t abc = 0;
  if (llvm::MulOverflow(ab, c, abc))
    return false;
  bytes = abc;
  return true;
}

CandidateLegalityResult checkCandidateLegality(const KernelCandidate &candidate,
                                                const NPUTargetConfig &target) {
  CandidateLegalityResult result;
  result.candidateId = candidate.candidateId;
  const Conv2DProblemShape &problem = candidate.problem;
  const TileShape &tile = candidate.tile;

  // ---- precision / dataflow / PE-array hardware support ----
  bool precisionSupported = (candidate.precision == Precision::INT8)
                                 ? target.supportsInt8
                                 : target.supportsFp16;
  if (!precisionSupported)
    result.reasons.push_back(LegalityReason::UnsupportedPrecision);

  if (!supportsDataflow(target, candidate.dataflow))
    result.reasons.push_back(LegalityReason::UnsupportedDataflow);

  if (!supportsPEArray(target, candidate.peArray))
    result.reasons.push_back(LegalityReason::UnsupportedPEArray);

  // ---- M/N tile-shape validity (must be strictly positive) ----
  bool tileShapeValid =
      tile.outputChannels > 0 && tile.height > 0 && tile.width > 0;
  if (!tileShapeValid)
    result.reasons.push_back(LegalityReason::InvalidTileShape);

  // ---- Slice 5: reduction-tile (tileK) validity ----
  // problemK is computed here (not reused from any cached value) so this
  // check is self-contained; see the doc comment above for why an
  // overflow here folds into InvalidReductionTile rather than
  // ByteFootprintOverflow (kept for the later byte-sum arithmetic only).
  int64_t problemK = 0;
  bool problemKOverflow =
      llvm::MulOverflow(problem.inputChannels, problem.kernelHeight, problemK);
  if (!problemKOverflow)
    problemKOverflow = llvm::MulOverflow(problemK, problem.kernelWidth, problemK);

  int64_t tileK = tile.reductionDepth;
  bool reductionTileValid =
      !problemKOverflow && tileK > 0 && tileK <= problemK;
  if (!reductionTileValid)
    result.reasons.push_back(LegalityReason::InvalidReductionTile);

  // ---- dynamic/unsupported conv2d output shape: fail closed ----
  if (!problem.outputShapeIsStaticAndSupported)
    result.reasons.push_back(LegalityReason::DynamicOrUnsupportedShape);

  // ---- byte-footprint-dependent checks (only meaningful for a valid M/N
  // tile shape AND a valid tileK; skipped otherwise -- there is no
  // sensible M/N/K to compute a footprint from) ----
  if (tileShapeValid && reductionTileValid) {
    // M/N/tileK per this module's documented mapping (see CandidateLegality.h).
    int64_t m = 0;
    bool mOverflow = llvm::MulOverflow(tile.height, tile.width, m);
    int64_t n = tile.outputChannels;
    int64_t k = tileK; // physical reduction-tile depth, NOT problemK

    int64_t inputElementBytes = (candidate.precision == Precision::INT8) ? 1 : 2;
    int64_t weightElementBytes = inputElementBytes;
    // Accumulator is 4 bytes either way (INT32 for INT8, FP32 for FP16) --
    // see the doc comment in CandidateLegality.h for why this is spelled
    // out explicitly rather than left implicit.
    int64_t accumulatorElementBytes = 4;

    int64_t inputBytes = 0, weightBytes = 0, outputBytes = 0, totalBytes = 0;
    bool overflow = mOverflow;
    if (!overflow)
      overflow |= !computeCheckedProduct3(m, k, inputElementBytes, inputBytes);
    if (!overflow)
      overflow |= !computeCheckedProduct3(k, n, weightElementBytes, weightBytes);
    if (!overflow)
      // Output accumulator tile bytes NEVER depend on tileK/numReductionTiles
      // -- see the doc comment above ("one resident accumulator tile").
      overflow |=
          !computeCheckedProduct3(m, n, accumulatorElementBytes, outputBytes);
    if (!overflow) {
      int64_t inputPlusWeight = 0;
      overflow |= llvm::AddOverflow(inputBytes, weightBytes, inputPlusWeight);
      if (!overflow)
        overflow |= llvm::AddOverflow(inputPlusWeight, outputBytes, totalBytes);
    }

    if (overflow) {
      result.reasons.push_back(LegalityReason::ByteFootprintOverflow);
      // Byte fields stay at their 0 default -- never a wrapped value.
    } else {
      result.inputTileBytes = static_cast<std::size_t>(inputBytes);
      result.weightTileBytes = static_cast<std::size_t>(weightBytes);
      result.outputTileBytes = static_cast<std::size_t>(outputBytes);
      result.totalScratchpadBytes = static_cast<std::size_t>(totalBytes);

      std::int64_t numReductionTiles = problemK / tileK + (problemK % tileK != 0 ? 1 : 0);
      result.reductionK = static_cast<std::uint64_t>(problemK);
      result.numReductionTiles = static_cast<std::uint64_t>(numReductionTiles);
      result.physicalReductionK = static_cast<std::uint64_t>(numReductionTiles * tileK);

      if (result.inputTileBytes > target.inputBufferBytes)
        result.reasons.push_back(LegalityReason::InputTileExceedsBuffer);
      if (result.weightTileBytes > target.weightBufferBytes)
        result.reasons.push_back(LegalityReason::WeightTileExceedsBuffer);
      if (result.outputTileBytes > target.outputBufferBytes)
        result.reasons.push_back(LegalityReason::OutputTileExceedsBuffer);
      if (result.totalScratchpadBytes > target.scratchpadBytes)
        result.reasons.push_back(LegalityReason::TotalScratchpadExceeded);

      int channelAlignment = (candidate.precision == Precision::INT8)
                                  ? target.channelAlignmentInt8
                                  : target.channelAlignmentFp16;
      if (channelAlignment > 0 && (n % channelAlignment != 0))
        result.reasons.push_back(LegalityReason::ChannelAlignmentViolation);
      if (channelAlignment > 0 && (k % channelAlignment != 0))
        result.reasons.push_back(LegalityReason::ReductionAlignmentViolation);

      bool dmaViolated = false;
      if (target.dmaAlignmentBytes > 0) {
        auto notAligned = [&](std::size_t bytes) {
          return (bytes % static_cast<std::size_t>(target.dmaAlignmentBytes)) != 0;
        };
        dmaViolated = notAligned(result.inputTileBytes) ||
                      notAligned(result.weightTileBytes) ||
                      notAligned(result.outputTileBytes);
      }
      if (dmaViolated)
        result.reasons.push_back(LegalityReason::DMAAlignmentViolation);
    }
  }

  result.isLegal = result.reasons.empty();
  return result;
}

std::vector<CandidateLegalityResult>
checkCandidateLegality(llvm::ArrayRef<KernelCandidate> candidates,
                       const NPUTargetConfig &target) {
  std::vector<CandidateLegalityResult> results;
  results.reserve(candidates.size());
  for (const KernelCandidate &candidate : candidates)
    results.push_back(checkCandidateLegality(candidate, target));
  return results;
}

LegalityPartition
partitionByLegality(llvm::ArrayRef<CandidateLegalityResult> results) {
  LegalityPartition partition;
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (results[i].isLegal)
      partition.legalIndices.push_back(i);
    else
      partition.illegalIndices.push_back(i);
  }
  return partition;
}

} // namespace mlir::costmodel
