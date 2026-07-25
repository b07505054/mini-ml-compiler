#include "costmodel/CandidateCostModel.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>

namespace mlir::costmodel {

bool CandidateCostEstimate::operator==(const CandidateCostEstimate &other) const {
  return candidateId == other.candidateId &&
         logicalOutputSpatial == other.logicalOutputSpatial &&
         logicalOutputChannels == other.logicalOutputChannels &&
         reductionK == other.reductionK &&
         physicalReductionK == other.physicalReductionK &&
         numReductionTiles == other.numReductionTiles &&
         logicalMacs == other.logicalMacs &&
         logicalOutputElements == other.logicalOutputElements &&
         numMTiles == other.numMTiles && numNTiles == other.numNTiles &&
         totalTiles == other.totalTiles && physicalMacs == other.physicalMacs &&
         physicalOutputElements == other.physicalOutputElements &&
         paddingMacs == other.paddingMacs &&
         spatialPaddingMacs == other.spatialPaddingMacs &&
         reductionPaddingMacs == other.reductionPaddingMacs &&
         activePELanes == other.activePELanes &&
         totalPELanes == other.totalPELanes &&
         arrayWavesPerTile == other.arrayWavesPerTile &&
         peUtilization == other.peUtilization &&
         computeCycles == other.computeCycles &&
         inputLoadCount == other.inputLoadCount &&
         weightLoadCount == other.weightLoadCount &&
         outputWriteCount == other.outputWriteCount &&
         outputReadCount == other.outputReadCount &&
         offChipInputBytes == other.offChipInputBytes &&
         offChipWeightBytes == other.offChipWeightBytes &&
         offChipOutputWriteBytes == other.offChipOutputWriteBytes &&
         offChipOutputReadBytes == other.offChipOutputReadBytes &&
         totalOffChipBytes == other.totalOffChipBytes &&
         dmaTransferCycles == other.dmaTransferCycles &&
         dmaSetupCycles == other.dmaSetupCycles &&
         dmaCycles == other.dmaCycles &&
         localInputBytes == other.localInputBytes &&
         localWeightBytes == other.localWeightBytes &&
         localAccumReadBytes == other.localAccumReadBytes &&
         localAccumWriteBytes == other.localAccumWriteBytes &&
         totalLocalBytes == other.totalLocalBytes &&
         localMemoryCycles == other.localMemoryCycles &&
         setupCycles == other.setupCycles &&
         synchronizationCycles == other.synchronizationCycles &&
         computeDmaOverlapApplied == other.computeDmaOverlapApplied &&
         totalEstimatedCycles == other.totalEstimatedCycles;
}

namespace {

// Every integer computation in this file goes through this tracker so
// overflow is checked at every single step (never silently wrapped) but
// without threading llvm::Expected<uint64_t> through dozens of
// individual arithmetic call sites. `overflowed` is checked exactly once
// at the end of estimateCandidateCost().
struct CheckedArith {
  bool overflowed = false;

  std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
    std::uint64_t result = 0;
    if (__builtin_mul_overflow(a, b, &result)) {
      overflowed = true;
      return 0;
    }
    return result;
  }

  std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    std::uint64_t result = 0;
    if (__builtin_add_overflow(a, b, &result)) {
      overflowed = true;
      return 0;
    }
    return result;
  }

  std::uint64_t sub(std::uint64_t a, std::uint64_t b) {
    if (b > a) {
      overflowed = true;
      return 0;
    }
    return a - b;
  }

  // Overflow-free ceiling division (never computes a+b-1, which could
  // itself overflow near UINT64_MAX).
  std::uint64_t ceilDiv(std::uint64_t a, std::uint64_t b) {
    if (b == 0) {
      overflowed = true;
      return 0;
    }
    return a / b + (a % b != 0 ? 1 : 0);
  }
};

// dma cycles = ceil(totalBytes * clockHz / bandwidthBytesPerSecond),
// computed via a 128-bit intermediate so the multiplication cannot
// silently overflow 64 bits before the division runs (Section 9
// explicitly warns against a lossy `bandwidth / clockHz` truncation;
// this avoids the symmetric risk of the numerator overflowing instead).
// Returns false (and leaves `result` untouched) if the true 128-bit
// result does not fit back into a uint64_t.
bool checkedDmaTransferCycles(std::uint64_t totalBytes, std::uint64_t clockHz,
                               std::uint64_t bandwidthBytesPerSecond,
                               std::uint64_t &result) {
  if (bandwidthBytesPerSecond == 0)
    return false;
  unsigned __int128 numerator =
      static_cast<unsigned __int128>(totalBytes) * static_cast<unsigned __int128>(clockHz);
  unsigned __int128 quotient = numerator / bandwidthBytesPerSecond;
  unsigned __int128 remainder = numerator % bandwidthBytesPerSecond;
  unsigned __int128 ceiled = quotient + (remainder != 0 ? 1 : 0);
  if (ceiled > static_cast<unsigned __int128>(std::numeric_limits<std::uint64_t>::max()))
    return false;
  result = static_cast<std::uint64_t>(ceiled);
  return true;
}

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Error validateTargetForCostEstimation(const NPUTargetConfig &target,
                                             Precision precision) {
  if (target.clockHz == 0)
    return makeError("estimateCandidateCost: target.clockHz must be nonzero");
  if (target.offChipBandwidthBytesPerSecond == 0)
    return makeError(
        "estimateCandidateCost: target.offChipBandwidthBytesPerSecond must be nonzero");
  if (target.localMemoryBandwidthBytesPerCycle == 0)
    return makeError(
        "estimateCandidateCost: target.localMemoryBandwidthBytesPerCycle must be nonzero");
  std::uint64_t macsPerPE = (precision == Precision::INT8)
                                ? target.int8MacsPerPEPerCycle
                                : target.fp16MacsPerPEPerCycle;
  if (macsPerPE == 0)
    return makeError(
        "estimateCandidateCost: target's per-precision MACs-per-PE-per-cycle "
        "must be nonzero for the candidate's precision");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<CandidateCostEstimate>
estimateCandidateCost(const KernelCandidate &candidate,
                      const CandidateLegalityResult &legality,
                      const NPUTargetConfig &target) {
  // ---- fail-closed preconditions ----
  if (legality.candidateId != candidate.candidateId)
    return makeError(
        "estimateCandidateCost: legality.candidateId does not match candidate.candidateId "
        "(refusing to associate a legality result with the wrong candidate)");
  if (!legality.isLegal)
    return makeError("estimateCandidateCost: candidate '" + candidate.candidateId +
                      "' is not legal; illegal candidates are never estimated");
  // Defense in depth: independent of what `legality` claims, refuse to
  // estimate a candidate whose conv2d output shape was not statically
  // resolvable (see extractConv2DProblemShape).
  if (!candidate.problem.outputShapeIsStaticAndSupported)
    return makeError(
        "estimateCandidateCost: candidate's conv2d problem has a dynamic/unsupported "
        "output shape; cost estimation fails closed");

  const Conv2DProblemShape &problem = candidate.problem;
  if (problem.batch <= 0 || problem.outputHeight <= 0 || problem.outputWidth <= 0 ||
      problem.outputChannels <= 0 || problem.inputChannels <= 0 ||
      problem.kernelHeight <= 0 || problem.kernelWidth <= 0)
    return makeError(
        "estimateCandidateCost: conv2d problem has a non-positive logical extent");

  if (candidate.peArray.rows <= 0 || candidate.peArray.cols <= 0)
    return makeError("estimateCandidateCost: candidate's PE array has a non-positive dimension");

  if (candidate.tile.height <= 0 || candidate.tile.width <= 0 ||
      candidate.tile.outputChannels <= 0 || candidate.tile.reductionDepth <= 0)
    return makeError("estimateCandidateCost: candidate's tile shape has a non-positive "
                      "dimension (including reductionDepth/tileK)");

  if (llvm::Error err = validateTargetForCostEstimation(target, candidate.precision))
    return std::move(err);

  CheckedArith ck;

  // ---- Section 4: logical extents ----
  std::uint64_t batch = static_cast<std::uint64_t>(problem.batch);
  std::uint64_t outH = static_cast<std::uint64_t>(problem.outputHeight);
  std::uint64_t outW = static_cast<std::uint64_t>(problem.outputWidth);
  std::uint64_t outC = static_cast<std::uint64_t>(problem.outputChannels);
  std::uint64_t inC = static_cast<std::uint64_t>(problem.inputChannels);
  std::uint64_t kH = static_cast<std::uint64_t>(problem.kernelHeight);
  std::uint64_t kW = static_cast<std::uint64_t>(problem.kernelWidth);

  std::uint64_t logicalOutputSpatial = ck.mul(ck.mul(batch, outH), outW);
  std::uint64_t logicalOutputChannels = outC;
  std::uint64_t reductionK = ck.mul(ck.mul(inC, kH), kW); // logical K (problemK)
  std::uint64_t logicalMacs =
      ck.mul(ck.mul(logicalOutputSpatial, logicalOutputChannels), reductionK);
  std::uint64_t logicalOutputElements = ck.mul(logicalOutputSpatial, logicalOutputChannels);

  // ---- Section 5: tiling / padding (Slice 5: tileK is now explicit) ----
  std::uint64_t tileM =
      ck.mul(static_cast<std::uint64_t>(candidate.tile.height),
             static_cast<std::uint64_t>(candidate.tile.width));
  std::uint64_t tileN = static_cast<std::uint64_t>(candidate.tile.outputChannels);
  std::uint64_t tileK = static_cast<std::uint64_t>(candidate.tile.reductionDepth);

  std::uint64_t numMTiles = ck.ceilDiv(logicalOutputSpatial, tileM);
  std::uint64_t numNTiles = ck.ceilDiv(logicalOutputChannels, tileN);
  std::uint64_t numReductionTiles = ck.ceilDiv(reductionK, tileK);
  std::uint64_t totalTiles = ck.mul(numMTiles, numNTiles);

  std::uint64_t physicalM = ck.mul(numMTiles, tileM);
  std::uint64_t physicalN = ck.mul(numNTiles, tileN);
  std::uint64_t physicalReductionK = ck.mul(numReductionTiles, tileK);
  std::uint64_t physicalMacs = ck.mul(ck.mul(physicalM, physicalN), physicalReductionK);
  std::uint64_t physicalOutputElements = ck.mul(physicalM, physicalN);

  // Padding split by source (see CandidateCostModel.h): spatial (M/N)
  // padding alone, holding K at its logical value, then reduction (K)
  // padding on top of that.
  std::uint64_t macsAtLogicalK = ck.mul(ck.mul(physicalM, physicalN), reductionK);
  std::uint64_t spatialPaddingMacs = ck.sub(macsAtLogicalK, logicalMacs);
  std::uint64_t reductionPaddingMacs = ck.sub(physicalMacs, macsAtLogicalK);
  std::uint64_t paddingMacs = ck.sub(physicalMacs, logicalMacs);

  // ---- Section 6: PE-array utilization (unchanged by K) ----
  std::uint64_t peRows = static_cast<std::uint64_t>(candidate.peArray.rows);
  std::uint64_t peCols = static_cast<std::uint64_t>(candidate.peArray.cols);
  std::uint64_t activeRows = std::min(tileM, peRows);
  std::uint64_t activeCols = std::min(tileN, peCols);
  std::uint64_t activePELanes = ck.mul(activeRows, activeCols);
  std::uint64_t totalPELanes = ck.mul(peRows, peCols);
  std::uint64_t mWaves = ck.ceilDiv(tileM, peRows);
  std::uint64_t nWaves = ck.ceilDiv(tileN, peCols);
  std::uint64_t arrayWavesPerTile = ck.mul(mWaves, nWaves);

  // ---- Section 8: compute cycles (wave-based, now K-tile-aware) ----
  std::uint64_t macsPerPEPerCycle = (candidate.precision == Precision::INT8)
                                         ? target.int8MacsPerPEPerCycle
                                         : target.fp16MacsPerPEPerCycle;
  std::uint64_t reductionCyclesPerKTile = ck.ceilDiv(tileK, macsPerPEPerCycle);
  std::uint64_t computeCyclesPerMNTile =
      ck.mul(ck.mul(arrayWavesPerTile, numReductionTiles), reductionCyclesPerKTile);
  std::uint64_t computeCycles = ck.mul(totalTiles, computeCyclesPerMNTile);

  std::uint64_t peCapacityMacs = ck.mul(ck.mul(computeCycles, totalPELanes), macsPerPEPerCycle);
  double peUtilization = 0.0;
  if (!ck.overflowed) {
    if (peCapacityMacs == 0)
      return makeError("estimateCandidateCost: degenerate zero PE capacity");
    peUtilization = static_cast<double>(logicalMacs) / static_cast<double>(peCapacityMacs);
  }

  // ---- Section 9-10: dataflow-specific off-chip traffic (K-tile-aware) ----
  std::uint64_t inputLoadCount = 0, weightLoadCount = 0;
  std::uint64_t outputWriteCount = 0, outputReadCount = 0;
  switch (candidate.dataflow) {
  case Dataflow::WeightStationary:
    inputLoadCount = ck.mul(ck.mul(numMTiles, numNTiles), numReductionTiles);
    weightLoadCount = ck.mul(numNTiles, numReductionTiles);
    outputWriteCount = ck.mul(totalTiles, numReductionTiles);
    outputReadCount = ck.mul(totalTiles, ck.sub(numReductionTiles, 1));
    break;
  case Dataflow::OutputStationary:
    inputLoadCount = ck.mul(ck.mul(numMTiles, numNTiles), numReductionTiles);
    weightLoadCount = ck.mul(ck.mul(numMTiles, numNTiles), numReductionTiles);
    outputWriteCount = totalTiles;
    outputReadCount = 0;
    break;
  case Dataflow::InputStationary:
    inputLoadCount = ck.mul(numMTiles, numReductionTiles);
    weightLoadCount = ck.mul(ck.mul(numMTiles, numNTiles), numReductionTiles);
    outputWriteCount = ck.mul(totalTiles, numReductionTiles);
    outputReadCount = ck.mul(totalTiles, ck.sub(numReductionTiles, 1));
    break;
  }

  std::uint64_t offChipInputBytes =
      ck.mul(inputLoadCount, static_cast<std::uint64_t>(legality.inputTileBytes));
  std::uint64_t offChipWeightBytes =
      ck.mul(weightLoadCount, static_cast<std::uint64_t>(legality.weightTileBytes));
  std::uint64_t offChipOutputWriteBytes =
      ck.mul(outputWriteCount, static_cast<std::uint64_t>(legality.outputTileBytes));
  std::uint64_t offChipOutputReadBytes =
      ck.mul(outputReadCount, static_cast<std::uint64_t>(legality.outputTileBytes));
  std::uint64_t totalOffChipBytes = ck.add(
      ck.add(ck.add(offChipInputBytes, offChipWeightBytes), offChipOutputReadBytes),
      offChipOutputWriteBytes);

  // ---- Section 9 continued: DMA cycles ----
  std::uint64_t dmaTransferCycles = 0;
  if (!ck.overflowed &&
      !checkedDmaTransferCycles(totalOffChipBytes, target.clockHz,
                                 target.offChipBandwidthBytesPerSecond, dmaTransferCycles))
    ck.overflowed = true;
  std::uint64_t dmaOperationCount = ck.add(
      ck.add(ck.add(inputLoadCount, weightLoadCount), outputReadCount), outputWriteCount);
  std::uint64_t dmaSetupCycles = ck.mul(dmaOperationCount, target.dmaSetupCycles);
  std::uint64_t dmaCycles = ck.add(dmaTransferCycles, dmaSetupCycles);

  // ---- Section 11: local-memory traffic (hierarchy model) ----
  std::uint64_t localInputBytes =
      ck.mul(inputLoadCount, static_cast<std::uint64_t>(legality.inputTileBytes));
  std::uint64_t localWeightBytes =
      ck.mul(weightLoadCount, static_cast<std::uint64_t>(legality.weightTileBytes));
  std::uint64_t localAccumReadBytes =
      ck.mul(outputReadCount, static_cast<std::uint64_t>(legality.outputTileBytes));
  std::uint64_t localAccumWriteBytes =
      ck.mul(outputWriteCount, static_cast<std::uint64_t>(legality.outputTileBytes));
  std::uint64_t totalLocalBytes = ck.add(
      ck.add(ck.add(localInputBytes, localWeightBytes), localAccumReadBytes),
      localAccumWriteBytes);
  std::uint64_t localMemoryCycles =
      ck.ceilDiv(totalLocalBytes, target.localMemoryBandwidthBytesPerCycle);

  // ---- Section 12: setup / synchronization (K-tile progression) ----
  std::uint64_t setupCycles = target.kernelLaunchCycles;
  std::uint64_t synchronizationCycles =
      ck.mul(ck.mul(totalTiles, numReductionTiles), target.tileSynchronizationCycles);

  // ---- Section 13: compute/DMA overlap policy ----
  bool overlapApplied = target.supportsComputeDmaOverlap;
  std::uint64_t totalEstimatedCycles;
  if (overlapApplied) {
    std::uint64_t corePipelineCycles = std::max(computeCycles, dmaTransferCycles);
    totalEstimatedCycles = ck.add(
        ck.add(ck.add(ck.add(corePipelineCycles, dmaSetupCycles), localMemoryCycles), setupCycles),
        synchronizationCycles);
  } else {
    totalEstimatedCycles = ck.add(
        ck.add(ck.add(ck.add(ck.add(computeCycles, dmaTransferCycles), dmaSetupCycles),
                       localMemoryCycles),
               setupCycles),
        synchronizationCycles);
  }

  if (ck.overflowed)
    return makeError("estimateCandidateCost: integer overflow during cost estimation for '" +
                      candidate.candidateId + "'");

  CandidateCostEstimate estimate;
  estimate.candidateId = candidate.candidateId;
  estimate.logicalOutputSpatial = logicalOutputSpatial;
  estimate.logicalOutputChannels = logicalOutputChannels;
  estimate.reductionK = reductionK;
  estimate.physicalReductionK = physicalReductionK;
  estimate.numReductionTiles = numReductionTiles;
  estimate.logicalMacs = logicalMacs;
  estimate.logicalOutputElements = logicalOutputElements;
  estimate.numMTiles = numMTiles;
  estimate.numNTiles = numNTiles;
  estimate.totalTiles = totalTiles;
  estimate.physicalMacs = physicalMacs;
  estimate.physicalOutputElements = physicalOutputElements;
  estimate.paddingMacs = paddingMacs;
  estimate.spatialPaddingMacs = spatialPaddingMacs;
  estimate.reductionPaddingMacs = reductionPaddingMacs;
  estimate.activePELanes = activePELanes;
  estimate.totalPELanes = totalPELanes;
  estimate.arrayWavesPerTile = arrayWavesPerTile;
  estimate.peUtilization = peUtilization;
  estimate.computeCycles = computeCycles;
  estimate.inputLoadCount = inputLoadCount;
  estimate.weightLoadCount = weightLoadCount;
  estimate.outputWriteCount = outputWriteCount;
  estimate.outputReadCount = outputReadCount;
  estimate.offChipInputBytes = offChipInputBytes;
  estimate.offChipWeightBytes = offChipWeightBytes;
  estimate.offChipOutputWriteBytes = offChipOutputWriteBytes;
  estimate.offChipOutputReadBytes = offChipOutputReadBytes;
  estimate.totalOffChipBytes = totalOffChipBytes;
  estimate.dmaTransferCycles = dmaTransferCycles;
  estimate.dmaSetupCycles = dmaSetupCycles;
  estimate.dmaCycles = dmaCycles;
  estimate.localInputBytes = localInputBytes;
  estimate.localWeightBytes = localWeightBytes;
  estimate.localAccumReadBytes = localAccumReadBytes;
  estimate.localAccumWriteBytes = localAccumWriteBytes;
  estimate.totalLocalBytes = totalLocalBytes;
  estimate.localMemoryCycles = localMemoryCycles;
  estimate.setupCycles = setupCycles;
  estimate.synchronizationCycles = synchronizationCycles;
  estimate.computeDmaOverlapApplied = overlapApplied;
  estimate.totalEstimatedCycles = totalEstimatedCycles;
  return estimate;
}

llvm::Expected<std::vector<CandidateCostEstimate>>
estimateCandidateCosts(llvm::ArrayRef<KernelCandidate> candidates,
                       llvm::ArrayRef<CandidateLegalityResult> legalityResults,
                       const NPUTargetConfig &target) {
  if (candidates.size() != legalityResults.size())
    return makeError(
        "estimateCandidateCosts: candidates.size() != legalityResults.size()");

  std::vector<CandidateCostEstimate> estimates;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (legalityResults[i].candidateId != candidates[i].candidateId)
      return makeError(
          "estimateCandidateCosts: legalityResults[" + llvm::Twine(i) +
          "].candidateId does not match candidates[" + llvm::Twine(i) + "].candidateId");
    if (!legalityResults[i].isLegal)
      continue; // documented: illegal candidates are skipped, not estimated
    llvm::Expected<CandidateCostEstimate> estimate =
        estimateCandidateCost(candidates[i], legalityResults[i], target);
    if (!estimate)
      return estimate.takeError();
    estimates.push_back(std::move(*estimate));
  }
  return estimates;
}

} // namespace mlir::costmodel
