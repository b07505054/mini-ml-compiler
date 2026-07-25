#include "costmodel/SchedulePlan.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::costmodel {

llvm::StringRef toString(ScheduleLoopDimension dim) {
  switch (dim) {
  case ScheduleLoopDimension::M:
    return "M";
  case ScheduleLoopDimension::N:
    return "N";
  case ScheduleLoopDimension::K:
    return "K";
  }
  llvm_unreachable("unhandled ScheduleLoopDimension");
}

llvm::StringRef toString(ScheduleOperationKind kind) {
  switch (kind) {
  case ScheduleOperationKind::LoadInputTile:
    return "load_input_tile";
  case ScheduleOperationKind::LoadWeightTile:
    return "load_weight_tile";
  case ScheduleOperationKind::ReloadPartialOutput:
    return "reload_partial_output";
  case ScheduleOperationKind::ComputeTile:
    return "compute_tile";
  case ScheduleOperationKind::StorePartialOutput:
    return "store_partial_output";
  case ScheduleOperationKind::StoreFinalOutput:
    return "store_final_output";
  case ScheduleOperationKind::Synchronize:
    return "synchronize";
  }
  llvm_unreachable("unhandled ScheduleOperationKind");
}

bool SchedulePlan::operator==(const SchedulePlan &other) const {
  if (candidateId != other.candidateId || precision != other.precision ||
      dataflow != other.dataflow || !(peArray == other.peArray) || !(tile == other.tile) ||
      logicalM != other.logicalM || logicalN != other.logicalN || logicalK != other.logicalK ||
      numMTiles != other.numMTiles || numNTiles != other.numNTiles ||
      numKTiles != other.numKTiles || loopOrder != other.loopOrder ||
      !(residency == other.residency) || inputLoadCount != other.inputLoadCount ||
      weightLoadCount != other.weightLoadCount ||
      partialOutputReloadCount != other.partialOutputReloadCount ||
      computeCount != other.computeCount ||
      partialOutputStoreCount != other.partialOutputStoreCount ||
      finalOutputStoreCount != other.finalOutputStoreCount ||
      synchronizationCount != other.synchronizationCount)
    return false;
  if (operations.size() != other.operations.size())
    return false;
  for (std::size_t i = 0; i < operations.size(); ++i)
    if (!(operations[i] == other.operations[i]))
      return false;
  return true;
}

namespace {

llvm::Error makeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

// Same checked-arithmetic philosophy as CandidateCostModel.cpp: every
// operation is checked; `overflowed` is inspected once by the caller.
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
  std::uint64_t ceilDiv(std::uint64_t a, std::uint64_t b) {
    if (b == 0) {
      overflowed = true;
      return 0;
    }
    return a / b + (a % b != 0 ? 1 : 0);
  }
};

// Never allows underflow (Section 4): returns min(tileDim, logicalDim -
// checked(tileIndex * tileDim)). `ok` is set false (and 0 returned) on
// any overflow or if tileIndex*tileDim >= logicalDim -- which should
// never happen for an index in the valid [0, numTiles) range produced by
// ceilDiv, but is guarded defensively rather than assumed.
std::uint64_t checkedValidExtent(std::uint64_t tileIndex, std::uint64_t tileDim,
                                  std::uint64_t logicalDim, bool &ok) {
  std::uint64_t offset = 0;
  if (__builtin_mul_overflow(tileIndex, tileDim, &offset)) {
    ok = false;
    return 0;
  }
  if (offset >= logicalDim) {
    ok = false;
    return 0;
  }
  std::uint64_t remaining = logicalDim - offset;
  return std::min(tileDim, remaining);
}

} // namespace

llvm::Expected<SchedulePlan>
materializeCandidateSchedule(const KernelCandidate &candidate,
                            const CandidateLegalityResult &legality,
                            const CandidateCostEstimate &cost,
                            const Conv2DProblemShape &problem,
                            const NPUTargetConfig &target) {
  (void)target; // not needed by this slice's schedule materialization itself

  // ---- Section 2: fail-closed cross-stage identity validation ----
  if (legality.candidateId != candidate.candidateId)
    return makeError("materializeCandidateSchedule: legality.candidateId does not match "
                      "candidate.candidateId");
  if (cost.candidateId != candidate.candidateId)
    return makeError(
        "materializeCandidateSchedule: cost.candidateId does not match candidate.candidateId");
  if (!legality.isLegal)
    return makeError("materializeCandidateSchedule: candidate '" + candidate.candidateId +
                      "' is not legal; illegal candidates are never scheduled");
  if (!problem.outputShapeIsStaticAndSupported)
    return makeError(
        "materializeCandidateSchedule: conv2d problem's output shape is not statically "
        "supported; schedule materialization fails closed");
  if (problem.batch <= 0 || problem.outputHeight <= 0 || problem.outputWidth <= 0 ||
      problem.outputChannels <= 0 || problem.inputChannels <= 0 || problem.kernelHeight <= 0 ||
      problem.kernelWidth <= 0)
    return makeError("materializeCandidateSchedule: conv2d problem has a non-positive logical "
                      "extent");
  if (candidate.tile.height <= 0 || candidate.tile.width <= 0 ||
      candidate.tile.outputChannels <= 0 || candidate.tile.reductionDepth <= 0)
    return makeError("materializeCandidateSchedule: candidate's tile shape has a non-positive "
                      "dimension (including reductionDepth/tileK)");
  if (candidate.peArray.rows <= 0 || candidate.peArray.cols <= 0)
    return makeError("materializeCandidateSchedule: candidate's PE array has a non-positive "
                      "dimension");

  CheckedArith ck;
  std::uint64_t tileM = ck.mul(static_cast<std::uint64_t>(candidate.tile.height),
                                static_cast<std::uint64_t>(candidate.tile.width));
  std::uint64_t tileN = static_cast<std::uint64_t>(candidate.tile.outputChannels);
  std::uint64_t tileK = static_cast<std::uint64_t>(candidate.tile.reductionDepth);

  std::uint64_t logicalM =
      ck.mul(ck.mul(static_cast<std::uint64_t>(problem.batch),
                    static_cast<std::uint64_t>(problem.outputHeight)),
             static_cast<std::uint64_t>(problem.outputWidth));
  std::uint64_t logicalN = static_cast<std::uint64_t>(problem.outputChannels);
  std::uint64_t logicalK =
      ck.mul(ck.mul(static_cast<std::uint64_t>(problem.inputChannels),
                    static_cast<std::uint64_t>(problem.kernelHeight)),
             static_cast<std::uint64_t>(problem.kernelWidth));

  std::uint64_t MT = ck.ceilDiv(logicalM, tileM);
  std::uint64_t NT = ck.ceilDiv(logicalN, tileN);
  std::uint64_t KT = ck.ceilDiv(logicalK, tileK);

  if (ck.overflowed)
    return makeError("materializeCandidateSchedule: integer overflow computing tile counts");

  // "recomputed tile counts match the cost estimate" (Section 2).
  if (MT != cost.numMTiles || NT != cost.numNTiles || KT != cost.numReductionTiles)
    return makeError(
        "materializeCandidateSchedule: recomputed MT/NT/KT do not match the supplied cost "
        "estimate (stale or inconsistent stage results)");

  std::uint64_t I = static_cast<std::uint64_t>(legality.inputTileBytes);
  std::uint64_t W = static_cast<std::uint64_t>(legality.weightTileBytes);
  std::uint64_t O = static_cast<std::uint64_t>(legality.outputTileBytes);

  SchedulePlan plan;
  plan.candidateId = candidate.candidateId;
  plan.precision = candidate.precision;
  plan.dataflow = candidate.dataflow;
  plan.peArray = candidate.peArray;
  plan.tile = candidate.tile;
  plan.logicalM = logicalM;
  plan.logicalN = logicalN;
  plan.logicalK = logicalK;
  plan.numMTiles = MT;
  plan.numNTiles = NT;
  plan.numKTiles = KT;

  switch (candidate.dataflow) {
  case Dataflow::WeightStationary:
    plan.loopOrder = {ScheduleLoopDimension::N, ScheduleLoopDimension::K, ScheduleLoopDimension::M};
    plan.residency.weightResidentAcrossM = true;
    break;
  case Dataflow::InputStationary:
    plan.loopOrder = {ScheduleLoopDimension::M, ScheduleLoopDimension::K, ScheduleLoopDimension::N};
    plan.residency.inputResidentAcrossN = true;
    break;
  case Dataflow::OutputStationary:
    plan.loopOrder = {ScheduleLoopDimension::M, ScheduleLoopDimension::N, ScheduleLoopDimension::K};
    plan.residency.outputResidentAcrossK = true;
    break;
  }

  std::uint64_t opIndex = 0;
  bool extentOk = true;

  auto emit = [&](ScheduleOperationKind kind, std::uint64_t m, std::uint64_t n, std::uint64_t k,
                  std::uint64_t bytes) {
    ScheduleOperation op;
    op.operationIndex = opIndex++;
    op.kind = kind;
    op.coordinate = TileCoordinate{m, n, k};
    op.extent.physicalM = tileM;
    op.extent.physicalN = tileN;
    op.extent.physicalK = tileK;
    op.extent.validM = checkedValidExtent(m, tileM, logicalM, extentOk);
    op.extent.validN = checkedValidExtent(n, tileN, logicalN, extentOk);
    op.extent.validK = checkedValidExtent(k, tileK, logicalK, extentOk);
    op.isBoundaryTile = op.extent.validM < op.extent.physicalM ||
                        op.extent.validN < op.extent.physicalN ||
                        op.extent.validK < op.extent.physicalK;
    op.isFinalReductionTile = (k + 1 == KT);
    op.bytes = bytes;
    plan.operations.push_back(op);
  };

  switch (candidate.dataflow) {
  case Dataflow::WeightStationary:
    for (std::uint64_t n = 0; n < NT; ++n) {
      for (std::uint64_t k = 0; k < KT; ++k) {
        emit(ScheduleOperationKind::LoadWeightTile, /*m=*/0, n, k, W);
        ++plan.weightLoadCount;
        for (std::uint64_t m = 0; m < MT; ++m) {
          emit(ScheduleOperationKind::LoadInputTile, m, n, k, I);
          ++plan.inputLoadCount;
          if (k > 0) {
            emit(ScheduleOperationKind::ReloadPartialOutput, m, n, k, O);
            ++plan.partialOutputReloadCount;
          }
          emit(ScheduleOperationKind::ComputeTile, m, n, k, 0);
          ++plan.computeCount;
          if (k + 1 < KT) {
            emit(ScheduleOperationKind::StorePartialOutput, m, n, k, O);
            ++plan.partialOutputStoreCount;
          } else {
            emit(ScheduleOperationKind::StoreFinalOutput, m, n, k, O);
            ++plan.finalOutputStoreCount;
          }
          emit(ScheduleOperationKind::Synchronize, m, n, k, 0);
          ++plan.synchronizationCount;
        }
      }
    }
    break;
  case Dataflow::InputStationary:
    for (std::uint64_t m = 0; m < MT; ++m) {
      for (std::uint64_t k = 0; k < KT; ++k) {
        emit(ScheduleOperationKind::LoadInputTile, m, /*n=*/0, k, I);
        ++plan.inputLoadCount;
        for (std::uint64_t n = 0; n < NT; ++n) {
          emit(ScheduleOperationKind::LoadWeightTile, m, n, k, W);
          ++plan.weightLoadCount;
          if (k > 0) {
            emit(ScheduleOperationKind::ReloadPartialOutput, m, n, k, O);
            ++plan.partialOutputReloadCount;
          }
          emit(ScheduleOperationKind::ComputeTile, m, n, k, 0);
          ++plan.computeCount;
          if (k + 1 < KT) {
            emit(ScheduleOperationKind::StorePartialOutput, m, n, k, O);
            ++plan.partialOutputStoreCount;
          } else {
            emit(ScheduleOperationKind::StoreFinalOutput, m, n, k, O);
            ++plan.finalOutputStoreCount;
          }
          emit(ScheduleOperationKind::Synchronize, m, n, k, 0);
          ++plan.synchronizationCount;
        }
      }
    }
    break;
  case Dataflow::OutputStationary:
    for (std::uint64_t m = 0; m < MT; ++m) {
      for (std::uint64_t n = 0; n < NT; ++n) {
        for (std::uint64_t k = 0; k < KT; ++k) {
          emit(ScheduleOperationKind::LoadInputTile, m, n, k, I);
          ++plan.inputLoadCount;
          emit(ScheduleOperationKind::LoadWeightTile, m, n, k, W);
          ++plan.weightLoadCount;
          emit(ScheduleOperationKind::ComputeTile, m, n, k, 0);
          ++plan.computeCount;
          emit(ScheduleOperationKind::Synchronize, m, n, k, 0);
          ++plan.synchronizationCount;
        }
        emit(ScheduleOperationKind::StoreFinalOutput, m, n, KT - 1, O);
        ++plan.finalOutputStoreCount;
      }
    }
    break;
  }

  if (!extentOk)
    return makeError("materializeCandidateSchedule: overflow or invariant violation computing "
                      "a tile's valid extent");

  if (llvm::Error err = validateScheduleAgainstCost(plan, cost))
    return std::move(err);

  return plan;
}

llvm::Expected<SchedulePlan> materializeSelectedSchedule(const PlanSelectionResult &selection,
                                                          const Conv2DProblemShape &problem,
                                                          const NPUTargetConfig &target) {
  if (selection.selectedCandidate.candidateId != selection.selectedCandidateId)
    return makeError("materializeSelectedSchedule: selection.selectedCandidate does not match "
                      "selection.selectedCandidateId");
  if (selection.selectedCost.candidateId != selection.selectedCandidateId)
    return makeError("materializeSelectedSchedule: selection.selectedCost does not match "
                      "selection.selectedCandidateId");

  // Recomputed via the existing, unmodified Slice 2 API -- checkCandidateLegality
  // is a pure function of (candidate, target), so this reproduces byte-
  // identical results to whatever legality was used to build `selection`.
  CandidateLegalityResult legality = checkCandidateLegality(selection.selectedCandidate, target);
  return materializeCandidateSchedule(selection.selectedCandidate, legality,
                                      selection.selectedCost, problem, target);
}

llvm::Error validateScheduleAgainstCost(const SchedulePlan &schedule,
                                        const CandidateCostEstimate &cost) {
  if (schedule.inputLoadCount != cost.inputLoadCount)
    return makeError("validateScheduleAgainstCost: inputLoadCount mismatch");
  if (schedule.weightLoadCount != cost.weightLoadCount)
    return makeError("validateScheduleAgainstCost: weightLoadCount mismatch");
  if (schedule.partialOutputReloadCount != cost.outputReadCount)
    return makeError("validateScheduleAgainstCost: partialOutputReloadCount != "
                      "cost.outputReadCount");

  CheckedArith ck;
  std::uint64_t totalWrites =
      ck.add(schedule.partialOutputStoreCount, schedule.finalOutputStoreCount);
  if (ck.overflowed)
    return makeError("validateScheduleAgainstCost: overflow summing output-write counts");
  if (totalWrites != cost.outputWriteCount)
    return makeError("validateScheduleAgainstCost: total output-write count != "
                      "cost.outputWriteCount");

  std::uint64_t scheduleDmaOps =
      ck.add(ck.add(ck.add(schedule.inputLoadCount, schedule.weightLoadCount),
                    schedule.partialOutputReloadCount),
             totalWrites);
  std::uint64_t costDmaOps =
      ck.add(ck.add(ck.add(cost.inputLoadCount, cost.weightLoadCount), cost.outputReadCount),
             cost.outputWriteCount);
  if (ck.overflowed)
    return makeError("validateScheduleAgainstCost: overflow summing DMA operation counts");
  if (scheduleDmaOps != costDmaOps)
    return makeError("validateScheduleAgainstCost: total DMA operation count mismatch");

  std::uint64_t expectedTileOps = ck.mul(ck.mul(cost.numMTiles, cost.numNTiles), cost.numReductionTiles);
  if (ck.overflowed)
    return makeError("validateScheduleAgainstCost: overflow computing MT*NT*KT");
  if (schedule.synchronizationCount != expectedTileOps)
    return makeError("validateScheduleAgainstCost: synchronizationCount != MT*NT*KT");
  if (schedule.computeCount != expectedTileOps)
    return makeError("validateScheduleAgainstCost: computeCount != MT*NT*KT");

  std::uint64_t inputBytesSum = 0, weightBytesSum = 0, reloadBytesSum = 0, storeBytesSum = 0;
  for (const ScheduleOperation &op : schedule.operations) {
    switch (op.kind) {
    case ScheduleOperationKind::LoadInputTile:
      inputBytesSum = ck.add(inputBytesSum, op.bytes);
      break;
    case ScheduleOperationKind::LoadWeightTile:
      weightBytesSum = ck.add(weightBytesSum, op.bytes);
      break;
    case ScheduleOperationKind::ReloadPartialOutput:
      reloadBytesSum = ck.add(reloadBytesSum, op.bytes);
      break;
    case ScheduleOperationKind::StorePartialOutput:
    case ScheduleOperationKind::StoreFinalOutput:
      storeBytesSum = ck.add(storeBytesSum, op.bytes);
      break;
    case ScheduleOperationKind::ComputeTile:
    case ScheduleOperationKind::Synchronize:
      break;
    }
  }
  if (ck.overflowed)
    return makeError("validateScheduleAgainstCost: overflow summing operation bytes");

  if (inputBytesSum != cost.offChipInputBytes)
    return makeError("validateScheduleAgainstCost: summed input bytes != cost.offChipInputBytes");
  if (weightBytesSum != cost.offChipWeightBytes)
    return makeError(
        "validateScheduleAgainstCost: summed weight bytes != cost.offChipWeightBytes");
  if (reloadBytesSum != cost.offChipOutputReadBytes)
    return makeError(
        "validateScheduleAgainstCost: summed reload bytes != cost.offChipOutputReadBytes");
  if (storeBytesSum != cost.offChipOutputWriteBytes)
    return makeError(
        "validateScheduleAgainstCost: summed store bytes != cost.offChipOutputWriteBytes");

  std::uint64_t totalBytes =
      ck.add(ck.add(ck.add(inputBytesSum, weightBytesSum), reloadBytesSum), storeBytesSum);
  if (ck.overflowed)
    return makeError("validateScheduleAgainstCost: overflow summing total off-chip bytes");
  if (totalBytes != cost.totalOffChipBytes)
    return makeError("validateScheduleAgainstCost: summed total bytes != cost.totalOffChipBytes");

  return llvm::Error::success();
}

std::string serializeSchedulePlanToJson(const SchedulePlan &plan) {
  std::string out;
  llvm::raw_string_ostream os(out);
  std::uint64_t tileM = static_cast<std::uint64_t>(plan.tile.height) *
                        static_cast<std::uint64_t>(plan.tile.width);
  os << "{\n";
  os << "  \"candidate_id\": \"" << plan.candidateId << "\",\n";
  os << "  \"dataflow\": \"" << toString(plan.dataflow) << "\",\n";
  os << "  \"loop_order\": [";
  for (std::size_t i = 0; i < plan.loopOrder.size(); ++i) {
    if (i)
      os << ", ";
    os << "\"" << toString(plan.loopOrder[i]) << "\"";
  }
  os << "],\n";
  os << "  \"tile\": {\"m\": " << tileM << ", \"n\": " << plan.tile.outputChannels
     << ", \"k\": " << plan.tile.reductionDepth << ", \"height\": " << plan.tile.height
     << ", \"width\": " << plan.tile.width
     << ", \"output_channels\": " << plan.tile.outputChannels << "},\n";
  os << "  \"tile_counts\": {\"m\": " << plan.numMTiles << ", \"n\": " << plan.numNTiles
     << ", \"k\": " << plan.numKTiles << "},\n";
  os << "  \"residency\": {\"input_across_n\": "
     << (plan.residency.inputResidentAcrossN ? "true" : "false")
     << ", \"weight_across_m\": " << (plan.residency.weightResidentAcrossM ? "true" : "false")
     << ", \"output_across_k\": " << (plan.residency.outputResidentAcrossK ? "true" : "false")
     << "},\n";
  os << "  \"operation_counts\": {\"input_loads\": " << plan.inputLoadCount
     << ", \"weight_loads\": " << plan.weightLoadCount
     << ", \"partial_output_reloads\": " << plan.partialOutputReloadCount
     << ", \"compute\": " << plan.computeCount
     << ", \"partial_output_stores\": " << plan.partialOutputStoreCount
     << ", \"final_output_stores\": " << plan.finalOutputStoreCount
     << ", \"synchronizations\": " << plan.synchronizationCount << "}\n";
  os << "}";
  os.flush();
  return out;
}

} // namespace mlir::costmodel
