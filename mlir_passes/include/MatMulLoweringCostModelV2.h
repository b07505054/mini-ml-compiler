#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::hir {

enum class FusionMode { Unfused, Fused };
enum class ScheduleMode {
  Scalar,
  WholeShapeVector,
  TiledVector
};
enum class TailStrategy {
  None,
  WholeShapeMaterializedPadding,
  TileMaterializedPadding,
  DirectScalarCleanup,
  DirectVectorCleanup,
  SpecializedMicrokernel,
  MaskedTransfer
};

struct NativeCostCalibration {
  std::string target = "conservative-generic";
  std::string dtype = "f32";
  double effectiveScalarOpsPerNs = 0.25;
  double effectiveVectorOpsPerNs = 1.0;
  double vectorUtilization = 0.5;
  int64_t vectorWidthBits = 128;
  double effectiveBandwidthBytesPerNs = 8.0;
  double allocationFixedNs = 120.0;
  double allocationInitBytesPerNs = 16.0;
  double branchNs = 1.5;
  double internalFreeNs = 35.0;
  double fusedVectorInteractionFraction = 0.0;
  bool supportsVector = false;
};

inline NativeCostCalibration raspberryPi5CortexA76F32Calibration() {
  NativeCostCalibration c;
  c.target = "raspberry-pi5-cortex-a76";
  c.effectiveScalarOpsPerNs = 0.52;
  c.effectiveVectorOpsPerNs = 14.5;
  c.vectorUtilization = 0.82;
  c.vectorWidthBits = 128;
  c.effectiveBandwidthBytesPerNs = 12.0;
  c.allocationFixedNs = 120.0;
  c.allocationInitBytesPerNs = 20.0;
  c.branchNs = 1.5;
  c.internalFreeNs = 35.0;
  // Residual locality/scheduling correction only. It is deliberately capped
  // below and is not the source of the modeled fusion benefit.
  c.fusedVectorInteractionFraction = 0.03;
  c.supportsVector = true;
  return c;
}

struct NativeCostBreakdown {
  double computeNs = 0.0;
  double memoryTrafficNs = 0.0;
  double allocationNs = 0.0;
  double paddingComputeNs = 0.0;
  double paddingCopyNs = 0.0;
  double cropCopyNs = 0.0;
  double controlOverheadNs = 0.0;
  double interactionCorrectionNs = 0.0;
  double totalNs = 0.0;

  void sum() {
    totalNs = computeNs + memoryTrafficNs + allocationNs +
              paddingComputeNs + paddingCopyNs + cropCopyNs +
              controlOverheadNs + interactionCorrectionNs;
  }
};

struct MatMulLoweringCandidate {
  std::string id;
  FusionMode fusion = FusionMode::Unfused;
  ScheduleMode schedule = ScheduleMode::Scalar;
  TailStrategy mTailStrategy = TailStrategy::None;
  TailStrategy nTailStrategy = TailStrategy::None;
  TailStrategy kTailStrategy = TailStrategy::None;
  int64_t originalM = 0, originalN = 0, originalK = 0;
  int64_t executedM = 0, executedN = 0, executedK = 0;
  std::string dtype = "f32";
  int64_t tileM = 0, tileN = 0, tileK = 0;
  int64_t vectorWidth = 1;
  int64_t mRemainder = 0, nRemainder = 0, kRemainder = 0;
  int64_t mTileCount = 0, nTileCount = 0, kTileCount = 0;
  int64_t estimatedTailInvocationCount = 0;
  bool usesWholeShapeMaterializedPadding = false;
  bool usesTileLocalMaterializedPadding = false;
  bool usesMaterializedPadding = false;
  bool usesMaskedLoads = false;
  bool usesMaskedStores = false;
  bool usesScalarCleanup = false;
  bool usesDirectVectorCleanup = false;
  bool usesSpecializedTailMicrokernel = false;
  bool requiresPadding = false;
  bool requiresCrop = false;
  bool loweringComplete = true;
  int64_t temporaryAllocationCount = 0;
  int64_t temporaryAllocatedBytes = 0;
  int64_t intermediateReadCount = 0;
  int64_t intermediateWriteCount = 0;
  int64_t inputBytes = 0, outputBytes = 0, intermediateBytes = 0;
  int64_t computeOperations = 0;
  int64_t paddingInputBytes = 0, cropBytes = 0;
  int64_t branchCount = 0, internalFreeCount = 0;
  int64_t codeSizeRisk = 0;
  int64_t fullTileCount = 0, edgeTileCount = 0, tailElementCount = 0;
  int64_t materializedPaddingBytes = 0, maskedLaneCount = 0;
  int64_t zeroFillBytes = 0, copyBytes = 0;
  int64_t scalarCleanupOperations = 0;
  int64_t directVectorCleanupOperations = 0;
  int64_t estimatedCodeSizeBytes = 0;
  double codeSizePenaltyNs = 0.0;
  double registerPressureRisk = 0.0, spillRiskPenaltyNs = 0.0;
  std::string rejectionReason;
  NativeCostBreakdown cost;

  bool valid() const { return loweringComplete && rejectionReason.empty(); }
};

struct MatMulLoweringProblem {
  int64_t m = 0, n = 0, k = 0;
  std::string dtype = "f32";
  bool fusionLegal = true;
  std::string fusionRejectionReason;
  bool fusedScalarRequiresTile = true;
  bool fusedVectorRequiresTile = true;
  // The AArch64 vector pipeline lowers tensor.pad's post-bufferization
  // zero-fill linalg maps to loops before the LLVM conversion boundary.
  // Targets/pipelines without that cleanup can still override this capability.
  bool paddedFusedVectorLoweringComplete = true;
  int64_t tileM = 16, tileN = 16, tileK = 32;
};

struct MatMulLoweringSelection {
  std::vector<MatMulLoweringCandidate> candidates;
  int selectedIndex = -1;
  const MatMulLoweringCandidate *winner() const {
    return selectedIndex < 0 ? nullptr : &candidates[selectedIndex];
  }
};

inline int64_t roundUpV2(int64_t x, int64_t multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

inline const char *fusionName(FusionMode m) {
  return m == FusionMode::Fused ? "fused" : "unfused";
}
inline const char *scheduleName(ScheduleMode m) {
  switch (m) {
  case ScheduleMode::Scalar: return "scalar";
  case ScheduleMode::WholeShapeVector: return "whole_shape_vector";
  case ScheduleMode::TiledVector: return "tiled_vector";
  }
  return "unknown";
}
inline const char *tailName(TailStrategy t) {
  switch (t) {
  case TailStrategy::None: return "none";
  case TailStrategy::WholeShapeMaterializedPadding:
    return "whole_shape_materialized_padding";
  case TailStrategy::TileMaterializedPadding:
    return "tile_materialized_padding";
  case TailStrategy::DirectScalarCleanup: return "direct_scalar_cleanup";
  case TailStrategy::DirectVectorCleanup: return "direct_vector_cleanup";
  case TailStrategy::SpecializedMicrokernel: return "specialized_microkernel";
  case TailStrategy::MaskedTransfer: return "masked_transfer";
  }
  return "unknown";
}

inline void estimateCandidate(MatMulLoweringCandidate &c,
                              const NativeCostCalibration &cal) {
  if (!c.valid())
    return;
  const int64_t elementBytes = c.dtype == "f16" ? 2 : 4;
  const int64_t originalOps =
      2 * c.originalM * c.originalN * c.originalK +
      2 * c.originalM * c.originalN;
  c.computeOperations =
      2 * c.executedM * c.executedN * c.executedK +
      2 * c.executedM * c.executedN;
  c.inputBytes = (c.originalM * c.originalK +
                  c.originalK * c.originalN +
                  c.originalM * c.originalN) * elementBytes;
  c.outputBytes = c.originalM * c.originalN * elementBytes;
  if (c.fusion == FusionMode::Unfused) {
    c.intermediateReadCount = 2;
    c.intermediateWriteCount = 2;
    c.intermediateBytes = 4 * c.originalM * c.originalN * elementBytes;
    c.temporaryAllocationCount = 2;
    c.temporaryAllocatedBytes = 2 * c.originalM * c.originalN * elementBytes;
    c.internalFreeCount = 2;
    c.branchCount = 18;
  } else if (c.requiresPadding) {
    c.temporaryAllocationCount = 4;
    c.temporaryAllocatedBytes =
        (c.executedM * c.executedK + c.executedK * c.executedN +
         c.executedM * c.executedN * 2) * elementBytes;
    c.internalFreeCount = 4;
    c.branchCount = 35;
    c.paddingInputBytes =
        ((c.originalM * c.originalK + c.originalK * c.originalN +
          c.originalM * c.originalN) +
         (c.executedM * c.executedK + c.executedK * c.executedN +
          c.executedM * c.executedN)) * elementBytes;
    c.cropBytes = (c.executedM * c.executedN +
                   c.originalM * c.originalN) * elementBytes;
  } else if (c.kTailStrategy == TailStrategy::TileMaterializedPadding ||
             c.mTailStrategy == TailStrategy::TileMaterializedPadding ||
             c.nTailStrategy == TailStrategy::TileMaterializedPadding) {
    c.internalFreeCount = c.temporaryAllocationCount;
    c.branchCount = 24 + c.edgeTileCount;
    c.paddingInputBytes = c.zeroFillBytes + c.copyBytes;
  } else {
    c.temporaryAllocationCount =
        c.schedule == ScheduleMode::Scalar ? 1 : 0;
    c.temporaryAllocatedBytes =
        c.temporaryAllocationCount * c.originalM * c.originalN * elementBytes;
    c.internalFreeCount = c.temporaryAllocationCount;
    c.branchCount = c.schedule == ScheduleMode::Scalar ? 11 : 3;
  }
  if (c.schedule == ScheduleMode::WholeShapeVector) {
    c.estimatedCodeSizeBytes =
        std::max<int64_t>(256, c.executedM * c.executedN * c.executedK / 2);
    c.registerPressureRisk =
        static_cast<double>(c.executedM * c.executedN) / 128.0;
  } else if (c.schedule == ScheduleMode::TiledVector) {
    c.estimatedCodeSizeBytes = 4096;
    c.registerPressureRisk =
        static_cast<double>(c.tileM * c.tileN) / 128.0;
  } else {
    c.estimatedCodeSizeBytes = 512;
  }
  c.codeSizeRisk = c.estimatedCodeSizeBytes;
  c.codeSizePenaltyNs =
      std::max(0.0, static_cast<double>(c.estimatedCodeSizeBytes - 16384)) /
      64.0;
  c.spillRiskPenaltyNs = std::max(0.0, c.registerPressureRisk - 2.0) * 20.0;

  const double throughput =
      c.schedule != ScheduleMode::Scalar
          ? cal.effectiveVectorOpsPerNs * cal.vectorUtilization
          : cal.effectiveScalarOpsPerNs;
  c.cost.computeNs = static_cast<double>(originalOps) / throughput;
  c.cost.paddingComputeNs =
      static_cast<double>(c.computeOperations - originalOps) / throughput;
  c.cost.memoryTrafficNs =
      static_cast<double>(c.inputBytes + c.outputBytes +
                          c.intermediateBytes) /
      cal.effectiveBandwidthBytesPerNs;
  c.cost.allocationNs =
      c.temporaryAllocationCount * cal.allocationFixedNs +
      static_cast<double>(c.temporaryAllocatedBytes) /
          cal.allocationInitBytesPerNs;
  c.cost.paddingCopyNs =
      static_cast<double>(c.paddingInputBytes) /
      cal.effectiveBandwidthBytesPerNs;
  c.cost.cropCopyNs =
      static_cast<double>(c.cropBytes) / cal.effectiveBandwidthBytesPerNs;
  c.cost.controlOverheadNs =
      c.branchCount * cal.branchNs + c.internalFreeCount * cal.internalFreeNs;
  c.cost.controlOverheadNs +=
      c.codeSizePenaltyNs + c.spillRiskPenaltyNs +
      static_cast<double>(c.scalarCleanupOperations) /
          cal.effectiveScalarOpsPerNs +
      static_cast<double>(c.directVectorCleanupOperations) /
          (cal.effectiveVectorOpsPerNs * cal.vectorUtilization);
  if (c.fusion == FusionMode::Fused &&
      c.schedule != ScheduleMode::Scalar) {
    const double fraction =
        std::clamp(cal.fusedVectorInteractionFraction, 0.0, 0.05);
    c.cost.interactionCorrectionNs =
        -fraction * (c.cost.computeNs + c.cost.memoryTrafficNs);
  }
  c.cost.sum();
}

inline MatMulLoweringSelection
selectMatMulLowering(const MatMulLoweringProblem &p,
                     const NativeCostCalibration &cal,
                     double tieTolerance = 0.01) {
  MatMulLoweringSelection result;
  for (FusionMode fusion : {FusionMode::Unfused, FusionMode::Fused}) {
    for (ScheduleMode schedule :
         {ScheduleMode::Scalar, ScheduleMode::WholeShapeVector,
          ScheduleMode::TiledVector}) {
      std::vector<TailStrategy> tails = {TailStrategy::None};
      const bool hasTileTail =
          p.m % p.tileM != 0 || p.n % p.tileN != 0 || p.k % p.tileK != 0;
      if (schedule == ScheduleMode::TiledVector && hasTileTail)
        tails = {TailStrategy::None, TailStrategy::TileMaterializedPadding,
                 TailStrategy::DirectVectorCleanup};
      for (TailStrategy tail : tails) {
        MatMulLoweringCandidate c;
        c.fusion = fusion;
        c.schedule = schedule;
        c.id = std::string(fusionName(fusion)) + "_" +
               scheduleName(schedule);
        if (tail != TailStrategy::None)
          c.id += "_" + std::string(tailName(tail));
        c.originalM = c.executedM = p.m;
        c.originalN = c.executedN = p.n;
        c.originalK = c.executedK = p.k;
        c.dtype = p.dtype;
        if (fusion == FusionMode::Fused && !p.fusionLegal) {
          c.loweringComplete = false;
          c.rejectionReason = p.fusionRejectionReason.empty()
                                  ? "fusion_illegal"
                                  : p.fusionRejectionReason;
        }
        if (schedule != ScheduleMode::Scalar && !cal.supportsVector) {
          c.loweringComplete = false;
          c.rejectionReason = "target_has_no_vector_capability";
        }
        const bool wholePadding =
            fusion == FusionMode::Fused &&
            ((schedule == ScheduleMode::WholeShapeVector &&
              p.fusedVectorRequiresTile) ||
             (schedule == ScheduleMode::Scalar &&
              p.fusedScalarRequiresTile));
        if (wholePadding) {
          c.tileM = p.tileM;
          c.tileN = p.tileN;
          c.tileK = p.tileK;
          c.executedM = roundUpV2(p.m, p.tileM);
          c.executedN = roundUpV2(p.n, p.tileN);
          c.executedK = roundUpV2(p.k, p.tileK);
          c.requiresPadding = c.executedM != p.m || c.executedN != p.n ||
                              c.executedK != p.k;
          c.requiresCrop = c.requiresPadding;
          if (c.requiresPadding) {
            c.mTailStrategy = c.nTailStrategy = c.kTailStrategy =
                TailStrategy::WholeShapeMaterializedPadding;
            c.usesWholeShapeMaterializedPadding = true;
            c.usesMaterializedPadding = true;
          }
        }
        if (schedule == ScheduleMode::TiledVector) {
          c.tileM = p.tileM;
          c.tileN = p.tileN;
          c.tileK = std::min<int64_t>(p.tileK, 8);
          c.vectorWidth = 4;
          c.mRemainder = p.m % c.tileM;
          c.nRemainder = p.n % c.tileN;
          c.kRemainder = p.k % c.tileK;
          c.mTileCount = (p.m + c.tileM - 1) / c.tileM;
          c.nTileCount = (p.n + c.tileN - 1) / c.tileN;
          c.kTileCount = (p.k + c.tileK - 1) / c.tileK;
          c.edgeTileCount =
              (c.mRemainder || c.nRemainder || c.kRemainder)
                  ? c.mTileCount * c.nTileCount * c.kTileCount
                  : 0;
          c.fullTileCount = c.mTileCount * c.nTileCount * c.kTileCount -
                            c.edgeTileCount;
          c.tailElementCount =
              c.mTileCount * c.tileM * c.nTileCount * c.tileN *
                  c.kTileCount * c.tileK -
              p.m * p.n * p.k;
          if (c.mRemainder)
            c.mTailStrategy = tail;
          if (c.nRemainder)
            c.nTailStrategy = tail;
          if (c.kRemainder)
            c.kTailStrategy = tail;
          c.estimatedTailInvocationCount =
              (c.kRemainder ? c.mTileCount * c.nTileCount : 0) +
              (c.mRemainder ? c.nTileCount * c.kTileCount : 0) +
              (c.nRemainder ? c.mTileCount * c.kTileCount : 0);
          if (tail == TailStrategy::None && c.edgeTileCount) {
            c.loweringComplete = false;
            c.rejectionReason = "tiled_vector_requires_tail_strategy";
          } else if (tail == TailStrategy::TileMaterializedPadding &&
                     c.edgeTileCount) {
            c.usesMaterializedPadding = true;
            c.usesTileLocalMaterializedPadding = true;
            c.materializedPaddingBytes =
                (c.tileM * c.tileK + c.tileK * c.tileN +
                 2 * c.tileM * c.tileN) * 4;
            c.zeroFillBytes =
                c.materializedPaddingBytes * c.estimatedTailInvocationCount;
            c.copyBytes =
                c.materializedPaddingBytes * 2 *
                c.estimatedTailInvocationCount;
            c.temporaryAllocationCount = 4;
            c.temporaryAllocatedBytes = c.materializedPaddingBytes;
          } else if (tail == TailStrategy::DirectVectorCleanup &&
                     c.edgeTileCount) {
            // This slice implements only full M/N tiles with a direct K tail.
            if (c.mRemainder || c.nRemainder) {
              c.loweringComplete = false;
              c.rejectionReason =
                  "direct_vector_cleanup_supports_k_tail_only";
            } else {
              c.usesDirectVectorCleanup = true;
              c.directVectorCleanupOperations =
                  2 * p.m * p.n * c.kRemainder;
              c.estimatedTailInvocationCount =
                  c.mTileCount * c.nTileCount;
            }
          }
        }
        if (c.requiresPadding && fusion == FusionMode::Fused &&
            schedule == ScheduleMode::WholeShapeVector &&
            !p.paddedFusedVectorLoweringComplete) {
          c.loweringComplete = false;
          c.rejectionReason =
              "padded_fused_vector_padding_fill_lowering_unavailable";
        }
        estimateCandidate(c, cal);
        result.candidates.push_back(std::move(c));
      }
    }
  }

  auto betterTieBreak = [](const MatMulLoweringCandidate &a,
                           const MatMulLoweringCandidate &b) {
    if (a.requiresPadding != b.requiresPadding)
      return !a.requiresPadding;
    if (a.temporaryAllocationCount != b.temporaryAllocationCount)
      return a.temporaryAllocationCount < b.temporaryAllocationCount;
    if (a.codeSizeRisk != b.codeSizeRisk)
      return a.codeSizeRisk < b.codeSizeRisk;
    if (a.schedule != b.schedule)
      return a.schedule == ScheduleMode::Scalar;
    return a.id < b.id;
  };
  for (int i = 0, e = static_cast<int>(result.candidates.size()); i < e; ++i) {
    const auto &candidate = result.candidates[i];
    if (!candidate.valid())
      continue;
    if (result.selectedIndex < 0) {
      result.selectedIndex = i;
      continue;
    }
    const auto &selected = result.candidates[result.selectedIndex];
    const double scale = std::max(candidate.cost.totalNs, selected.cost.totalNs);
    const bool tied =
        std::abs(candidate.cost.totalNs - selected.cost.totalNs) <=
        tieTolerance * scale;
    if (candidate.cost.totalNs < selected.cost.totalNs && !tied)
      result.selectedIndex = i;
    else if (tied && betterTieBreak(candidate, selected))
      result.selectedIndex = i;
  }
  return result;
}

} // namespace mlir::hir
