#include "MatMulLoweringCostModelV2.h"

#include <cassert>
#include <cmath>
#include <string>

using namespace mlir::hir;

static const MatMulLoweringCandidate &
find(const MatMulLoweringSelection &s, const char *id) {
  for (const auto &c : s.candidates)
    if (c.id == id)
      return c;
  assert(false && "candidate not found");
  return s.candidates.front();
}

int main() {
  NativeCostCalibration pi = raspberryPi5CortexA76F32Calibration();
  MatMulLoweringProblem aligned{16, 16, 32};
  auto all = selectMatMulLowering(aligned, pi);
  assert(all.candidates.size() == 6);
  for (const auto &c : all.candidates)
    assert(c.valid());
  assert(find(all, "unfused_scalar_baseline").intermediateBytes >
         find(all, "fused_scalar_baseline").intermediateBytes);
  assert(find(all, "unfused_tiled_vector_full_tiles").cost.computeNs <
         find(all, "unfused_scalar_baseline").cost.computeNs);
  assert(find(all, "fused_tiled_vector_full_tiles").fullTileCount > 0);
  assert(find(all, "fused_whole_shape_vector_no_padding")
             .estimatedCodeSizeBytes >=
         find(all, "fused_tiled_vector_full_tiles").estimatedCodeSizeBytes);

  NativeCostCalibration scalarOnly = pi;
  scalarOnly.supportsVector = false;
  auto noVector = selectMatMulLowering(aligned, scalarOnly);
  assert(!find(noVector, "unfused_whole_shape_vector_no_padding").valid());
  assert(!find(noVector, "fused_tiled_vector_full_tiles").valid());

  MatMulLoweringProblem noFusion = aligned;
  noFusion.fusionLegal = false;
  noFusion.fusionRejectionReason = "matmul_result_not_one_use";
  auto unfusedOnly = selectMatMulLowering(noFusion, pi);
  assert(!find(unfusedOnly, "fused_scalar_baseline").valid());
  assert(!find(unfusedOnly, "fused_tiled_vector_full_tiles").valid());

  MatMulLoweringProblem padded{15, 15, 31};
  auto pad = selectMatMulLowering(padded, pi);
  const auto &paddedScalar = find(pad, "fused_scalar_baseline");
  assert(paddedScalar.valid() && paddedScalar.requiresPadding);
  assert(paddedScalar.executedM == 16 && paddedScalar.executedN == 16 &&
         paddedScalar.executedK == 32);
  const auto &paddedVector =
      find(pad, "fused_tiled_vector_materialized_tail");
  assert(paddedVector.valid() && !paddedVector.requiresPadding);
  assert(paddedVector.loweringComplete && !paddedVector.usesScalarCleanup);
  assert(paddedVector.usesMaterializedPadding);
  assert(paddedVector.paddingPolicy == PaddingPolicy::TileMaterialized);
  assert(paddedVector.zeroFillBytes > 0 && paddedVector.copyBytes > 0);
  MatMulLoweringProblem kTail{8, 8, 15};
  kTail.tileM = kTail.tileN = kTail.tileK = 8;
  auto directK = selectMatMulLowering(kTail, pi);
  const auto &materializedK =
      find(directK, "fused_tiled_vector_materialized_tail");
  const auto &direct =
      find(directK,
           "fused_tiled_vector_direct_cleanup_direct_vector_cleanup");
  assert(direct.valid() && direct.kRemainder == 7);
  assert(direct.mRemainder == 0 && direct.nRemainder == 0);
  assert(direct.usesDirectVectorCleanup);
  assert(direct.temporaryAllocationCount == 0);
  assert(materializedK.valid() && materializedK.zeroFillBytes > 0);
  assert(direct.cost.directTailNs > 0.0);
  assert(materializedK.cost.zeroFillNs > 0.0 &&
         materializedK.cost.copyNs > 0.0);
  assert(find(directK, "fused_whole_shape_vector_no_padding").valid());
  assert(find(directK,
              "fused_whole_shape_vector_materialized_padding").valid());
  assert(!find(directK,
               "fused_tiled_vector_direct_cleanup_direct_scalar_cleanup")
              .valid());
  assert(!find(directK, "fused_tiled_vector_specialized_tail").valid());
  assert(paddedVector.edgeTileCount > 0);
  assert(paddedVector.materializedPaddingBytes < 4096);
  assert(paddedVector.tiling == TilingKind::Tiled);
  assert(paddedVector.vectorization == VectorizationKind::TiledVector);
  assert(direct.paddingPolicy == PaddingPolicy::None);
  assert(!direct.requiresFullKTile && direct.requiresFullMTile &&
         direct.requiresFullNTile);

  MatMulLoweringProblem tiny{1, 1, 1};
  auto tinySelection = selectMatMulLowering(tiny, pi);
  const auto &tinyWhole =
      find(tinySelection, "fused_whole_shape_vector_no_padding");
  assert(!tinyWhole.valid());
  assert(tinyWhole.rejectionReason ==
         "whole_shape_vector_smaller_than_target_vector_width");

  MatMulLoweringProblem unsupportedPaddingCleanup = padded;
  unsupportedPaddingCleanup.paddedFusedVectorLoweringComplete = false;
  auto unsupported = selectMatMulLowering(unsupportedPaddingCleanup, pi);
  assert(!find(unsupported,
               "fused_whole_shape_vector_materialized_padding").valid());
  assert(find(unsupported,
              "fused_whole_shape_vector_materialized_padding")
             .rejectionReason ==
         "padded_fused_vector_padding_fill_lowering_unavailable");

  const auto &b = find(all, "fused_scalar_baseline");
  const double explicitSum =
      b.cost.computeNs + b.cost.memoryTrafficNs + b.cost.allocationNs +
      b.cost.paddingComputeNs + b.cost.paddingCopyNs + b.cost.cropCopyNs +
      b.cost.controlOverheadNs + b.cost.zeroFillNs + b.cost.copyNs +
      b.cost.directTailNs + b.cost.specializedTailNs +
      b.cost.codeSizePenaltyNs + b.cost.registerSpillPenaltyNs +
      b.cost.interactionCorrectionNs;
  assert(std::abs(explicitSum - b.cost.totalNs) < 1e-9);

  NativeCostCalibration tie = scalarOnly;
  tie.effectiveScalarOpsPerNs = 1.0;
  tie.effectiveBandwidthBytesPerNs = 1e30;
  tie.allocationFixedNs = 0.0;
  tie.allocationInitBytesPerNs = 1e30;
  tie.branchNs = 0.0;
  tie.internalFreeNs = 0.0;
  MatMulLoweringProblem exactTie = aligned;
  exactTie.fusedScalarRequiresTile = false;
  auto t1 = selectMatMulLowering(exactTie, tie);
  auto t2 = selectMatMulLowering(exactTie, tie);
  assert(t1.winner() && t2.winner());
  assert(t1.winner()->id == t2.winner()->id);
  assert(t1.winner()->id == "fused_scalar_baseline");
  assert(t1.usedFullComparison == t2.usedFullComparison);

  NativeCostCalibration loaded = raspberryPi5CortexA76F32Calibration();
  std::string error;
  bool loadedConfig = loadNativeCostCalibration(
      "../configs/calibration/cortex_a76_fp32_matmul_bias_relu_v1.json",
      loaded, &error);
  if (!loadedConfig)
    loadedConfig = loadNativeCostCalibration(
        "configs/calibration/cortex_a76_fp32_matmul_bias_relu_v1.json",
        loaded, &error);
  assert(loadedConfig && error.empty());
  assert(loaded.directVectorTailBaseNs == 53.0);
  assert(loaded.wholeShapeCodeSizeThresholdBytes == 16384);
  return 0;
}
