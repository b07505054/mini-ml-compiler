#pragma once

#include "MatMulLoweringCostModelV2.h"
#include "../../configs/cost_model/cortex_a76_fp32_matmul_bias_relu_gbdt_v2/generated_model.h"

#include <array>
#include <cmath>
#include <string>

namespace mlir::hir {

enum class MatMulCostModelMode { Analytical, GBDT, Hybrid };

inline const char *costModelModeName(MatMulCostModelMode mode) {
  switch (mode) {
  case MatMulCostModelMode::Analytical: return "analytical";
  case MatMulCostModelMode::GBDT: return "gbdt";
  case MatMulCostModelMode::Hybrid: return "hybrid";
  }
  return "analytical";
}

inline MatMulCostModelMode parseCostModelMode(const std::string &value) {
  if (value == "gbdt") return MatMulCostModelMode::GBDT;
  if (value == "hybrid") return MatMulCostModelMode::Hybrid;
  return MatMulCostModelMode::Analytical;
}

struct GBDTPrediction {
  bool available = false;
  bool ood = true;
  std::string reason;
  double logNs = 0.0;
  double ns = 0.0;
};

inline int candidateOneHotIndex(CandidateKind kind) {
  switch (kind) {
  case CandidateKind::ScalarBaseline: return 0;
  case CandidateKind::TiledVectorDirectCleanup: return 1;
  case CandidateKind::TiledVectorFullTiles: return 2;
  case CandidateKind::TiledVectorMaterializedTail: return 3;
  case CandidateKind::WholeShapeVectorMaterializedPadding: return 4;
  case CandidateKind::WholeShapeVectorNoPadding: return 5;
  case CandidateKind::TiledVectorSpecializedTail: return -1;
  }
  return -1;
}

inline GBDTPrediction
predictGBDTLatency(const MatMulLoweringCandidate &c,
                   const NativeCostCalibration &cal,
                   const char *schemaVersion =
                       gbdt_v2::kSchemaVersion,
                   const char *modelVersion =
                       gbdt_v2::kModelVersion,
                   const char *operatorKind = "matmul_bias_relu") {
  GBDTPrediction result;
  if (std::string(schemaVersion) != gbdt_v2::kSchemaVersion) {
    result.reason = "schema_version_mismatch";
    return result;
  }
  if (std::string(modelVersion) != gbdt_v2::kModelVersion) {
    result.reason = "model_version_mismatch";
    return result;
  }
  if (std::string(operatorKind) != "matmul_bias_relu") {
    result.reason = "operator_mismatch";
    return result;
  }
  if (cal.target != "raspberry-pi5-cortex-a76") {
    result.reason = "target_mismatch";
    return result;
  }
  if (c.dtype != "f32") {
    result.reason = "dtype_mismatch";
    return result;
  }
  if (c.fusion != FusionMode::Fused) {
    result.reason = "unfused_candidate_absent_from_training_data";
    return result;
  }
  if (c.originalM < 1 || c.originalM > 1024 || c.originalN < 1 ||
      c.originalN > 512 || c.originalK < 1 || c.originalK > 2048) {
    result.reason = "shape_out_of_training_domain";
    return result;
  }
  if (c.schedule == ScheduleKind::TiledVector &&
      (c.tileM != 8 || c.tileN != 8 || c.tileK != 8)) {
    result.reason = "tile_family_mismatch";
    return result;
  }
  int kindIndex = candidateOneHotIndex(c.kind);
  if (kindIndex < 0) {
    result.reason = "unsupported_candidate_category";
    return result;
  }

  constexpr double elementBytes = 4.0;
  const double m = c.originalM, n = c.originalN, k = c.originalK;
  const double totalFlops = 2.0 * m * n * k + 2.0 * m * n;
  if (totalFlops > 33949186.0) {
    result.reason = "flops_out_of_training_domain";
    return result;
  }
  const double inputBytes = elementBytes * (m * k + k * n + m * n);
  const double outputBytes = elementBytes * m * n;
  const double intermediateBytes =
      c.fusion == FusionMode::Fused ? 0.0 : 2.0 * m * n * elementBytes;
  const double arithmeticIntensity =
      totalFlops / (inputBytes + outputBytes + intermediateBytes);
  const bool tiled = c.schedule == ScheduleKind::TiledVector;
  const bool vector = c.schedule != ScheduleKind::Scalar;
  const bool whole = c.schedule == ScheduleKind::WholeShapeVector;
  const double paddedM =
      c.paddingPolicy == PaddingPolicy::WholeShapeMaterialized ? c.executedM : m;
  const double paddedN =
      c.paddingPolicy == PaddingPolicy::WholeShapeMaterialized ? c.executedN : n;
  const double paddedK =
      c.paddingPolicy == PaddingPolicy::WholeShapeMaterialized ? c.executedK : k;
  const double paddedFlops =
      c.paddingPolicy == PaddingPolicy::WholeShapeMaterialized
          ? 2.0 * paddedM * paddedN * paddedK + 2.0 * paddedM * paddedN
          : totalFlops;
  const double paddedElements =
      c.paddingPolicy == PaddingPolicy::WholeShapeMaterialized
          ? paddedM * paddedK + paddedK * paddedN + paddedM * paddedN -
                (m * k + k * n + m * n)
          : 0.0;
  const double vectorElements = vector ? paddedM * paddedN * paddedK : 0.0;
  const double registerRisk =
      std::min(3.0, (tiled ? 64.0 : std::max(m * n, 1.0)) / 64.0);
  if (c.temporaryAllocatedBytes > 2816) {
    result.reason = "temporary_bytes_out_of_training_domain";
    return result;
  }

  std::array<double, gbdt_v2::kFeatureCount> f{};
  size_t i = 0;
  auto add = [&](double value) { f[i++] = value; };
  add(m); add(n); add(k);
  add(std::log2(m + 1)); add(std::log2(n + 1)); add(std::log2(k + 1));
  add(m * n); add(k); add(totalFlops); add(inputBytes); add(outputBytes);
  add(arithmeticIntensity); add(1); add(!vector); add(vector); add(whole);
  add(tiled); add(c.tileM); add(c.tileN); add(c.tileK); add(c.vectorWidth);
  const bool fullTile = c.kind == CandidateKind::TiledVectorFullTiles;
  const bool directTail = c.kind == CandidateKind::TiledVectorDirectCleanup;
  add(fullTile || directTail); add(fullTile || directTail); add(fullTile);
  add(c.fullTileCount); add(c.mTileCount); add(c.nTileCount);
  add(c.kTileCount); add(c.mRemainder); add(c.nRemainder); add(c.kRemainder);
  add(c.tileM ? double(c.mRemainder) / c.tileM : 0);
  add(c.tileN ? double(c.nRemainder) / c.tileN : 0);
  add(c.tileK ? double(c.kRemainder) / c.tileK : 0);
  add(c.mRemainder ? c.nTileCount * c.kTileCount : 0);
  add(c.nRemainder ? c.mTileCount * c.kTileCount : 0);
  add(c.kRemainder ? c.mTileCount * c.nTileCount : 0);
  add(paddedM); add(paddedN); add(paddedK); add(paddedElements);
  add(paddedFlops); add(paddedFlops / totalFlops);
  add(c.temporaryAllocatedBytes); add(c.zeroFillBytes); add(c.copyBytes);
  add(intermediateBytes); add(2.0 * m * n * elementBytes);
  add(c.directVectorCleanupOperations); add(c.maskedLaneCount);
  add(512.0 + vectorElements * 2.0);
  add(512.0 + vectorElements * 4.0);
  add(64.0 + std::floor(vectorElements / 4.0));
  add(registerRisk); add(whole && m * n > 256);
  add(tiled ? c.mTileCount * c.nTileCount * c.kTileCount : 1);
  add(vector ? 10 : 8);
  for (int category = 0; category < 6; ++category) add(category == kindIndex);
  for (int category = 0; category < 3; ++category)
    add(category == (c.schedule == ScheduleKind::Scalar ? 0 :
                     c.schedule == ScheduleKind::TiledVector ? 1 : 2));
  for (int category = 0; category < 3; ++category)
    add(category == (c.schedule == ScheduleKind::Scalar ? 0 :
                     c.schedule == ScheduleKind::WholeShapeVector ? 1 : 2));
  for (int category = 0; category < 3; ++category)
    add(category == (!vector ? 0 : whole ? 1 : 2));
  for (int category = 0; category < 5; ++category)
    add(category == (!vector ? 0 : 4));
  for (int category = 0; category < 3; ++category)
    add(category == (c.paddingPolicy == PaddingPolicy::None ? 0 :
                     c.paddingPolicy == PaddingPolicy::TileMaterialized ? 1 : 2));
  auto addTail = [&](TailStrategy strategy, bool remainder) {
    int category =
        c.kind == CandidateKind::TiledVectorMaterializedTail && remainder ? 1 :
        strategy == TailStrategy::None ? 0 :
        strategy == TailStrategy::DirectScalarCleanup ? 2 :
        strategy == TailStrategy::DirectVectorCleanup ? 3 :
        strategy == TailStrategy::SpecializedMicrokernel ? 4 : 5;
    for (int value = 0; value < 6; ++value) add(value == category);
  };
  addTail(c.mTailStrategy, c.mRemainder);
  addTail(c.nTailStrategy, c.nRemainder);
  addTail(c.kTailStrategy, c.kRemainder);
  if (i != gbdt_v2::kFeatureCount) {
    result.reason = "feature_count_mismatch";
    return result;
  }
  if (!gbdt_v2::evaluate(f, result.logNs)) {
    result.reason = "missing_or_nonfinite_feature";
    return result;
  }
  result.ns = std::exp(result.logNs);
  result.available = true;
  result.ood = false;
  return result;
}

} // namespace mlir::hir
