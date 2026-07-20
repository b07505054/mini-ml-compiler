#include "MatMulLatencyGBDT.h"
#include "../../configs/cost_model/cortex_a76_fp32_matmul_bias_relu_gbdt_v2/generated_model_test_vectors.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>

int main() {
  using namespace mlir::hir::gbdt_v2;
  for (size_t row = 0; row < test::kRowCount; ++row) {
    std::array<double, kFeatureCount> feature{};
    for (size_t column = 0; column < kFeatureCount; ++column)
      feature[column] = test::kFeatures[row * kFeatureCount + column];
    double prediction = 0.0;
    assert(evaluate(feature, prediction));
    assert(std::abs(prediction - test::kPredictions[row]) <= 1e-12);
  }

  std::array<double, kFeatureCount> boundary{};
  double prediction = 0.0;
  assert(evaluate(boundary, prediction));
  boundary[0] = std::numeric_limits<double>::quiet_NaN();
  assert(!evaluate(boundary, prediction));
  boundary[0] = std::numeric_limits<double>::infinity();
  assert(!evaluate(boundary, prediction));

  std::array<double, kFeatureCount> representative{};
  for (size_t column = 0; column < kFeatureCount; ++column)
    representative[column] = test::kFeatures[column];
  constexpr int iterations = 100000;
  auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    volatile bool ok = evaluate(representative, prediction);
    assert(ok);
  }
  auto elapsed = std::chrono::duration<double, std::nano>(
      std::chrono::steady_clock::now() - begin).count();
  // A deliberately loose regression ceiling: the measured value is emitted
  // by the test runner; this only catches accidental process/runtime calls.
  assert(elapsed / iterations < 10000.0);

  mlir::hir::MatMulLoweringCandidate candidate;
  candidate.kind = mlir::hir::CandidateKind::TiledVectorDirectCleanup;
  candidate.fusion = mlir::hir::FusionMode::Fused;
  candidate.schedule = mlir::hir::ScheduleKind::TiledVector;
  candidate.tiling = mlir::hir::TilingKind::Tiled;
  candidate.vectorization = mlir::hir::VectorizationKind::TiledVector;
  candidate.vectorizedDimension = mlir::hir::VectorizedDimension::Multiple;
  candidate.paddingPolicy = mlir::hir::PaddingPolicy::None;
  candidate.kTailStrategy = mlir::hir::TailStrategy::DirectVectorCleanup;
  candidate.originalM = candidate.executedM = 8;
  candidate.originalN = candidate.executedN = 8;
  candidate.originalK = candidate.executedK = 15;
  candidate.tileM = candidate.tileN = candidate.tileK = 8;
  candidate.vectorWidth = 4;
  candidate.mTileCount = candidate.nTileCount = 1;
  candidate.kTileCount = 2;
  candidate.kRemainder = 7;
  candidate.fullTileCount = 1;
  candidate.directVectorCleanupOperations = 896;
  candidate.requiresFullMTile = true;
  candidate.requiresFullNTile = true;
  candidate.requiresFullKTile = false;
  auto calibration = mlir::hir::raspberryPi5CortexA76F32Calibration();
  auto learned = mlir::hir::predictGBDTLatency(candidate, calibration);
  assert(learned.available && !learned.ood && learned.ns > 0);
  assert(candidate.temporaryAllocatedBytes == 0);
  assert(candidate.zeroFillBytes == 0 && candidate.copyBytes == 0);

  double wrapperNs[3] = {};
  for (int candidateCount : {1, 3, 6}) {
    auto wrapperBegin = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (int iteration = 0; iteration < iterations; ++iteration)
      for (int index = 0; index < candidateCount; ++index)
        sum += mlir::hir::predictGBDTLatency(candidate, calibration).ns;
    auto wrapperElapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - wrapperBegin).count();
    wrapperNs[candidateCount == 1 ? 0 : candidateCount == 3 ? 1 : 2] =
        wrapperElapsed / iterations;
    assert(sum > 0.0);
  }
  std::printf(
      "gbdt_raw_ns_per_candidate=%.3f wrapper_1_candidate_ns=%.3f "
      "wrapper_3_candidates_ns=%.3f wrapper_6_candidates_ns=%.3f\n",
      elapsed / iterations, wrapperNs[0], wrapperNs[1], wrapperNs[2]);

  auto mismatch = calibration;
  mismatch.target = "other-target";
  assert(mlir::hir::predictGBDTLatency(candidate, mismatch).reason ==
         "target_mismatch");
  candidate.dtype = "f16";
  assert(mlir::hir::predictGBDTLatency(candidate, calibration).reason ==
         "dtype_mismatch");
  candidate.dtype = "f32";
  candidate.originalM = candidate.executedM = 1025;
  assert(mlir::hir::predictGBDTLatency(candidate, calibration).reason ==
         "shape_out_of_training_domain");
  candidate.originalM = candidate.executedM = 8;
  assert(mlir::hir::predictGBDTLatency(
             candidate, calibration, "wrong-schema").reason ==
         "schema_version_mismatch");
  assert(mlir::hir::predictGBDTLatency(
             candidate, calibration, mlir::hir::gbdt_v2::kSchemaVersion,
             mlir::hir::gbdt_v2::kModelVersion, "other_operator").reason ==
         "operator_mismatch");
  return 0;
}
