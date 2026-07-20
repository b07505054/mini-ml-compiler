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
  assert(all.candidates.size() == 4);
  for (const auto &c : all.candidates)
    assert(c.valid());
  assert(find(all, "unfused_scalar").intermediateBytes >
         find(all, "fused_scalar").intermediateBytes);
  assert(find(all, "unfused_vectorized").cost.computeNs <
         find(all, "unfused_scalar").cost.computeNs);

  NativeCostCalibration scalarOnly = pi;
  scalarOnly.supportsVector = false;
  auto noVector = selectMatMulLowering(aligned, scalarOnly);
  assert(!find(noVector, "unfused_vectorized").valid());
  assert(!find(noVector, "fused_vectorized").valid());

  MatMulLoweringProblem noFusion = aligned;
  noFusion.fusionLegal = false;
  noFusion.fusionRejectionReason = "matmul_result_not_one_use";
  auto unfusedOnly = selectMatMulLowering(noFusion, pi);
  assert(!find(unfusedOnly, "fused_scalar").valid());
  assert(!find(unfusedOnly, "fused_vectorized").valid());

  MatMulLoweringProblem padded{15, 15, 31};
  auto pad = selectMatMulLowering(padded, pi);
  const auto &paddedScalar = find(pad, "fused_scalar");
  assert(paddedScalar.valid() && paddedScalar.requiresPadding);
  assert(paddedScalar.executedM == 16 && paddedScalar.executedN == 16 &&
         paddedScalar.executedK == 32);
  assert(!find(pad, "fused_vectorized").valid());
  assert(pad.winner() && !pad.winner()->requiresPadding);

  const auto &b = find(all, "fused_scalar");
  const double explicitSum =
      b.cost.computeNs + b.cost.memoryTrafficNs + b.cost.allocationNs +
      b.cost.paddingComputeNs + b.cost.paddingCopyNs + b.cost.cropCopyNs +
      b.cost.controlOverheadNs + b.cost.interactionCorrectionNs;
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
  assert(t1.winner()->id == "fused_scalar");
  return 0;
}
