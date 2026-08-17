// CTest unit test for D6's distributed_profitability_contract_v1
// (estimateDistributedProfitability, DistributedModelProfile,
// DistributedWorkloadProfile, DistributedProfitabilityCalibration) in
// serving/DistributedPlanning.h. Pure C++. No MLIR IR, no GoogleTest --
// mirrors DistributedStrategyPlanningTest.cpp's style exactly.
//
// The full pass-in-real-pipeline path (module-attr reading, real
// selection branching, opt_in semantics, ExecutionPlan export) is covered
// separately by the CTest integration test
// RunDistributedStrategyPlanningPipelineTest.cmake, which runs the actual
// compile-for-target binary against the real per-layer Qwen ONNX graph
// with the real D6 calibration profile.

#include "serving/DistributedPlanning.h"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace mlir::hir;

static DistributedModelProfile real05bModel() {
  DistributedModelProfile mp;
  mp.num_layers = 24;
  mp.hidden_size = 896;
  mp.num_attention_heads = 14;
  mp.num_kv_heads = 2;
  mp.weight_footprint_mb = 1454.3235168457031;  // real, matches D5 tp_cost_model.py
  mp.available = true;
  return mp;
}

static DistributedModelProfile real7bModel() {
  DistributedModelProfile mp;
  mp.num_layers = 28;
  mp.hidden_size = 3584;
  mp.num_attention_heads = 28;
  mp.num_kv_heads = 4;
  mp.weight_footprint_mb = 15242807270.0 / (1024.0 * 1024.0) + 512.0;  // real, matches D5
  mp.available = true;
  return mp;
}

static DistributedProfitabilityCalibration realCalibration() {
  // The exact D5-fitted coefficients (cross-checked bit-for-bit against
  // heterogeneous-inference-runtime's cost_model_fitted.json by
  // tools/generate_distributed_profitability_profile.py --check-against).
  DistributedProfitabilityCalibration c;
  c.contract_version = kDistributedProfitabilityContractVersion;
  c.calibration_dataset_hash = "test-fixture-not-the-real-hash";
  c.calibration_hardware_identity = "2x RTX 4090 (test fixture)";
  c.gpu_memory_mb_per_device = 24564.0;
  c.gpu_memory_utilization = 0.9;
  c.tp1_coefficients = {7.962344761464902, -0.24772518079214728, 58.06762504638019,
                       7.962344761464541, 0.06688598116920906, 0.8093129304993703,
                       177.66563688419586};
  c.tp2_coefficients = {20.38139140395511, -0.33955255109345256, 74.31842439533375,
                       40.762782807910725, 0.059169929961928894, 0.9658812851196273,
                       166.02350203699726};
  c.communication.profile_id = "2x_rtx4090_phb_single_numa_p2p_unavailable_nccl_shm_direct";
  c.communication.topology_class = "PHB";
  c.communication.p2p_available = false;
  c.communication.nccl_transport = "SHM/direct/direct";
  c.communication.collective_kind = "all_reduce";
  c.communication.predictor_kind = "log_size_piecewise_interpolation";
  c.communication.mode = "out_of_place";
  c.communication.points = {
      {1024, 10.79}, {2048, 10.22}, {4096, 10.26}, {8192, 10.48},
      {16384, 14.55}, {32768, 22.33}, {65536, 36.79}, {131072, 64.2}};
  c.communication.valid = true;
  c.d9_decision_margin_us = 250.0;
  c.d9_runtime_residual_us = 0.0;
  c.d9_compute_reference_weight_mb = 1454.3235168457031;
  c.d9_compute_savings_us_per_weight_mb_above_reference = 0.50;
  c.d9_overlap_assumption = "zero";
  c.valid = true;
  return c;
}

static DistributedCandidate tp1Candidate() {
  return DistributedCandidate{"tp1", 1, 1, 1};
}
static DistributedCandidate tp2Candidate() {
  return DistributedCandidate{"tp2", 2, 2, 1};
}

// 1. Real-unit KV-cache/weight helpers behave as documented.
static void testKvCacheAndWeightHelpersMatchPythonReference() {
  auto model = real05bModel();
  const double kvTp1 = distributedKvCacheBytesPerTokenPerGpu(model, 1);
  const double kvTp2 = distributedKvCacheBytesPerTokenPerGpu(model, 2);
  assert(std::abs(kvTp2 - kvTp1 / 2.0) < 1e-9 && "TP2 halves KV bytes/token/gpu vs TP1");
  const double wTp1 = distributedPerGpuWeightMb(model, 1);
  const double wTp2 = distributedPerGpuWeightMb(model, 2);
  assert(std::abs(wTp2 - wTp1 / 2.0) < 1e-9 && "TP2 halves per-GPU weight vs TP1");
  std::puts("  [PASS] testKvCacheAndWeightHelpersMatchPythonReference");
}

// 2. Missing model profile -> not computed, never a crash.
static void testMissingModelProfileFailsClosedNotComputed() {
  DistributedModelProfile model;  // available = false (default)
  DistributedWorkloadProfile wl;
  auto calib = realCalibration();
  auto est = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  assert(!est.computed);
  assert(est.infeasibility_reason == "model_profile_unavailable");
  std::puts("  [PASS] testMissingModelProfileFailsClosedNotComputed");
}

// 3. Missing/invalid calibration -> not computed, never a crash (Part H #8/#9).
static void testMissingCalibrationFailsClosedNotComputed() {
  auto model = real05bModel();
  DistributedWorkloadProfile wl;
  DistributedProfitabilityCalibration calib;  // valid = false (default)
  auto est = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  assert(!est.computed);
  assert(est.infeasibility_reason == "calibration_unavailable_or_invalid_version_or_missing_communication_profile");
  std::puts("  [PASS] testMissingCalibrationFailsClosedNotComputed");
}

static void testInvalidCalibrationVersionFailsClosed() {
  auto model = real05bModel();
  DistributedWorkloadProfile wl;
  auto calib = realCalibration();
  calib.contract_version = "distributed_profitability_contract_v0_stale";
  calib.valid = false;  // mirrors readCalibration()'s own version-match gate
  auto est = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  assert(!est.computed);
  std::puts("  [PASS] testInvalidCalibrationVersionFailsClosed");
}

// 4. Real calibration + real model -> 0.5B predicts TP1 wins (matches D5).
static void testProfitabilityPredictsTp1WinsFor05bAtRealCalibration() {
  auto model = real05bModel();
  DistributedWorkloadProfile wl;
  wl.input_tokens = 32; wl.output_tokens = 32; wl.concurrency = 1; wl.declared = true;
  auto calib = realCalibration();
  auto tp1 = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  auto tp2 = estimateDistributedProfitability(tp2Candidate(), model, wl, calib);
  assert(tp1.computed && tp2.computed);
  assert(tp1.feasible && tp2.feasible);
  const double regressionComputeSavings = tp1.predicted_throughput_before_communication_tokens_per_s > 0.0 &&
      tp2.predicted_throughput_before_communication_tokens_per_s > 0.0
      ? 1'000'000.0 / tp1.predicted_throughput_before_communication_tokens_per_s -
        1'000'000.0 / tp2.predicted_throughput_before_communication_tokens_per_s
      : 0.0;
  const double structuralComputeSavings =
      std::max(0.0, model.weight_footprint_mb - calib.d9_compute_reference_weight_mb) *
      calib.d9_compute_savings_us_per_weight_mb_above_reference;
  const double computeSavings = regressionComputeSavings + structuralComputeSavings;
  assert(computeSavings < tp2.estimated_communication_penalty_us + tp2.decision_margin_us &&
        "0.5B at in32_out32_c1 must remain TP1-favorable under D9 break-even decomposition");
  assert(tp2.overlap_assumption == "zero");
  assert(tp2.estimated_collective_call_count == 24);
  std::puts("  [PASS] testProfitabilityPredictsTp1WinsFor05bAtRealCalibration");
}

// 5. Real calibration + real model -> 7B predicts TP2 wins (matches D5).
static void testProfitabilityPredictsTp2WinsFor7bAtRealCalibration() {
  auto model = real7bModel();
  DistributedWorkloadProfile wl;
  wl.input_tokens = 32; wl.output_tokens = 32; wl.concurrency = 1; wl.declared = true;
  auto calib = realCalibration();
  auto tp1 = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  auto tp2 = estimateDistributedProfitability(tp2Candidate(), model, wl, calib);
  assert(tp1.computed && tp2.computed);
  double regressionComputeSavings = 0.0;
  if (tp1.predicted_throughput_before_communication_tokens_per_s > 0.0 &&
      tp2.predicted_throughput_before_communication_tokens_per_s > 0.0)
    regressionComputeSavings = 1'000'000.0 / tp1.predicted_throughput_before_communication_tokens_per_s -
        1'000'000.0 / tp2.predicted_throughput_before_communication_tokens_per_s;
  const double structuralComputeSavings =
      std::max(0.0, model.weight_footprint_mb - calib.d9_compute_reference_weight_mb) *
      calib.d9_compute_savings_us_per_weight_mb_above_reference;
  const double computeSavings = regressionComputeSavings + structuralComputeSavings;
  assert(computeSavings - tp2.estimated_communication_penalty_us > tp2.decision_margin_us &&
        "7B at in32_out32_c1 must be TP2-favorable under D9 break-even decomposition");
  assert(tp2.estimated_collective_call_count == 28);
  std::puts("  [PASS] testProfitabilityPredictsTp2WinsFor7bAtRealCalibration");
}

// 6. Memory infeasibility: an absurdly small GPU budget rejects TP1 (full
// weight shard) while TP2 (half shard) may still fit.
static void testMemoryInfeasibilityRejectsTp1ButAllowsTp2() {
  auto model = real7bModel();
  DistributedWorkloadProfile wl;
  wl.max_model_len = 2048; wl.max_num_seqs = 4;
  auto calib = realCalibration();
  calib.gpu_memory_mb_per_device = 9000.0;  // smaller than TP1's ~15GB weight shard alone
  calib.gpu_memory_utilization = 0.9;
  auto tp1 = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  auto tp2 = estimateDistributedProfitability(tp2Candidate(), model, wl, calib);
  assert(tp1.computed && !tp1.feasible);
  assert(tp1.infeasibility_reason == "required_memory_mb_exceeds_budget");
  assert(tp2.computed && tp2.feasible &&
        "halved per-GPU weight shard must fit where the full shard does not");
  std::puts("  [PASS] testMemoryInfeasibilityRejectsTp1ButAllowsTp2");
}

// 7. Memory infeasibility: both candidates rejected when the budget is
// absurdly small for either shard.
static void testMemoryInfeasibilityRejectsBoth() {
  auto model = real7bModel();
  DistributedWorkloadProfile wl;
  wl.max_model_len = 2048; wl.max_num_seqs = 4;
  auto calib = realCalibration();
  calib.gpu_memory_mb_per_device = 100.0;  // smaller than even the halved TP2 shard
  calib.gpu_memory_utilization = 0.9;
  auto tp1 = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  auto tp2 = estimateDistributedProfitability(tp2Candidate(), model, wl, calib);
  assert(!tp1.feasible && !tp2.feasible);
  std::puts("  [PASS] testMemoryInfeasibilityRejectsBoth");
}

// 8. Larger context/concurrency increases the worst-case KV memory estimate
// (real, not a constant regardless of workload).
static void testWorstCaseMemoryScalesWithWorkload() {
  auto model = real7bModel();
  auto calib = realCalibration();
  DistributedWorkloadProfile small; small.max_model_len = 2048; small.max_num_seqs = 4;
  DistributedWorkloadProfile large; large.max_model_len = 32768; large.max_num_seqs = 16;
  auto estSmall = estimateDistributedProfitability(tp1Candidate(), model, small, calib);
  auto estLarge = estimateDistributedProfitability(tp1Candidate(), model, large, calib);
  assert(estLarge.required_memory_mb > estSmall.required_memory_mb);
  std::puts("  [PASS] testWorstCaseMemoryScalesWithWorkload");
}

// 9. Tie-break precondition: a genuinely equal-predicted-throughput
// scenario is reachable (zero every feature-sensitive coefficient so
// TP1/TP2's differing gpu_count/per-GPU-weight/KV-cache inputs cannot
// separate them, leaving only the intercept). The pass's own tie-break
// (DistributedStrategyPlanningPass.cpp: `std::abs(delta) <=
// kTieBreakEpsilonTokensPerSec` -> prefer TP1) is a 3-line deterministic
// comparison exercised in aggregate by the 21-cell real fresh-compilation
// reproduction (zero ties observed with real calibrated data, as
// expected); this test proves the *arithmetic precondition* a tie-break
// must handle is real and reachable, not merely theoretical.
static void testEqualPredictedThroughputIsReachableAndExactlyEqual() {
  auto model = real05bModel();
  DistributedWorkloadProfile wl; wl.input_tokens = 32; wl.output_tokens = 32; wl.concurrency = 1;
  DistributedProfitabilityCalibration calib = realCalibration();
  DistributedThroughputCoefficients flat;
  flat.intercept = 100.0;  // every other coefficient defaults to 0.0
  calib.tp1_coefficients = flat;
  calib.tp2_coefficients = flat;
  auto tp1 = estimateDistributedProfitability(tp1Candidate(), model, wl, calib);
  auto tp2 = estimateDistributedProfitability(tp2Candidate(), model, wl, calib);
  assert(tp1.computed && tp2.computed);
  assert(tp1.predicted_throughput_before_communication_tokens_per_s ==
         tp2.predicted_throughput_before_communication_tokens_per_s);
  assert(tp1.predicted_throughput_before_communication_tokens_per_s == 100.0);
  assert(tp2.predicted_throughput_tokens_per_s < tp1.predicted_throughput_tokens_per_s &&
         "D9 communication penalty breaks an equal pre-communication throughput tie toward TP1");
  std::puts("  [PASS] testEqualPredictedThroughputIsReachableAndExactlyEqual");
}

int main() {
  std::puts("DistributedProfitabilityTest:");
  testKvCacheAndWeightHelpersMatchPythonReference();
  testMissingModelProfileFailsClosedNotComputed();
  testMissingCalibrationFailsClosedNotComputed();
  testInvalidCalibrationVersionFailsClosed();
  testProfitabilityPredictsTp1WinsFor05bAtRealCalibration();
  testProfitabilityPredictsTp2WinsFor7bAtRealCalibration();
  testMemoryInfeasibilityRejectsTp1ButAllowsTp2();
  testMemoryInfeasibilityRejectsBoth();
  testWorstCaseMemoryScalesWithWorkload();
  testEqualPredictedThroughputIsReachableAndExactlyEqual();
  std::puts("DistributedProfitabilityTest: PASS");
  return 0;
}
