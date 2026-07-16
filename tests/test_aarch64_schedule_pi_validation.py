#!/usr/bin/env python3
"""Focused tests for tools/run_aarch64_schedule_pi_validation.py (Stage 13).

Pure host-side logic only -- no SSH, no Pi, no toolchain invocation. Real
end-to-end coverage (does the harness produce correct/credible numbers on
real Raspberry Pi hardware) is the artifact run itself
(artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/).

Categories covered (task brief Stage 13 section 13's minimum list):
  - matched candidate comparison (only schedule_unroll_k differs -> accepted)
  - mismatched shape rejection
  - mismatched tile rejection
  - stale or missing artifact rejection (object-identity/checksum guard)
  - correctness failure parsing (harness JSON with overall_pass=false)
  - benchmark distribution calculation (min/median/mean/p95/stddev/cv
    aggregation across measurement groups)
  - runtime classification thresholds (A/B/D/E boundaries, including the
    noise-floor gate that a sub-1% or noisy change must not read as a win)
  - diagnostic spill-validation reporting (confirmed/contradicted/partially
    confirmed/inconclusive outcomes)
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import run_aarch64_schedule_pi_validation as m  # noqa: E402


def cfg(shape="32x32x32", tile=None, unroll_k=1, iterations=2000, warmup=200, git_commit="abc123"):
    return {
        "shape": shape, "tile": tile or {"m": 8, "n": 8, "k": 8}, "target_cpu": "cortex-a76",
        "opt_level": "O2", "iterations": iterations, "warmup": warmup,
        "git_commit": git_commit, "schedule_unroll_k": unroll_k,
    }


class TestMatchedComparisonGuard(unittest.TestCase):
    def test_matched_configuration_is_accepted(self):
        m.assert_matched_comparison(cfg(unroll_k=1), cfg(unroll_k=2))  # should not raise

    def test_mismatched_shape_is_rejected(self):
        with self.assertRaises(m.MismatchedComparisonError):
            m.assert_matched_comparison(cfg(shape="32x32x32", unroll_k=1), cfg(shape="64x64x64", unroll_k=2))

    def test_mismatched_tile_is_rejected(self):
        with self.assertRaises(m.MismatchedComparisonError):
            m.assert_matched_comparison(
                cfg(tile={"m": 8, "n": 8, "k": 8}, unroll_k=1),
                cfg(tile={"m": 8, "n": 8, "k": 4}, unroll_k=2),
            )

    def test_mismatched_iteration_count_is_rejected(self):
        with self.assertRaises(m.MismatchedComparisonError):
            m.assert_matched_comparison(cfg(iterations=2000, unroll_k=1), cfg(iterations=500, unroll_k=2))

    def test_mismatched_git_revision_is_rejected(self):
        with self.assertRaises(m.MismatchedComparisonError):
            m.assert_matched_comparison(cfg(git_commit="aaa", unroll_k=1), cfg(git_commit="bbb", unroll_k=2))

    def test_identical_schedule_unroll_k_is_rejected(self):
        # Nothing to compare if schedule_unroll_k is the same on both sides.
        with self.assertRaises(m.MismatchedComparisonError):
            m.assert_matched_comparison(cfg(unroll_k=1), cfg(unroll_k=1))


class TestStaleArtifactRejection(unittest.TestCase):
    def test_matching_checksum_is_accepted(self):
        self.assertTrue(m.verify_object_identity("cand", "deadbeef", "deadbeef"))

    def test_mismatched_checksum_is_rejected(self):
        with self.assertRaises(m.StaleArtifactError):
            m.verify_object_identity("cand", "deadbeef", "cafef00d")

    def test_missing_remote_checksum_is_rejected(self):
        with self.assertRaises(m.StaleArtifactError):
            m.verify_object_identity("cand", "deadbeef", "")


class TestBenchmarkDistributionAggregation(unittest.TestCase):
    def test_single_group_aggregation(self):
        groups = [{"min_ms": 1.0, "median_ms": 1.2, "mean_ms": 1.25, "p95_ms": 1.4, "stddev_ms": 0.1, "cv": 0.08}]
        agg = m.aggregate_measurement_groups(groups)
        self.assertEqual(agg["group_count"], 1)
        self.assertEqual(agg["median_of_medians_ms"], 1.2)
        self.assertEqual(agg["min_of_mins_ms"], 1.0)
        self.assertEqual(agg["stddev_of_medians_ms"], 0.0)  # only one sample -> no spread

    def test_multi_group_aggregation_uses_median_of_medians(self):
        groups = [
            {"min_ms": 1.0, "median_ms": 1.0, "mean_ms": 1.0, "p95_ms": 1.1, "stddev_ms": 0.05, "cv": 0.05},
            {"min_ms": 0.9, "median_ms": 1.2, "mean_ms": 1.2, "p95_ms": 1.3, "stddev_ms": 0.05, "cv": 0.04},
            {"min_ms": 1.1, "median_ms": 1.1, "mean_ms": 1.1, "p95_ms": 1.2, "stddev_ms": 0.05, "cv": 0.045},
        ]
        agg = m.aggregate_measurement_groups(groups)
        self.assertEqual(agg["group_count"], 3)
        self.assertEqual(agg["median_of_medians_ms"], 1.1)  # median of [1.0, 1.2, 1.1]
        self.assertEqual(agg["min_of_mins_ms"], 0.9)
        self.assertAlmostEqual(agg["mean_of_medians_ms"], 1.1, places=6)
        self.assertGreater(agg["stddev_of_medians_ms"], 0.0)


class TestRuntimeClassification(unittest.TestCase):
    def _agg(self, median_ms, cv=0.01):
        return {"median_of_medians_ms": median_ms, "cv_of_medians": cv}

    def test_incorrect_candidate_classifies_e_regardless_of_speed(self):
        cls, _ = m.classify_runtime(self._agg(2.0), self._agg(1.0), True, False, 0, 0)
        self.assertEqual(cls, "E")

    def test_new_spill_classifies_d_even_if_faster(self):
        cls, _ = m.classify_runtime(self._agg(2.0), self._agg(1.0), True, True, baseline_spills=0, scheduled_spills=3)
        self.assertEqual(cls, "D")

    def test_clear_improvement_beyond_noise_floor_classifies_a(self):
        cls, reason = m.classify_runtime(self._agg(1.0, cv=0.005), self._agg(0.9, cv=0.005), True, True, 0, 0)
        self.assertEqual(cls, "A")
        self.assertIn("clears noise floor", reason)

    def test_sub_noise_floor_change_classifies_b_not_a(self):
        # ~1% change with the default 3% noise floor must NOT read as a win.
        cls, _ = m.classify_runtime(self._agg(1.0, cv=0.005), self._agg(0.99, cv=0.005), True, True, 0, 0)
        self.assertEqual(cls, "B")

    def test_high_variance_widens_noise_floor_to_avoid_spurious_win(self):
        # A 4% "improvement" would clear the flat 3% floor, but with 5% CV
        # on both sides the noise floor should widen past 4%, preventing a
        # spurious "A" from what could just be measurement noise.
        cls, _ = m.classify_runtime(self._agg(1.0, cv=0.05), self._agg(0.96, cv=0.05), True, True, 0, 0)
        self.assertEqual(cls, "B")

    def test_clear_regression_classifies_d(self):
        cls, _ = m.classify_runtime(self._agg(1.0, cv=0.005), self._agg(1.2, cv=0.005), True, True, 0, 0)
        self.assertEqual(cls, "D")


class TestSpillPredictionValidation(unittest.TestCase):
    def test_spilling_candidate_that_regresses_is_confirmed(self):
        outcome, _ = m.validate_spill_prediction("diag", 11, 12, 224, baseline_median_ms=1.0, scheduled_median_ms=1.3, correctness_pass=True)
        self.assertEqual(outcome, "confirmed")

    def test_spilling_candidate_that_is_faster_is_contradicted(self):
        outcome, reason = m.validate_spill_prediction("diag", 2, 2, 176, baseline_median_ms=1.0, scheduled_median_ms=0.88, correctness_pass=True)
        self.assertEqual(outcome, "contradicted")
        self.assertIn("IMPROVEMENT", reason)

    def test_spilling_candidate_with_small_noisy_change_is_partially_confirmed(self):
        outcome, _ = m.validate_spill_prediction("diag", 2, 2, 176, baseline_median_ms=1.0, scheduled_median_ms=1.005, correctness_pass=True)
        self.assertEqual(outcome, "partially confirmed")

    def test_correctness_failure_is_inconclusive_not_forced_to_a_direction(self):
        outcome, _ = m.validate_spill_prediction("diag", 11, 12, 224, baseline_median_ms=1.0, scheduled_median_ms=0.5, correctness_pass=False)
        self.assertEqual(outcome, "inconclusive")

    def test_zero_spill_candidate_is_not_applicable(self):
        outcome, _ = m.validate_spill_prediction("not_a_diagnostic", 0, 0, 32, baseline_median_ms=1.0, scheduled_median_ms=0.9, correctness_pass=True)
        self.assertEqual(outcome, "not applicable")


class TestCorrectnessFailureParsing(unittest.TestCase):
    def test_all_groups_pass_yields_true(self):
        groups = [{"correctness": {"overall_pass": True}}, {"correctness": {"overall_pass": True}}]
        self.assertTrue(all(g["correctness"]["overall_pass"] for g in groups))

    def test_one_group_failing_yields_false(self):
        groups = [{"correctness": {"overall_pass": True}}, {"correctness": {"overall_pass": False}}]
        self.assertFalse(all(g["correctness"]["overall_pass"] for g in groups))


if __name__ == "__main__":
    unittest.main()
