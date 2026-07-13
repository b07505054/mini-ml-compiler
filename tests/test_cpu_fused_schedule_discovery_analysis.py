"""Unit tests for the Phase 1 CPU fused schedule discovery analysis layer.

Tests the noise-aware classification, winner-region aggregation, and
static-policy regret logic against small synthetic (but internally
consistent) measurement fixtures — not the real hardware run, which is
exercised separately by the C++/CTest smoke tests and the real discovery
run documented in DOC/result/CPU_FUSED_SCHEDULE_CANDIDATE_DISCOVERY.md.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_cpu_fused_schedule_discovery as analysis  # noqa: E402


def candidate(candidate_id: str, mean_ms: float, cv: float, passed: bool = True) -> dict:
    return {
        "candidate_id": candidate_id,
        "correctness": {"passed": passed, "max_abs_error": 0.0, "max_rel_error": 0.0,
                        "contains_nan": False, "contains_inf": False},
        "stats": {"mean_ms": mean_ms, "coefficient_of_variation": cv},
        "samples_ms": [mean_ms] * 5,
    }


class TestClassifyWorkload(unittest.TestCase):
    def test_stable_winner_when_margin_clears_noise_floor(self):
        candidates = [
            candidate("bm32", 1.0, 0.01),
            candidate("bm8", 1.05, 0.01),  # 5% margin, well above the 2% floor
        ]
        result = analysis.classify_workload(candidates)
        self.assertEqual(result["classification"], "stable_winner")
        self.assertEqual(result["oracle_winner"], "bm32")

    def test_noisy_inconclusive_when_margin_below_cv_scaled_threshold(self):
        candidates = [
            candidate("bm32", 1.000, 0.02),   # 2% CV
            candidate("bm8", 1.005, 0.02),    # 0.5% margin << 3*2%=6% threshold
        ]
        result = analysis.classify_workload(candidates)
        self.assertEqual(result["classification"], "noisy_inconclusive")

    def test_near_tie_between_half_and_full_threshold(self):
        # threshold = max(2.0, 3*cv*100); with cv=0 threshold=2.0; margin=1.2 -> near_tie
        candidates = [
            candidate("bm32", 1.000, 0.0),
            candidate("bm8", 1.012, 0.0),
        ]
        result = analysis.classify_workload(candidates)
        self.assertEqual(result["classification"], "near_tie")

    def test_correctness_failure_short_circuits_ranking(self):
        candidates = [
            candidate("bm32", 1.0, 0.01),
            candidate("bm8", 0.5, 0.01, passed=False),  # would "win" on latency but is wrong
        ]
        result = analysis.classify_workload(candidates)
        self.assertEqual(result["classification"], "correctness_failure")
        self.assertIsNone(result["oracle_winner"])
        self.assertIn("bm8", result["failing_candidates"])

    def test_noise_threshold_formula_matches_documented_rule(self):
        self.assertEqual(analysis.noise_threshold_pct(0.001, 0.001), 2.0)
        self.assertAlmostEqual(analysis.noise_threshold_pct(0.05, 0.01), 15.0)


class TestWinnerRegions(unittest.TestCase):
    def _measurements(self):
        return {
            "workloads": [
                {"workload_id": "w1", "family": "fam_a", "m": 8, "n": 8, "k": 8, "flops": 1024.0,
                 "candidates": [candidate("bm32", 1.0, 0.001), candidate("bm8", 2.0, 0.001)]},
                {"workload_id": "w2", "family": "fam_a", "m": 8, "n": 8, "k": 8, "flops": 1024.0,
                 "candidates": [candidate("bm32", 1.0, 0.001), candidate("bm8", 2.0, 0.001)]},
                {"workload_id": "w3", "family": "fam_b", "m": 8, "n": 8, "k": 8, "flops": 1024.0,
                 "candidates": [candidate("bm8", 1.0, 0.001), candidate("bm32", 2.0, 0.001)]},
            ]
        }

    def test_dominant_candidate_and_distinct_stable_winners(self):
        measurements = self._measurements()
        oracle_winners = analysis.build_oracle_winners(measurements)
        regions = analysis.build_winner_regions(oracle_winners, ["bm32", "bm8"])
        self.assertEqual(regions["distinct_stable_winner_count"], 2)
        self.assertEqual(set(regions["candidates_with_at_least_one_stable_win"]), {"bm32", "bm8"})
        self.assertIn(regions["dominant_candidate"], {"bm32", "bm8"})
        self.assertEqual(regions["by_family"]["fam_a"]["stable_winner_counts"]["bm32"], 2)
        self.assertEqual(regions["by_family"]["fam_b"]["stable_winner_counts"]["bm8"], 1)

    def test_single_dominant_candidate_detected(self):
        measurements = {
            "workloads": [
                {"workload_id": f"w{i}", "family": "fam", "m": 8, "n": 8, "k": 8, "flops": 1024.0,
                 "candidates": [candidate("bm32", 1.0, 0.001), candidate("bm8", 5.0, 0.001)]}
                for i in range(5)
            ]
        }
        oracle_winners = analysis.build_oracle_winners(measurements)
        regions = analysis.build_winner_regions(oracle_winners, ["bm32", "bm8"])
        self.assertEqual(regions["distinct_stable_winner_count"], 1)
        self.assertEqual(regions["dominant_candidate"], "bm32")
        self.assertEqual(regions["dominant_candidate_stable_share_pct"], 100.0)


class TestStaticPolicyComparison(unittest.TestCase):
    def test_regret_zero_for_always_oracle_policy(self):
        measurements = {
            "workloads": [
                {"workload_id": "w1", "family": "f", "m": 1, "n": 1, "k": 1, "flops": 1.0,
                 "candidates": [candidate("bm32", 1.0, 0.0), candidate("bm8", 2.0, 0.0)]},
            ]
        }
        static_policy = analysis.build_static_policy_comparison(measurements, ["bm32", "bm8"])
        self.assertEqual(static_policy["policies"]["bm32"]["mean_regret"], 0.0)
        self.assertAlmostEqual(static_policy["policies"]["bm8"]["mean_regret"], 1.0)  # 100% slower

    def test_correctness_failure_workloads_excluded(self):
        measurements = {
            "workloads": [
                {"workload_id": "w1", "family": "f", "m": 1, "n": 1, "k": 1, "flops": 1.0,
                 "candidates": [candidate("bm32", 1.0, 0.0), candidate("bm8", 2.0, 0.0, passed=False)]},
            ]
        }
        static_policy = analysis.build_static_policy_comparison(measurements, ["bm32", "bm8"])
        self.assertEqual(static_policy["excluded_correctness_failure_workloads"], 1)
        self.assertEqual(static_policy["policies"]["bm32"]["workload_count"], 0)


class TestSummaryVerdict(unittest.TestCase):
    def _env(self):
        return {
            "cpu_model": {"value": "test-cpu"}, "os": {"value": "test-os"},
            "arch": {"value": "test-arch"}, "compiler": {"value": "test-compiler"},
            "benchmark_thread_count": {"value": "1"},
        }

    def test_verdict_success_when_two_distinct_stable_winners(self):
        measurements = {
            "workloads": [
                {"workload_id": "w1", "family": "fam_a", "m": 1, "n": 1, "k": 1, "flops": 1.0,
                 "candidates": [candidate("bm32", 1.0, 0.0), candidate("bm8", 2.0, 0.0)]},
                {"workload_id": "w2", "family": "fam_b", "m": 1, "n": 1, "k": 1, "flops": 1.0,
                 "candidates": [candidate("bm8", 1.0, 0.0), candidate("bm32", 2.0, 0.0)]},
            ],
            "fusion_attribution_baseline": {},
        }
        oracle_winners = analysis.build_oracle_winners(measurements)
        regions = analysis.build_winner_regions(oracle_winners, ["bm32", "bm8"])
        static_policy = analysis.build_static_policy_comparison(measurements, ["bm32", "bm8"])
        summary = analysis.build_summary(
            measurements, oracle_winners, regions, static_policy,
            self._env(), {"candidates": [{}, {}]},
            {"total_override_count": 0},
        )
        self.assertEqual(summary["phase1_verdict"], "SUCCESS")

    def test_verdict_failed_foundation_when_one_candidate_dominates(self):
        measurements = {
            "workloads": [
                {"workload_id": f"w{i}", "family": "fam", "m": 1, "n": 1, "k": 1, "flops": 1.0,
                 "candidates": [candidate("bm32", 1.0, 0.0), candidate("bm8", 5.0, 0.0)]}
                for i in range(10)
            ],
            "fusion_attribution_baseline": {},
        }
        oracle_winners = analysis.build_oracle_winners(measurements)
        regions = analysis.build_winner_regions(oracle_winners, ["bm32", "bm8"])
        static_policy = analysis.build_static_policy_comparison(measurements, ["bm32", "bm8"])
        summary = analysis.build_summary(
            measurements, oracle_winners, regions, static_policy,
            self._env(), {"candidates": [{}, {}]},
            {"total_override_count": 0},
        )
        self.assertEqual(summary["phase1_verdict"], "FAILED_FOUNDATION")

    def test_verdict_inconclusive_when_no_stable_winners(self):
        measurements = {
            "workloads": [
                {"workload_id": "w1", "family": "fam", "m": 1, "n": 1, "k": 1, "flops": 1.0,
                 "candidates": [candidate("bm32", 1.000, 0.02), candidate("bm8", 1.005, 0.02)]},
            ],
            "fusion_attribution_baseline": {},
        }
        oracle_winners = analysis.build_oracle_winners(measurements)
        regions = analysis.build_winner_regions(oracle_winners, ["bm32", "bm8"])
        static_policy = analysis.build_static_policy_comparison(measurements, ["bm32", "bm8"])
        summary = analysis.build_summary(
            measurements, oracle_winners, regions, static_policy,
            self._env(), {"candidates": [{}, {}]},
            {"total_override_count": 0},
        )
        self.assertEqual(summary["phase1_verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
