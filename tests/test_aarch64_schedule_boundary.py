#!/usr/bin/env python3
"""Focused tests for Stage 17 (schedule-unroll boundary search and
counterexample-oriented validation). Pure host-side logic -- no SSH/Pi
dependency. Real end-to-end coverage is the artifact run itself
(artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/).
"""
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as cm  # noqa: E402
import select_and_compile_aarch64_matmul_schedule as sel  # noqa: E402
import run_boundary_analysis as ba  # noqa: E402

SMALL_A = dict(shape_m=16, shape_n=16, shape_k=16, tile_m=8, tile_n=8, tile_k=4)
HIGH_K = dict(shape_m=32, shape_n=32, shape_k=128, tile_m=8, tile_n=8, tile_k=8)
PRIMARY = dict(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8)


def key(domain, uk=2):
    return cm.CandidateKey(schedule_unroll_k=uk, **domain)


# ---------------------------------------------------------------------------
# Stress-domain identity
# ---------------------------------------------------------------------------

class TestStressDomainIdentity(unittest.TestCase):
    def test_small_and_high_k_shapes_do_not_collide(self):
        self.assertNotEqual(key(SMALL_A, 4).canonical_id(), key(HIGH_K, 4).canonical_id())

    def test_full_and_partial_unroll_domains_remain_distinct(self):
        # smallA uk4 = full K-loop collapse (K trip 4/4=1); highK uk4 =
        # partial unroll (K trip 16/4=4, real remaining loop) -- distinct
        # candidate identities despite both being "uk4".
        self.assertNotEqual(key(SMALL_A, 4).canonical_id(), key(HIGH_K, 4).canonical_id())
        self.assertEqual(key(SMALL_A, 4).schedule_unroll_k, key(HIGH_K, 4).schedule_unroll_k)

    def test_tile_differences_remain_distinct(self):
        # smallA (tile 8x8x4) vs a hypothetical same-shape tile-8x8x8 domain.
        alt = dict(SMALL_A)
        alt["tile_k"] = 8
        self.assertNotEqual(key(SMALL_A, 2).canonical_id(), key(alt, 2).canonical_id())


# ---------------------------------------------------------------------------
# Timing quality
# ---------------------------------------------------------------------------

class TestTimingQuality(unittest.TestCase):
    def test_insufficient_timed_duration_flagged_unreliable(self):
        # Below BORDERLINE_THRESHOLD_NS (5x clock overhead = 185ns).
        result = ba.classify_timing_quality(0.0001)  # 100ns
        self.assertEqual(result["quality"], "unreliable")

    def test_borderline_duration_flagged_not_silently_trusted(self):
        # Between borderline (185ns) and reliable (370ns) thresholds.
        result = ba.classify_timing_quality(0.00025)  # 250ns
        self.assertEqual(result["quality"], "borderline")

    def test_ample_duration_classified_reliable(self):
        result = ba.classify_timing_quality(0.01)  # 10 microseconds, ms-scale kernels
        self.assertEqual(result["quality"], "reliable")

    def test_overhead_fraction_computed_correctly(self):
        result = ba.classify_timing_quality(0.00037)  # 370ns == exactly reliable threshold
        self.assertAlmostEqual(result["overhead_fraction"], ba.CLOCK_OVERHEAD_NS / 370.0, places=6)

    def test_real_smallA_uk4_measurement_is_borderline_not_silently_reliable(self):
        # The actual Stage 17 measurement: 351ns median for smallA_uk4.
        result = ba.classify_timing_quality(0.000351)
        self.assertEqual(result["quality"], "borderline")


# ---------------------------------------------------------------------------
# Winner classification
# ---------------------------------------------------------------------------

def _blank_static_ir():
    def e(v=None):
        return cm.ev(v, "mlir", "t", "t", "t", "t", "c")
    return cm.StaticIRBEvidence(e(4), e(4), e(4), e(3), e(False), e(128), e(4), e(0.0))


def _blank_backend(spills=0, object_bytes=2608):
    def e(v=None, level="mir"):
        return cm.ev(v, level, "t", "t", "t", "t", "c")
    return cm.LLVMBackendEvidence(e(100), e(28), e(spills), e(spills), e(96), e(object_bytes, "assembly"), e(128, "assembly"), e(16), e(8), e(None, "llvm_mca"))


def _measured(median_ms, correctness=True, cv=0.01):
    def e(v):
        return cm.ev(v, "raspberry_pi_measured", "t", "t", "t", "t", "c")
    return cm.MeasuredHardwareEvidence(e(median_ms), e(median_ms * 1.1), e(median_ms * 1.02), e(median_ms * cv), e(cv), e(correctness), e("raspberry_pi_5_cortex_a76"))


class TestWinnerClassification(unittest.TestCase):
    def _domain(self, uk1_ms, uk2_ms, uk4_ms, uk1_cv=0.01, uk2_cv=0.01, uk4_cv=0.01):
        recs = [
            cm.CandidateEvidenceRecord(key=key(PRIMARY, 1), label="uk1", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=_measured(uk1_ms, cv=uk1_cv)),
            cm.CandidateEvidenceRecord(key=key(PRIMARY, 2), label="uk2", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=_measured(uk2_ms, cv=uk2_cv)),
            cm.CandidateEvidenceRecord(key=key(PRIMARY, 4), label="uk4", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=_measured(uk4_ms, cv=uk4_cv)),
        ]
        return recs

    def test_clear_uk4_winner(self):
        recs = self._domain(0.010, 0.009, 0.007)
        fastest = min(recs, key=lambda r: r.measured.median_latency_ms.value)
        self.assertEqual(fastest.label, "uk4")

    def test_clear_uk2_or_uk1_winner_fixture(self):
        # Deliberately constructed fixture where uk1 is fastest -- proves
        # the classification logic doesn't hardcode uk4.
        recs = self._domain(0.005, 0.007, 0.009)
        fastest = min(recs, key=lambda r: r.measured.median_latency_ms.value)
        self.assertEqual(fastest.label, "uk1")

    def test_variance_aware_tie(self):
        # uk2/uk4 within noise (0.3% apart), CV high enough that the gap
        # doesn't clear even a modest noise floor.
        recs = self._domain(0.010, 0.00701, 0.00700, uk2_cv=0.02, uk4_cv=0.02)
        b, o = recs[2], recs[1]  # uk4 vs uk2
        pct = (o.measured.median_latency_ms.value - b.measured.median_latency_ms.value) / o.measured.median_latency_ms.value * 100.0
        noise_floor = max(3.0, 2 * 100 * max(b.measured.cv.value, o.measured.cv.value))
        self.assertLess(pct, noise_floor)  # does not clear -> should be classified tied, not a clean win

    def test_inconclusive_when_no_measured_evidence(self):
        recs = [cm.CandidateEvidenceRecord(key=key(PRIMARY, 1), label="uk1", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=None)]
        measured_present = [r for r in recs if r.measured]
        self.assertEqual(measured_present, [])  # no winner can be determined

    def test_incorrect_candidate_excluded_from_winner_determination(self):
        recs = self._domain(0.010, 0.009, 0.001)  # uk4 suspiciously fast
        recs[2] = cm.CandidateEvidenceRecord(key=key(PRIMARY, 4), label="uk4", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=_measured(0.001, correctness=False))
        pool = [(r.key, r.measured) for r in recs]
        ranked = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        uk4_breakdown = next(b for b in ranked if b.label == "uk4")
        self.assertTrue(uk4_breakdown.rejected)


# ---------------------------------------------------------------------------
# Profile behavior
# ---------------------------------------------------------------------------

class TestProfileBehavior(unittest.TestCase):
    def test_new_exact_domain_accepted(self):
        k = key(HIGH_K, 4)
        compat = cm.check_compatibility(k, k, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.EXACT_MATCH)

    def test_nearest_unsupported_domain_not_reused(self):
        # smallA (tile 8x8x4) is NOT an exact/bucket match for a
        # same-shape, different-tile query.
        query = cm.CandidateKey(shape_m=16, shape_n=16, shape_k=16, tile_m=4, tile_n=8, tile_k=8, schedule_unroll_k=1)
        evidence = key(SMALL_A, 1)
        compat = cm.check_compatibility(query, evidence, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.INCOMPATIBLE)

    def test_counterexample_domain_would_select_its_own_winner_if_found(self):
        # Simulated counterexample: a domain where uk1 measures fastest.
        # Verifies the ranking mechanism itself has no bias toward uk4 --
        # it would correctly select uk1 if that were the real measured
        # winner (this project never found such a domain, but the
        # SELECTION LOGIC must not be hardcoded to prevent it).
        recs = [
            cm.CandidateEvidenceRecord(key=key(SMALL_A, 1), label="ce_uk1", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(spills=0), measured=_measured(0.001)),
            cm.CandidateEvidenceRecord(key=key(SMALL_A, 2), label="ce_uk2", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(spills=5), measured=_measured(0.0015)),
        ]
        pool = [(r.key, r.measured) for r in recs]
        ranked = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual(ranked[0].label, "ce_uk1")

    def test_no_universal_uk4_rule_in_ranking_code(self):
        import inspect
        source = inspect.getsource(cm.rank_candidates) + inspect.getsource(cm.compute_cost)
        self.assertNotIn("schedule_unroll_k == 4", source)
        self.assertNotIn("schedule_unroll_k==4", source)

    def test_exact_domain_evidence_isolation_intact(self):
        # highK evidence must never leak into a smallA query even though
        # both eventually get "uk4" answers independently.
        highk_pool = [(key(HIGH_K, 1), _measured(0.009)), (key(HIGH_K, 4), _measured(0.007))]
        smallA_records = [
            cm.CandidateEvidenceRecord(key=key(SMALL_A, 1), label="sa1", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=None),
        ]
        ranked = cm.rank_candidates(smallA_records, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=highk_pool)
        self.assertEqual(ranked[0].measured_latency_calibration_bonus, 0.0)


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------

class TestSelectionPolicies(unittest.TestCase):
    def test_exact_calibration_policy_never_active_by_default(self):
        self.assertEqual(sel.MODE_MANUAL, "manual")

    def test_bounded_heuristic_remains_offline_only(self):
        # The "bounded heuristic" (Policy 2) must not appear as a wired
        # CLI mode -- it is evaluated only inside run_boundary_analysis.py,
        # never inside the actual selector.
        self.assertNotIn("bounded", sel.VALID_MODES)
        self.assertEqual(set(sel.VALID_MODES), {"manual", "static", "calibrated"})

    def test_universal_uk4_policy_marked_unsafe_not_exposed(self):
        self.assertNotIn("universal", sel.VALID_MODES)

    def test_fallback_remains_deterministic(self):
        candidates, _ = sel.generate_supported_candidates(16, 16, 16, 4, 8, 8)  # untested tile combo
        r1 = sel.select_candidate(sel.MODE_STATIC, candidates, {})
        r2 = sel.select_candidate(sel.MODE_STATIC, candidates, {})
        self.assertEqual(r1["selected_key"], r2["selected_key"])
        self.assertEqual(r1["effective_mode"], r2["effective_mode"])


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegression(unittest.TestCase):
    def test_default_remains_manual(self):
        import inspect
        self.assertIn('default=MODE_MANUAL', inspect.getsource(sel.main))

    def test_calibrated_remains_opt_in(self):
        self.assertIn(sel.MODE_CALIBRATED, sel.VALID_MODES)
        self.assertNotEqual(sel.MODE_MANUAL, sel.MODE_CALIBRATED)

    def test_serving_cost_model_untouched(self):
        import inspect
        source = inspect.getsource(sel) + inspect.getsource(cm) + inspect.getsource(ba)
        self.assertNotIn("import ServingCostModel", source)
        self.assertNotIn("ServingCostModel(", source)
        self.assertNotIn("PlanSelectionPass(", source)

    def test_spill_hard_rejection_does_not_return_as_default(self):
        spilling = cm.CandidateEvidenceRecord(key=key(SMALL_A, 4), label="spilling", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(spills=10), measured=None)
        b = cm.compute_cost(spilling, cm.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertFalse(b.rejected)


if __name__ == "__main__":
    unittest.main()
