#!/usr/bin/env python3
"""Focused tests for tools/aarch64_schedule_candidate_model.py (Stage 14).

Pure logic tests -- no SSH, no Pi, no toolchain, no Stage 12/13 artifact
dependency (synthetic fixtures throughout, so these don't overfit to one
candidate name). Real end-to-end coverage is the artifact run itself
(artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model/).
"""
import copy
import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as m  # noqa: E402


def make_key(**overrides):
    defaults = dict(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=1)
    defaults.update(overrides)
    return m.CandidateKey(**defaults)


def make_ev(value, source_level="mir", estimated=False):
    return m.ev(value, source_level, "test_artifact.json", "test_tool_v1", "aarch64-linux-gnu/cortex-a76",
                "deadbeef", "test_candidate", estimated=estimated)


def make_static_ir(**overrides):
    fields = {
        "m_trip_count": make_ev(4, "mlir"), "n_trip_count": make_ev(4, "mlir"),
        "k_trip_count_post_unroll": make_ev(4, "mlir"), "surviving_loop_count": make_ev(3, "mlir"),
        "k_loop_collapsed": make_ev(False, "mlir"), "static_vector_contract_count": make_ev(128, "llvm_ir"),
        "estimated_dynamic_k_loop_iterations": make_ev(4, "mlir", estimated=True),
        "estimated_dynamic_loop_control_reduction_pct": make_ev(0.0, "mlir", estimated=True),
    }
    fields.update(overrides)
    return m.StaticIRBEvidence(**fields)


def make_backend(spills=0, reloads=0, object_bytes=2608, stack_bytes=96, physical_vec=28, **overrides):
    fields = {
        "pre_ra_approx_peak_live_vector_registers": make_ev(113, "mir"),
        "physical_vector_registers_post_ra": make_ev(physical_vec, "mir"),
        "spill_store_count": make_ev(spills, "mir"), "reload_load_count": make_ev(reloads, "mir"),
        "stack_frame_bytes": make_ev(stack_bytes, "mir"), "object_bytes": make_ev(object_bytes, "assembly"),
        "static_fmla_asm_count": make_ev(128, "assembly"), "accumulator_chains": make_ev(16, "mir"),
        "max_accumulator_chain_length": make_ev(8, "mir"), "llvm_mca_block_rthroughput": make_ev(None, "llvm_mca"),
    }
    fields.update(overrides)
    return m.LLVMBackendEvidence(**fields)


def make_measured(median_ms=1.0, correctness=True, cv=0.01, **overrides):
    fields = {
        "median_latency_ms": make_ev(median_ms, "raspberry_pi_measured"),
        "p95_latency_ms": make_ev(median_ms * 1.1, "raspberry_pi_measured"),
        "mean_latency_ms": make_ev(median_ms * 1.02, "raspberry_pi_measured"),
        "stddev_latency_ms": make_ev(median_ms * cv, "raspberry_pi_measured"),
        "cv": make_ev(cv, "raspberry_pi_measured"),
        "correctness_pass": make_ev(correctness, "raspberry_pi_measured"),
        "hardware_identity": make_ev("raspberry_pi_5_cortex_a76", "raspberry_pi_measured"),
    }
    fields.update(overrides)
    return m.MeasuredHardwareEvidence(**fields)


def make_record(label="cand", key=None, static_ir=None, backend=None, measured=None):
    return m.CandidateEvidenceRecord(
        key=key or make_key(), label=label,
        static_ir=static_ir or make_static_ir(), llvm_backend=backend or make_backend(), measured=measured,
    )


# ---------------------------------------------------------------------------
# Candidate identity
# ---------------------------------------------------------------------------

class TestCandidateIdentity(unittest.TestCase):
    def test_identical_configuration_produces_identical_key(self):
        self.assertEqual(make_key(), make_key())
        self.assertEqual(make_key().canonical_id(), make_key().canonical_id())

    def test_different_unroll_gives_different_key(self):
        self.assertNotEqual(make_key(schedule_unroll_k=1), make_key(schedule_unroll_k=2))
        self.assertNotEqual(make_key(schedule_unroll_k=1).canonical_id(), make_key(schedule_unroll_k=2).canonical_id())

    def test_different_tile_gives_different_key(self):
        self.assertNotEqual(make_key(tile_k=8), make_key(tile_k=4))

    def test_different_shape_gives_different_key(self):
        self.assertNotEqual(make_key(shape_m=32), make_key(shape_m=64))

    def test_labels_do_not_affect_identity(self):
        r1 = make_record(label="my_candidate_A")
        r2 = make_record(label="totally_different_label_for_same_config")
        self.assertEqual(r1.key, r2.key)
        self.assertEqual(r1.canonical_id(), r2.canonical_id())
        self.assertNotEqual(r1.label, r2.label)

    def test_key_is_deterministic_and_serializable(self):
        key = make_key()
        d = key.as_dict()
        self.assertEqual(m.CandidateKey(**d), key)


# ---------------------------------------------------------------------------
# Evidence typing / provenance
# ---------------------------------------------------------------------------

class TestEvidenceTyping(unittest.TestCase):
    def test_measured_latency_cannot_be_recorded_as_llvm_mca_evidence(self):
        with self.assertRaises(ValueError):
            m.ev(1.23, "raspberry_pi_measured_but_typo", "a", "t", "x", "r", "c")
        # a correctly-typed measured value must use the real source level
        ok = m.ev(1.23, "raspberry_pi_measured", "a", "t", "x", "r", "c")
        self.assertEqual(ok.provenance.source_level, "raspberry_pi_measured")

    def test_llvm_mca_is_a_distinct_source_level_from_measured(self):
        mca_value = m.ev(128.0, "llvm_mca", "a", "t", "x", "r", "c")
        measured_value = m.ev(0.002, "raspberry_pi_measured", "a", "t", "x", "r", "c")
        self.assertNotEqual(mca_value.provenance.source_level, measured_value.provenance.source_level)

    def test_static_and_measured_evidence_remain_separate_categories(self):
        r = make_record(measured=make_measured())
        # llvm_backend and measured must never be the same object/merged dict
        self.assertIsNot(r.llvm_backend, r.measured)
        self.assertNotIn("median_latency_ms", vars(r.llvm_backend))
        self.assertNotIn("spill_store_count", vars(r.measured))

    def test_provenance_survives_serialization(self):
        r = make_record(measured=make_measured())
        d = m.record_to_dict(r)
        prov = d["llvm_backend"]["spill_store_count"]["provenance"]
        self.assertEqual(prov["source_level"], "mir")
        self.assertEqual(prov["tool_version"], "test_tool_v1")
        measured_prov = d["measured"]["median_latency_ms"]["provenance"]
        self.assertEqual(measured_prov["source_level"], "raspberry_pi_measured")

    def test_missing_evidence_is_none_not_fabricated_zero(self):
        ir = make_static_ir(estimated_dynamic_k_loop_iterations=make_ev(None, "mlir", estimated=True))
        self.assertIsNone(ir.estimated_dynamic_k_loop_iterations.value)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

class TestCompatibility(unittest.TestCase):
    def test_exact_target_accepted(self):
        q = make_key()
        result = m.check_compatibility(q, q, m.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(result["level"], m.EXACT_MATCH)
        self.assertEqual(result["confidence"], 1.0)

    def test_wrong_cpu_rejected(self):
        q = make_key()
        other = m.CandidateKey(**{**q.as_dict(), "target_cpu": "cortex-a55"})
        result = m.check_compatibility(q, other, m.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(result["level"], m.INCOMPATIBLE)
        self.assertEqual(result["confidence"], 0.0)

    def test_wrong_tile_rejected_for_exact_match_evidence(self):
        q = make_key(tile_k=8)
        other = make_key(tile_k=4)
        result = m.check_compatibility(q, other, m.BENCHMARK_METHODOLOGY_VERSION)
        self.assertNotEqual(result["level"], m.EXACT_MATCH)

    def test_same_schedule_different_shape_is_cross_shape_compatible(self):
        q = make_key(shape_m=64, shape_n=64, shape_k=64)
        other = make_key(shape_m=32, shape_n=32, shape_k=32)
        result = m.check_compatibility(q, other, m.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(result["level"], m.CROSS_SHAPE_SAME_SCHEDULE)
        self.assertGreater(result["confidence"], 0.0)
        self.assertLess(result["confidence"], 1.0)

    def test_stale_methodology_version_is_incompatible_regardless_of_everything_else(self):
        q = make_key()
        result = m.check_compatibility(q, q, "some_old_v0_methodology")
        self.assertEqual(result["level"], m.INCOMPATIBLE)
        self.assertEqual(result["confidence"], 0.0)

    def test_missing_evidence_lowers_confidence_not_silently_zero(self):
        record = make_record(static_ir=make_static_ir(estimated_dynamic_k_loop_iterations=make_ev(None, "mlir", estimated=True)))
        breakdown = m.compute_cost(record, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertIn("estimated_dynamic_k_loop_iterations", breakdown.missing_evidence_fields)
        self.assertLess(breakdown.confidence, 1.0)
        self.assertGreater(breakdown.confidence, 0.0)  # lowered, not zeroed out


# ---------------------------------------------------------------------------
# Cost logic
# ---------------------------------------------------------------------------

class TestCostLogic(unittest.TestCase):
    def test_spill_adds_penalty_but_does_not_reject_in_soft_mode(self):
        clean = make_record(backend=make_backend(spills=0))
        spilling = make_record(backend=make_backend(spills=5))
        b_clean = m.compute_cost(clean, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        b_spill = m.compute_cost(spilling, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertFalse(b_spill.rejected)
        self.assertGreater(b_spill.spill_penalty, b_clean.spill_penalty)
        self.assertGreater(b_spill.total_cost, b_clean.total_cost)

    def test_reload_adds_penalty(self):
        no_reload = make_record(backend=make_backend(reloads=0))
        reload_ = make_record(backend=make_backend(reloads=4))
        b1 = m.compute_cost(no_reload, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        b2 = m.compute_cost(reload_, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertGreater(b2.reload_penalty, b1.reload_penalty)

    def test_code_size_adds_penalty(self):
        small = make_record(backend=make_backend(object_bytes=2000))
        large = make_record(backend=make_backend(object_bytes=8000))
        b1 = m.compute_cost(small, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        b2 = m.compute_cost(large, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertGreater(b2.code_size_penalty, b1.code_size_penalty)

    def test_reduced_loop_control_can_offset_other_costs(self):
        # A candidate with a much lower loop_control_cost but a modest
        # code-size penalty should be able to beat a candidate with high
        # loop control cost and zero code growth, in the SAME static mode.
        low_loop_control = make_record(
            static_ir=make_static_ir(estimated_dynamic_k_loop_iterations=make_ev(1, "mlir", estimated=True)),
            backend=make_backend(object_bytes=6000),
        )
        high_loop_control = make_record(
            static_ir=make_static_ir(estimated_dynamic_k_loop_iterations=make_ev(50, "mlir", estimated=True)),
            backend=make_backend(object_bytes=2608),
        )
        b1 = m.compute_cost(low_loop_control, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        b2 = m.compute_cost(high_loop_control, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertLess(b1.total_cost, b2.total_cost)

    def test_measured_profitable_candidate_remains_selected_despite_spills(self):
        # Mirrors the real Stage 13 finding: a spilling candidate that
        # measures faster must win in calibrated mode.
        baseline_key = make_key(schedule_unroll_k=1)
        spilling_key = make_key(schedule_unroll_k=4)
        baseline = make_record(label="baseline", key=baseline_key, backend=make_backend(spills=0), measured=make_measured(median_ms=1.0))
        spilling_faster = make_record(label="spilling_faster", key=spilling_key, backend=make_backend(spills=11, reloads=12), measured=make_measured(median_ms=0.8))
        pool = [(baseline.key, baseline.measured), (spilling_faster.key, spilling_faster.measured)]
        ranked = m.rank_candidates([baseline, spilling_faster], m.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual(ranked[0].label, "spilling_faster")
        self.assertFalse(ranked[0].rejected)

    def test_incorrect_candidate_is_always_rejected(self):
        bad = make_record(label="bad_cand", measured=make_measured(median_ms=0.1, correctness=False))  # suspiciously fast AND wrong
        good = make_record(label="good_cand", key=make_key(schedule_unroll_k=2), measured=make_measured(median_ms=1.0, correctness=True))
        pool = [(bad.key, bad.measured), (good.key, good.measured)]
        ranked = m.rank_candidates([bad, good], m.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        bad_breakdown = next(b for b in ranked if b.label == bad.label)
        self.assertTrue(bad_breakdown.rejected)
        self.assertIn("INCORRECT", bad_breakdown.rejection_reason)
        # correctness rejection must also apply in static modes, not only calibrated
        static_breakdown = m.compute_cost(bad, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertTrue(static_breakdown.rejected)

    def test_unsupported_candidate_is_always_rejected(self):
        unsupported = make_record(backend=make_backend(object_bytes=None))
        b = m.compute_cost(unsupported, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertTrue(b.rejected)
        self.assertIn("UNSUPPORTED", b.rejection_reason)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestRanking(unittest.TestCase):
    def test_uk2_ranks_above_uk1_in_calibrated_mode_when_measured_faster(self):
        uk1 = make_record(label="uk1", key=make_key(schedule_unroll_k=1), measured=make_measured(median_ms=1.0))
        uk2 = make_record(label="uk2", key=make_key(schedule_unroll_k=2), backend=make_backend(object_bytes=3248), measured=make_measured(median_ms=0.9))
        pool = [(uk1.key, uk1.measured), (uk2.key, uk2.measured)]
        ranked = m.rank_candidates([uk1, uk2], m.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual(ranked[0].label, "uk2")
        self.assertEqual(ranked[0].rank, 1)

    def test_backend_costly_but_measured_profitable_is_representable_not_hidden(self):
        costly_but_fast = make_record(label="costly_fast", backend=make_backend(spills=2, reloads=2), measured=make_measured(median_ms=0.8))
        cls = m.full_classification(costly_but_fast.llvm_backend,
                                     make_measured(median_ms=1.0), costly_but_fast.measured)
        self.assertEqual(cls["backend_safety"], m.BACKEND_COSTLY)
        self.assertEqual(cls["hardware_confirmation"], m.HW_CONFIRMED_PROFITABLE)
        self.assertIn("Backend-costly", cls["combined_human_summary"])
        self.assertIn("profitable", cls["combined_human_summary"])

    def test_static_only_mode_does_not_pretend_to_know_unsupported_measured_behavior(self):
        r = make_record(measured=None)
        b = m.compute_cost(r, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertEqual(b.measured_latency_calibration_bonus, 0.0)
        self.assertNotIn("raspberry_pi", b.truth_boundary)  # static truth_boundary must not claim hardware truth
        self.assertIn("static", b.truth_boundary)

    def test_deterministic_tie_breaking(self):
        # Two GENUINELY DIFFERENT candidates (different canonical_id --
        # different schedule_unroll_k) whose evidence happens to compute
        # to the exact same total_cost must still produce a stable,
        # deterministic order regardless of input list order -- tie-broken
        # by canonical_id, not by whichever happened to be listed first.
        a = make_record(label="a_candidate", key=make_key(schedule_unroll_k=1))
        b = make_record(label="b_candidate", key=make_key(schedule_unroll_k=2))
        cost_a = m.compute_cost(a, m.RANKING_MODE_STATIC_SOFT_PENALTY).total_cost
        cost_b = m.compute_cost(b, m.RANKING_MODE_STATIC_SOFT_PENALTY).total_cost
        self.assertEqual(cost_a, cost_b, "fixture must produce a genuine cost tie for this test to be meaningful")
        self.assertNotEqual(a.canonical_id(), b.canonical_id())

        order1 = [x.label for x in m.rank_candidates([a, b], m.RANKING_MODE_STATIC_SOFT_PENALTY)]
        order2 = [x.label for x in m.rank_candidates([b, a], m.RANKING_MODE_STATIC_SOFT_PENALTY)]
        self.assertEqual(order1, order2, "tie-break order must not depend on input list order")
        expected_first = "a_candidate" if a.canonical_id() < b.canonical_id() else "b_candidate"
        self.assertEqual(order1[0], expected_first, "tie-break must follow canonical_id ordering")


# ---------------------------------------------------------------------------
# Regression: the old hard-rejection policy must not be the default
# ---------------------------------------------------------------------------

class TestNoRegressionToHardRejectDefault(unittest.TestCase):
    def test_default_static_mode_is_soft_penalty_not_hard_reject(self):
        spilling = make_record(backend=make_backend(spills=11, reloads=12))
        b = m.compute_cost(spilling, m.RANKING_MODE_STATIC_SOFT_PENALTY)
        self.assertFalse(b.rejected, "spill must not cause rejection under the corrected default policy")

    def test_hard_reject_mode_exists_only_for_the_comparison_experiment(self):
        spilling = make_record(backend=make_backend(spills=11, reloads=12))
        b = m.compute_cost(spilling, m.RANKING_MODE_STATIC_HARD_REJECT)
        self.assertTrue(b.rejected)
        self.assertIn("LEGACY", b.rejection_reason)

    def test_ranking_modes_available_do_not_default_to_hard_reject(self):
        self.assertNotEqual(m.RANKING_MODE_STATIC_SOFT_PENALTY, m.RANKING_MODE_STATIC_HARD_REJECT)
        weights = m.DEFAULT_WEIGHTS
        self.assertFalse(weights.hard_reject_on_any_spill)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

class TestAttribution(unittest.TestCase):
    def test_matched_pair_produces_attribution(self):
        baseline = make_record(label="base", key=make_key(schedule_unroll_k=1))
        scheduled = make_record(label="sched", key=make_key(schedule_unroll_k=2), backend=make_backend(object_bytes=3248))
        attr = m.build_attribution(baseline, scheduled)
        self.assertIn("benefit_signals", attr)
        self.assertIn("cost_signals", attr)
        self.assertIn("k_loop_dynamic_trip_count_reduction", attr["benefit_signals"])

    def test_mismatched_shape_attribution_is_rejected(self):
        baseline = make_record(key=make_key(shape_m=32, schedule_unroll_k=1))
        scheduled = make_record(key=make_key(shape_m=64, schedule_unroll_k=2))
        with self.assertRaises(ValueError):
            m.build_attribution(baseline, scheduled)

    def test_identical_unroll_attribution_is_rejected(self):
        baseline = make_record(key=make_key(schedule_unroll_k=1))
        scheduled = make_record(key=make_key(schedule_unroll_k=1))
        with self.assertRaises(ValueError):
            m.build_attribution(baseline, scheduled)


if __name__ == "__main__":
    unittest.main()
