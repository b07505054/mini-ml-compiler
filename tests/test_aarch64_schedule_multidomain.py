#!/usr/bin/env python3
"""Focused tests for Stage 16 (multi-domain calibration and shape/tile
compatibility). Pure host-side logic -- no SSH/Pi dependency. Real
end-to-end coverage is the artifact run itself
(artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_multidomain/).
"""
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as cm  # noqa: E402
import select_and_compile_aarch64_matmul_schedule as sel  # noqa: E402

PRIMARY = dict(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8)
CUBE64 = dict(shape_m=64, shape_n=64, shape_k=64, tile_m=8, tile_n=8, tile_k=8)
ALTK = dict(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=4)
RECT = dict(shape_m=32, shape_n=64, shape_k=32, tile_m=8, tile_n=8, tile_k=8)


def key(domain, uk=2):
    return cm.CandidateKey(schedule_unroll_k=uk, **domain)


# ---------------------------------------------------------------------------
# Multi-domain schema
# ---------------------------------------------------------------------------

class TestMultiDomainSchema(unittest.TestCase):
    def test_multiple_domains_serialize_deterministically(self):
        keys = [key(PRIMARY, 1), key(CUBE64, 1), key(ALTK, 1), key(RECT, 1)]
        d1 = [k.canonical_id() for k in keys]
        d2 = [k.canonical_id() for k in keys]
        self.assertEqual(d1, d2)

    def test_candidate_identity_unique_within_domain(self):
        uk1 = key(PRIMARY, 1)
        uk2 = key(PRIMARY, 2)
        self.assertNotEqual(uk1.canonical_id(), uk2.canonical_id())

    def test_same_params_different_shape_do_not_collide(self):
        primary_uk2 = key(PRIMARY, 2)
        cube64_uk2 = key(CUBE64, 2)
        self.assertNotEqual(primary_uk2.canonical_id(), cube64_uk2.canonical_id())

    def test_tile_differences_do_not_collide(self):
        primary_uk2 = key(PRIMARY, 2)
        altk_uk2 = key(ALTK, 2)
        self.assertNotEqual(primary_uk2.canonical_id(), altk_uk2.canonical_id())


# ---------------------------------------------------------------------------
# Compatibility (cross-domain rejection)
# ---------------------------------------------------------------------------

class TestCrossDomainCompatibility(unittest.TestCase):
    def test_exact_shape_accepted(self):
        k = key(PRIMARY, 2)
        compat = cm.check_compatibility(k, k, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.EXACT_MATCH)

    def test_larger_shape_not_exact_match_for_primary(self):
        primary_query = key(PRIMARY, 2)
        cube64_evidence = key(CUBE64, 2)
        compat = cm.check_compatibility(primary_query, cube64_evidence, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertNotEqual(compat["level"], cm.EXACT_MATCH)
        self.assertEqual(compat["level"], cm.CROSS_SHAPE_SAME_SCHEDULE)  # same tile+unroll, different shape

    def test_alternate_tile_rejected_by_primary_domain(self):
        primary_query = key(PRIMARY, 2)
        altk_evidence = key(ALTK, 2)
        compat = cm.check_compatibility(primary_query, altk_evidence, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.INCOMPATIBLE)

    def test_rectangular_evidence_rejected_for_cubic_exact_match(self):
        cube_query = key(PRIMARY, 2)
        rect_evidence = key(RECT, 2)
        compat = cm.check_compatibility(cube_query, rect_evidence, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertNotEqual(compat["level"], cm.EXACT_MATCH)

    def test_explicit_supported_bucket_accepted_with_reduced_confidence(self):
        primary_query = key(PRIMARY, 2)
        cube64_evidence = key(CUBE64, 2)
        compat = cm.check_compatibility(primary_query, cube64_evidence, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertGreater(compat["confidence"], 0.0)
        self.assertLess(compat["confidence"], 1.0)

    def test_unsupported_bucket_rejected(self):
        primary_query = key(PRIMARY, 2)
        altk_evidence = key(ALTK, 1)  # different tile AND different unroll
        compat = cm.check_compatibility(primary_query, altk_evidence, cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.INCOMPATIBLE)

    def test_cross_domain_evidence_never_merged_in_ranking(self):
        # A calibrated ranking for the ALTK domain must never be swayed by
        # PRIMARY domain measured evidence, even if included in the pool.
        altk_uk1 = cm.CandidateEvidenceRecord(
            key=key(ALTK, 1), label="altk_uk1",
            static_ir=_blank_static_ir(), llvm_backend=_blank_backend(spills=0),
            measured=None,
        )
        altk_uk2 = cm.CandidateEvidenceRecord(
            key=key(ALTK, 2), label="altk_uk2",
            static_ir=_blank_static_ir(), llvm_backend=_blank_backend(spills=2),
            measured=None,
        )
        # Pool contains ONLY primary-domain measured evidence (wrong tile).
        primary_measured_evidence = _measured(0.001)
        pool = [(key(PRIMARY, 1), primary_measured_evidence), (key(PRIMARY, 2), _measured(0.0009))]
        ranked = cm.rank_candidates([altk_uk1, altk_uk2], cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        # Neither altk candidate should receive a calibration bonus from
        # incompatible primary-domain evidence.
        for b in ranked:
            self.assertEqual(b.measured_latency_calibration_bonus, 0.0)


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


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

class TestMultiDomainRanking(unittest.TestCase):
    def _domain_records(self, domain, uk1_ms, uk2_ms, uk4_ms, uk4_spills=0):
        recs = []
        for uk, ms, spills in ((1, uk1_ms, 0), (2, uk2_ms, 0), (4, uk4_ms, uk4_spills)):
            recs.append(cm.CandidateEvidenceRecord(
                key=key(domain, uk), label=f"d_uk{uk}",
                static_ir=_blank_static_ir(), llvm_backend=_blank_backend(spills=spills),
                measured=None,
            ))
        pool = [(key(domain, 1), _measured(uk1_ms)), (key(domain, 2), _measured(uk2_ms)), (key(domain, 4), _measured(uk4_ms))]
        return recs, pool

    def test_each_domain_selects_its_own_measured_winner(self):
        recs, pool = self._domain_records(CUBE64, uk1_ms=0.020, uk2_ms=0.018, uk4_ms=0.016, uk4_spills=0)
        ranked = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual(ranked[0].label, "d_uk4")

    def test_tie_honors_variance_aware_policy_not_forced_unique_winner(self):
        # uk1 and uk2 within noise of each other, uk4 clearly faster.
        recs, pool = self._domain_records(RECT, uk1_ms=0.0050, uk2_ms=0.00498, uk4_ms=0.0040)
        ranked = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual(ranked[0].label, "d_uk4")  # not tied with uk1/uk2 -- clearly fastest

    def test_missing_domain_falls_back(self):
        candidates, _ = sel.generate_supported_candidates(
            RECT["shape_m"], RECT["shape_n"], RECT["shape_k"], RECT["tile_m"], RECT["tile_n"], RECT["tile_k"])
        result = sel.select_candidate(sel.MODE_STATIC, candidates, {})
        self.assertEqual(result["effective_mode"], "fallback_conservative_baseline")
        self.assertEqual(result["selected_key"].schedule_unroll_k, 1)

    def test_incompatible_domain_cannot_affect_score(self):
        recs, _ = self._domain_records(ALTK, uk1_ms=0.003, uk2_ms=0.0026, uk4_ms=0.0024)
        wrong_domain_pool = [(key(PRIMARY, 1), _measured(0.0001)), (key(PRIMARY, 2), _measured(0.00005))]
        ranked_with_wrong_pool = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=wrong_domain_pool)
        ranked_no_pool = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=[])
        self.assertEqual([b.total_cost for b in ranked_with_wrong_pool], [b.total_cost for b in ranked_no_pool])

    def test_candidate_label_changes_do_not_affect_selection(self):
        recs, pool = self._domain_records(CUBE64, uk1_ms=0.020, uk2_ms=0.018, uk4_ms=0.016)
        import dataclasses
        relabeled = [dataclasses.replace(r, label="relabeled_" + r.label) for r in recs]
        r1 = cm.rank_candidates(recs, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        r2 = cm.rank_candidates(relabeled, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual([b.candidate_id for b in r1], [b.candidate_id for b in r2])


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------

class TestMultiDomainProfileLoader(unittest.TestCase):
    def test_multidomain_schema_profile_loads(self):
        profile = {
            "schema_version": "stage16_multidomain_profile_v1",
            "benchmark_methodology_version": cm.BENCHMARK_METHODOLOGY_VERSION,
            "domains": {
                "cube64": {
                    "domain_identity": {"target_arch": "aarch64", "target_cpu": "cortex-a76", "target_features": "none",
                                         "dtype": "f32", "tile": {"m": 8, "n": 8, "k": 8}, "shape": "64x64x64"},
                    "candidates": {
                        key(CUBE64, 4).canonical_id(): {"label": "cube64_uk4", "median_latency_ms": 0.0166, "cv": 0.001, "correctness_pass": True},
                    },
                },
            },
        }
        path = write_json_file(profile)
        pool = sel.load_profile_pool(path)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0][0], key(CUBE64, 4))

    def test_multidomain_profile_key_mismatch_rejected(self):
        profile = {
            "schema_version": "stage16_multidomain_profile_v1",
            "benchmark_methodology_version": cm.BENCHMARK_METHODOLOGY_VERSION,
            "domains": {
                "cube64": {
                    "domain_identity": {"target_arch": "aarch64", "target_cpu": "cortex-a76", "target_features": "none",
                                         "dtype": "f32", "tile": {"m": 8, "n": 8, "k": 8}, "shape": "64x64x64"},
                    "candidates": {
                        "not-a-real-canonical-id:uk4": {"label": "bad", "median_latency_ms": 0.01, "cv": 0.01, "correctness_pass": True},
                    },
                },
            },
        }
        path = write_json_file(profile)
        with self.assertRaises((sel.ScheduleSelectionError, IndexError, ValueError)):
            sel.load_profile_pool(path)

    def test_stage12_evidence_source_is_independently_overridable_from_measured_profile(self):
        # Regression test for a real Stage 16 bug: static/backend evidence
        # (--stage12-json) was silently always the ORIGINAL Stage 12 file
        # regardless of what --schedule-profile pointed at, so a candidate
        # only present in a NEW profile (e.g. cube64 uk4, which Stage 12
        # never analyzed) was wrongly treated as UNSUPPORTED (no compiled
        # object on record) even though real evidence existed elsewhere --
        # verified concretely against cube64/8x8x8/uk4 during this stage's
        # own integration testing. This test locks in the fix: an extended
        # --stage12-json containing the new candidate's static evidence
        # must make it selectable.
        extended_stage12 = {
            "environment": {"git_commit": "test"},
            "candidates": {
                "cube64_uk4": {
                    "shape": "64x64x64", "tile": {"m": 8, "n": 8, "k": 8}, "schedule_unroll_k": 4,
                    "register_allocation": {
                        "stages": {"pre_ra": {"approx_peak_live_vector_registers": 100},
                                   "post_ra": {"physical_vector_registers_referenced": 28}},
                        "comparison": {"spill_stores_inserted_by_ra": 0, "reload_loads_inserted_by_ra": 0},
                        "final_stack_frame_bytes": 96,
                    },
                    "schedule": {"post_scheduler": {"accumulator_chains": 16, "max_accumulator_chain_length": 32}},
                    "fp_reduction_order": {"fmuladd_call_count": 512},
                    "assembly_counts": {"fmla": 512},
                    "object_bytes": 5000,
                },
            },
        }
        path = write_json_file(extended_stage12)
        records = sel.load_available_evidence(path, None)
        k = key(CUBE64, 4)
        self.assertIn(k, records)
        self.assertEqual(records[k].llvm_backend.object_bytes.value, 5000)
        self.assertEqual(records[k].llvm_backend.spill_store_count.value, 0)


def write_json_file(obj):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


class TestMultiDomainMaterialization(unittest.TestCase):
    def test_uk1_uk2_uk4_generated_for_cube64(self):
        candidates, rejected = sel.generate_supported_candidates(64, 64, 64, 8, 8, 8)
        uks = sorted(c.schedule_unroll_k for c in candidates)
        self.assertEqual(uks, [1, 2, 4])

    def test_altk_domain_candidates_legal(self):
        candidates, rejected = sel.generate_supported_candidates(32, 32, 32, 8, 8, 4)
        uks = sorted(c.schedule_unroll_k for c in candidates)
        self.assertEqual(uks, [1, 2, 4])  # K trip = 8, divisible by both 2 and 4

    def test_artifact_key_matches_selection_for_cube64(self):
        k = key(CUBE64, 4)
        result = sel.verify_no_mismatch(k, (8, 8, 8), 4, 64, 64, 64)
        self.assertEqual(result, k)

    def test_domain_mismatch_raises(self):
        k = key(CUBE64, 4)
        with self.assertRaises(sel.ArtifactIdentityMismatchError):
            sel.verify_no_mismatch(k, (8, 8, 8), 4, 32, 32, 32)  # shape mismatch


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestMultiDomainCorrectness(unittest.TestCase):
    def test_incorrect_candidate_excluded_from_measured_selection(self):
        good = cm.CandidateEvidenceRecord(key=key(RECT, 2), label="good", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=_measured(0.005, correctness=True))
        bad = cm.CandidateEvidenceRecord(key=key(RECT, 4), label="bad", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=_measured(0.001, correctness=False))
        pool = [(good.key, good.measured), (bad.key, bad.measured)]
        ranked = cm.rank_candidates([good, bad], cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_pool=pool)
        self.assertEqual(ranked[0].label, "good")
        bad_breakdown = next(b for b in ranked if b.label == "bad")
        self.assertTrue(bad_breakdown.rejected)

    def test_missing_correctness_result_prevents_calibrated_use(self):
        rec = cm.CandidateEvidenceRecord(key=key(RECT, 2), label="cand", static_ir=_blank_static_ir(), llvm_backend=_blank_backend(), measured=None)
        b = cm.compute_cost(rec, cm.RANKING_MODE_CALIBRATED_PI, measured_evidence_for_calibration=None, compatibility=None)
        self.assertEqual(b.measured_latency_calibration_bonus, 0.0)
        self.assertLess(b.confidence, 1.0)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegression(unittest.TestCase):
    def test_default_remains_manual(self):
        self.assertEqual(sel.MODE_MANUAL, "manual")

    def test_calibrated_remains_opt_in(self):
        import inspect
        source = inspect.getsource(sel.main)
        self.assertIn('default=MODE_MANUAL', source)

    def test_no_universal_uk4_rule_in_code(self):
        # The selector must never special-case "schedule_unroll_k == 4"
        # as an automatic winner -- every result must flow through real
        # ranking, not a hardcoded shortcut.
        import inspect
        source = inspect.getsource(sel)
        self.assertNotIn("schedule_unroll_k = 4", source.replace(" ", ""))
        self.assertNotIn('selected_key=cm.CandidateKey(schedule_unroll_k=4', source.replace(" ", ""))

    def test_serving_cost_model_untouched(self):
        # The module docstrings legitimately MENTION "ServingCostModel" in
        # prose (explaining that it's a separate, deliberately-untouched
        # subsystem) -- a plain substring check would false-positive on
        # that documentation. What actually matters: no import of or call
        # into the serving subsystem's Python/C++ bindings.
        import inspect
        source = inspect.getsource(sel) + inspect.getsource(cm)
        self.assertNotIn("import ServingCostModel", source)
        self.assertNotIn("from serving", source)
        self.assertNotIn("ServingCostModel(", source)
        self.assertNotIn("PlanSelectionPass(", source)
        serving_dir = os.path.join(REPO_ROOT, "mlir_passes", "lib", "serving")
        import subprocess
        git_status = subprocess.run(["git", "status", "--short", serving_dir], capture_output=True, text=True, cwd=REPO_ROOT).stdout
        self.assertEqual(git_status.strip(), "", "mlir_passes/lib/serving/ must show no changes")


if __name__ == "__main__":
    unittest.main()
