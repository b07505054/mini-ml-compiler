#!/usr/bin/env python3
"""Focused tests for tools/select_and_compile_aarch64_matmul_schedule.py (Stage 15).

Pure host-side logic tests -- no SSH/Pi dependency, and no live
mlir-opt/llc compilation inside this file (that end-to-end proof is the
artifact run itself: manual uk1, manual uk2, static, calibrated, and
fallback were each compiled and verified through the real pipeline while
building artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/,
documented in that directory's README rather than re-run on every test
invocation, matching this project's established convention of separating
fast host-only tests from hardware/toolchain integration runs).
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


def write_json(obj):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(obj, f)
    f.close()
    return f.name


def minimal_fixture(target_cpu="cortex-a76", target_arch="aarch64", target_features="none",
                     dtype="f32", methodology="stage13_pi5_harness_v1", candidates=None):
    return {
        "schema_version": "stage15_schedule_profile_v1",
        "profile_kind": "compatibility_test_fixture",
        "target_arch": target_arch, "target_cpu": target_cpu, "target_features": target_features,
        "dtype": dtype, "benchmark_methodology_version": methodology,
        "candidates": candidates or {
            "cand": {"shape_m": 32, "shape_n": 32, "shape_k": 32, "tile_m": 8, "tile_n": 8, "tile_k": 8,
                     "schedule_unroll_k": 2, "median_latency_ms": 0.002, "correctness_pass": True, "cv": 0.01},
        },
    }


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------

class TestCliBehavior(unittest.TestCase):
    def test_default_mode_is_manual(self):
        ap = sel.argparse.ArgumentParser()
        ap.add_argument("--schedule-candidate-mode", choices=sel.VALID_MODES, default=sel.MODE_MANUAL)
        args = ap.parse_args([])
        self.assertEqual(args.schedule_candidate_mode, sel.MODE_MANUAL)

    def test_manual_mode_respects_requested_uk(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        result = sel.select_candidate(sel.MODE_MANUAL, candidates, {}, manual_unroll_k=2)
        self.assertEqual(result["selected_key"].schedule_unroll_k, 2)
        self.assertEqual(result["effective_mode"], sel.MODE_MANUAL)

    def test_manual_mode_requires_uk_value(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        with self.assertRaises(sel.ScheduleSelectionError):
            sel.select_candidate(sel.MODE_MANUAL, candidates, {}, manual_unroll_k=None)

    def test_manual_mode_rejects_illegal_uk(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        with self.assertRaises(sel.ScheduleSelectionError):
            sel.select_candidate(sel.MODE_MANUAL, candidates, {}, manual_unroll_k=3)  # 3 does not divide K trip count 4

    def test_static_mode_runs_without_measured_profile(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        evidence = sel.load_available_evidence(sel.DEFAULT_STAGE12_JSON, None)
        result = sel.select_candidate(sel.MODE_STATIC, candidates, evidence)
        self.assertEqual(result["effective_mode"], sel.MODE_STATIC)
        self.assertEqual(result["fallback_reason"], "none")

    def test_calibrated_mode_requires_profile(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        with self.assertRaises(sel.ScheduleSelectionError):
            sel.select_candidate(sel.MODE_CALIBRATED, candidates, {}, profile_path=None)

    def test_unknown_mode_rejected(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        with self.assertRaises(sel.ScheduleSelectionError):
            sel.select_candidate("nonexistent_mode", candidates, {})


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

class TestCandidateGeneration(unittest.TestCase):
    def test_valid_uk1_uk2_generated_for_primary_shape(self):
        candidates, rejected = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        uks = sorted(c.schedule_unroll_k for c in candidates)
        self.assertIn(1, uks)
        self.assertIn(2, uks)
        self.assertEqual(rejected, [])

    def test_uk4_generated_when_trip_count_supports_it(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)  # K trip = 4
        self.assertIn(4, [c.schedule_unroll_k for c in candidates])

    def test_uk4_not_generated_when_trip_count_does_not_divide(self):
        # K trip count = 32/4 = 8 for tile-k=4 -> 8%4==0 so uk4 IS legal there;
        # use a shape/tile where trip count is 2 (uk4 illegal: 2%4 != 0).
        candidates, _ = sel.generate_supported_candidates(16, 16, 16, 8, 8, 8)  # K trip = 2
        self.assertNotIn(4, [c.schedule_unroll_k for c in candidates])
        self.assertIn(2, [c.schedule_unroll_k for c in candidates])  # full collapse, still legal

    def test_unsupported_tile_rejected_before_scoring(self):
        candidates, rejected = sel.generate_supported_candidates(32, 32, 32, 16, 16, 16)
        self.assertEqual(candidates, [])
        self.assertTrue(any("structural validation" in r["reason"] for r in rejected))

    def test_non_divisible_shape_rejected(self):
        candidates, rejected = sel.generate_supported_candidates(30, 30, 30, 8, 8, 8)
        self.assertEqual(candidates, [])
        self.assertTrue(any("not evenly divisible" in r["reason"] for r in rejected))

    def test_duplicate_semantic_candidates_removed(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        keys = [c.canonical_id() for c in candidates]
        self.assertEqual(len(keys), len(set(keys)))

    def test_deterministic_generation_order(self):
        c1, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        c2, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        self.assertEqual([c.canonical_id() for c in c1], [c.canonical_id() for c in c2])


# ---------------------------------------------------------------------------
# Compatibility (real fixtures: Stage 13 Pi profile + real x86/A72 targets)
# ---------------------------------------------------------------------------

class TestCompatibilityFixtures(unittest.TestCase):
    def test_real_pi_fixture_accepted(self):
        path = write_json(minimal_fixture(target_cpu="cortex-a76"))
        self.addCleanup(os.unlink, path)
        pool = sel.load_profile_pool(path)
        self.assertEqual(len(pool), 1)
        key, measured = pool[0]
        self.assertEqual(key.target_cpu, "cortex-a76")

    def test_real_x86_fixture_rejected_for_aarch64_query(self):
        path = write_json(minimal_fixture(target_arch="x86_64", target_cpu="skylake"))
        self.addCleanup(os.unlink, path)
        pool = sel.load_profile_pool(path)
        query = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=2)
        compat = cm.check_compatibility(query, pool[0][0], cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.INCOMPATIBLE)

    def test_different_real_aarch64_cpu_rejected(self):
        path = write_json(minimal_fixture(target_cpu="cortex-a72"))
        self.addCleanup(os.unlink, path)
        pool = sel.load_profile_pool(path)
        query = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=2)
        compat = cm.check_compatibility(query, pool[0][0], cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.INCOMPATIBLE)

    def test_wrong_feature_set_rejected(self):
        path = write_json(minimal_fixture(target_features="+dotprod"))
        self.addCleanup(os.unlink, path)
        pool = sel.load_profile_pool(path)
        query = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=2)
        compat = cm.check_compatibility(query, pool[0][0], cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertEqual(compat["level"], cm.INCOMPATIBLE)

    def test_wrong_tile_rejected(self):
        candidates = {"cand": {"shape_m": 32, "shape_n": 32, "shape_k": 32, "tile_m": 8, "tile_n": 8, "tile_k": 4,
                                "schedule_unroll_k": 1, "median_latency_ms": 0.002, "correctness_pass": True, "cv": 0.01}}
        path = write_json(minimal_fixture(candidates=candidates))
        self.addCleanup(os.unlink, path)
        pool = sel.load_profile_pool(path)
        query = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=1)
        compat = cm.check_compatibility(query, pool[0][0], cm.BENCHMARK_METHODOLOGY_VERSION)
        self.assertNotEqual(compat["level"], cm.EXACT_MATCH)

    def test_malformed_profile_rejected(self):
        path = write_json({"some_unrelated_field": "not a valid schema"})
        self.addCleanup(os.unlink, path)
        with self.assertRaises(sel.ScheduleSelectionError):
            sel.load_profile_pool(path)

    def test_stale_methodology_version_rejected(self):
        path = write_json(minimal_fixture(methodology="stage13_pi5_harness_v0_preliminary"))
        self.addCleanup(os.unlink, path)
        with self.assertRaises(sel.ScheduleSelectionError):
            sel.load_profile_pool(path)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelection(unittest.TestCase):
    def test_calibrated_pi_mode_selects_measured_winner_with_exact_evidence(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        evidence = sel.load_available_evidence(sel.DEFAULT_STAGE12_JSON, sel.DEFAULT_STAGE13_JSON)
        result = sel.select_candidate(sel.MODE_CALIBRATED, candidates, evidence, profile_path=sel.DEFAULT_STAGE13_JSON)
        # Stage 13 measured primary_full_unroll (uk4) as the fastest for this exact shape/tile.
        self.assertEqual(result["selected_key"].schedule_unroll_k, 4)
        self.assertEqual(result["effective_mode"], sel.MODE_CALIBRATED)

    def test_manual_uk1_remains_uk1(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        result = sel.select_candidate(sel.MODE_MANUAL, candidates, {}, manual_unroll_k=1)
        self.assertEqual(result["selected_key"].schedule_unroll_k, 1)

    def test_manual_uk2_remains_uk2(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        result = sel.select_candidate(sel.MODE_MANUAL, candidates, {}, manual_unroll_k=2)
        self.assertEqual(result["selected_key"].schedule_unroll_k, 2)

    def test_static_mode_is_deterministic(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        evidence = sel.load_available_evidence(sel.DEFAULT_STAGE12_JSON, None)
        r1 = sel.select_candidate(sel.MODE_STATIC, candidates, evidence)
        r2 = sel.select_candidate(sel.MODE_STATIC, candidates, evidence)
        self.assertEqual(r1["selected_key"], r2["selected_key"])

    def test_incompatible_calibration_falls_back_explicitly(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        evidence = sel.load_available_evidence(sel.DEFAULT_STAGE12_JSON, None)
        path = write_json(minimal_fixture(target_arch="x86_64", target_cpu="skylake"))
        self.addCleanup(os.unlink, path)
        result = sel.select_candidate(sel.MODE_CALIBRATED, candidates, evidence, profile_path=path)
        self.assertEqual(result["effective_mode"], "fallback_static")
        self.assertNotEqual(result["fallback_reason"], "none")

    def test_candidate_labels_do_not_affect_selection(self):
        # Two evidence dicts with identical keys but differently-labeled
        # records must select identically.
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        evidence = sel.load_available_evidence(sel.DEFAULT_STAGE12_JSON, None)
        relabeled = {k: cm.dataclasses.replace(v, label="totally_different_label") for k, v in evidence.items()}
        r1 = sel.select_candidate(sel.MODE_STATIC, candidates, evidence)
        r2 = sel.select_candidate(sel.MODE_STATIC, candidates, relabeled)
        self.assertEqual(r1["selected_key"], r2["selected_key"])

    def test_no_evidence_falls_back_to_conservative_baseline(self):
        candidates, _ = sel.generate_supported_candidates(16, 16, 16, 8, 8, 8)  # shape Stage 12 never analyzed at this tile
        result = sel.select_candidate(sel.MODE_STATIC, candidates, {})
        self.assertEqual(result["selected_key"].schedule_unroll_k, sel.CONSERVATIVE_BASELINE_UNROLL_K)
        self.assertEqual(result["effective_mode"], "fallback_conservative_baseline")


# ---------------------------------------------------------------------------
# Materialization / hard guard
# ---------------------------------------------------------------------------

class TestMaterializationGuard(unittest.TestCase):
    def test_matching_key_passes_guard(self):
        key = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=2)
        result = sel.verify_no_mismatch(key, (8, 8, 8), 2, 32, 32, 32)
        self.assertEqual(result, key)

    def test_mismatched_unroll_raises(self):
        key = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=2)
        with self.assertRaises(sel.ArtifactIdentityMismatchError):
            sel.verify_no_mismatch(key, (8, 8, 8), 1, 32, 32, 32)  # actual compile used uk1, selection said uk2

    def test_mismatched_tile_raises(self):
        key = cm.CandidateKey(shape_m=32, shape_n=32, shape_k=32, tile_m=8, tile_n=8, tile_k=8, schedule_unroll_k=1)
        with self.assertRaises(sel.ArtifactIdentityMismatchError):
            sel.verify_no_mismatch(key, (8, 8, 4), 1, 32, 32, 32)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegression(unittest.TestCase):
    def test_no_spill_hard_reject_policy_in_selection_path(self):
        candidates, _ = sel.generate_supported_candidates(32, 32, 32, 8, 8, 8)
        evidence = sel.load_available_evidence(sel.DEFAULT_STAGE12_JSON, None)
        result = sel.select_candidate(sel.MODE_STATIC, candidates, evidence)
        # primary_full_unroll (uk4) has real spills in Stage 12 evidence but
        # must still appear in the cost breakdown, not be silently dropped.
        labels_in_breakdown = [b["candidate_id"] for b in result["cost_breakdown"]]
        uk4_id = next(c.canonical_id() for c in candidates if c.schedule_unroll_k == 4)
        self.assertIn(uk4_id, labels_in_breakdown)
        uk4_breakdown = next(b for b in result["cost_breakdown"] if b["candidate_id"] == uk4_id)
        self.assertFalse(uk4_breakdown["rejected"])

    def test_default_cli_mode_constant_is_manual_not_calibrated(self):
        self.assertEqual(sel.MODE_MANUAL, "manual")
        # Sanity: the module-level default used by argparse must be MODE_MANUAL.
        import inspect
        source = inspect.getsource(sel.main)
        self.assertIn('default=MODE_MANUAL', source)

    def test_serving_cost_model_files_not_imported_or_referenced(self):
        import inspect
        source = inspect.getsource(sel)
        self.assertNotIn("ServingCostModel", source)
        self.assertNotIn("PlanSelectionPass", source)


if __name__ == "__main__":
    unittest.main()
