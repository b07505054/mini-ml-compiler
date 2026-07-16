#!/usr/bin/env python3
"""test_tile_candidate_selection.py

Host-side unit tests for the AArch64 tile-candidate selection slice
(Stage 17 of the task brief). Runs with no network access and no
Raspberry Pi -- pure Python logic tests against
tools/generate_aarch64_matmul_tile_candidates.py and
tools/select_aarch64_matmul_tile_candidate.py, plus a couple of subprocess
tests against mlir_passes/tools/generate_tiled_transform.sh.

Usage:
  python3 -m pytest tests/test_tile_candidate_selection.py -v
  (or: python3 tests/test_tile_candidate_selection.py)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import generate_aarch64_matmul_tile_candidates as gen  # noqa: E402
import select_aarch64_matmul_tile_candidate as sel  # noqa: E402


def make_candidate(shape=(32, 32, 32), tile=(4, 8, 8), median_ms=1.0, object_bytes=2000,
                    spills=0, reloads=0, registers=20, legal=True, correct=True):
    return {
        "shape": list(shape),
        "tile": {"m": tile[0], "n": tile[1], "k": tile[2]},
        "legality": {"legal": legal, "rejection_reasons": [] if legal else ["test rejection"]},
        "correctness": {"passed": correct, "max_abs_error": 0.0 if correct else 1.0},
        "performance": {"median_ms": median_ms, "p95_ms": median_ms * 1.01},
        "backend": {
            "object_bytes": object_bytes,
            "hot_loop_vector_spills": spills,
            "hot_loop_vector_reloads": reloads,
            "vector_registers_referenced": registers,
        },
    }


class TileParameterValidation(unittest.TestCase):
    """Item 1: tile parameter validation."""

    def setUp(self):
        # mkdtemp (not TemporaryDirectory's `with`) so the directory
        # survives past run_generator() returning -- a `with` block here
        # would delete the directory (and the file we just asked the
        # caller to check for) before the caller ever sees it.
        self._tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self._tmpdir, ignore_errors=True)

    def run_generator(self, tile_m, tile_n, tile_k):
        out = os.path.join(self._tmpdir, "out.mlir")
        cmd = ["bash", os.path.join(REPO_ROOT, "mlir_passes", "tools", "generate_tiled_transform.sh"),
               "--tile-m", str(tile_m), "--tile-n", str(tile_n), "--tile-k", str(tile_k),
               "--output", out]
        return subprocess.run(cmd, capture_output=True, text=True), out

    def test_rejects_zero(self):
        proc, _ = self.run_generator(0, 8, 8)
        self.assertNotEqual(proc.returncode, 0)

    def test_rejects_negative(self):
        proc, _ = self.run_generator(-4, 8, 8)
        self.assertNotEqual(proc.returncode, 0)

    def test_rejects_non_integer(self):
        proc, _ = self.run_generator("abc", 8, 8)
        self.assertNotEqual(proc.returncode, 0)

    def test_accepts_positive_integers(self):
        proc, out = self.run_generator(4, 8, 8)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.isfile(out))


class ParameterizedTransformGeneration(unittest.TestCase):
    """Item 3: parameterized Transform generation produces correct substitution."""

    def test_substitution_correctness(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mlir")
            cmd = ["bash", os.path.join(REPO_ROOT, "mlir_passes", "tools", "generate_tiled_transform.sh"),
                   "--tile-m", "6", "--tile-n", "12", "--tile-k", "3", "--output", out]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out) as f:
                text = f.read()
            self.assertIn("tile_sizes [6, 12]", text)
            self.assertIn("tile_sizes [0, 0, 3]", text)
            # No leftover placeholder tokens.
            self.assertNotIn("TILE_M", text)
            self.assertNotIn("TILE_N", text)
            self.assertNotIn("TILE_K", text)


class CandidateLegality(unittest.TestCase):
    """Item 2: candidate legality; item 9: rejection of illegal candidates."""

    def test_divisible_shape_is_legal(self):
        result = gen.static_legality((32, 32, 32), (4, 8, 8))
        self.assertTrue(result["legal"])
        self.assertEqual(result["rejection_reasons"], [])

    def test_non_divisible_m_is_illegal(self):
        result = gen.static_legality((10, 32, 32), (4, 8, 8))
        self.assertFalse(result["legal"])
        self.assertTrue(any("M=10" in r for r in result["rejection_reasons"]))

    def test_non_divisible_n_is_illegal(self):
        result = gen.static_legality((32, 10, 32), (4, 8, 8))
        self.assertFalse(result["legal"])
        self.assertTrue(any("N=10" in r for r in result["rejection_reasons"]))

    def test_non_divisible_k_is_illegal(self):
        result = gen.static_legality((32, 32, 10), (4, 8, 8))
        self.assertFalse(result["legal"])
        self.assertTrue(any("K=10" in r for r in result["rejection_reasons"]))

    def test_register_estimate_hard_limit_rejection(self):
        # A synthetic absurd tile that legitimately exceeds the hard limit.
        result = gen.static_legality((64, 64, 64), (32, 64, 4))
        self.assertFalse(result["legal"])
        self.assertTrue(any("register demand" in r for r in result["rejection_reasons"]))

    def test_all_required_candidates_legal_for_required_shapes(self):
        shapes = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (32, 64, 32), (64, 32, 64)]
        tiles = [(4, 4, 4), (4, 8, 4), (4, 8, 8), (8, 4, 4), (8, 4, 8), (8, 8, 4), (8, 8, 8)]
        for shape in shapes:
            for tile in tiles:
                result = gen.static_legality(shape, tile)
                self.assertTrue(result["legal"], f"expected legal: shape={shape} tile={tile}: {result}")


class ScoringPolicy(unittest.TestCase):
    """Item 8: selection score calculation; item 11: deterministic tie-breaking."""

    def test_faster_candidate_scores_lower(self):
        fast = make_candidate(median_ms=1.0)
        slow = make_candidate(median_ms=2.0)
        s_fast, _ = sel.score(fast, best_latency_ms=1.0, smallest_object_bytes=2000)
        s_slow, _ = sel.score(slow, best_latency_ms=1.0, smallest_object_bytes=2000)
        self.assertLess(s_fast, s_slow)

    def test_spill_penalty_applied(self):
        clean = make_candidate(median_ms=1.0, spills=0)
        spilling = make_candidate(median_ms=1.0, spills=2)
        s_clean, _ = sel.score(clean, best_latency_ms=1.0, smallest_object_bytes=2000)
        s_spilling, _ = sel.score(spilling, best_latency_ms=1.0, smallest_object_bytes=2000)
        self.assertAlmostEqual(s_spilling - s_clean, 2 * sel.WEIGHT_SPILL, places=6)

    def test_small_latency_win_does_not_beat_a_spill(self):
        # A 5% faster candidate with 1 spill should score worse than a
        # slightly slower, zero-spill candidate -- 0.05 < WEIGHT_SPILL (0.20).
        faster_with_spill = make_candidate(median_ms=0.95, spills=1)
        slower_clean = make_candidate(median_ms=1.0, spills=0)
        s1, _ = sel.score(faster_with_spill, best_latency_ms=0.95, smallest_object_bytes=2000)
        s2, _ = sel.score(slower_clean, best_latency_ms=0.95, smallest_object_bytes=2000)
        self.assertGreater(s1, s2)

    def test_large_latency_win_beats_a_spill(self):
        # A 30% faster candidate with 1 spill SHOULD beat a zero-spill
        # candidate -- 0.30 > WEIGHT_SPILL (0.20).
        much_faster_with_spill = make_candidate(median_ms=0.70, spills=1)
        slower_clean = make_candidate(median_ms=1.0, spills=0)
        s1, _ = sel.score(much_faster_with_spill, best_latency_ms=0.70, smallest_object_bytes=2000)
        s2, _ = sel.score(slower_clean, best_latency_ms=0.70, smallest_object_bytes=2000)
        self.assertLess(s1, s2)

    def test_tie_break_prefers_lower_latency(self):
        a = make_candidate(median_ms=1.0, object_bytes=2000)
        b = make_candidate(median_ms=0.9, object_bytes=2000)
        self.assertLess(sel.tie_break_key(b), sel.tie_break_key(a))

    def test_tie_break_prefers_zero_spills_when_latency_equal(self):
        a = make_candidate(median_ms=1.0, spills=1)
        b = make_candidate(median_ms=1.0, spills=0)
        self.assertLess(sel.tie_break_key(b), sel.tie_break_key(a))

    def test_tie_break_prefers_smaller_object_when_latency_and_spills_equal(self):
        a = make_candidate(median_ms=1.0, spills=0, object_bytes=3000)
        b = make_candidate(median_ms=1.0, spills=0, object_bytes=2000)
        self.assertLess(sel.tie_break_key(b), sel.tie_break_key(a))

    def test_tie_break_is_deterministic(self):
        a = make_candidate(median_ms=1.0, object_bytes=2000, tile=(4, 8, 8))
        b = make_candidate(median_ms=1.0, object_bytes=2000, tile=(4, 8, 8))
        self.assertEqual(sel.tie_break_key(a), sel.tie_break_key(b))


class EligibilityFiltering(unittest.TestCase):
    """Item 9: rejection of illegal candidates; item 10: rejection of incorrect candidates."""

    def test_illegal_candidate_is_ineligible(self):
        c = make_candidate(legal=False)
        ok, reason = sel.is_eligible(c)
        self.assertFalse(ok)
        self.assertIn("illegal", reason)

    def test_incorrect_candidate_is_ineligible(self):
        c = make_candidate(correct=False)
        ok, reason = sel.is_eligible(c)
        self.assertFalse(ok)
        self.assertIn("correctness", reason)

    def test_legal_correct_candidate_is_eligible(self):
        c = make_candidate()
        ok, reason = sel.is_eligible(c)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_missing_benchmark_is_ineligible(self):
        c = make_candidate()
        c["performance"]["median_ms"] = None
        ok, reason = sel.is_eligible(c)
        self.assertFalse(ok)

    def test_incorrect_candidate_excluded_not_kept_as_penalized(self):
        """Per the task brief: an incorrect candidate must be excluded
        entirely, never kept as a high-penalty scored option."""
        candidates = [make_candidate(tile=(4, 8, 8), median_ms=1.0, correct=True),
                      make_candidate(tile=(8, 8, 8), median_ms=0.5, correct=False)]
        result, rejected = sel.select_for_shape(candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result["selected_tile"], [4, 8, 8])
        # The incorrect candidate must not appear among the SCORED
        # rejected_candidates (which only lists eligible-but-not-chosen
        # candidates) -- it is filtered out before scoring entirely.
        scored_tiles = [tuple(rc["tile"]) for rc in result["rejected_candidates"]]
        self.assertNotIn((8, 8, 8), scored_tiles)


class CandidateJsonSchema(unittest.TestCase):
    """Item 7: candidate JSON schema."""

    def test_required_top_level_keys(self):
        c = make_candidate()
        for key in ("shape", "tile", "legality", "correctness", "performance", "backend"):
            self.assertIn(key, c)

    def test_tile_has_m_n_k(self):
        c = make_candidate()
        for key in ("m", "n", "k"):
            self.assertIn(key, c["tile"])

    def test_selection_output_schema(self):
        candidates = [make_candidate(tile=(4, 8, 8), median_ms=1.0),
                      make_candidate(tile=(8, 8, 8), median_ms=0.9)]
        result, _ = sel.select_for_shape(candidates)
        for key in ("shape", "selected_tile", "score", "selection_reasons",
                    "fastest_measured", "selected_matches_fastest", "rejected_candidates"):
            self.assertIn(key, result)


if __name__ == "__main__":
    unittest.main()
