#!/usr/bin/env python3
"""Focused tests for tools/compare_aarch64_schedule_variants.py (Stage 12).

These exercise the pure parsing/logic functions with small synthetic
fixtures -- no llc/mlir-opt/llvm-mca invocation, no Raspberry Pi, no
network -- so they run fast and do not depend on the toolchain being
installed. Real end-to-end coverage (does the tool produce credible
numbers on genuine compiler output) is the artifact run itself
(artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/), not
these tests.

Categories covered (task brief Stage 12 section 13's minimum list):
  - named versus numbered LLVM IR SSA values in the FP-chain reconstruction
    (an @llvm.fmuladd's accumulator operand can be a named value like the
    zero-initializer, not just a numbered %N)
  - a spill-free register-comparison classifies as A/B, never D
  - a synthetic MIR/register-comparison fixture with a real spill pattern
    classifies as D
  - collapsed-loop assembly (no innermost-loop-header comment at all, as
    happens when full unroll eliminates the loop) is handled without a
    crash, reporting ok=False with a clear reason
  - schedule-unroll-k=1 baseline vs itself is a true no-op classification
    (identical metrics on both sides -> B, never A or D)
  - mismatched shape/tile comparisons are rejected outright, not silently
    scored
"""
import json
import os
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import compare_aarch64_schedule_variants as m  # noqa: E402


def make_register_allocation(virtual_vec_pre_ra, physical_vec_post_ra, spill_stores, reload_loads, approx_peak):
    return {
        "stages": {
            "pre_ra": {"approx_peak_live_vector_registers": approx_peak},
            "post_ra": {"physical_vector_registers_referenced": physical_vec_post_ra},
        },
        "comparison": {
            "virtual_vector_registers_before_ra": virtual_vec_pre_ra,
            "spill_stores_inserted_by_ra": spill_stores,
            "reload_loads_inserted_by_ra": reload_loads,
        },
    }


def make_schedule(accumulator_chains, same_acc_median, load_use_median):
    return {
        "post_scheduler": {
            "accumulator_chains": accumulator_chains,
            "same_accumulator_distance": {"median": same_acc_median},
            "load_to_use_distance": {"median": load_use_median},
        }
    }


def make_record(shape, tile, unroll_k, reg, sched, object_bytes):
    return {
        "shape": shape, "tile": tile, "schedule_unroll_k": unroll_k,
        "register_allocation": reg, "schedule": sched, "object_bytes": object_bytes,
    }


class TestFpReductionOrder(unittest.TestCase):
    def _write_ll(self, body):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".ll", delete=False)
        f.write(body)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_named_accumulator_operand_is_a_valid_chain_root(self):
        # The FIRST fmuladd in a reduction accumulates into a named value
        # (%cst, the zero-initializer), not a numbered %N -- this must be
        # recognized as a chain root, not silently dropped by a regex that
        # only accepts digit-only SSA names.
        ll = self._write_ll(
            "define void @f() {\n"
            "  %1 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %cst)\n"
            "  %2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %c, <4 x float> %d, <4 x float> %1)\n"
            "  ret void\n"
            "}\n"
        )
        result = m.check_fp_reduction_order(ll)
        self.assertEqual(result["fmuladd_call_count"], 2)
        self.assertEqual(result["accumulator_chain_count"], 1)
        self.assertEqual(result["max_accumulator_chain_length"], 2)
        self.assertTrue(result["all_fmuladd_calls_accounted_for"])
        self.assertEqual(result["fmf_flags_found_on_fmuladd_calls"], [])

    def test_bareword_zeroinitializer_accumulator_operand_is_counted(self):
        # Regression test for a real bug found while running this tool on
        # a full-K-unroll candidate: LLVM sometimes emits the literal
        # keyword `zeroinitializer` (no leading %) as the first fmuladd's
        # accumulator operand instead of a named %cst value. A regex that
        # only accepted "%<name>" silently dropped these calls -- verified
        # concretely on the 32x32x32/tile-8x8x8/schedule-unroll-k=4
        # candidate: `grep -c llvm.fmuladd` found 256 real calls, the
        # buggy regex matched only 248.
        ll = self._write_ll(
            "define void @f() {\n"
            "  %1 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %a, <8 x float> %b, <8 x float> zeroinitializer)\n"
            "  %2 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %c, <8 x float> %d, <8 x float> %1)\n"
            "  ret void\n"
            "}\n"
        )
        result = m.check_fp_reduction_order(ll)
        self.assertEqual(result["fmuladd_call_count"], 2)
        self.assertEqual(result["accumulator_chain_count"], 1)
        self.assertEqual(result["max_accumulator_chain_length"], 2)
        self.assertTrue(result["all_fmuladd_calls_accounted_for"])

    def test_fast_math_flag_is_detected_when_present(self):
        ll = self._write_ll(
            "define void @f() {\n"
            "  %1 = call reassoc <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %cst)\n"
            "  ret void\n"
            "}\n"
        )
        result = m.check_fp_reduction_order(ll)
        self.assertIn("reassoc", result["fmf_flags_found_on_fmuladd_calls"])

    def test_independent_chains_are_not_merged(self):
        # Two unrelated reductions (e.g. two output tile positions) must
        # be reported as two separate chains, not one -- merging them
        # would hide a real reassociation-into-parallel-sums defect if one
        # ever existed.
        ll = self._write_ll(
            "define void @f() {\n"
            "  %1 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %cst0)\n"
            "  %2 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %c, <4 x float> %d, <4 x float> %cst1)\n"
            "  ret void\n"
            "}\n"
        )
        result = m.check_fp_reduction_order(ll)
        self.assertEqual(result["accumulator_chain_count"], 2)
        self.assertEqual(result["accumulator_chain_lengths"], [1, 1])


class TestInnermostLoopRegion(unittest.TestCase):
    def test_picks_deepest_depth_not_first_match(self):
        # Regression test for the real bug found while building this tool:
        # "This (?:Inner )?Loop Header" matches the OUTER loop's "This Loop
        # Header: Depth=1" comment too, since "Inner" is optional -- taking
        # the first match locks onto the wrong (outer) region.
        lines = [
            ".LBB0_1:                                // =>This Loop Header: Depth=1",
            "        nop",
            ".LBB0_2:                                //   Parent Loop BB0_1 Depth=1",
            "                                        // =>  This Inner Loop Header: Depth=2",
            "        fmla v0.4s, v1.4s, v2.4s",
            "        b.gt .LBB0_2",
            "        b.gt .LBB0_1",
        ]
        label, header_idx, back_idx = m.find_innermost_loop_region(lines)
        self.assertEqual(label, ".LBB0_2")
        self.assertEqual(header_idx, 2)
        self.assertEqual(back_idx, 5)

    def test_no_loop_header_comment_returns_none_not_crash(self):
        # A fully-collapsed loop (e.g. full K-unroll eliminating the
        # innermost scf.for, verified in Stage 11) may leave assembly with
        # fewer loop levels than expected, or none at all in a synthetic
        # fixture -- must degrade to (None, None, None), not raise.
        lines = ["nop", "fmla v0.4s, v1.4s, v2.4s", "ret"]
        label, header_idx, back_idx = m.find_innermost_loop_region(lines)
        self.assertIsNone(label)
        self.assertIsNone(header_idx)
        self.assertIsNone(back_idx)

    def test_extract_hot_loop_region_handles_missing_loop_gracefully(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".s", delete=False)
        f.write("nop\nfmla v0.4s, v1.4s, v2.4s\nret\n")
        f.close()
        self.addCleanup(os.unlink, f.name)
        out_path = f.name + ".marked"
        self.addCleanup(lambda: os.path.exists(out_path) and os.unlink(out_path))
        result = m.extract_hot_loop_region(f.name, out_path)
        self.assertIsNone(result)


class TestClassifyPair(unittest.TestCase):
    def test_no_spill_improved_overlap_classifies_a(self):
        shape, tile = "32x32x32", {"m": 8, "n": 8, "k": 8}
        baseline = make_record(
            shape, tile, 1,
            make_register_allocation(242, 28, 0, 0, 113),
            make_schedule(16, 18.0, 1.0),
            2608,
        )
        scheduled = make_record(
            shape, tile, 2,
            make_register_allocation(402, 28, 0, 0, 145),
            make_schedule(16, 18.0, 1.0),
            3248,
        )
        result = m.classify_pair(baseline, scheduled)
        self.assertEqual(result["classification"], "A")
        self.assertEqual(result["spill_stores_delta"], 0)

    def test_real_spill_pattern_classifies_d_regardless_of_overlap(self):
        shape, tile = "32x32x32", {"m": 8, "n": 8, "k": 8}
        baseline = make_record(
            shape, tile, 1,
            make_register_allocation(242, 28, 0, 0, 113),
            make_schedule(16, 18.0, 1.0),
            2608,
        )
        scheduled = make_record(
            shape, tile, 4,
            make_register_allocation(689, 32, 11, 12, 300),
            make_schedule(16, 30.0, 1.0),  # even with "improved" overlap numbers
            5200,
        )
        result = m.classify_pair(baseline, scheduled)
        self.assertEqual(result["classification"], "D")
        self.assertEqual(result["spill_stores_delta"], 11)
        self.assertEqual(result["reload_loads_delta"], 12)

    def test_identical_metrics_at_unroll_1_is_neutral_not_a_or_d(self):
        # schedule-unroll-k=1 vs itself: a true no-op should never be
        # reported as either a win or a regression.
        shape, tile = "32x32x32", {"m": 4, "n": 8, "k": 8}
        reg = make_register_allocation(150, 20, 0, 0, 80)
        sched = make_schedule(8, 12.0, 2.0)
        baseline = make_record(shape, tile, 1, reg, sched, 1704)
        scheduled = make_record(shape, tile, 1, dict(reg), dict(sched), 1704)
        result = m.classify_pair(baseline, scheduled)
        self.assertEqual(result["classification"], "B")

    def test_mismatched_shape_is_rejected(self):
        baseline = make_record(
            "32x32x32", {"m": 8, "n": 8, "k": 8}, 1,
            make_register_allocation(242, 28, 0, 0, 113), make_schedule(16, 18.0, 1.0), 2608,
        )
        scheduled = make_record(
            "64x64x64", {"m": 8, "n": 8, "k": 8}, 2,
            make_register_allocation(242, 28, 0, 0, 113), make_schedule(16, 18.0, 1.0), 2608,
        )
        with self.assertRaises(m.MismatchedComparisonError):
            m.classify_pair(baseline, scheduled)

    def test_mismatched_tile_is_rejected(self):
        baseline = make_record(
            "32x32x32", {"m": 8, "n": 8, "k": 8}, 1,
            make_register_allocation(242, 28, 0, 0, 113), make_schedule(16, 18.0, 1.0), 2608,
        )
        scheduled = make_record(
            "32x32x32", {"m": 4, "n": 8, "k": 8}, 2,
            make_register_allocation(242, 28, 0, 0, 113), make_schedule(16, 18.0, 1.0), 2608,
        )
        with self.assertRaises(m.MismatchedComparisonError):
            m.classify_pair(baseline, scheduled)


if __name__ == "__main__":
    unittest.main()
