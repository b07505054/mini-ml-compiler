"""Deterministic tests for profile-guided MatMul post-op kernel selection.

All profile evidence in this file is synthetic fixture data. No test asserts
a latency threshold; tests assert selection, matching, validation, fallback,
and ExecutionPlan emission behavior only.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import mlir_fusion_to_runtime_json as bridge


SHAPE = {"m": 128, "n": 128, "k": 128, "dtype": "f32"}
CONFIG = dict(bridge.DEFAULT_MATMUL_KERNEL_CONFIG)

FUSED_MLIR = (
    '%0 = "hir.fused_matmul_bias_relu"(%arg0, %arg1, %arg2) : '
    "(tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<128x128xf32>"
)


def variant_record(kernel_id, mean, p50=None, p95=None, cv=0.005, tile_size=32, passed=True):
    return {
        "kernel_id": kernel_id,
        "statistics": {
            "mean_ms": mean,
            "p50_ms": p50 if p50 is not None else mean,
            "p95_ms": p95 if p95 is not None else mean * 1.01,
            "stddev_ms": mean * cv,
            "coefficient_of_variation": cv,
        },
        "correctness": {"passed": passed},
        "implementation_properties": {"tile_size": tile_size},
    }


def bias_profile_payload():
    return {
        "schema": "kernel_benchmark_profile",
        "schema_version": 2,
        "benchmark": "matmul_postop_relu",
        "mode": "sweep-candidates",
        "machine": {"hostname": "fixture-host"},
        "configuration": {
            "warmup": 50,
            "iterations": 300,
            "repeats": 5,
            "dtype": "f32",
            "m": 128,
            "n": 128,
            "k": 128,
            "tile_size": 32,
        },
        "patterns": {
            "bias": {
                "postop_semantics": "bias_shape_N",
                "variants": {
                    "naive_unfused": variant_record(
                        "cpu_naive_matmul_bias_relu_unfused_f32", 1.75, tile_size=0
                    ),
                    "tiled_unfused": variant_record(
                        "cpu_tiled_matmul_bias_relu_unfused_f32", 0.4852
                    ),
                    "naive_one_pass_fused": variant_record(
                        "cpu_naive_matmul_bias_relu_one_pass_f32", 1.77, tile_size=0
                    ),
                    "tiled_one_pass_fused": variant_record(
                        "cpu_tiled_matmul_bias_relu_one_pass_f32", 0.4149
                    ),
                },
            }
        },
    }


class ProfileSelectionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="profile_selection_"))

    def load_profile(self, payload):
        path = self.tmpdir / "profile.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return bridge.load_kernel_profiles([str(path)])

    def select_bias(self, payload, shape=None, config=None):
        profile = self.load_profile(payload)
        return bridge.select_matmul_kernel(
            "matmul_bias_relu", shape or SHAPE, config or CONFIG, profile
        )

    def assert_fallback(self, selection, reason):
        self.assertEqual(selection["selection"]["policy"], "safe_fallback")
        self.assertTrue(selection["selection"]["fallback_used"])
        self.assertEqual(selection["selection"]["fallback_reason"], reason)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_unfused_f32")
        self.assertFalse(selection["profile_calibrated"])

    def test_exact_match_selects_fastest_valid_candidate(self):
        selection = self.select_bias(bias_profile_payload())
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_one_pass_f32")
        self.assertEqual(selection["selection"]["policy"], "profile_guided_latency")
        self.assertEqual(selection["selection"]["metric"], "mean_latency_ms")
        self.assertEqual(selection["selection"]["profile_match"], "exact")
        self.assertFalse(selection["selection"]["fallback_used"])
        self.assertAlmostEqual(selection["selection"]["selected_value"], 0.4149)
        ranks = {c["kernel_id"]: c["rank"] for c in selection["kernel_candidates"]}
        self.assertEqual(ranks["cpu_tiled_matmul_bias_relu_one_pass_f32"], 1)
        self.assertEqual(ranks["cpu_tiled_matmul_bias_relu_unfused_f32"], 2)

    def test_wrong_shape_profile_is_not_used(self):
        payload = bias_profile_payload()
        payload["configuration"]["m"] = 256
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "no_exact_shape_match")

    def test_bias_profile_not_used_for_elementwise_add(self):
        profile = self.load_profile(bias_profile_payload())
        selection = bridge.select_matmul_kernel("matmul_add_relu", SHAPE, CONFIG, profile)
        self.assertEqual(selection["selection"]["fallback_reason"], "no_matching_pattern")
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_add_relu_unfused_f32")

    def test_wrong_dtype_is_rejected(self):
        payload = bias_profile_payload()
        payload["configuration"]["dtype"] = "f16"
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "no_exact_shape_match")

    def test_failed_correctness_record_is_ignored(self):
        payload = bias_profile_payload()
        payload["patterns"]["bias"]["variants"]["tiled_one_pass_fused"]["correctness"]["passed"] = False
        selection = self.select_bias(payload)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_unfused_f32")
        self.assertEqual(selection["selection"]["policy"], "profile_guided_latency")
        by_id = {c["kernel_id"]: c for c in selection["kernel_candidates"]}
        entry = by_id["cpu_tiled_matmul_bias_relu_one_pass_f32"]
        self.assertFalse(entry["eligible"])
        self.assertEqual(entry["ineligible_reason"], "correctness_failed")

    def test_all_failed_correctness_uses_explicit_fallback(self):
        payload = bias_profile_payload()
        for variant in payload["patterns"]["bias"]["variants"].values():
            variant["correctness"]["passed"] = False
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "no_correctness_passing_candidate")

    def test_invalid_latency_is_rejected(self):
        for bad_value in (0.0, -1.0, float("nan")):
            payload = bias_profile_payload()
            variants = payload["patterns"]["bias"]["variants"]
            variants["tiled_one_pass_fused"]["statistics"]["mean_ms"] = bad_value
            selection = self.select_bias(payload)
            self.assertEqual(
                selection["selected_kernel"],
                "cpu_tiled_matmul_bias_relu_unfused_f32",
                f"bad latency {bad_value} must not win",
            )
            by_id = {c["kernel_id"]: c for c in selection["kernel_candidates"]}
            entry = by_id["cpu_tiled_matmul_bias_relu_one_pass_f32"]
            self.assertFalse(entry["eligible"])
            self.assertIn("invalid_profile_measurement", entry["ineligible_reason"])

    def test_zero_warmup_profile_is_rejected(self):
        payload = bias_profile_payload()
        payload["configuration"]["warmup"] = 0
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "invalid_profile_measurement")

    def test_unsupported_schema_uses_explicit_fallback(self):
        payload = bias_profile_payload()
        payload["schema_version"] = 99
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "unsupported_profile_schema")

    def test_missing_profile_uses_explicit_fallback(self):
        profile = bridge.load_kernel_profiles(None)
        selection = bridge.select_matmul_kernel("matmul_bias_relu", SHAPE, CONFIG, profile)
        self.assert_fallback(selection, "profile_not_provided")
        profile = bridge.load_kernel_profiles([str(self.tmpdir / "does_not_exist.json")])
        selection = bridge.select_matmul_kernel("matmul_bias_relu", SHAPE, CONFIG, profile)
        self.assert_fallback(selection, "profile_not_provided")

    def test_illegal_candidate_kernel_is_not_selected(self):
        payload = bias_profile_payload()
        variants = payload["patterns"]["bias"]["variants"]
        # A measured kernel that is not in the legal candidate set for this op
        # must never be selected, even if it is the fastest record.
        variants["rogue"] = variant_record("cpu_tiled_matmul_bias_relu_one_pass_f16", 0.0001)
        selection = self.select_bias(payload)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_one_pass_f32")
        self.assertNotIn(
            "cpu_tiled_matmul_bias_relu_one_pass_f16",
            [c["kernel_id"] for c in selection["kernel_candidates"]],
        )

    def test_only_illegal_candidates_uses_explicit_fallback(self):
        payload = bias_profile_payload()
        payload["patterns"]["bias"]["variants"] = {
            "rogue": variant_record("cpu_tiled_matmul_bias_relu_one_pass_f16", 0.1)
        }
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "all_profiled_candidates_illegal")

    def test_kernel_config_mismatch_is_not_an_exact_match(self):
        payload = bias_profile_payload()
        for variant in payload["patterns"]["bias"]["variants"].values():
            if variant["implementation_properties"]["tile_size"]:
                variant["implementation_properties"]["tile_size"] = 16
        selection = self.select_bias(payload)
        # Tiled kernels were measured at tile 16, plan requires tile 32; only
        # the naive kernels remain eligible.
        self.assertEqual(selection["selected_kernel"], "cpu_naive_matmul_bias_relu_unfused_f32")
        by_id = {c["kernel_id"]: c for c in selection["kernel_candidates"]}
        self.assertEqual(
            by_id["cpu_tiled_matmul_bias_relu_one_pass_f32"]["ineligible_reason"],
            "kernel_config_mismatch",
        )

    def test_deterministic_tie_breaking(self):
        payload = bias_profile_payload()
        variants = payload["patterns"]["bias"]["variants"]
        # Equal mean: lower p95 must win.
        variants["tiled_unfused"] = variant_record(
            "cpu_tiled_matmul_bias_relu_unfused_f32", 0.4, p95=0.41
        )
        variants["tiled_one_pass_fused"] = variant_record(
            "cpu_tiled_matmul_bias_relu_one_pass_f32", 0.4, p95=0.42
        )
        selection = self.select_bias(payload)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_unfused_f32")

        # Equal mean/p95/CV: fewer runtime dispatches must win.
        payload = bias_profile_payload()
        variants = payload["patterns"]["bias"]["variants"]
        variants["tiled_unfused"] = variant_record(
            "cpu_tiled_matmul_bias_relu_unfused_f32", 0.4, p95=0.41, cv=0.005
        )
        variants["tiled_one_pass_fused"] = variant_record(
            "cpu_tiled_matmul_bias_relu_one_pass_f32", 0.4, p95=0.41, cv=0.005
        )
        selection = self.select_bias(payload)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_one_pass_f32")
        self.assertEqual(
            selection["selection"]["tie_breaker_order"],
            [
                "p95_ms",
                "coefficient_of_variation",
                "runtime_dispatch_count",
                "intermediate_tensor_count",
                "kernel_id_lexical",
            ],
        )

    def test_cv_above_threshold_is_rejected(self):
        payload = bias_profile_payload()
        variants = payload["patterns"]["bias"]["variants"]
        variants["tiled_one_pass_fused"]["statistics"]["coefficient_of_variation"] = 0.5
        selection = self.select_bias(payload)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_unfused_f32")

    def test_use_plan_output_is_rejected_as_selection_evidence(self):
        payload = bias_profile_payload()
        payload["mode"] = "use-plan"
        selection = self.select_bias(payload)
        self.assert_fallback(selection, "unsupported_profile_schema")
        self.assertIn(
            "use_plan_output_rejected_as_selection_evidence",
            selection["selection"].get("fallback_detail", ""),
        )

    def test_later_profile_overrides_earlier(self):
        first = bias_profile_payload()
        second = copy.deepcopy(first)
        second["patterns"]["bias"]["variants"]["tiled_one_pass_fused"]["correctness"]["passed"] = False
        path_a = self.tmpdir / "a.json"
        path_b = self.tmpdir / "b.json"
        path_a.write_text(json.dumps(first), encoding="utf-8")
        path_b.write_text(json.dumps(second), encoding="utf-8")
        profile = bridge.load_kernel_profiles([str(path_a), str(path_b)])
        selection = bridge.select_matmul_kernel("matmul_bias_relu", SHAPE, CONFIG, profile)
        self.assertEqual(selection["selected_kernel"], "cpu_tiled_matmul_bias_relu_unfused_f32")


class ExecutionPlanEmissionTestCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="profile_selection_plan_"))

    def build_plan(self, profile_payload):
        profile_paths = None
        if profile_payload is not None:
            path = self.tmpdir / "profile.json"
            path.write_text(json.dumps(profile_payload), encoding="utf-8")
            profile_paths = [str(path)]
        profile = bridge.load_kernel_profiles(profile_paths)
        matches = bridge.detect_fused_matmul(FUSED_MLIR)
        self.assertEqual(len(matches), 1)
        lowered = bridge.build_lowered_graph(matches, [], [], "fixture.mlir", profile, FUSED_MLIR, "CUDA")
        return bridge.build_execution_plan(lowered)

    def test_selection_policy_and_evidence_appear_in_execution_plan(self):
        plan = self.build_plan(bias_profile_payload())
        self.assertEqual(len(plan["operations"]), 1)
        operation = plan["operations"][0]
        self.assertEqual(operation["selected_kernel"], "cpu_tiled_matmul_bias_relu_one_pass_f32")
        self.assertEqual(operation["selection"]["policy"], "profile_guided_latency")
        self.assertEqual(operation["selection"]["metric"], "mean_latency_ms")
        self.assertFalse(operation["selection"]["fallback_used"])
        self.assertEqual(operation["selection"]["profile_match"], "exact")
        ranked = [c for c in operation["kernel_candidates"] if c["eligible"]]
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertEqual(ranked[0]["kernel_id"], "cpu_tiled_matmul_bias_relu_one_pass_f32")
        # The runtime dispatch contract must carry the same kernel.
        step = plan["steps"][0]
        self.assertEqual(step["runtime_kernel"], "cpu_tiled_matmul_bias_relu_one_pass_f32")
        self.assertEqual(
            step["runtime_dispatch_contract"]["runtime_kernel"],
            "cpu_tiled_matmul_bias_relu_one_pass_f32",
        )

    def test_fallback_policy_and_reason_appear_in_execution_plan(self):
        plan = self.build_plan(None)
        operation = plan["operations"][0]
        self.assertEqual(operation["selected_kernel"], "cpu_tiled_matmul_bias_relu_unfused_f32")
        self.assertEqual(operation["selection"]["policy"], "safe_fallback")
        self.assertTrue(operation["selection"]["fallback_used"])
        self.assertEqual(operation["selection"]["fallback_reason"], "profile_not_provided")
        self.assertEqual(plan["kernel_profile"]["status"], "not_provided")

    def test_plan_remains_loadable_by_runtime_contract_fields(self):
        plan = self.build_plan(bias_profile_payload())
        operation = plan["operations"][0]
        for field in ("op_id", "op_type", "backend", "selected_kernel", "kernel_config", "inputs", "outputs"):
            self.assertIn(field, operation)
        self.assertEqual(operation["op_type"], "FusedMatMulBiasRelu")
        self.assertEqual(operation["backend"], "cpu")
        self.assertEqual(operation["kernel_config"], {"tile_m": 32, "tile_n": 32, "tile_k": 32})
        self.assertEqual(plan["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
