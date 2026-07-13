from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILE_FOR_TARGET = REPO_ROOT / "build-mlir" / "compile-for-target"
BASE_PROFILE = REPO_ROOT / "configs" / "target_profiles" / "raspberry_pi5_cortex_a76_cpu.json"
POLICY_ID = "raspberry_pi5_fused_matmul_bias_relu_thread_schedule_p1d1"


def _write_mlir(path: Path, m: str, n: str, k: str, *, op: str = "hir.fused_matmul_bias_relu", dtype: str = "f32") -> None:
    path.write_text(
        f'''module {{
  func.func @matmul_bias_relu_main(%a: tensor<{m}x{k}x{dtype}>)
      -> tensor<{m}x{n}x{dtype}> attributes {{cv.semantic_annotation.status = "completed"}} {{
    %b = "test.weight"() {{weight.is_constant = true}} : () -> tensor<{k}x{n}x{dtype}>
    %bias = "test.weight"() {{weight.is_constant = true}} : () -> tensor<{n}x{dtype}>
    %0 = "{op}"(%a, %b, %bias)
        : (tensor<{m}x{k}x{dtype}>, tensor<{k}x{n}x{dtype}>, tensor<{n}x{dtype}>) -> tensor<{m}x{n}x{dtype}>
    return %0 : tensor<{m}x{n}x{dtype}>
  }}
}}
'''
    )


def _profile(tmp_path: Path, mutate=None) -> Path:
    profile = json.loads(BASE_PROFILE.read_text())
    if "threadSchedulePolicy" in profile:
        profile["threadSchedulePolicy"]["path"] = str(
            REPO_ROOT
            / "configs"
            / "thread_schedule_policies"
            / "raspberry_pi5_cortex_a76_p1d1_thread_policy.json"
        )
    if mutate:
        mutate(profile, tmp_path)
    out = tmp_path / "profile.json"
    out.write_text(json.dumps(profile, indent=2) + "\n")
    return out


def _compile(tmp_path: Path, profile: Path, m: str, n: str, k: str, **mlir_kwargs) -> dict:
    mlir = tmp_path / "input.mlir"
    out = tmp_path / "execution_plan.json"
    _write_mlir(mlir, m, n, k, **mlir_kwargs)
    completed = subprocess.run(
        [str(COMPILE_FOR_TARGET), "--device-profile", str(profile), "--mlir", str(mlir), "--out", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(completed.stderr)
    return json.loads(out.read_text())


def _op(plan: dict, suffix: str = "fused_matmul_bias_relu") -> dict:
    for fp in plan.get("function_plans", []):
        for op in fp.get("per_op_decisions", []):
            if op.get("op_type", "").endswith(suffix):
                return op
    raise AssertionError(f"no op ending in {suffix}")


def _thread(plan: dict) -> dict:
    return _op(plan)["thread_schedule"]


class P1D1ThreadSchedulePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (COMPILE_FOR_TARGET.exists() and BASE_PROFILE.exists()):
            raise unittest.SkipTest("requires built compile-for-target and Raspberry Pi target profile")

    def run_case(self, func):
        with tempfile.TemporaryDirectory() as td:
            return func(Path(td))

    def test_tiny_shape_selects_serial(self):
        def case(tmp_path):
            ts = _thread(_compile(tmp_path, _profile(tmp_path), "8", "8", "8"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertEqual(ts["partition_axis"], "none")
            self.assertEqual(ts["policy_metric_value"], 512)
            self.assertEqual(ts["policy_selection_reason"], "metric_below_threshold_select_serial")
        self.run_case(case)

    def test_below_threshold_shape_selects_serial(self):
        def case(tmp_path):
            ts = _thread(_compile(tmp_path, _profile(tmp_path), "16", "16", "127"))
            self.assertEqual(ts["policy_metric_value"], 32512)
            self.assertLess(ts["policy_metric_value"], ts["policy_threshold"])
            self.assertEqual(ts["thread_count"], 1)
        self.run_case(case)

    def test_exact_threshold_selects_parallel(self):
        def case(tmp_path):
            ts = _thread(_compile(tmp_path, _profile(tmp_path), "64", "64", "64"))
            self.assertEqual(ts["policy_metric_value"], 262144)
            self.assertEqual(ts["policy_threshold"], 262144)
            self.assertEqual(ts["thread_count"], 4)
            self.assertEqual(ts["partition_axis"], "m")
            self.assertEqual(ts["partition_strategy"], "contiguous_chunks")
        self.run_case(case)

    def test_above_threshold_selects_split_m_4thread(self):
        def case(tmp_path):
            ts = _thread(_compile(tmp_path, _profile(tmp_path), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 4)
            self.assertEqual(ts["partition_axis"], "m")
            self.assertEqual(ts["policy_id"], POLICY_ID)
            self.assertTrue(ts["policy_evidence_sha256"].startswith("sha256:"))
        self.run_case(case)

    def test_wrong_op_does_not_activate_policy(self):
        def case(tmp_path):
            plan = _compile(tmp_path, _profile(tmp_path), "64", "64", "64", op="hir.not_fused_matmul_bias_relu")
            op = _op(plan, "not_fused_matmul_bias_relu")
            self.assertEqual(op["kernel_selection"]["status"], "rejected_no_kernel_for_op")
            self.assertNotIn("thread_schedule", op)
        self.run_case(case)

    def test_wrong_dtype_does_not_activate_policy(self):
        def case(tmp_path):
            plan = _compile(tmp_path, _profile(tmp_path), "64", "64", "64", dtype="f16")
            op = _op(plan)
            self.assertEqual(op["kernel_selection"]["status"], "rejected_dtype_unsupported")
            self.assertNotIn("thread_schedule", op)
        self.run_case(case)

    def test_wrong_kernel_uses_serial_policy_fallback(self):
        def case(tmp_path):
            def mutate(profile, _tmp):
                profile["runtimeKernels"][0]["kernelId"] = "portable_fused_matmul_bias_relu_bm32_bn32_bk32"
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertEqual(ts["policy_selection_reason"], "policy_kernel_mismatch_serial_fallback")
        self.run_case(case)

    def test_missing_policy_preserves_existing_serial_behavior(self):
        def case(tmp_path):
            def mutate(profile, _tmp):
                profile.pop("threadSchedulePolicy", None)
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertNotIn("policy_id", ts)
        self.run_case(case)

    def test_missing_physical_compute_units_prevents_parallel(self):
        def case(tmp_path):
            def mutate(profile, _tmp):
                profile.pop("hardwareExecutionProfile", None)
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertEqual(ts["policy_selection_reason"], "metric_at_or_above_threshold_but_missing_compute_units_serial_fallback")
        self.run_case(case)

    def test_physical_compute_units_less_than_four_prevents_parallel(self):
        def case(tmp_path):
            def mutate(profile, _tmp):
                profile["hardwareExecutionProfile"]["physicalComputeUnits"] = 2
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertEqual(ts["policy_selection_reason"], "metric_at_or_above_threshold_but_compute_units_insufficient_serial_fallback")
        self.run_case(case)

    def test_missing_four_thread_candidate_prevents_parallel(self):
        def case(tmp_path):
            def mutate(profile, _tmp):
                schedules = profile["runtimeKernels"][0]["threadSchedules"]
                profile["runtimeKernels"][0]["threadSchedules"] = [
                    s for s in schedules if not (s["threadCount"] == 4 and s["partitionAxis"] == "m")
                ]
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertIn("policy_parallel_schedule_not_declared", ts["rejection_reasons"])
        self.run_case(case)

    def test_static_unknown_dimensions_do_not_select_parallel(self):
        def case(tmp_path):
            plan = _compile(tmp_path, _profile(tmp_path), "?", "64", "64")
            op = _op(plan)
            self.assertEqual(op["kernel_selection"]["status"], "deferred_dynamic_shape")
            self.assertNotIn("thread_schedule", op)
        self.run_case(case)

    def test_policy_target_profile_mismatch_ignores_policy(self):
        def case(tmp_path):
            base_policy_path = REPO_ROOT / "configs" / "thread_schedule_policies" / "raspberry_pi5_cortex_a76_p1d1_thread_policy.json"
            def mutate(profile, tmp):
                policy = json.loads(base_policy_path.read_text())
                policy["target_profile_id"] = "wrong-target"
                policy_path = tmp / "policy.json"
                policy_path.write_text(json.dumps(policy, indent=2) + "\n")
                profile["threadSchedulePolicy"]["path"] = str(policy_path)
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertNotIn("policy_id", ts)
        self.run_case(case)

    def test_wrong_evidence_hash_ignores_policy(self):
        def case(tmp_path):
            def mutate(profile, _tmp):
                profile["threadSchedulePolicy"]["expectedEvidenceSha256"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
            ts = _thread(_compile(tmp_path, _profile(tmp_path, mutate), "64", "64", "64"))
            self.assertEqual(ts["thread_count"], 1)
            self.assertNotIn("policy_id", ts)
        self.run_case(case)

    def test_execution_plan_preserves_policy_provenance(self):
        def case(tmp_path):
            ts = _thread(_compile(tmp_path, _profile(tmp_path), "64", "64", "64"))
            self.assertEqual(ts["policy_id"], POLICY_ID)
            self.assertEqual(ts["policy_version"], "p1d1.v1")
            self.assertEqual(ts["policy_metric"], "matmul_mnk")
            self.assertEqual(ts["policy_threshold"], 262144)
            self.assertEqual(ts["policy_boundary_rule"], "at_or_above_threshold_selects_parallel")
            self.assertIn("p1d_raw_measurements.json", ts["policy_evidence_ref"])
            self.assertIn("offline_calibration", ts["policy_truth_boundary"])
        self.run_case(case)


if __name__ == "__main__":
    unittest.main()
