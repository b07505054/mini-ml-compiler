from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MLIR_OPT = Path("/home/allen/Desktop/Project/.deps/mlir21-root/usr/lib/llvm-21/bin/mlir-opt")
PLUGIN = REPO_ROOT / "build-mlir" / "libHIRMatMulBiasReluFusionPass.so"

KERNEL_ID = "portable_fused_matmul_bias_relu_bm32_bn128_bk32"
POLICY_ID = "raspberry_pi5_fused_matmul_bias_relu_thread_schedule_p1d1"


def _fixture(
    m: str,
    n: str,
    k: str,
    *,
    physical_units: int | None = 4,
    include_parallel_schedule: bool = True,
    policy_kernel_id: str = KERNEL_ID,
    policy_dtype: str = "fp32",
    descriptor_dtype: str = "fp32",
    duplicate_policy_schedules: bool = False,
) -> str:
    physical = (
        "" if physical_units is None
        else f"target.hardware.physical_compute_units = {physical_units} : i64,"
    )
    schedules = (
        '{thread_count = 1 : i64, partition_axis = "none", partition_strategy = "serial"}'
    )
    if include_parallel_schedule:
        schedules += (
            ', {thread_count = 4 : i64, partition_axis = "m", '
            'partition_strategy = "contiguous_chunks"}'
        )
    above_axis = "none" if duplicate_policy_schedules else "m"
    above_strategy = "serial" if duplicate_policy_schedules else "contiguous_chunks"
    above_threads = 1 if duplicate_policy_schedules else 4
    return f'''
module attributes {{
  target.profile_id = "raspberry-pi5-cortex-a76-cpu",
  {physical}
  target.thread_schedule_policy.policy_id = "{POLICY_ID}",
  target.thread_schedule_policy.policy_version = "p1d1.v1",
  target.thread_schedule_policy.target_profile_id = "raspberry-pi5-cortex-a76-cpu",
  target.thread_schedule_policy.fused_region_identity = "fused_matmul_bias_relu",
  target.thread_schedule_policy.dtype = "{policy_dtype}",
  target.thread_schedule_policy.kernel_id = "{policy_kernel_id}",
  target.thread_schedule_policy.metric = "matmul_mnk",
  target.thread_schedule_policy.threshold = 262144 : i64,
  target.thread_schedule_policy.boundary_rule = "at_or_above_threshold_selects_parallel",
  target.thread_schedule_policy.below_threshold.thread_count = 1 : i64,
  target.thread_schedule_policy.below_threshold.partition_axis = "none",
  target.thread_schedule_policy.below_threshold.partition_strategy = "serial",
  target.thread_schedule_policy.at_or_above_threshold.thread_count = {above_threads} : i64,
  target.thread_schedule_policy.at_or_above_threshold.partition_axis = "{above_axis}",
  target.thread_schedule_policy.at_or_above_threshold.partition_strategy = "{above_strategy}",
  target.thread_schedule_policy.calibration_evidence_ref = "test_evidence",
  target.thread_schedule_policy.evidence_sha256 = "sha256:test",
  target.thread_schedule_policy.truth_boundary = "offline_calibration_test",
  target.runtime_kernels = [{{
    kernel_id = "{KERNEL_ID}",
    op_name = "fused_matmul_bias_relu",
    backend = "cpu",
    supported_dtypes = ["{descriptor_dtype}"],
    supported_quant_modes = ["none"],
    supported_thread_schedules = [{schedules}],
    requires_static_shape = true,
    source = "test_descriptor",
    truth_boundary = "test_kernel_descriptor"
  }}]
}} {{
  func.func @matmul_bias_relu_main(%a: tensor<{m}x{k}xf32>) -> tensor<{m}x{n}xf32>
      attributes {{representation.source_backend = "cpu", representation.effective_dtype = "f32"}} {{
    %b = "test.weight"() {{weight.is_constant = true}} : () -> tensor<{k}x{n}xf32>
    %bias = "test.weight"() {{weight.is_constant = true}} : () -> tensor<{n}xf32>
    %0 = "hir.fused_matmul_bias_relu"(%a, %b, %bias)
        : (tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>, tensor<{n}xf32>) -> tensor<{m}x{n}xf32>
    return %0 : tensor<{m}x{n}xf32>
  }}
}}
'''


def _run_kernel_selection(mlir_text: str) -> str:
    with tempfile.TemporaryDirectory() as td:
      path = Path(td) / "input.mlir"
      path.write_text(mlir_text)
      completed = subprocess.run(
          [
              str(MLIR_OPT),
              "--allow-unregistered-dialect",
              f"--load-pass-plugin={PLUGIN}",
              "--pass-pipeline=builtin.module(kernel-selection-pipeline)",
              str(path),
          ],
          capture_output=True,
          text=True,
          check=False,
      )
      if completed.returncode != 0:
          raise AssertionError(completed.stderr)
      return completed.stdout


def _thread_op(output: str) -> str:
    for line in output.splitlines():
        if '"hir.fused_matmul_bias_relu"' in line:
            return line
    raise AssertionError(output)


def _selected_candidate(line: str) -> str:
    match = re.search(r'thread_schedule.selected_candidate_id = "([^"]+)"', line)
    if not match:
        raise AssertionError(line)
    return match.group(1)


def _attr(line: str, name: str) -> str:
    match = re.search(rf'{re.escape(name)} = "([^"]+)"', line)
    if not match:
        raise AssertionError(f"{name} missing from: {line}")
    return match.group(1)


class A2ThreadScheduleCandidateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MLIR_OPT.exists() or not PLUGIN.exists():
            raise unittest.SkipTest("requires mlir-opt and built HIR plugin")

    def test_tiny_selects_serial_candidate(self):
        line = _thread_op(_run_kernel_selection(_fixture("8", "8", "8")))
        self.assertIn("thread_schedule.thread_count = 1", line)
        self.assertIn("metric_below_threshold_select_serial", line)
        selected = _selected_candidate(line)
        self.assertEqual(_attr(line, "implementation_candidate.backend"), "cpu")
        self.assertEqual(
            _attr(line, "implementation_candidate.implementation_kind"),
            "opaque_portable_cpu_native_kernel",
        )
        self.assertEqual(
            _attr(line, "implementation_candidate.runtime_contract_kind"),
            "portable_cpu_kernel_adapter_contract",
        )
        self.assertEqual(_attr(line, "implementation_candidate.kernel_id"), KERNEL_ID)
        self.assertEqual(_attr(line, "implementation_candidate.dtype"), "fp32")
        self.assertEqual(
            _attr(line, "implementation_candidate.tile_identity"),
            "bm32_bn128_bk32",
        )
        self.assertIn("threads=1", selected)
        self.assertIn("scope=fused_region", selected)
        self.assertIn("backend=cpu", selected)
        self.assertIn("opaque_portable_cpu_native_kernel", selected)
        self.assertIn("contract=portable_cpu_kernel_adapter_contract", selected)
        self.assertIn("tile=bm32_bn128_bk32", selected)
        self.assertIn("dtype=fp32", selected)
        self.assertIn("axis=none", selected)
        self.assertIn("strategy=serial", selected)
        self.assertIn("threads=4", line)
        self.assertIn("metric_below_threshold", line)

    def test_threshold_selects_parallel_candidate(self):
        line = _thread_op(_run_kernel_selection(_fixture("64", "64", "64")))
        self.assertIn("thread_schedule.thread_count = 4", line)
        self.assertIn("metric_at_or_above_threshold_select_parallel", line)
        selected = _selected_candidate(line)
        self.assertIn("threads=4", selected)
        self.assertIn("axis=m", selected)
        self.assertIn("contiguous_chunks", selected)
        self.assertIn("kernel=" + KERNEL_ID, selected)
        self.assertIn("tile=bm32_bn128_bk32", selected)
        self.assertIn("dtype=fp32", selected)
        self.assertIn("metric_at_or_above_threshold", line)

    def test_materialized_kernel_tile_dtype_come_from_selected_candidate(self):
        line = _thread_op(_run_kernel_selection(_fixture("64", "64", "64")))
        selected = _selected_candidate(line)
        self.assertEqual(_attr(line, "implementation_candidate.selected_id"), selected)
        self.assertEqual(_attr(line, "kernel_selection.selected_id"), KERNEL_ID)
        self.assertIn("implementation_candidate.tile_block_m = 32", line)
        self.assertIn("implementation_candidate.tile_block_n = 128", line)
        self.assertIn("implementation_candidate.tile_block_k = 32", line)

    def test_candidate_ids_are_deterministic(self):
        line_a = _thread_op(_run_kernel_selection(_fixture("64", "64", "64")))
        line_b = _thread_op(_run_kernel_selection(_fixture("64", "64", "64")))
        self.assertEqual(_selected_candidate(line_a), _selected_candidate(line_b))

    def test_parallel_rejected_when_compute_units_insufficient(self):
        line = _thread_op(
            _run_kernel_selection(_fixture("64", "64", "64", physical_units=2))
        )
        self.assertIn("thread_schedule.thread_count = 1", line)
        self.assertIn("policy_parallel_rejected_exceeds_compute_units", line)
        self.assertIn("rejected_exceeds_compute_units", line)

    def test_parallel_rejected_when_schedule_missing(self):
        line = _thread_op(
            _run_kernel_selection(
                _fixture("64", "64", "64", include_parallel_schedule=False)
            )
        )
        self.assertIn("thread_schedule.thread_count = 1", line)
        self.assertIn("policy_parallel_schedule_not_declared", line)

    def test_wrong_policy_kernel_selects_serial_candidate(self):
        line = _thread_op(
            _run_kernel_selection(
                _fixture("64", "64", "64", policy_kernel_id="wrong_kernel")
            )
        )
        self.assertIn("thread_schedule.thread_count = 1", line)
        self.assertIn("policy_kernel_mismatch_serial_fallback", line)
        self.assertIn("policy_kernel_mismatch", line)

    def test_wrong_dtype_rejects_policy_activation(self):
        line = _thread_op(
            _run_kernel_selection(_fixture("64", "64", "64", policy_dtype="f16"))
        )
        self.assertIn("thread_schedule.thread_count = 1", line)
        self.assertIn("policy_dtype_mismatch_serial_fallback", line)
        self.assertIn("policy_dtype_mismatch", line)

    def test_unknown_shape_defers_before_thread_candidates(self):
        line = _thread_op(_run_kernel_selection(_fixture("?", "64", "64")))
        self.assertIn("kernel_selection.status = \"deferred_dynamic_shape\"", line)
        self.assertNotIn("thread_schedule.selected_candidate_id", line)

    def test_candidate_id_collision_is_rejected(self):
        line = _thread_op(
            _run_kernel_selection(
                _fixture("64", "64", "64", duplicate_policy_schedules=True)
            )
        )
        self.assertIn(
            "thread_schedule.status = \"rejected_thread_schedule_candidate_id_collision\"",
            line,
        )
        self.assertIn("thread_schedule_candidate_id_collision", line)


if __name__ == "__main__":
    unittest.main()
