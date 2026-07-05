import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import export_vllm_execution_plan as exporter  # noqa: E402


class TestVLLMExecutionPlanExport(unittest.TestCase):
    def test_vllm_execution_plan_can_be_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            output = tmp_path / "qwen_0_5b_gtx1650_plan.json"
            plan = exporter.build_plan(
                created_at_utc="2026-07-04T00:00:00+00:00",
                git_commit="test-commit",
                capabilities_root=_capability_root(tmp_path),
            )
            exporter.write_plan(output, plan)

            self.assertTrue(output.exists())
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["artifact_type"], "vllm_execution_plan")
            self.assertTrue(loaded["truth_boundary"])
            self.assertEqual(loaded["model"]["model_id"], "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(loaded["runtime_config"]["tensor_parallel_size"], 1)
            self.assertEqual(loaded["runtime_config"]["pipeline_parallel_size"], 1)
            self.assertFalse(loaded["speculative_policy"]["enabled"])

    def test_invalid_artifact_type_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))
            plan["artifact_type"] = "serving_execution_plan"

            with self.assertRaisesRegex(exporter.VLLMExecutionPlanExportError, "artifact_type"):
                exporter.validate_plan(plan)

    def test_invalid_schema_version_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))
            plan["schema_version"] = "0.1"

            with self.assertRaisesRegex(exporter.VLLMExecutionPlanExportError, "schema_version"):
                exporter.validate_plan(plan)

    def test_producer_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))

        self.assertEqual(plan["producer"]["repo"], "ml-graph-compiler-runtime")
        self.assertEqual(plan["producer"]["tool"], "tools/export_vllm_execution_plan.py")
        self.assertEqual(plan["producer"]["git_commit"], "test-commit")
        self.assertEqual(plan["producer"]["created_at_utc"], "2026-07-04T00:00:00+00:00")

    def test_source_artifacts_include_shared_capability_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))

        source_artifacts = plan["source_artifacts"]
        self.assertIn("configs/models/qwen_0_5b_spec.json", source_artifacts)
        self.assertTrue(any(path.endswith("hardware/nvidia_gtx1650_maxq.json") for path in source_artifacts))
        self.assertTrue(any(path.endswith("backend/vllm.json") for path in source_artifacts))
        self.assertTrue(any(path.endswith("kernels/flashattention2.json") for path in source_artifacts))
        self.assertTrue(any(path.endswith("kernels/cublas.json") for path in source_artifacts))

    def test_required_profile_and_policy_sections_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))

        for section in [
            "model",
            "hardware_profile",
            "backend_profile",
            "kernel_profile",
            "workload_profile",
            "batch_policy",
            "prefix_policy",
            "memory_policy",
            "quantization_policy",
            "speculative_policy",
            "runtime_config",
        ]:
            self.assertIn(section, plan)
        self.assertEqual(plan["hardware_profile"]["gpu_name"], "NVIDIA GeForce GTX 1650 Max-Q")
        self.assertEqual(plan["backend_profile"]["backend"], "vllm")
        self.assertTrue(plan["kernel_profile"]["available_kernels"])

    def test_plan_has_no_measured_speedup_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))

        self.assertEqual(
            plan["truth_boundary"],
            "Execution planning artifact only; not measured performance.",
        )
        self.assertNotIn("speedup", plan)
        self.assertNotIn("measured_speedup", plan)
        self.assertNotIn("performance_claim", plan)
        self.assertIn("no_measured_speedup_claim", plan["non_claims"])

    def test_measured_speedup_claim_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = _plan(Path(tmp))
            plan["measured_speedup"] = 1.2

            with self.assertRaisesRegex(exporter.VLLMExecutionPlanExportError, "speedup"):
                exporter.validate_plan(plan)


def _plan(tmp_path: Path) -> dict:
    return exporter.build_plan(
        created_at_utc="2026-07-04T00:00:00+00:00",
        git_commit="test-commit",
        capabilities_root=_capability_root(tmp_path),
    )


def _capability_root(tmp_path: Path) -> Path:
    root = tmp_path / "capabilities" / "profiles"
    (root / "hardware").mkdir(parents=True)
    (root / "backend").mkdir(parents=True)
    (root / "kernels").mkdir(parents=True)
    _write_json(root / "hardware" / "nvidia_gtx1650_maxq.json", _hardware_profile())
    _write_json(root / "backend" / "vllm.json", _backend_profile())
    for provider in ["flashattention2", "cutlass", "cublas", "triton", "xformers"]:
        _write_json(root / "kernels" / f"{provider}.json", _kernel_profile(provider))
    return root


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _hardware_profile() -> dict:
    return {
        "hardware_id": "nvidia_gtx1650_maxq",
        "vendor": "NVIDIA",
        "family": "Turing",
        "model": "NVIDIA GeForce GTX 1650 Max-Q",
        "memory": {"vram_mb": 4096, "vram_gb": 4},
        "attributes": {
            "cuda_runtime": "13.2",
            "compute_capability": "7.5",
            "tensor_core_support": "unknown",
            "fp16": True,
            "bf16": False,
            "fp8": False,
            "nvfp4": False,
            "mxfp4": False,
            "int8": False,
            "int4": False,
            "multi_gpu": False,
        },
    }


def _backend_profile() -> dict:
    return {
        "backend_id": "vllm",
        "backend": "vLLM",
        "backend_api": "OpenAI-compatible HTTP API",
        "supports": {
            "features": ["OpenAI-compatible serving", "continuous batching", "paged attention"],
            "quantization": ["none"],
            "speculative_decoding": {"supported": False, "measured": False},
            "prefix_cache": {"supported": True, "measured": False},
            "chunked_prefill": {"supported": True, "measured": False},
            "paged_attention": {"supported": True, "measured": False},
            "tensor_parallel": {"supported": False, "max_size": 1, "measured": False},
            "pipeline_parallel": {"supported": False, "max_size": 1, "measured": False},
        },
        "supported_kernel_libraries": ["flashattention2", "cutlass", "cublas", "triton", "xformers"],
        "does_not_claim": ["this repo implements vLLM"],
    }


def _kernel_profile(provider: str) -> dict:
    return {
        "profile_id": provider,
        "backend_id": "cuda",
        "kernels": [
            {
                "operation": "Attention" if provider in {"flashattention2", "triton", "xformers"} else "MatMul",
                "availability": "unsupported" if provider == "flashattention2" else "opaque",
                "support_status": "provider_profile_fixture",
                "supported_precisions": ["float16"],
                "supported_features": ["paged_attention"] if provider in {"flashattention2", "triton", "xformers"} else [],
                "measured": False,
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
