import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import generate_vllm_serving_plan as gvsp  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "tiny_gpt_llm_config.json"
TRACE_PATH = REPO_ROOT / "configs" / "vllm_serving_request_trace.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class TestVllmServingPlanGeneration(unittest.TestCase):
    """Tests that generate_all() writes all four artifacts with correct structure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._tmp.name)
        cls.model_cfg = _load(CONFIG_PATH)
        cls.request_trace = _load(TRACE_PATH)
        gvsp.generate_all(cls.model_cfg, cls.request_trace, cls.out_dir)
        cls.artifacts = {p.name: _load(p) for p in sorted(cls.out_dir.glob("*.json"))}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _get(self, name: str) -> dict:
        self.assertIn(name, self.artifacts, f"Missing artifact: {name}")
        return self.artifacts[name]

    def test_all_four_artifacts_present(self) -> None:
        for name in [
            "serving_analysis.json",
            "cuda_graph_bucket_plan.json",
            "kv_cache_layout_plan.json",
            "runtime_replan_report.json",
        ]:
            self._get(name)

    def test_all_artifacts_have_truth_boundary(self) -> None:
        for name, data in self.artifacts.items():
            self.assertIn("truth_boundary", data, f"{name} missing truth_boundary")
            self.assertIsInstance(data["truth_boundary"], str)

    def test_serving_analysis_schema(self) -> None:
        a = self._get("serving_analysis.json")
        self.assertEqual(a["artifact_type"], "serving_analysis")
        self.assertEqual(a["shape_classification"]["prefill"], "dynamic")
        self.assertEqual(a["shape_classification"]["decode"], "static")
        self.assertIsInstance(a["batch_shape_buckets"], list)
        self.assertGreater(len(a["batch_shape_buckets"]), 0)
        self.assertIsInstance(a["request_classes"], list)
        self.assertGreater(len(a["request_classes"]), 0)

    def test_serving_analysis_covers_all_requests(self) -> None:
        a = self._get("serving_analysis.json")
        all_ids = {r["request_id"] for r in self.request_trace["requests"]}
        classified_ids: set[str] = set()
        for rc in a["request_classes"]:
            classified_ids.update(rc["request_ids"])
        self.assertEqual(all_ids, classified_ids)

    def test_cuda_graph_bucket_plan_schema(self) -> None:
        a = self._get("cuda_graph_bucket_plan.json")
        self.assertEqual(a["artifact_type"], "cuda_graph_bucket_plan")
        buckets = a["decode_buckets"]
        self.assertGreater(len(buckets), 0)
        safe_values = {b["replay_safe"] for b in buckets}
        self.assertIn(True, safe_values, "Expected at least one replay_safe=True bucket")
        self.assertIn(False, safe_values, "Expected at least one replay_safe=False bucket")

    def test_cuda_graph_bucket_query_len_always_one(self) -> None:
        a = self._get("cuda_graph_bucket_plan.json")
        for b in a["decode_buckets"]:
            self.assertEqual(b["query_len"], 1)

    def test_cuda_graph_bucket_unsafe_has_fallback_reason(self) -> None:
        a = self._get("cuda_graph_bucket_plan.json")
        for b in a["decode_buckets"]:
            if not b["replay_safe"]:
                self.assertIsNotNone(b["fallback_reason"])
            else:
                self.assertIsNone(b["fallback_reason"])

    def test_kv_cache_layout_plan_schema(self) -> None:
        a = self._get("kv_cache_layout_plan.json")
        self.assertEqual(a["artifact_type"], "kv_cache_layout_plan")
        self.assertGreater(a["aggregate"]["total_estimated_kv_mb"], 0)
        entries = a["per_request_entries"]
        self.assertEqual(len(entries), len(self.request_trace["requests"]))

    def test_kv_cache_pd_split_threshold_consistent(self) -> None:
        a = self._get("kv_cache_layout_plan.json")
        threshold = a["pd_split_threshold_prompt_tokens"]
        for e in a["per_request_entries"]:
            if e["prompt_tokens"] >= threshold:
                self.assertTrue(e["pd_split_candidate"])
                self.assertEqual(e["recommendation"], "disaggregated")
            else:
                self.assertFalse(e["pd_split_candidate"])
                self.assertEqual(e["recommendation"], "colocated")

    def test_kv_cache_recommendation_values(self) -> None:
        a = self._get("kv_cache_layout_plan.json")
        valid = {"colocated", "disaggregated"}
        for e in a["per_request_entries"]:
            self.assertIn(e["recommendation"], valid)

    def test_runtime_replan_report_schema(self) -> None:
        a = self._get("runtime_replan_report.json")
        self.assertEqual(a["artifact_type"], "runtime_replan_report")
        self.assertIn("observed_metrics_note", a)
        for key in ["TTFT_ms", "TPOT_ms", "replay_hit_rate", "kv_transfer_latency_ms", "queue_wait_ms"]:
            self.assertIn(key, a["observed_metrics"])
        self.assertIsInstance(a["triggered_rules"], list)
        self.assertIsInstance(a["actions"], list)
        self.assertEqual(len(a["triggered_rules"]), len(a["actions"]))

    def test_runtime_replan_split_buckets_triggered(self) -> None:
        a = self._get("runtime_replan_report.json")
        self.assertIn("split_buckets", a["triggered_rules"])


class TestVllmServingPlanBuilders(unittest.TestCase):
    """Unit tests for individual builder functions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_cfg = _load(CONFIG_PATH)
        cls.request_trace = _load(TRACE_PATH)
        cls.analysis = gvsp.build_serving_analysis(cls.model_cfg, cls.request_trace)

    def test_cuda_graph_replay_safety_rule(self) -> None:
        plan = gvsp.build_cuda_graph_bucket_plan(self.analysis, self.model_cfg)
        for b in plan["decode_buckets"]:
            expected = b["batch_size"] <= 8 and b["context_bucket"] <= 2048
            self.assertEqual(b["replay_safe"], expected, f"bucket {b['bucket_id']} safety mismatch")

    def test_kv_bytes_positive_for_all_requests(self) -> None:
        plan = gvsp.build_kv_cache_layout_plan(self.request_trace, self.model_cfg)
        for e in plan["per_request_entries"]:
            self.assertGreater(e["estimated_kv_bytes"], 0)

    def test_kv_aggregate_equals_sum_of_entries(self) -> None:
        plan = gvsp.build_kv_cache_layout_plan(self.request_trace, self.model_cfg)
        total_bytes = sum(e["estimated_kv_bytes"] for e in plan["per_request_entries"])
        expected_mb = round(total_bytes / (1024 * 1024), 4)
        self.assertAlmostEqual(plan["aggregate"]["total_estimated_kv_mb"], expected_mb, places=3)

    def test_evaluate_rules_split_buckets_on_high_ttft(self) -> None:
        triggered, _ = gvsp._evaluate_rules({
            "TTFT_ms": 150.0,
            "TPOT_ms": 5.0,
            "replay_hit_rate": 0.5,
            "kv_transfer_latency_ms": 2.0,
            "queue_wait_ms": 5.0,
        })
        self.assertIn("split_buckets", triggered)

    def test_evaluate_rules_merge_buckets_on_high_hit_rate(self) -> None:
        triggered, _ = gvsp._evaluate_rules({
            "TTFT_ms": 50.0,
            "TPOT_ms": 5.0,
            "replay_hit_rate": 0.95,
            "kv_transfer_latency_ms": 2.0,
            "queue_wait_ms": 5.0,
        })
        self.assertIn("merge_buckets", triggered)

    def test_evaluate_rules_no_trigger_on_healthy_metrics(self) -> None:
        triggered, _ = gvsp._evaluate_rules({
            "TTFT_ms": 40.0,
            "TPOT_ms": 10.0,
            "replay_hit_rate": 0.7,
            "kv_transfer_latency_ms": 3.0,
            "queue_wait_ms": 5.0,
        })
        self.assertEqual(triggered, [])


if __name__ == "__main__":
    unittest.main()
