import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import compiler_serving_cost_model as cscm  # noqa: E402
import generate_vllm_serving_plan as gvsp    # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "tiny_gpt_llm_config.json"
TRACE_PATH = REPO_ROOT / "configs" / "vllm_serving_request_trace.json"

# Cross-repo artifact (skipped when absent)
GPU_ARTIFACT_PATH = (
    REPO_ROOT.parent / "heterogeneous-inference-runtime" / "results" /
    "llm_runtime_artifacts" / "gpu_decode_batch_scaling_gtx1650maxq.json"
)

_COST_COMPONENT_KEYS = ["compute_ms", "kv_transfer_ms", "queue_ms", "replay_gain_ms", "memory_penalty_ms"]

# Minimal in-memory artifact for unit tests — no disk I/O, no cross-repo dependency
_MINIMAL_ARTIFACT: dict = {
    "results": {
        "decode": [
            {"batch_size": 1, "context_tokens": 128, "latency_ms": {"mean_ms": 10.0}},
            {"batch_size": 1, "context_tokens": 512, "latency_ms": {"mean_ms": 12.0}},
            {"batch_size": 1, "context_tokens": 1024, "latency_ms": {"mean_ms": 15.0}},
        ],
        "prefill": [
            {"batch_size": 1, "prefill_tokens": 128, "latency_ms": {"mean_ms": 100.0}},
            {"batch_size": 1, "prefill_tokens": 512, "latency_ms": {"mean_ms": 500.0}},
            {"batch_size": 1, "prefill_tokens": 1024, "latency_ms": {"mean_ms": 1000.0}},
        ],
    },
    "hardware": {"gpu_name": "Test GPU"},
    "truth_boundary": "measured",
    "config": {"hidden_size": 4096},
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _req(rid: str, prompt: int, output: int) -> dict:
    return {"request_id": rid, "prompt_tokens": prompt, "output_tokens": output}


def _kv(kv_mb: float, kv_transfer_mb: float) -> dict:
    return {"estimated_kv_mb": kv_mb, "estimated_kv_transfer_mb": kv_transfer_mb}


class TestCostReport(unittest.TestCase):
    """Integration tests: generate fresh base artifacts, run estimate_all, validate structure."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.plan_dir = Path(cls._tmp.name)

        model_cfg = _load(CONFIG_PATH)
        request_trace = _load(TRACE_PATH)
        gvsp.generate_all(model_cfg, request_trace, cls.plan_dir)

        model_name, requests, kv_by_id, replay_safe_by_id = cscm.load_cost_inputs(
            cls.plan_dir, TRACE_PATH
        )
        cls.result = cscm.estimate_all(model_name, requests, kv_by_id, replay_safe_by_id)
        cls.costs = {r["request_id"]: r for r in cls.result["per_request_costs"]}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_artifact_type(self) -> None:
        self.assertEqual(self.result["artifact_type"], "compiler_serving_cost_report")

    def test_truth_boundary_present(self) -> None:
        self.assertIn("truth_boundary", self.result)
        self.assertEqual(self.result["truth_boundary"], cscm.TRUTH_BOUNDARY)
        self.assertGreater(len(self.result["truth_boundary"]), 0)

    def test_per_request_count_matches_trace(self) -> None:
        trace = _load(TRACE_PATH)
        self.assertEqual(len(self.result["per_request_costs"]), len(trace["requests"]))

    def test_every_request_has_both_cost_paths(self) -> None:
        for rid, entry in self.costs.items():
            self.assertIn("colocated_cost", entry, f"{rid} missing colocated_cost")
            self.assertIn("pd_split_cost", entry, f"{rid} missing pd_split_cost")

    def test_summary_counts_consistent(self) -> None:
        s = self.result["summary"]
        costs = self.result["per_request_costs"]
        self.assertEqual(s["total_requests"], len(costs))
        self.assertEqual(
            s["colocated_wins"],
            sum(1 for r in costs if r["recommended_execution_mode"] == "colocated"),
        )
        self.assertEqual(
            s["pd_split_wins"],
            sum(1 for r in costs if r["recommended_execution_mode"] == "pd_split"),
        )
        self.assertEqual(s["colocated_wins"] + s["pd_split_wins"], s["total_requests"])

    def test_assumptions_block_present(self) -> None:
        self.assertIn("assumptions", self.result)
        for key in [
            "prefill_ms_per_token", "decode_ms_per_token", "pd_bandwidth_mb_per_ms",
            "pd_coordination_ms", "replay_gain_ms", "memory_penalty_threshold_mb",
        ]:
            self.assertIn(key, self.result["assumptions"], f"assumptions missing {key}")

    def test_r3_cost_recommends_pd_split(self) -> None:
        # r3 (prompt=512, kv_transfer=27 MB): heuristic planner chose colocated,
        # cost model should recommend pd_split because queue savings exceed transfer cost.
        entry = self.costs["r3"]
        self.assertEqual(entry["recommended_execution_mode"], "pd_split")
        self.assertLess(
            entry["pd_split_cost"]["total_ms"],
            entry["colocated_cost"]["total_ms"],
        )

    def test_r0_cost_recommends_colocated(self) -> None:
        # r0 (prompt=32): tiny request where PD coordination overhead dominates
        entry = self.costs["r0"]
        self.assertEqual(entry["recommended_execution_mode"], "colocated")

    def test_r2_pd_split_has_nonzero_kv_transfer(self) -> None:
        # r2 has kv_transfer_mb=13.5 → pd_split_cost must reflect transfer latency
        pd = self.costs["r2"]["pd_split_cost"]
        self.assertGreater(pd["kv_transfer_ms"], 0.0)

    def test_r3_high_kv_has_memory_penalty(self) -> None:
        # r3 kv_mb=27 > threshold=15 → memory_penalty_ms > 0 in both paths
        for path in ("colocated_cost", "pd_split_cost"):
            self.assertGreater(self.costs["r3"][path]["memory_penalty_ms"], 0.0)

    def test_confidence_field_present_and_valid(self) -> None:
        valid = {"low", "medium", "high"}
        for rid, entry in self.costs.items():
            self.assertIn("confidence", entry, f"{rid} missing confidence")
            self.assertIn(entry["confidence"], valid, f"{rid} invalid confidence")

    def test_cost_explanation_present_and_nonempty(self) -> None:
        for rid, entry in self.costs.items():
            self.assertIn("cost_explanation", entry, f"{rid} missing cost_explanation")
            self.assertGreater(len(entry["cost_explanation"]), 0, f"{rid} empty cost_explanation")

    def test_confidence_reasons_present_and_nonempty(self) -> None:
        for rid, entry in self.costs.items():
            self.assertIn("confidence_reasons", entry, f"{rid} missing confidence_reasons")
            self.assertIsInstance(entry["confidence_reasons"], list, f"{rid} not a list")
            self.assertGreater(len(entry["confidence_reasons"]), 0, f"{rid} empty confidence_reasons")

    def test_compute_lookup_present_in_cost_paths(self) -> None:
        for rid, entry in self.costs.items():
            for path in ("colocated_cost", "pd_split_cost"):
                self.assertIn("compute_lookup", entry[path], f"{rid}/{path} missing compute_lookup")

    def test_formula_mode_compute_lookup_mode(self) -> None:
        # No artifact loaded → all compute_lookup should be formula mode
        for rid, entry in self.costs.items():
            col_lookup = entry["colocated_cost"]["compute_lookup"]
            self.assertEqual(col_lookup["mode"], "formula", f"{rid} expected formula mode")
            self.assertIsNone(col_lookup["lookup_method"], f"{rid} expected null lookup_method")


class TestCostComponents(unittest.TestCase):
    """Unit tests for individual estimator functions using synthetic inputs."""

    def test_colocated_total_ms_equals_component_sum(self) -> None:
        b = cscm.estimate_colocated(_req("tx", 100, 100), _kv(10.0, 0.0), replay_safe=True)
        expected = b.compute_ms + b.kv_transfer_ms + b.queue_ms - b.replay_gain_ms + b.memory_penalty_ms
        self.assertAlmostEqual(b.total_ms, expected, places=10)

    def test_pd_split_total_ms_equals_component_sum(self) -> None:
        b = cscm.estimate_pd_split(_req("tx", 256, 128), _kv(13.5, 13.5), replay_safe=True)
        expected = b.compute_ms + b.kv_transfer_ms + b.queue_ms - b.replay_gain_ms + b.memory_penalty_ms
        self.assertAlmostEqual(b.total_ms, expected, places=10)

    def test_pd_split_kv_transfer_ms_positive_when_transfer_mb_positive(self) -> None:
        b = cscm.estimate_pd_split(_req("tx", 256, 128), _kv(13.5, 13.5), replay_safe=True)
        self.assertGreater(b.kv_transfer_ms, 0.0)
        self.assertAlmostEqual(b.kv_transfer_ms, 13.5 / cscm.PD_BANDWIDTH_MB_PER_MS, places=6)

    def test_pd_split_kv_transfer_ms_zero_when_transfer_mb_zero(self) -> None:
        b = cscm.estimate_pd_split(_req("tx", 48, 320), _kv(12.9, 0.0), replay_safe=True)
        self.assertEqual(b.kv_transfer_ms, 0.0)

    def test_colocated_kv_transfer_ms_always_zero(self) -> None:
        # Even when the kv_entry carries a nonzero transfer MB, colocated has no transfer
        b = cscm.estimate_colocated(_req("tx", 256, 128), _kv(13.5, 13.5), replay_safe=True)
        self.assertEqual(b.kv_transfer_ms, 0.0)

    def test_replay_safe_gives_nonzero_replay_gain(self) -> None:
        col = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True)
        pd = cscm.estimate_pd_split(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True)
        self.assertGreater(col.replay_gain_ms, 0.0)
        self.assertGreater(pd.replay_gain_ms, 0.0)
        self.assertAlmostEqual(col.replay_gain_ms, cscm.REPLAY_GAIN_MS, places=6)

    def test_not_replay_safe_gives_zero_replay_gain(self) -> None:
        col = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=False)
        pd = cscm.estimate_pd_split(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=False)
        self.assertEqual(col.replay_gain_ms, 0.0)
        self.assertEqual(pd.replay_gain_ms, 0.0)

    def test_high_kv_gives_memory_penalty(self) -> None:
        kv_mb = 27.0  # > threshold 15.0
        b = cscm.estimate_colocated(_req("tx", 32, 64), _kv(kv_mb, 0.0), replay_safe=True)
        self.assertGreater(b.memory_penalty_ms, 0.0)
        expected = (kv_mb - cscm.MEMORY_PENALTY_THRESHOLD_MB) * cscm.MEMORY_PENALTY_FACTOR
        self.assertAlmostEqual(b.memory_penalty_ms, expected, places=6)

    def test_low_kv_gives_zero_memory_penalty(self) -> None:
        b = cscm.estimate_colocated(_req("tx", 32, 64), _kv(10.0, 0.0), replay_safe=True)
        self.assertEqual(b.memory_penalty_ms, 0.0)

    def test_recommended_mode_is_lower_total_colocated(self) -> None:
        # Small request: coordination overhead makes pd_split more expensive
        result = cscm.estimate_request_cost(_req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True)
        col_total = result["colocated_cost"]["total_ms"]
        pd_total = result["pd_split_cost"]["total_ms"]
        self.assertEqual(result["recommended_execution_mode"], "colocated")
        self.assertLessEqual(col_total, pd_total)

    def test_recommended_mode_is_lower_total_pd_split(self) -> None:
        # Large prompt with transfer: queue savings > transfer + coordination overhead
        result = cscm.estimate_request_cost(_req("tx", 512, 256), _kv(27.0, 27.0), replay_safe=True)
        col_total = result["colocated_cost"]["total_ms"]
        pd_total = result["pd_split_cost"]["total_ms"]
        self.assertEqual(result["recommended_execution_mode"], "pd_split")
        self.assertLess(pd_total, col_total)

    def test_confidence_low(self) -> None:
        # diff_pct = |101 - 100| / 100 = 0.01 < 0.05
        self.assertEqual(cscm._confidence(100.0, 101.0), "low")

    def test_confidence_medium(self) -> None:
        # diff_pct = |109 - 100| / 100 = 0.09, in [0.05, 0.15)
        self.assertEqual(cscm._confidence(100.0, 109.0), "medium")

    def test_confidence_high(self) -> None:
        # diff_pct = |120 - 100| / 100 = 0.20 >= 0.15
        self.assertEqual(cscm._confidence(100.0, 120.0), "high")

    def test_cost_sources_all_keys_present_colocated(self) -> None:
        b = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True)
        for key in _COST_COMPONENT_KEYS:
            self.assertIn(key, b.cost_sources, f"colocated cost_sources missing {key}")

    def test_cost_sources_all_keys_present_pd_split(self) -> None:
        b = cscm.estimate_pd_split(_req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True)
        for key in _COST_COMPONENT_KEYS:
            self.assertIn(key, b.cost_sources, f"pd_split cost_sources missing {key}")

    def test_cost_sources_kv_transfer_label_differs_by_mode(self) -> None:
        col = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True)
        pd = cscm.estimate_pd_split(_req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True)
        self.assertEqual(col.cost_sources["kv_transfer_ms"], "not_applicable")
        self.assertEqual(pd.cost_sources["kv_transfer_ms"], "bandwidth_formula")

    def test_to_dict_includes_cost_sources(self) -> None:
        b = cscm.estimate_colocated(_req("tx", 64, 64), _kv(5.0, 0.0), replay_safe=True)
        d = b.to_dict()
        self.assertIn("cost_sources", d)
        self.assertIsInstance(d["cost_sources"], dict)
        for key in _COST_COMPONENT_KEYS:
            self.assertIn(key, d["cost_sources"])


class TestGpuCalibration(unittest.TestCase):
    """Tests for GPU batch-scaling artifact loader, lookup helpers, and proxy compute path."""

    # -----------------------------------------------------------------------
    # load_gpu_batch_scaling error paths
    # -----------------------------------------------------------------------

    def test_load_none_path_returns_none(self) -> None:
        data, meta = cscm.load_gpu_batch_scaling(None)
        self.assertIsNone(data)
        self.assertEqual(meta, {})

    def test_load_missing_file_returns_none(self) -> None:
        data, meta = cscm.load_gpu_batch_scaling(Path("/nonexistent/path/artifact.json"))
        self.assertIsNone(data)
        self.assertEqual(meta, {})

    def test_load_invalid_json_returns_none(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{not valid json")
            tmp = Path(f.name)
        try:
            data, meta = cscm.load_gpu_batch_scaling(tmp)
            self.assertIsNone(data)
            self.assertEqual(meta, {})
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_missing_results_decode_returns_none(self) -> None:
        bad = {"results": {"prefill": []}}  # decode missing
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(bad, f)
            tmp = Path(f.name)
        try:
            data, meta = cscm.load_gpu_batch_scaling(tmp)
            self.assertIsNone(data)
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_missing_results_prefill_returns_none(self) -> None:
        bad = {"results": {"decode": []}}  # prefill missing
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(bad, f)
            tmp = Path(f.name)
        try:
            data, meta = cscm.load_gpu_batch_scaling(tmp)
            self.assertIsNone(data)
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_valid_artifact_returns_expected_meta(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(_MINIMAL_ARTIFACT, f)
            tmp = Path(f.name)
        try:
            data, meta = cscm.load_gpu_batch_scaling(tmp)
            self.assertIsNotNone(data)
            self.assertEqual(meta["compute_cost_source"], "gpu_measured_proxy")
            self.assertEqual(meta["lookup_method"], "nearest_neighbor")
            self.assertIn("proxy_model_note", meta)
            self.assertIn("compute_calibration_truth_boundary", meta)
            self.assertIn("gpu_batch_scaling_artifact", meta)
            self.assertIn("artifact_hardware", meta)
            self.assertEqual(meta["artifact_hardware"], "Test GPU")
        finally:
            tmp.unlink(missing_ok=True)

    # -----------------------------------------------------------------------
    # Nearest-neighbor lookup helpers
    # -----------------------------------------------------------------------

    def test_nearest_prefill_exact_match(self) -> None:
        ms = cscm._nearest_prefill_ms(_MINIMAL_ARTIFACT, 128)
        self.assertAlmostEqual(ms, 100.0)

    def test_nearest_prefill_extrapolates_lower(self) -> None:
        # prompt=32 is below all buckets; nearest is 128
        ms = cscm._nearest_prefill_ms(_MINIMAL_ARTIFACT, 32)
        self.assertAlmostEqual(ms, 100.0)

    def test_nearest_prefill_mid_picks_upper(self) -> None:
        # prompt=800: |800-512|=288, |800-1024|=224 → 1024
        ms = cscm._nearest_prefill_ms(_MINIMAL_ARTIFACT, 800)
        self.assertAlmostEqual(ms, 1000.0)

    def test_nearest_decode_exact_match(self) -> None:
        ms = cscm._nearest_decode_step_ms(_MINIMAL_ARTIFACT, 512)
        self.assertAlmostEqual(ms, 12.0)

    def test_nearest_decode_tie_picks_smaller_bucket(self) -> None:
        # context=320: |320-128|=192, |320-512|=192 → tie → pick 128 (smaller)
        ms = cscm._nearest_decode_step_ms(_MINIMAL_ARTIFACT, 320)
        self.assertAlmostEqual(ms, 10.0)

    # -----------------------------------------------------------------------
    # Compute source: formula vs gpu_proxy
    # -----------------------------------------------------------------------

    def test_no_artifact_compute_source_is_formula_synthetic(self) -> None:
        col = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True)
        self.assertEqual(col.cost_sources["compute_ms"], "formula_synthetic")

    def test_artifact_compute_source_is_gpu_measured_proxy(self) -> None:
        col = cscm.estimate_colocated(
            _req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        self.assertEqual(col.cost_sources["compute_ms"], "gpu_measured_proxy")

    def test_pd_split_artifact_compute_source_is_gpu_measured_proxy(self) -> None:
        pd = cscm.estimate_pd_split(
            _req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        self.assertEqual(pd.cost_sources["compute_ms"], "gpu_measured_proxy")

    def test_measured_proxy_compute_differs_from_formula(self) -> None:
        # prompt=128, output=64: formula = 0.08*128 + 0.12*64 = 10.24 + 7.68 = 17.92
        # proxy: prefill_row=128→100.0, avg_ctx=int(128+32)=160→nearest=128→10.0
        #        compute = 100.0 + 10.0*64 = 740.0
        col_formula = cscm.estimate_colocated(_req("tx", 128, 64), _kv(5.0, 0.0), replay_safe=True)
        col_proxy = cscm.estimate_colocated(
            _req("tx", 128, 64), _kv(5.0, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        self.assertAlmostEqual(col_formula.compute_ms, 17.92, places=4)
        self.assertAlmostEqual(col_proxy.compute_ms, 740.0, places=4)
        self.assertGreater(col_proxy.compute_ms, col_formula.compute_ms * 10)

    # -----------------------------------------------------------------------
    # compute_lookup field
    # -----------------------------------------------------------------------

    def test_compute_lookup_present_in_breakdown(self) -> None:
        col = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True)
        self.assertIsInstance(col.compute_lookup, dict)

    def test_compute_lookup_mode_formula_without_artifact(self) -> None:
        col = cscm.estimate_colocated(_req("tx", 32, 64), _kv(3.0, 0.0), replay_safe=True)
        self.assertEqual(col.compute_lookup["mode"], "formula")
        self.assertIsNone(col.compute_lookup["lookup_method"])

    def test_compute_lookup_mode_gpu_proxy_with_artifact(self) -> None:
        col = cscm.estimate_colocated(
            _req("tx", 128, 64), _kv(5.0, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        self.assertEqual(col.compute_lookup["mode"], "gpu_proxy")
        self.assertEqual(col.compute_lookup["lookup_method"], "nearest_neighbor")

    def test_compute_lookup_has_bucket_token_fields(self) -> None:
        col = cscm.estimate_colocated(
            _req("tx", 128, 64), _kv(5.0, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        self.assertIn("prefill_lookup_tokens", col.compute_lookup)
        self.assertIn("decode_lookup_context_tokens", col.compute_lookup)
        self.assertIn("avg_decode_context_tokens", col.compute_lookup)
        # prefill exact match
        self.assertEqual(col.compute_lookup["prefill_lookup_tokens"], 128)

    def test_compute_lookup_in_to_dict(self) -> None:
        col = cscm.estimate_colocated(
            _req("tx", 128, 64), _kv(5.0, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        d = col.to_dict()
        self.assertIn("compute_lookup", d)
        self.assertEqual(d["compute_lookup"]["mode"], "gpu_proxy")

    # -----------------------------------------------------------------------
    # estimate_all with artifact meta
    # -----------------------------------------------------------------------

    def test_estimate_all_includes_artifact_meta_when_provided(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(_MINIMAL_ARTIFACT, f)
            tmp = Path(f.name)
        try:
            artifact, meta = cscm.load_gpu_batch_scaling(tmp)
            model_cfg = json.loads(CONFIG_PATH.read_text())
            trace = json.loads(TRACE_PATH.read_text())
            with tempfile.TemporaryDirectory() as plan_tmp:
                gvsp.generate_all(model_cfg, trace, Path(plan_tmp))
                mn, reqs, kv_by_id, rs_by_id = cscm.load_cost_inputs(
                    Path(plan_tmp), TRACE_PATH
                )
            result = cscm.estimate_all(mn, reqs, kv_by_id, rs_by_id, artifact, meta)
            self.assertIn("gpu_batch_scaling_artifact", result)
            self.assertEqual(result["compute_cost_source"], "gpu_measured_proxy")
            self.assertIn("compute_calibration_truth_boundary", result)
            self.assertIn("proxy_model_note", result)
            self.assertEqual(result["lookup_method"], "nearest_neighbor")
        finally:
            tmp.unlink(missing_ok=True)

    def test_estimate_all_no_artifact_no_meta_fields(self) -> None:
        model_cfg = json.loads(CONFIG_PATH.read_text())
        trace = json.loads(TRACE_PATH.read_text())
        with tempfile.TemporaryDirectory() as plan_tmp:
            gvsp.generate_all(model_cfg, trace, Path(plan_tmp))
            mn, reqs, kv_by_id, rs_by_id = cscm.load_cost_inputs(Path(plan_tmp), TRACE_PATH)
        result = cscm.estimate_all(mn, reqs, kv_by_id, rs_by_id)
        self.assertNotIn("gpu_batch_scaling_artifact", result)
        self.assertNotIn("compute_cost_source", result)

    # -----------------------------------------------------------------------
    # confidence_reasons
    # -----------------------------------------------------------------------

    def test_confidence_reasons_present_in_estimate_request_cost(self) -> None:
        result = cscm.estimate_request_cost(
            _req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True
        )
        self.assertIn("confidence_reasons", result)
        self.assertIsInstance(result["confidence_reasons"], list)
        self.assertGreater(len(result["confidence_reasons"]), 0)

    def test_low_confidence_formula_no_mismatch_reason(self) -> None:
        # prompt=200, output=100, kv_transfer=0:
        #   col=30.5ms, pd=30.1ms, diff_pct=0.013 → "low"
        #   No artifact → mismatch reason must NOT appear
        result = cscm.estimate_request_cost(
            _req("tx", 200, 100), _kv(5.0, 0.0), replay_safe=True
        )
        self.assertEqual(result["confidence"], "low")
        self.assertIn("decision_margin_below_5_percent", result["confidence_reasons"])
        self.assertNotIn(
            "proxy_model_mismatch_can_dominate_small_policy_margins",
            result["confidence_reasons"],
        )

    def test_low_confidence_gpu_proxy_includes_mismatch_reason(self) -> None:
        # With minimal artifact, compute dominates → low confidence + mismatch reason
        result = cscm.estimate_request_cost(
            _req("tx", 128, 64), _kv(3.375, 0.0), replay_safe=True, artifact=_MINIMAL_ARTIFACT
        )
        self.assertEqual(result["confidence"], "low")
        self.assertIn("proxy_model_mismatch_can_dominate_small_policy_margins",
                      result["confidence_reasons"])

    def test_high_confidence_formula_has_margin_reason(self) -> None:
        # prompt=32, output=64: col≈9.38ms, pd≈11.02ms, diff_pct=0.175 → "high"
        result = cscm.estimate_request_cost(
            _req("tx", 32, 64), _kv(3.375, 0.0), replay_safe=True
        )
        self.assertEqual(result["confidence"], "high")
        self.assertIn("decision_margin_at_least_15_percent", result["confidence_reasons"])

    # -----------------------------------------------------------------------
    # Integration test with real cross-repo artifact (optional)
    # -----------------------------------------------------------------------

    @unittest.skipUnless(
        GPU_ARTIFACT_PATH.exists(),
        "GPU batch-scaling artifact not found in sibling heterogeneous-inference-runtime repo",
    )
    def test_real_artifact_load_and_proxy_compute(self) -> None:
        artifact, meta = cscm.load_gpu_batch_scaling(GPU_ARTIFACT_PATH)
        self.assertIsNotNone(artifact)
        self.assertEqual(meta["compute_cost_source"], "gpu_measured_proxy")
        self.assertIn("4096", meta["proxy_model_note"])
        # For prompt=512, proxy compute_ms >> formula compute_ms
        col_formula = cscm.estimate_colocated(_req("tx", 512, 256), _kv(27.0, 27.0), replay_safe=True)
        col_proxy = cscm.estimate_colocated(
            _req("tx", 512, 256), _kv(27.0, 27.0), replay_safe=True, artifact=artifact
        )
        self.assertGreater(col_proxy.compute_ms, col_formula.compute_ms * 10)
        self.assertEqual(col_proxy.compute_lookup["mode"], "gpu_proxy")
        self.assertEqual(col_proxy.compute_lookup["prefill_lookup_tokens"], 512)


if __name__ == "__main__":
    unittest.main()
