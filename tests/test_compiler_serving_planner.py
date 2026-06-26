import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import compiler_serving_cost_model as cscm  # noqa: E402
import compiler_serving_planner as csp      # noqa: E402
import generate_vllm_serving_plan as gvsp   # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "tiny_gpt_llm_config.json"
TRACE_PATH = REPO_ROOT / "configs" / "vllm_serving_request_trace.json"

SAMPLE_BUCKETS: list[dict] = [
    {"bucket_id": 0, "batch_size": 1, "query_len": 1, "context_bucket": 128,  "replay_safe": True,  "fallback_reason": None},
    {"bucket_id": 1, "batch_size": 1, "query_len": 1, "context_bucket": 256,  "replay_safe": True,  "fallback_reason": None},
    {"bucket_id": 2, "batch_size": 1, "query_len": 1, "context_bucket": 512,  "replay_safe": True,  "fallback_reason": None},
    {"bucket_id": 3, "batch_size": 1, "query_len": 1, "context_bucket": 1024, "replay_safe": True,  "fallback_reason": None},
    {"bucket_id": 4, "batch_size": 1, "query_len": 1, "context_bucket": 2048, "replay_safe": True,  "fallback_reason": None},
    {"bucket_id": 5, "batch_size": 1, "query_len": 1, "context_bucket": 4096, "replay_safe": False, "fallback_reason": "context_exceeds_safe_capture_range"},
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _req(rid: str, prompt: int, output: int) -> dict:
    return {"request_id": rid, "prompt_tokens": prompt, "output_tokens": output}


def _kv(pd_candidate: bool, kv_mb: float, kv_transfer_mb: float = 0.0) -> dict:
    return {
        "pd_split_candidate": pd_candidate,
        "estimated_kv_mb": kv_mb,
        "estimated_kv_transfer_mb": kv_transfer_mb,
    }


def _plan(rid: str, prompt: int, output: int, pd_candidate: bool, kv_mb: float,
          kv_transfer_mb: float = 0.0, buckets: list[dict] | None = None) -> dict:
    return csp.plan_request(
        _req(rid, prompt, output),
        _kv(pd_candidate, kv_mb, kv_transfer_mb),
        buckets if buckets is not None else SAMPLE_BUCKETS,
    )


# ---------------------------------------------------------------------------
# Existing tests — unchanged
# ---------------------------------------------------------------------------

class TestPlanAll(unittest.TestCase):
    """Integration tests: generate base artifacts fresh, then run plan_all."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.plan_dir = Path(cls._tmp.name)

        model_cfg = _load(CONFIG_PATH)
        request_trace = _load(TRACE_PATH)
        gvsp.generate_all(model_cfg, request_trace, cls.plan_dir)

        model_name, requests, kv_by_id, buckets = csp.load_plan_inputs(cls.plan_dir, TRACE_PATH)
        cls.result = csp.plan_all(model_name, requests, kv_by_id, buckets)
        cls.decisions = {d["request_id"]: d for d in cls.result["per_request_decisions"]}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_artifact_type(self) -> None:
        self.assertEqual(self.result["artifact_type"], "compiler_serving_plan")

    def test_truth_boundary_at_top_level(self) -> None:
        self.assertEqual(self.result["truth_boundary"], csp.TRUTH_BOUNDARY)

    def test_per_request_count_matches_trace(self) -> None:
        trace = _load(TRACE_PATH)
        self.assertEqual(len(self.result["per_request_decisions"]), len(trace["requests"]))

    def test_summary_counts_consistent(self) -> None:
        s = self.result["summary"]
        ds = self.result["per_request_decisions"]
        self.assertEqual(s["total_requests"], len(ds))
        self.assertEqual(s["colocated_count"], sum(1 for d in ds if d["execution_mode"] == "colocated"))
        self.assertEqual(s["pd_split_count"], sum(1 for d in ds if d["execution_mode"] == "pd_split"))
        self.assertEqual(s["fallback_count"], sum(1 for d in ds if d["execution_mode"] == "fallback"))
        self.assertEqual(s["replay_safe_count"], sum(1 for d in ds if d["replay_safe"]))

    def test_r3_kv_transfer_override_in_integration(self) -> None:
        d = self.decisions["r3"]
        self.assertEqual(d["execution_mode"], "colocated")
        self.assertTrue(d["input_pd_split_candidate"])
        self.assertFalse(d["final_pd_split_decision"])
        self.assertIn("kv_transfer_cost_exceeds_threshold_colocated_fallback", d["decision_reasons"])

    def test_r0_short_prompt_colocated_in_integration(self) -> None:
        d = self.decisions["r0"]
        self.assertEqual(d["execution_mode"], "colocated")
        self.assertFalse(d["final_pd_split_decision"])

    def test_thresholds_present_in_output(self) -> None:
        self.assertIn("thresholds", self.result)
        self.assertIn("kv_transfer_colocated_override_mb", self.result["thresholds"])
        self.assertIn("kv_layout_paged_threshold_mb", self.result["thresholds"])


class TestPlanRequestRules(unittest.TestCase):
    """Unit tests for individual decision functions using synthetic inputs."""

    def test_short_prompt_colocated(self) -> None:
        d = _plan("r0", 32, 64, False, 3.375)
        self.assertEqual(d["execution_mode"], "colocated")
        self.assertIn("short_prompt_colocated", d["decision_reasons"])
        self.assertFalse(d["final_pd_split_decision"])

    def test_pd_candidate_pd_split(self) -> None:
        d = _plan("r2", 256, 128, True, 13.5, kv_transfer_mb=13.5)
        self.assertEqual(d["execution_mode"], "pd_split")
        self.assertTrue(d["input_pd_split_candidate"])
        self.assertTrue(d["final_pd_split_decision"])
        self.assertIn("pd_split_candidate_from_kv_layout_plan", d["decision_reasons"])

    def test_decode_heavy_pd_split(self) -> None:
        d = _plan("r4", 48, 320, False, 12.9375)
        self.assertEqual(d["execution_mode"], "pd_split")
        self.assertFalse(d["input_pd_split_candidate"])
        self.assertTrue(d["final_pd_split_decision"])
        self.assertIn("decode_heavy_pd_split", d["decision_reasons"])

    def test_high_kv_transfer_override_colocated(self) -> None:
        d = _plan("r3", 512, 256, True, 27.0, kv_transfer_mb=27.0)
        self.assertEqual(d["execution_mode"], "colocated")
        self.assertTrue(d["input_pd_split_candidate"])
        self.assertFalse(d["final_pd_split_decision"])
        self.assertIn("kv_transfer_cost_exceeds_threshold_colocated_fallback", d["decision_reasons"])

    def test_high_kv_transfer_overrides_decode_heavy_too(self) -> None:
        # decode-heavy request with high KV transfer → still forced colocated
        d = _plan("rx", 48, 400, False, 30.0, kv_transfer_mb=26.0)
        self.assertEqual(d["execution_mode"], "colocated")
        self.assertFalse(d["final_pd_split_decision"])

    def test_replay_safe_bucket_selected(self) -> None:
        d = _plan("r1", 64, 128, False, 6.75)
        # total_tokens = 192 → smallest fitting bucket is context_bucket=256
        self.assertTrue(d["replay_safe"])
        self.assertIsNotNone(d["cuda_graph_bucket"])
        self.assertEqual(d["cuda_graph_bucket"]["context_bucket"], 256)
        self.assertEqual(d["cuda_graph_bucket"]["batch_size"], 1)
        self.assertIsNone(d["replay_fallback_reason"])

    def test_replay_fallback_too_large_context(self) -> None:
        # total_tokens = 2100 > max safe context 2048
        d = _plan("rx", 1600, 500, False, 50.0)
        self.assertFalse(d["replay_safe"])
        self.assertIsNone(d["cuda_graph_bucket"])
        self.assertEqual(d["replay_fallback_reason"], "no_replay_safe_bucket_for_context_size")

    def test_replay_fallback_all_unsafe_buckets(self) -> None:
        unsafe_only = [
            {"bucket_id": 0, "batch_size": 1, "query_len": 1, "context_bucket": 512,
             "replay_safe": False, "fallback_reason": "batch_size_exceeds_safe_capture_range"},
        ]
        d = _plan("rx", 32, 64, False, 3.0, buckets=unsafe_only)
        self.assertFalse(d["replay_safe"])
        self.assertIsNone(d["cuda_graph_bucket"])

    def test_kv_layout_paged(self) -> None:
        d = _plan("r5", 128, 384, False, 18.0)
        self.assertEqual(d["kv_layout"], "paged")

    def test_kv_layout_contiguous(self) -> None:
        d = _plan("r0", 32, 64, False, 3.375)
        self.assertEqual(d["kv_layout"], "contiguous")

    def test_kv_layout_at_exact_threshold_is_paged(self) -> None:
        d = _plan("rx", 32, 64, False, 15.0)
        self.assertEqual(d["kv_layout"], "paged")

    def test_schema_has_both_pd_fields(self) -> None:
        d = _plan("r2", 256, 128, True, 13.5, kv_transfer_mb=13.5)
        self.assertIn("input_pd_split_candidate", d)
        self.assertIn("final_pd_split_decision", d)
        self.assertIsInstance(d["input_pd_split_candidate"], bool)
        self.assertIsInstance(d["final_pd_split_decision"], bool)

    def test_pd_fields_diverge_on_kv_override(self) -> None:
        # r3: layout plan says pd_split_candidate=True, but compiler overrides to colocated
        d = _plan("r3", 512, 256, True, 27.0, kv_transfer_mb=27.0)
        self.assertTrue(d["input_pd_split_candidate"])
        self.assertFalse(d["final_pd_split_decision"])

    def test_truth_boundary_in_every_decision(self) -> None:
        d = _plan("r0", 32, 64, False, 3.375)
        self.assertIn("truth_boundary", d)
        self.assertEqual(d["truth_boundary"], csp.TRUTH_BOUNDARY)


# ---------------------------------------------------------------------------
# New: cost-report-driven mode tests
# ---------------------------------------------------------------------------

class TestCostReportDrivenPlanAll(unittest.TestCase):
    """Integration: generate cost report, run plan_all with it, validate schema + decisions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.plan_dir = Path(cls._tmp.name)

        # Generate base artifacts (A–D)
        model_cfg = _load(CONFIG_PATH)
        request_trace = _load(TRACE_PATH)
        gvsp.generate_all(model_cfg, request_trace, cls.plan_dir)

        # Generate cost report (F) into the same temp dir
        model_name_c, reqs_c, kv_by_id_c, rs_by_id_c = cscm.load_cost_inputs(
            cls.plan_dir, TRACE_PATH
        )
        cost_report = cscm.estimate_all(model_name_c, reqs_c, kv_by_id_c, rs_by_id_c)
        cls.cost_report_path = cls.plan_dir / "compiler_serving_cost_report.json"
        cscm.write_json(cls.cost_report_path, cost_report)

        # Load planner inputs and cost entries, then run plan_all
        model_name, requests, kv_by_id, buckets = csp.load_plan_inputs(cls.plan_dir, TRACE_PATH)
        cost_entries_by_id, meta = csp.load_cost_entries(cls.cost_report_path)
        cls.result = csp.plan_all(model_name, requests, kv_by_id, buckets, cost_entries_by_id, meta)
        cls.decisions = {d["request_id"]: d for d in cls.result["per_request_decisions"]}

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # --- Decision flip tests ---

    def test_r3_flips_to_pd_split(self) -> None:
        # Heuristic: colocated (KV transfer override rule). Cost: pd_split (queue savings win).
        self.assertEqual(self.decisions["r3"]["execution_mode"], "pd_split")
        self.assertTrue(self.decisions["r3"]["final_pd_split_decision"])

    def test_r4_flips_to_colocated(self) -> None:
        # Heuristic: pd_split (decode_heavy). Cost: colocated (coordination overhead wins).
        self.assertEqual(self.decisions["r4"]["execution_mode"], "colocated")
        self.assertFalse(self.decisions["r4"]["final_pd_split_decision"])

    def test_r5_flips_to_colocated(self) -> None:
        self.assertEqual(self.decisions["r5"]["execution_mode"], "colocated")

    def test_r6_flips_to_colocated(self) -> None:
        self.assertEqual(self.decisions["r6"]["execution_mode"], "colocated")

    def test_r0_r1_remain_colocated(self) -> None:
        for rid in ("r0", "r1"):
            self.assertEqual(self.decisions[rid]["execution_mode"], "colocated", f"{rid} should be colocated")

    def test_r2_r7_remain_pd_split(self) -> None:
        for rid in ("r2", "r7"):
            self.assertEqual(self.decisions[rid]["execution_mode"], "pd_split", f"{rid} should be pd_split")

    # --- decision_source ---

    def test_all_entries_have_cost_report_decision_source(self) -> None:
        for rid, d in self.decisions.items():
            self.assertEqual(
                d["decision_source"], csp.DECISION_SOURCE_COST_REPORT,
                f"{rid} has wrong decision_source",
            )

    # --- cost-driven fields present ---

    def test_confidence_present_and_valid(self) -> None:
        valid = {"low", "medium", "high"}
        for rid, d in self.decisions.items():
            self.assertIn("confidence", d, f"{rid} missing confidence")
            self.assertIn(d["confidence"], valid, f"{rid} invalid confidence value")

    def test_cost_summary_present(self) -> None:
        for rid, d in self.decisions.items():
            self.assertIn("cost_summary", d, f"{rid} missing cost_summary")
            cs = d["cost_summary"]
            for key in ("colocated_total_ms", "pd_split_total_ms",
                        "decision_margin_ms", "decision_margin_pct"):
                self.assertIn(key, cs, f"{rid} cost_summary missing {key}")

    def test_cost_explanation_present_and_nonempty(self) -> None:
        for rid, d in self.decisions.items():
            self.assertIn("cost_explanation", d, f"{rid} missing cost_explanation")
            self.assertGreater(len(d["cost_explanation"]), 0, f"{rid} empty cost_explanation")

    # --- decision_reasons absent in cost path ---

    def test_no_decision_reasons_in_cost_driven_entries(self) -> None:
        for rid, d in self.decisions.items():
            self.assertNotIn(
                "decision_reasons", d,
                f"{rid} should not have decision_reasons in cost-driven mode",
            )

    # --- decision_margin correctness ---

    def test_r3_cost_summary_margin_values_correct(self) -> None:
        cs = self.decisions["r3"]["cost_summary"]
        col = cs["colocated_total_ms"]
        pd = cs["pd_split_total_ms"]
        expected_margin_ms = round(abs(col - pd), 4)
        expected_margin_pct = round(expected_margin_ms / min(col, pd), 4)
        self.assertAlmostEqual(cs["decision_margin_ms"], expected_margin_ms, places=4)
        self.assertAlmostEqual(cs["decision_margin_pct"], expected_margin_pct, places=4)
        # pd_split is cheaper for r3
        self.assertGreater(cs["colocated_total_ms"], cs["pd_split_total_ms"])
        self.assertGreater(cs["decision_margin_ms"], 0.0)

    def test_cost_summary_margin_nonnegative_all_requests(self) -> None:
        for rid, d in self.decisions.items():
            cs = d["cost_summary"]
            self.assertGreaterEqual(cs["decision_margin_ms"], 0.0, f"{rid} negative margin_ms")
            self.assertGreaterEqual(cs["decision_margin_pct"], 0.0, f"{rid} negative margin_pct")

    # --- input_pd_split_candidate stays from layout plan ---

    def test_r3_input_pd_split_candidate_still_from_layout_plan(self) -> None:
        d = self.decisions["r3"]
        # kv_cache_layout_plan marks r3 pd_split_candidate=True (prompt=512 >= 256)
        self.assertTrue(d["input_pd_split_candidate"])
        # cost report also says pd_split, so final decision agrees
        self.assertTrue(d["final_pd_split_decision"])

    def test_r4_input_pd_split_candidate_false_from_layout_plan(self) -> None:
        d = self.decisions["r4"]
        # kv_cache_layout_plan marks r4 pd_split_candidate=False (prompt=48 < 256)
        self.assertFalse(d["input_pd_split_candidate"])
        # cost report says colocated, so final_pd_split_decision is False
        self.assertFalse(d["final_pd_split_decision"])

    # --- top-level metadata ---

    def test_top_level_decision_mode_cost_report_driven(self) -> None:
        self.assertEqual(self.result["decision_mode"], "cost_report_driven")

    def test_top_level_cost_model_version(self) -> None:
        self.assertIn("cost_model_version", self.result)

    def test_top_level_cost_model_truth_boundary(self) -> None:
        self.assertIn("cost_model_truth_boundary", self.result)
        self.assertGreater(len(self.result["cost_model_truth_boundary"]), 0)

    def test_top_level_cost_report_path(self) -> None:
        self.assertIn("cost_report_path", self.result)

    def test_summary_counts_consistent(self) -> None:
        s = self.result["summary"]
        ds = self.result["per_request_decisions"]
        self.assertEqual(s["total_requests"], len(ds))
        self.assertEqual(s["colocated_count"], sum(1 for d in ds if d["execution_mode"] == "colocated"))
        self.assertEqual(s["pd_split_count"], sum(1 for d in ds if d["execution_mode"] == "pd_split"))


# ---------------------------------------------------------------------------
# New: heuristic fallback tests
# ---------------------------------------------------------------------------

class TestHeuristicFallback(unittest.TestCase):
    """Unit tests for load_cost_entries error handling and heuristic-path schema."""

    def test_none_path_returns_none_entries(self) -> None:
        entries, meta = csp.load_cost_entries(None)
        self.assertIsNone(entries)
        self.assertEqual(meta, {})

    def test_missing_file_returns_none_entries(self) -> None:
        entries, meta = csp.load_cost_entries(
            Path(tempfile.gettempdir()) / "nonexistent_cost_report_xyz123.json"
        )
        self.assertIsNone(entries)

    def test_malformed_file_missing_key_returns_none(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"schema_version": "0.1"}, f)  # valid JSON, missing per_request_costs
            path = Path(f.name)
        try:
            entries, meta = csp.load_cost_entries(path)
            self.assertIsNone(entries)
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_file_invalid_json_returns_none(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = Path(f.name)
        try:
            entries, meta = csp.load_cost_entries(path)
            self.assertIsNone(entries)
        finally:
            path.unlink(missing_ok=True)

    def test_heuristic_plan_request_has_decision_source_field(self) -> None:
        d = _plan("r0", 32, 64, False, 3.375)
        self.assertIn("decision_source", d)
        self.assertEqual(d["decision_source"], csp.DECISION_SOURCE_HEURISTIC)

    def test_heuristic_plan_request_has_decision_reasons(self) -> None:
        d = _plan("r0", 32, 64, False, 3.375)
        self.assertIn("decision_reasons", d)
        self.assertIsInstance(d["decision_reasons"], list)
        self.assertGreater(len(d["decision_reasons"]), 0)

    def test_heuristic_plan_request_no_confidence_field(self) -> None:
        d = _plan("r0", 32, 64, False, 3.375)
        self.assertNotIn("confidence", d)
        self.assertNotIn("cost_summary", d)
        self.assertNotIn("cost_explanation", d)

    def test_plan_all_without_cost_entries_has_heuristic_decision_mode(self) -> None:
        # Minimal plan_all call — no cost entries
        req = _req("tx", 32, 64)
        kv = _kv(False, 3.375)
        result = csp.plan_all(
            "test-model",
            [req],
            {"tx": kv},
            SAMPLE_BUCKETS,
            cost_entries_by_id=None,
            cost_report_meta=None,
        )
        self.assertEqual(result["decision_mode"], "heuristic_rules")
        d = result["per_request_decisions"][0]
        self.assertEqual(d["decision_source"], csp.DECISION_SOURCE_HEURISTIC)
        self.assertIn("decision_reasons", d)

    def test_plan_all_without_cost_entries_no_cost_metadata_fields(self) -> None:
        req = _req("tx", 32, 64)
        kv = _kv(False, 3.375)
        result = csp.plan_all("test-model", [req], {"tx": kv}, SAMPLE_BUCKETS)
        self.assertNotIn("cost_report_path", result)
        self.assertNotIn("cost_model_version", result)
        self.assertNotIn("cost_model_truth_boundary", result)


if __name__ == "__main__":
    unittest.main()
