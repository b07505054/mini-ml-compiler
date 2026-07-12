"""Phase 26 regression: YOLO-Seg dispatch-unit materialization artifacts.

Validates the generated execution plan and dispatch-unit report:
- one dispatch unit per GenericGraphIR node (no helper-op units);
- classification totality over all top-level MLIR ops;
- typed tensor ABI (model input vs initializers, weights, biases, outputs);
- corrected memory metrics, including peak-live equality with the Phase 23
  lifetime analysis;
- CV postprocess contract;
- backend/kernel truth (no executable units without a registered kernel);
- Qwen plan schema untouched (no CV-only fields);
- runtime loader schema compatibility (mirrors the heterogeneous-inference-
  runtime validate_execution_plan required-field/no-measured-field contract
  without importing that repository).

Skips when the YOLO-Seg artifacts have not been generated.
"""

import json
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = REPO_ROOT / "artifacts" / "yoloseg_generic_frontend"
PLAN_PATH = ART_DIR / "yoloseg.execution_plan.json"
REPORT_PATH = ART_DIR / "yoloseg.dispatch_unit_report.json"
FACTS_PATH = ART_DIR / "yoloseg.cv_planning_facts.json"
QWEN_PLAN_PATH = REPO_ROOT / "artifacts" / "qwen" / "execution_plan.json"

MEASURED_FIELDS = {
    "measured_latency_ms",
    "actual_latency_ms",
    "measured_memory_mb",
    "measured_speedup",
    "speedup",
    "performance_claim",
    "runtime_result",
    "metrics",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_measured_field(payload) -> bool:
    if isinstance(payload, dict):
        if MEASURED_FIELDS.intersection(payload.keys()):
            return True
        return any(_contains_measured_field(v) for v in payload.values())
    if isinstance(payload, list):
        return any(_contains_measured_field(v) for v in payload)
    return False


@unittest.skipUnless(PLAN_PATH.exists(), "YOLO-Seg execution plan not generated")
class TestYolosegDispatchUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = _load(PLAN_PATH)
        cls.fp = cls.plan["function_plans"][0]
        cls.units = cls.fp["dispatch_units"]
        cls.cls = cls.fp["op_classification"]

    def test_one_unit_per_source_node(self):
        self.assertEqual(len(self.units), 268)
        self.assertEqual(self.cls["source_graph_node_count"], 268)
        # Every unit maps to exactly one source node today (no fusion
        # materialized), and unit ids are unique.
        ids = [u["dispatch_unit_id"] for u in self.units]
        self.assertEqual(len(set(ids)), 268)
        for unit in self.units:
            self.assertEqual(len(unit["source_graph_node_ids"]), 1)

    def test_family_groupings(self):
        families = Counter(u["operation_family"] for u in self.units)
        self.assertEqual(families["nn.conv2d"], 76)
        self.assertEqual(families["nn.sigmoid"], 67)
        self.assertEqual(families["nn.concat"], 18)
        self.assertEqual(families["nn.split"], 8)
        self.assertEqual(families["nn.softmax"], 1)
        # Conv folds constant/pad/empty/fill/conv/bias-add. The bias-free
        # unpadded DFL conv folds 4 ops; every other conv folds 6-7.
        for unit in self.units:
            if unit["operation_family"] == "nn.conv2d":
                self.assertGreaterEqual(len(unit["mlir_operation_refs"]), 4)
        # Softmax lowering (max/sub-exp/sum/div + fills/empties) is ONE unit.
        softmax = next(u for u in self.units
                       if u["operation_family"] == "nn.softmax")
        self.assertGreaterEqual(len(softmax["mlir_operation_refs"]), 10)
        self.assertEqual(len(softmax["output_tensor_ids"]), 1)
        # Split is one multi-output unit.
        for unit in self.units:
            if unit["operation_family"] == "nn.split":
                self.assertEqual(len(unit["output_tensor_ids"]), 2)

    def test_helpers_do_not_become_units(self):
        # 929 top-level MLIR ops classified; only 268 units exist.
        self.assertEqual(self.cls["total_mlir_operations"], 929)
        self.assertEqual(self.cls["allocation_helper"], 407)
        self.assertEqual(self.cls["scalar_helper"], 83)
        self.assertEqual(self.cls["tensor_contract_operation"], 50)
        self.assertEqual(self.cls["unresolved"], 0)
        self.assertEqual(self.cls["operations_assigned_to_units"], 929)
        total = (self.cls["dispatch_root"]
                 + self.cls["dispatch_internal_compute"]
                 + self.cls["tensor_contract_operation"]
                 + self.cls["allocation_helper"]
                 + self.cls["scalar_helper"]
                 + self.cls["view_operation"]
                 + self.cls["non_dispatch_metadata"]
                 + self.cls["unresolved"])
        self.assertEqual(total, self.cls["total_mlir_operations"])

    def test_per_op_decisions_suppressed_for_cv(self):
        self.assertEqual(self.fp["per_op_decisions"], [])

    def test_provenance_fields(self):
        for unit in self.units:
            self.assertTrue(unit["source_onnx_node_names"])
            self.assertTrue(unit["source_op_type"])
            self.assertTrue(unit["mlir_operation_refs"])
        conv0 = next(u for u in self.units if u["dispatch_unit_id"] == "du_0")
        self.assertEqual(conv0["source_op_type"], "Conv")
        self.assertEqual(conv0["source_onnx_node_names"], ["/model.0/conv/Conv"])

    def test_unit_tensor_graph_resolves(self):
        known = {f"arg_{i}" for i in range(158)}
        for unit in self.units:
            known.update(unit["output_tensor_ids"])
        for unit in self.units:
            for tid in unit["input_tensor_ids"] + unit["initializer_tensor_ids"]:
                self.assertIn(tid, known)

    def test_backend_and_kernel_truth(self):
        for unit in self.units:
            self.assertEqual(unit["backend_intent"]["backend"], "coreml")
            self.assertEqual(unit["backend_intent"]["intent_basis"],
                             "configured_preference")
            self.assertIn(unit["kernel_status"],
                          {"fallback_only", "unavailable", "deferred",
                           "lowering_only", "library_available"})
            self.assertFalse(unit["executable"])
            self.assertEqual(unit["non_executable_reason"],
                             "no_runtime_adapter_or_registered_kernel")

    def test_tensor_binding_abi(self):
        bindings = self.plan["tensor_bindings"]
        roles = Counter(b["role"] for b in bindings)
        self.assertEqual(roles["model_input"], 1)
        self.assertEqual(roles["weight"], 77)
        self.assertEqual(roles["bias"], 76)
        self.assertEqual(roles["initializer"], 4)
        self.assertEqual(roles["model_output"], 2)
        image = next(b for b in bindings if b["role"] == "model_input")
        self.assertEqual(image["tensor_id"], "arg_0")
        self.assertEqual(image["original_name"], "images")
        self.assertEqual(image["argument_index"], 0)
        self.assertEqual(image["ownership"], "caller")
        self.assertTrue(image["mutable"])
        weight = next(b for b in bindings if b["role"] == "weight")
        self.assertTrue(weight["original_name"].endswith(".weight"))
        self.assertEqual(weight["ownership"], "model_state")
        self.assertFalse(weight["mutable"])
        self.assertEqual(weight["model_artifact_reference"],
                         "models/yolo-seg.onnx")
        outputs = [b for b in bindings if b["role"] == "model_output"]
        self.assertEqual({b["tensor_id"] for b in outputs},
                         {"result_0", "result_1"})

    def test_memory_metrics(self):
        memory = self.plan["cv_extension"]["memory_summary"]
        self.assertEqual(memory["model_input_bytes"], 4915200)
        self.assertEqual(memory["model_output_bytes"], 7174400)
        self.assertEqual(memory["total_intermediate_write_bytes"],
                         self.plan["cv_extension"]["memory_estimates"]
                         ["estimated_temporary_bytes"])
        self.assertEqual(
            memory["model_input_bytes"] + memory["initializer_bytes"],
            self.plan["cv_extension"]["memory_estimates"]
            ["estimated_input_bytes"])
        self.assertIsNone(memory["planned_slot_bytes"])
        self.assertEqual(memory["workspace_bytes"], 0)
        # Peak live must be dramatically below cumulative write volume.
        self.assertLess(memory["peak_live_temporary_bytes"],
                        memory["total_intermediate_write_bytes"] // 10)

    @unittest.skipUnless(FACTS_PATH.exists(), "Phase 23 planning facts absent")
    def test_peak_live_matches_phase23_analysis(self):
        facts = _load(FACTS_PATH)
        memory = self.plan["cv_extension"]["memory_summary"]
        self.assertEqual(memory["peak_live_temporary_bytes"],
                         facts["memory_summary"]["peak_live_temporary_bytes"])
        self.assertEqual(memory["total_intermediate_tensor_bytes"],
                         facts["memory_summary"]["total_temporary_bytes"])

    def test_postprocess_contract(self):
        contract = self.plan["cv_extension"]["postprocess_contract"]
        self.assertEqual(contract["detection_tensor_id"], "result_0")
        self.assertEqual(contract["detection_shape"], [1, 116, 8400])
        self.assertEqual(contract["prototype_tensor_id"], "result_1")
        self.assertEqual(contract["prototype_shape"], [1, 32, 160, 160])
        self.assertEqual(contract["mask_coefficient_channel_start"], 84)
        self.assertEqual(contract["mask_coefficient_channel_end"], 116)
        self.assertEqual(contract["nms_required"], "true")
        self.assertTrue(contract["mask_decode_required"])
        self.assertEqual(contract["implementation_status"], "runtime_required")
        groups = {g["semantic"]: g for g in contract["detection_channel_groups"]}
        self.assertEqual(groups["box_regression"]["channel_start"], 0)
        self.assertEqual(groups["box_regression"]["channel_count"], 4)
        self.assertEqual(groups["class_scores"]["channel_start"], 4)
        self.assertEqual(groups["class_scores"]["channel_count"], 80)
        self.assertEqual(groups["mask_coefficients"]["channel_start"], 84)
        self.assertEqual(groups["mask_coefficients"]["channel_count"], 32)

    def test_model_artifact_reference(self):
        self.assertEqual(self.plan["provenance"]["model_spec_ref"],
                         "models/yolo-seg.onnx")

    @unittest.skipUnless(REPORT_PATH.exists(), "dispatch unit report absent")
    def test_dispatch_unit_report_reconciles(self):
        report = _load(REPORT_PATH)
        self.assertEqual(report["schema"], "dispatch_unit_report")
        fn = report["function_reports"][0]
        self.assertEqual(fn["dispatch_unit_count"], len(self.units))
        self.assertEqual(fn["executable_dispatch_unit_count"], 0)
        self.assertEqual(fn["non_executable_dispatch_unit_count"],
                         len(self.units))
        self.assertEqual(fn["op_classification"], self.cls)
        self.assertEqual(report["tensor_binding_count"],
                         len(self.plan["tensor_bindings"]))
        self.assertIn("real_yoloseg_dispatch_units_materialized",
                      report["truth_boundary"])

    def test_runtime_loader_schema_compatibility(self):
        # Mirrors heterogeneous-inference-runtime validate_execution_plan:
        # required fields present, no measured-performance fields anywhere.
        for key in ("plan_id", "provenance", "model_identity",
                    "global_decisions", "function_plans"):
            self.assertIn(key, self.plan)
        self.assertEqual(self.plan["schema"], "execution_plan")
        self.assertEqual(self.plan["schema_version"], "2.0.0")
        self.assertIn("capability_bundle", self.plan["provenance"])
        self.assertFalse(_contains_measured_field(self.plan))


@unittest.skipUnless(QWEN_PLAN_PATH.exists(), "Qwen plan artifact absent")
class TestQwenPlanUnaffected(unittest.TestCase):
    def test_no_cv_dispatch_fields_in_qwen_plan(self):
        plan = _load(QWEN_PLAN_PATH)
        self.assertNotIn("tensor_bindings", plan)
        for fp in plan["function_plans"]:
            self.assertNotIn("dispatch_units", fp)
            self.assertNotIn("op_classification", fp)
            # LLM per-op path is preserved.
            self.assertIn("per_op_decisions", fp)


if __name__ == "__main__":
    unittest.main()
