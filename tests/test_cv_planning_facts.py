import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "cv_planning_facts.py"

spec = importlib.util.spec_from_file_location("cv_planning_facts", TOOL_PATH)
cpf = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["cv_planning_facts"] = cpf
spec.loader.exec_module(cpf)


def build(text: str, shape_ir=None):
    return cpf.build_report(cpf.build_facts(text, shape_ir or {}, "test.mlir"))


CONV_SNIPPET = r"""
module {
  func.func @main(%arg0: tensor<1x3x8x8xf32>, %arg1: tensor<4x3x3x3xf32>) -> (tensor<1x4x6x6xf32>) attributes {cv.model_family = "test"} {
    %0 = tensor.empty() : tensor<1x4x6x6xf32>
    %1 = linalg.fill ins(%arg0 : tensor<1x3x8x8xf32>) outs(%0 : tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>
    %2 = linalg.conv_2d_nchw_fchw {cv.recognition_confidence = "high", cv.region_id = "cv.region.detection_head", cv.semantic_role = "detection_head"} ins(%arg0, %arg1 : tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) outs(%1 : tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>
    %3 = tensor.empty() : tensor<1x4x6x6xf32>
    %4 = linalg.generic {cv.output_role = "detection", cv.postprocess_boundary = "model_output_boundary", cv.recognition_confidence = "high", cv.region_id = "cv.region.detection_head", cv.semantic_role = "detection_output"} ins(%2 : tensor<1x4x6x6xf32>) outs(%3 : tensor<1x4x6x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %in : f32
      linalg.yield %5 : f32
    } -> tensor<1x4x6x6xf32>
    func.return %4 : tensor<1x4x6x6xf32>
  }
}
"""


ELEMENTWISE_SNIPPET = r"""
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>) attributes {cv.model_family = "test"} {
    %0 = tensor.empty() : tensor<1x4xf32>
    %1 = linalg.generic {cv.output_role = "detection", cv.recognition_confidence = "high", cv.region_id = "cv.region.detection_head", cv.semantic_role = "detection_output"} ins(%arg0 : tensor<1x4xf32>) outs(%0 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.addf %in, %in : f32
      linalg.yield %2 : f32
    } -> tensor<1x4xf32>
    func.return %1 : tensor<1x4xf32>
  }
}
"""


POOL_SOFTMAX_SNIPPET = r"""
module {
  func.func @main(%arg0: tensor<1x1x4x4xf32>) -> (tensor<1x1x2x2xf32>) attributes {cv.model_family = "test"} {
    %0 = tensor.empty() : tensor<1x1x2x2xf32>
    %1 = linalg.pooling_nchw_max {cv.recognition_confidence = "high", cv.region_id = "cv.region.pool", cv.semantic_role = "detection_head"} ins(%arg0 : tensor<1x1x4x4xf32>) outs(%0 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %2 = tensor.empty() : tensor<1x1x2x2xf32>
    %3 = linalg.generic {cv.output_role = "detection", cv.recognition_confidence = "high", cv.region_id = "cv.region.softmax", cv.semantic_role = "detection_output"} ins(%1 : tensor<1x1x2x2xf32>) outs(%2 : tensor<1x1x2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = math.exp %in : f32
      %5 = arith.divf %4, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<1x1x2x2xf32>
    func.return %3 : tensor<1x1x2x2xf32>
  }
}
"""


RESIZE_CONCAT_SNIPPET = r"""
module {
  func.func @main(%arg0: tensor<1x8x4x4xf32>, %arg1: tensor<1x8x8x8xf32>) -> (tensor<1x16x8x8xf32>) attributes {cv.model_family = "test"} {
    %0 = tensor.generate {cv.feature_scale = "8x8", cv.recognition_confidence = "medium", cv.region_id = "cv.region.feature_pyramid", cv.semantic_role = "feature_pyramid"} : tensor<1x8x8x8xf32>
    %1 = tensor.empty() : tensor<1x16x8x8xf32>
    %2 = tensor.insert_slice %0 into %1[0, 0, 0, 0] [1, 8, 8, 8] [1, 1, 1, 1] {cv.feature_scale = "8x8", cv.output_role = "detection", cv.recognition_confidence = "medium", cv.region_id = "cv.region.feature_pyramid", cv.semantic_role = "feature_pyramid"} : tensor<1x8x8x8xf32> into tensor<1x16x8x8xf32>
    func.return %2 : tensor<1x16x8x8xf32>
  }
}
"""


class TestCVPlanningFacts(unittest.TestCase):
    def test_semantic_region_collection_and_output_role(self):
        report = build(CONV_SNIPPET)
        self.assertEqual(report["model_family"], "test")
        self.assertEqual(report["regions"][0]["region_id"], "cv.region.detection_head")
        self.assertEqual(report["outputs"][0]["output_role"], "detection")

    def test_tensor_byte_size_and_producer_consumers(self):
        report = build(CONV_SNIPPET)
        tensors = {t["tensor_id"]: t for t in report["tensors"]}
        self.assertEqual(tensors["arg0"]["byte_size"], 1 * 3 * 8 * 8 * 4)
        self.assertIsNone(tensors["arg0"]["producer"])
        self.assertIn("op_0002", tensors["arg0"]["consumers"])

    def test_conv_flop_estimation(self):
        report = build(CONV_SNIPPET)
        region = report["regions"][0]
        self.assertGreaterEqual(region["estimated_flops"], 2 * 1 * 4 * 6 * 6 * 3 * 3 * 3)

    def test_elementwise_traffic_estimation(self):
        report = build(ELEMENTWISE_SNIPPET)
        region = report["regions"][0]
        self.assertGreaterEqual(region["estimated_flops"], 4)
        self.assertGreater(region["estimated_read_bytes"], 0)
        self.assertGreater(region["estimated_write_bytes"], 0)

    def test_pooling_and_softmax_estimation(self):
        report = build(POOL_SOFTMAX_SNIPPET)
        flops = {r["region_id"]: r["estimated_flops"] for r in report["regions"]}
        self.assertGreater(flops["cv.region.pool"], 0)
        self.assertGreater(flops["cv.region.softmax"], 0)

    def test_lifetime_and_peak_live_temporary_memory(self):
        report = build(CONV_SNIPPET)
        self.assertGreater(report["memory_summary"]["peak_live_temporary_bytes"], 0)
        self.assertGreaterEqual(report["memory_summary"]["total_temporary_bytes"], 1 * 4 * 6 * 6 * 4)

    def test_candidate_execution_domain_generation(self):
        report = build(CONV_SNIPPET)
        domains = {d["domain"] for d in report["regions"][0]["candidate_execution_domains"]}
        self.assertIn("accelerator_candidate", domains)
        self.assertIn("cpu_candidate", domains)

    def test_quantization_and_fusion_eligibility(self):
        report = build(CONV_SNIPPET)
        region = report["regions"][0]
        self.assertEqual(region["quantization_eligibility"]["status"], "eligible")
        self.assertIn("conv_plus_bias_or_activation_candidate", region["fusion_eligibility"]["candidate_patterns"])

    def test_resize_concat_fusion_eligibility_and_feature_scale(self):
        report = build(RESIZE_CONCAT_SNIPPET)
        region = report["regions"][0]
        self.assertEqual(region["feature_scales"], ["8x8"])
        self.assertIn("transfer_or_view_operation", {d["domain"] for d in region["candidate_execution_domains"]})

    def test_missing_semantic_attrs_record_unresolved(self):
        report = build("""
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>) {
    func.return %arg0 : tensor<1x4xf32>
  }
}
""")
        self.assertTrue(any(item["kind"] == "output_role" for item in report["unresolved_facts"]))
        self.assertTrue(any(item["kind"] == "semantic_regions" for item in report["unresolved_facts"]))

    def test_dynamic_shape_records_unresolved_byte_size(self):
        report = build("""
module {
  func.func @main(%arg0: tensor<?x4xf32>) -> (tensor<?x4xf32>) attributes {cv.model_family = "test"} {
    func.return %arg0 : tensor<?x4xf32>
  }
}
""")
        self.assertTrue(any(item["kind"] == "tensor_byte_size" for item in report["unresolved_facts"]))

    def test_unsupported_dtype_records_unknown_quantization(self):
        report = build("""
module {
  func.func @main(%arg0: tensor<1x4xi32>) -> (tensor<1x4xi32>) attributes {cv.model_family = "test"} {
    %0 = tensor.empty() : tensor<1x4xi32>
    %1 = linalg.generic {cv.output_role = "detection", cv.recognition_confidence = "high", cv.region_id = "cv.region.detection_head", cv.semantic_role = "detection_output"} ins(%arg0 : tensor<1x4xi32>) outs(%0 : tensor<1x4xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x4xi32>
    func.return %1 : tensor<1x4xi32>
  }
}
""")
        self.assertEqual(report["regions"][0]["quantization_eligibility"]["status"], "unknown")
        self.assertTrue(any(item["kind"] == "tensor_byte_size" for item in report["unresolved_facts"]))

    def test_real_yoloseg_planning_facts_regression_if_artifact_present(self):
        annotated = REPO_ROOT / "artifacts/yoloseg_generic_frontend/yoloseg.cv_annotated.mlir"
        shape_ir = REPO_ROOT / "artifacts/yoloseg_generic_frontend/yoloseg.shape_generic_graph_ir.json"
        if not annotated.exists():
            self.skipTest("YOLO-Seg annotated MLIR artifact not present")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "facts.json"
            cmd = [sys.executable, str(TOOL_PATH), "--in", str(annotated), "--out", str(out)]
            if shape_ir.exists():
                cmd.extend(["--shape-ir", str(shape_ir)])
            subprocess.run(cmd, check=True)
            report = json.loads(out.read_text())
        self.assertEqual(report["model_family"], "yoloseg")
        self.assertEqual(len(report["outputs"]), 2)
        self.assertEqual(len(report["regions"]), 4)
        self.assertEqual(report["memory_summary"]["total_initializer_bytes"], 13785524)
        self.assertEqual(report["unresolved_facts"], [])


if __name__ == "__main__":
    unittest.main()
