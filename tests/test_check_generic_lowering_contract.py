import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import check_generic_lowering_contract as contract  # noqa: E402
import run_generic_onnx_frontend as pipeline  # noqa: E402
from test_diagnose_generic_graph_ir import _shape_annotated_tiny_conv_add  # noqa: E402
from test_infer_generic_graph_shapes import _graph, _node, _value  # noqa: E402


class TestGenericLoweringContract(unittest.TestCase):
    def test_tiny_conv_add_is_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_ir = _shape_annotated_tiny_conv_add(Path(tmp))

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "ready_for_existing_mlir_lowering")
        self.assertEqual(report["supported_ops"], ["nn.add", "nn.conv2d"])
        self.assertEqual(report["unsupported_ops"], [])
        self.assertEqual(report["blocking_nodes"], [])
        self.assertIn("linalg.conv_2d_nchw_fchw or linalg.generic", report["preferred_mlir_targets"])

    def test_unknown_op_needs_lowering_support(self):
        node = _node(0, "nn.unknown", ["input"], ["out"])
        node["shape_inference_status"] = "inferred"
        graph_ir = _graph([node], [_value("input", [1]), _value("out", [1])])

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertEqual(report["unsupported_ops"], ["nn.unknown"])
        self.assertIn("no existing-MLIR lowering strategy selected", report["blocking_nodes"][0]["reasons"])

    def test_verifier_failure_is_invalid_generic_graph_ir(self):
        node = _node(0, "nn.identity", ["input"], ["out"])
        node["shape_inference_status"] = "inferred"
        del node["source_node_id"]
        graph_ir = _graph([node], [_value("input", [1]), _value("out", [1])])

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "invalid_generic_graph_ir")
        self.assertFalse(report["verifier"]["passed"])
        self.assertTrue(any("source_node_id" in error for error in report["verifier"]["errors"]))

    def test_missing_required_canonical_attr_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_ir = _shape_annotated_tiny_conv_add(Path(tmp))
        del graph_ir["nodes"][0]["canonical_attrs"]["strides"]

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertEqual(report["missing_required_attrs"][0]["attributes"], ["strides"])

    def test_missing_shape_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_ir = _shape_annotated_tiny_conv_add(Path(tmp))
        output_name = graph_ir["nodes"][1]["outputs"][0]
        value = next(item for item in graph_ir["values"] if item["name"] == output_name)
        value["shape"] = [{"kind": "unknown", "value": None}]

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertEqual(report["missing_shapes"][0]["values"], [output_name])

    def test_missing_dtype_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_ir = _shape_annotated_tiny_conv_add(Path(tmp))
        input_name = graph_ir["nodes"][0]["inputs"][0]
        value = next(item for item in graph_ir["values"] if item["name"] == input_name)
        value["dtype"] = "unknown"

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertEqual(report["missing_dtypes"][0]["values"], [input_name])

    @unittest.skipUnless((REPO_ROOT / "models" / "yolo-seg.onnx").exists(), "real model not available")
    def test_real_model_satisfies_selected_existing_mlir_strategies(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = pipeline.run_pipeline(
                REPO_ROOT / "models" / "yolo-seg.onnx",
                Path(tmp) / "frontend",
                "model",
            )
            shape_path = Path(report["artifact_paths"]["shapes"])
            graph_ir = json.loads(shape_path.read_text(encoding="utf-8"))

        contract_report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(contract_report["contract_status"], "ready_for_existing_mlir_lowering")
        self.assertEqual(contract_report["unsupported_ops"], [])
        self.assertEqual(contract_report["missing_required_attrs"], [])
        self.assertEqual(contract_report["missing_shapes"], [])
        self.assertEqual(contract_report["missing_dtypes"], [])
        self.assertTrue(contract_report["verifier"]["passed"])

    def test_unselected_resize_semantics_still_block(self):
        node = _node(
            0,
            "nn.resize",
            ["input"],
            ["out"],
            {
                "mode": "linear",
                "coordinate_transformation_mode": "half_pixel",
                "nearest_mode": "round_prefer_floor",
                "scales": [1.0, 1.0, 2.0, 2.0],
            },
        )
        node["shape_inference_status"] = "inferred"
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 16, 16])],
        )

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertEqual(report["unsupported_ops"], ["nn.resize"])
        self.assertIn("only nearest mode is selected", report["blocking_nodes"][0]["reasons"][0])

    def test_resize_non_2x_scale_still_blocks(self):
        node = _node(
            0,
            "nn.resize",
            ["input"],
            ["out"],
            {
                "mode": "nearest",
                "coordinate_transformation_mode": "asymmetric",
                "nearest_mode": "floor",
                "scales": [1.0, 1.0, 3.0, 3.0],
            },
        )
        node["shape_inference_status"] = "inferred"
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 24, 24])],
        )

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertIn("only static rank-4 2x spatial scale", report["blocking_nodes"][0]["reasons"][0])

    def test_resize_unsupported_coordinate_mode_still_blocks(self):
        node = _node(
            0,
            "nn.resize",
            ["input"],
            ["out"],
            {
                "mode": "nearest",
                "coordinate_transformation_mode": "half_pixel",
                "nearest_mode": "floor",
                "scales": [1.0, 1.0, 2.0, 2.0],
            },
        )
        node["shape_inference_status"] = "inferred"
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 16, 16])],
        )

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertIn("only asymmetric coordinate transformation", report["blocking_nodes"][0]["reasons"][0])

    def test_selected_resize_and_conv_transpose_subsets_are_ready(self):
        resize = _node(
            0,
            "nn.resize",
            ["input"],
            ["resize_out"],
            {
                "mode": "nearest",
                "coordinate_transformation_mode": "asymmetric",
                "nearest_mode": "floor",
                "scales": [1.0, 1.0, 2.0, 2.0],
            },
        )
        transpose_conv = _node(
            1,
            "nn.conv_transpose2d",
            ["deconv_input", "weight", "bias"],
            ["deconv_out"],
            {
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 1,
                "kernel_shape": [2, 2],
                "output_padding": [0, 0],
                "output_shape": [],
            },
        )
        resize["shape_inference_status"] = "inferred"
        transpose_conv["shape_inference_status"] = "inferred"
        graph_ir = _graph(
            [resize, transpose_conv],
            [
                _value("input", [1, 3, 8, 8]),
                _value("resize_out", [1, 3, 16, 16]),
                _value("deconv_input", [1, 4, 8, 8]),
                _value("deconv_out", [1, 4, 16, 16]),
            ],
            [_value("weight", [4, 4, 2, 2]), _value("bias", [4])],
            outputs=["resize_out", "deconv_out"],
        )

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "ready_for_existing_mlir_lowering")
        self.assertEqual(report["unsupported_ops"], [])
        self.assertEqual(report["blocking_nodes"], [])

    def test_overlapping_conv_transpose_semantics_still_block(self):
        node = _node(
            0,
            "nn.conv_transpose2d",
            ["input", "weight", "bias"],
            ["out"],
            {
                "pads": [0, 0, 0, 0],
                "strides": [1, 1],
                "dilations": [1, 1],
                "groups": 1,
                "kernel_shape": [3, 3],
                "output_padding": [0, 0],
                "output_shape": [],
            },
        )
        node["shape_inference_status"] = "inferred"
        graph_ir = _graph(
            [node],
            [_value("input", [1, 4, 8, 8]), _value("out", [1, 4, 10, 10])],
            [_value("weight", [4, 4, 3, 3]), _value("bias", [4])],
        )

        report = contract.check_lowering_contract(graph_ir)

        self.assertEqual(report["contract_status"], "needs_lowering_support")
        self.assertEqual(report["unsupported_ops"], ["nn.conv_transpose2d"])
        self.assertIn("kernel_shape equal to strides", report["blocking_nodes"][0]["reasons"][0])

    def test_conv_transpose_groups_padding_and_output_padding_still_block(self):
        variants = [
            ("groups", 2, "only groups=1"),
            ("pads", [1, 1, 1, 1], "only zero padding"),
            ("output_padding", [1, 1], "only zero output_padding"),
        ]
        for attr, value, expected in variants:
            with self.subTest(attr=attr):
                attrs = {
                    "pads": [0, 0, 0, 0],
                    "strides": [2, 2],
                    "dilations": [1, 1],
                    "groups": 1,
                    "kernel_shape": [2, 2],
                    "output_padding": [0, 0],
                    "output_shape": [],
                }
                attrs[attr] = value
                node = _node(
                    0,
                    "nn.conv_transpose2d",
                    ["input", "weight", "bias"],
                    ["out"],
                    attrs,
                )
                node["shape_inference_status"] = "inferred"
                graph_ir = _graph(
                    [node],
                    [_value("input", [1, 4, 8, 8]), _value("out", [1, 4, 16, 16])],
                    [_value("weight", [4, 4, 2, 2]), _value("bias", [4])],
                )

                report = contract.check_lowering_contract(graph_ir)

                self.assertEqual(report["contract_status"], "needs_lowering_support")
                self.assertIn(expected, report["blocking_nodes"][0]["reasons"][0])


if __name__ == "__main__":
    unittest.main()
