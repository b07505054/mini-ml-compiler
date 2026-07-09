import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import onnx_import_to_graph_ir as imported_ir  # noqa: E402
import imported_graph_ir_to_generic_graph_ir as generic_ir  # noqa: E402
import verify_graph_ir as verifier  # noqa: E402
from test_onnx_import_to_graph_ir import _build_tiny_conv_add_model  # noqa: E402


class TestImportedGraphIRToGenericGraphIR(unittest.TestCase):
    def test_tiny_conv_add_normalizes_to_generic_ops(self):
        model = _build_tiny_conv_add_model()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "tiny_conv_add.onnx"
            onnx.save(model, str(onnx_path))
            imported = imported_ir.import_onnx_to_graph_ir(onnx_path, onnx)

        generic = generic_ir.convert_imported_graph_ir(imported)

        result = verifier.verify_generic_graph_ir(generic)
        self.assertTrue(result.passed, result.errors)

        self.assertEqual(generic["schema"], "generic_graph_ir")
        self.assertEqual(generic["schema_version"], "0.1.0")
        self.assertEqual(generic["graph"]["name"], "tiny_conv_add")
        self.assertEqual(generic["graph"]["inputs"], ["input"])
        self.assertEqual(generic["graph"]["outputs"], ["output"])

        nodes = generic["nodes"]
        self.assertEqual([node["op"] for node in nodes], ["nn.conv2d", "nn.add"])
        self.assertEqual(nodes[0]["inputs"], ["input", "conv_weight", "conv_bias"])
        self.assertEqual(nodes[0]["outputs"], ["conv_out"])
        self.assertEqual(nodes[1]["inputs"], ["conv_out", "add_bias"])
        self.assertEqual(nodes[1]["outputs"], ["output"])

        self.assertEqual(nodes[0]["source_node_id"], 0)
        self.assertEqual(nodes[0]["source_op_type"], "Conv")
        self.assertEqual(nodes[0]["source_name"], "conv0")
        self.assertEqual(nodes[1]["source_node_id"], 1)
        self.assertEqual(nodes[1]["source_op_type"], "Add")
        self.assertEqual(nodes[1]["source_name"], "add0")

        values = {value["name"]: value for value in generic["values"]}
        self.assertEqual(values["input"]["dtype"], "float")
        self.assertEqual(values["output"]["dtype"], "float")
        self.assertEqual(values["conv_out"]["shape"][1], {"kind": "static", "value": 4})

        initializers = {init["name"]: init for init in generic["initializers"]}
        self.assertEqual(set(initializers), {"conv_weight", "conv_bias", "add_bias"})
        self.assertEqual(initializers["conv_weight"]["dtype"], "float")
        self.assertEqual(
            initializers["conv_weight"]["shape"],
            [
                {"kind": "static", "value": 4},
                {"kind": "static", "value": 3},
                {"kind": "static", "value": 3},
                {"kind": "static", "value": 3},
            ],
        )

        encoded = json.dumps(generic).lower()
        forbidden_terms = [
            "qwen",
            "llm",
            "yolo",
            "cv",
            "kv",
            "attention",
            "backbone",
            "neck",
            "head",
        ]
        for term in forbidden_terms:
            self.assertNotIn(term, encoded)

    def test_unknown_onnx_op_maps_to_nn_unknown(self):
        imported = {
            "schema": "imported_graph_ir",
            "schema_version": "0.1.0",
            "graph": {
                "name": "unknown_graph",
                "source_name": "unknown_graph",
                "inputs": ["x"],
                "outputs": ["y"],
                "nodes": [
                    {
                        "id": 0,
                        "name": "custom0",
                        "source_name": "custom0",
                        "op_type": "CustomOp",
                        "domain": "example",
                        "inputs": ["x"],
                        "outputs": ["y"],
                        "attributes": [],
                    }
                ],
                "values": [],
                "initializers": [],
            },
            "provenance": {
                "truth_boundary": "onnx_protobuf_metadata_preserved_no_domain_recognition",
            },
        }

        generic = generic_ir.convert_imported_graph_ir(imported)
        self.assertEqual(generic["nodes"][0]["op"], "nn.unknown")
        self.assertEqual(generic["nodes"][0]["source_op_type"], "CustomOp")
        self.assertEqual(generic["nodes"][0]["source_name"], "custom0")

    def test_yoloseg_gap_ops_map_to_model_agnostic_generic_ops(self):
        op_types = ["Div", "Sub", "MaxPool", "Split", "Slice", "ConvTranspose", "Constant"]
        imported = {
            "schema": "imported_graph_ir",
            "schema_version": "0.1.0",
            "graph": {
                "name": "gap_ops",
                "source_name": "gap_ops",
                "inputs": ["x"],
                "outputs": ["y5"],
                "nodes": [
                    {
                        "id": index,
                        "name": f"node{index}",
                        "source_name": f"node{index}",
                        "op_type": op_type,
                        "domain": "",
                        "inputs": ["x" if index == 0 else f"y{index - 1}"],
                        "outputs": [f"y{index}"],
                        "attributes": [],
                    }
                    for index, op_type in enumerate(op_types)
                ],
                "values": [
                    {"name": "x", "source_name": "x", "dtype": "float", "shape": []},
                    *[
                        {"name": f"y{index}", "source_name": f"y{index}", "dtype": "float", "shape": []}
                        for index in range(len(op_types))
                    ],
                ],
                "initializers": [],
            },
            "provenance": {
                "truth_boundary": "onnx_protobuf_metadata_preserved_no_domain_recognition",
            },
        }

        generic = generic_ir.convert_imported_graph_ir(imported)

        self.assertEqual(
            [node["op"] for node in generic["nodes"]],
            [
                "nn.div",
                "nn.sub",
                "nn.maxpool2d",
                "nn.split",
                "nn.slice",
                "nn.conv_transpose2d",
                "nn.constant",
            ],
        )
        self.assertTrue(verifier.verify_generic_graph_ir(generic).passed)

    def test_literal_values_are_preserved_in_generic_values_and_initializers(self):
        imported = {
            "schema": "imported_graph_ir",
            "schema_version": "0.1.0",
            "graph": {
                "name": "literal_graph",
                "source_name": "literal_graph",
                "inputs": ["x"],
                "outputs": ["y"],
                "nodes": [
                    {
                        "id": 0,
                        "name": "reshape0",
                        "source_name": "reshape0",
                        "op_type": "Reshape",
                        "domain": "",
                        "inputs": ["x", "shape"],
                        "outputs": ["y"],
                        "attributes": [],
                    }
                ],
                "values": [
                    {"name": "x", "source_name": "x", "dtype": "float", "shape": []},
                    {"name": "shape", "source_name": "shape", "dtype": "int64", "shape": [{"kind": "static", "value": 2}], "literal_values": [2, 3]},
                    {"name": "y", "source_name": "y", "dtype": "float", "shape": []},
                ],
                "initializers": [
                    {"name": "shape", "source_name": "shape", "dtype": "int64", "shape": [{"kind": "static", "value": 2}], "raw_data_bytes": 16, "literal_values": [2, 3]}
                ],
            },
            "provenance": {
                "truth_boundary": "onnx_protobuf_metadata_preserved_no_domain_recognition",
            },
        }

        generic = generic_ir.convert_imported_graph_ir(imported)

        values = {value["name"]: value for value in generic["values"]}
        initializers = {init["name"]: init for init in generic["initializers"]}
        self.assertEqual(values["shape"]["literal_values"], [2, 3])
        self.assertEqual(initializers["shape"]["literal_values"], [2, 3])

    def test_cli_writes_generic_graph_ir_json(self):
        model = _build_tiny_conv_add_model()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = tmp_path / "tiny_conv_add.onnx"
            imported_path = tmp_path / "imported_graph_ir.json"
            generic_path = tmp_path / "generic_graph_ir.json"
            onnx.save(model, str(onnx_path))

            imported = imported_ir.import_onnx_to_graph_ir(onnx_path, onnx)
            imported_path.write_text(json.dumps(imported), encoding="utf-8")

            old_argv = sys.argv
            sys.argv = [
                "imported_graph_ir_to_generic_graph_ir.py",
                "--in",
                str(imported_path),
                "--out",
                str(generic_path),
            ]
            try:
                rc = generic_ir.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            loaded = json.loads(generic_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], "generic_graph_ir")
            self.assertEqual([node["op"] for node in loaded["nodes"]], ["nn.conv2d", "nn.add"])


if __name__ == "__main__":
    unittest.main()
