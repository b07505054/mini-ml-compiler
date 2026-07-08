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
from test_onnx_import_to_graph_ir import _build_tiny_conv_add_model  # noqa: E402


class TestImportedGraphIRToGenericGraphIR(unittest.TestCase):
    def test_tiny_conv_add_normalizes_to_generic_ops(self):
        model = _build_tiny_conv_add_model()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "tiny_conv_add.onnx"
            onnx.save(model, str(onnx_path))
            imported = imported_ir.import_onnx_to_graph_ir(onnx_path, onnx)

        generic = generic_ir.convert_imported_graph_ir(imported)

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
