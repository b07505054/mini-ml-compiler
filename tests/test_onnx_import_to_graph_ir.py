import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import numpy as np  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

import onnx_import_to_graph_ir as importer  # noqa: E402
import verify_graph_ir as verifier  # noqa: E402


def _initializer(name: str, dtype: int, shape: list[int]) -> "onnx.TensorProto":
    count = 1
    for dim in shape:
        count *= dim
    if dtype == TensorProto.FLOAT:
        data = np.zeros(count, dtype=np.float32)
    else:
        raise AssertionError(f"unsupported test dtype: {dtype}")
    return helper.make_tensor(name, dtype, shape, data)


def _int64_tensor(name: str, values: list[int]) -> "onnx.TensorProto":
    return helper.make_tensor(name, TensorProto.INT64, [len(values)], values)


def _build_tiny_conv_add_model():
    conv = helper.make_node(
        "Conv",
        ["input", "conv_weight", "conv_bias"],
        ["conv_out"],
        name="conv0",
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    add = helper.make_node(
        "Add",
        ["conv_out", "add_bias"],
        ["output"],
        name="add0",
    )
    graph = helper.make_graph(
        [conv, add],
        "tiny_conv_add",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 8, 8])],
        initializer=[
            _initializer("conv_weight", TensorProto.FLOAT, [4, 3, 3, 3]),
            _initializer("conv_bias", TensorProto.FLOAT, [4]),
            _initializer("add_bias", TensorProto.FLOAT, [1, 4, 1, 1]),
        ],
        value_info=[
            helper.make_tensor_value_info("conv_out", TensorProto.FLOAT, [1, 4, 8, 8]),
        ],
    )
    return helper.make_model(
        graph,
        producer_name="test_onnx_import_to_graph_ir",
        opset_imports=[helper.make_opsetid("", 17)],
    )


class TestOnnxImportToGraphIR(unittest.TestCase):
    def test_tiny_conv_add_imports_generic_graph_metadata(self):
        model = _build_tiny_conv_add_model()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "tiny_conv_add.onnx"
            onnx.save(model, str(onnx_path))

            graph_ir = importer.import_onnx_to_graph_ir(onnx_path, onnx)

        result = verifier.verify_imported_graph_ir(graph_ir)
        self.assertTrue(result.passed, result.errors)

        self.assertEqual(graph_ir["schema"], "imported_graph_ir")
        self.assertEqual(graph_ir["schema_version"], "0.1.0")
        self.assertEqual(graph_ir["graph"]["name"], "tiny_conv_add")
        self.assertEqual(graph_ir["graph"]["opset_version"], 17)
        self.assertEqual(graph_ir["graph"]["inputs"], ["input"])
        self.assertEqual(graph_ir["graph"]["outputs"], ["output"])

        nodes = graph_ir["graph"]["nodes"]
        self.assertEqual([node["op_type"] for node in nodes], ["Conv", "Add"])
        self.assertEqual(nodes[0]["name"], "conv0")
        self.assertEqual(nodes[0]["inputs"], ["input", "conv_weight", "conv_bias"])
        self.assertEqual(nodes[0]["outputs"], ["conv_out"])
        self.assertEqual(nodes[1]["name"], "add0")
        self.assertEqual(nodes[1]["inputs"], ["conv_out", "add_bias"])
        self.assertEqual(nodes[1]["outputs"], ["output"])

        initializers = {init["name"]: init for init in graph_ir["graph"]["initializers"]}
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

        values = {value["name"]: value for value in graph_ir["graph"]["values"]}
        self.assertEqual(values["input"]["dtype"], "float")
        self.assertEqual(values["output"]["dtype"], "float")
        self.assertEqual(values["conv_out"]["dtype"], "float")
        self.assertIn("conv_weight", values)

        encoded = json.dumps(graph_ir).lower()
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

    def test_cli_writes_imported_graph_ir_json(self):
        model = _build_tiny_conv_add_model()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = tmp_path / "tiny_conv_add.onnx"
            out_path = tmp_path / "imported_graph_ir.json"
            onnx.save(model, str(onnx_path))

            old_argv = sys.argv
            sys.argv = [
                "onnx_import_to_graph_ir.py",
                "--onnx",
                str(onnx_path),
                "--out",
                str(out_path),
            ]
            try:
                rc = importer.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema"], "imported_graph_ir")
        self.assertEqual([node["op_type"] for node in loaded["graph"]["nodes"]], ["Conv", "Add"])

    def test_small_shape_initializers_are_inlined_but_large_initializers_are_not(self):
        reshape = helper.make_node("Reshape", ["input", "shape"], ["reshaped"], name="reshape0")
        graph = helper.make_graph(
            [reshape],
            "literal_initializer_graph",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [6])],
            [helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [2, 3])],
            initializer=[
                _int64_tensor("shape", [2, 3]),
                _int64_tensor("large_shape_like", list(range(65))),
            ],
        )
        model = helper.make_model(graph, producer_name="test_literals", opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "literals.onnx"
            onnx.save(model, str(onnx_path))
            graph_ir = importer.import_onnx_to_graph_ir(onnx_path, onnx)

        initializers = {init["name"]: init for init in graph_ir["graph"]["initializers"]}
        values = {value["name"]: value for value in graph_ir["graph"]["values"]}
        self.assertEqual(initializers["shape"]["literal_values"], [2, 3])
        self.assertEqual(values["shape"]["literal_values"], [2, 3])
        self.assertNotIn("literal_values", initializers["large_shape_like"])
        self.assertIn("raw_data_bytes", initializers["large_shape_like"])

    def test_constant_node_small_tensor_value_is_inlined(self):
        const = helper.make_node(
            "Constant",
            [],
            ["shape_const"],
            name="const_shape",
            value=_int64_tensor("const_shape_value", [1, 6]),
        )
        reshape = helper.make_node("Reshape", ["input", "shape_const"], ["reshaped"], name="reshape0")
        graph = helper.make_graph(
            [const, reshape],
            "constant_graph",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [6])],
            [helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [1, 6])],
        )
        model = helper.make_model(graph, producer_name="test_constant", opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "constant.onnx"
            onnx.save(model, str(onnx_path))
            graph_ir = importer.import_onnx_to_graph_ir(onnx_path, onnx)

        values = {value["name"]: value for value in graph_ir["graph"]["values"]}
        nodes = {node["name"]: node for node in graph_ir["graph"]["nodes"]}
        self.assertEqual(values["shape_const"]["literal_values"], [1, 6])
        self.assertEqual(nodes["const_shape"]["attributes"][0]["value"]["literal_values"], [1, 6])
        self.assertTrue(verifier.verify_imported_graph_ir(graph_ir).passed)


if __name__ == "__main__":
    unittest.main()
