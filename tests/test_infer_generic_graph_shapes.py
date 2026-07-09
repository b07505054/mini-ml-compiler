import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import infer_generic_graph_shapes as shape_infer  # noqa: E402
import verify_graph_ir as verifier  # noqa: E402


def _shape(dims):
    out = []
    for dim in dims:
        if isinstance(dim, int):
            out.append({"kind": "static", "value": dim})
        elif isinstance(dim, str):
            out.append({"kind": "symbolic", "value": dim})
        else:
            out.append({"kind": "unknown", "value": None})
    return out


def _value(name, dims, dtype="float"):
    return {"name": name, "source_name": name, "dtype": dtype, "shape": _shape(dims)}


def _node(node_id, op, inputs, outputs, canonical_attrs=None):
    return {
        "id": node_id,
        "name": f"node{node_id}",
        "op": op,
        "inputs": inputs,
        "outputs": outputs,
        "attributes": [],
        "source_attributes": [],
        "source_node_id": node_id,
        "source_op_type": op,
        "source_name": f"source_node{node_id}",
        "canonicalized": True,
        "canonicalization_version": "0.1.0",
        "canonical_attrs": canonical_attrs or {},
    }


def _graph(nodes, values, initializers=None, outputs=None):
    initializers = initializers or []
    output_names = outputs or nodes[-1]["outputs"]
    return {
        "schema": "generic_graph_ir",
        "schema_version": "0.1.0",
        "graph": {
            "name": "shape_test",
            "source_name": "shape_test",
            "inputs": ["input"],
            "outputs": output_names,
        },
        "nodes": nodes,
        "values": values + initializers,
        "initializers": initializers,
        "provenance": {
            "source_schema": "imported_graph_ir",
            "source_schema_version": "0.1.0",
            "truth_boundary": "imported_graph_ir_normalized_no_domain_recognition",
        },
    }


class TestInferGenericGraphShapes(unittest.TestCase):
    def test_conv2d_output_shape_inference(self):
        node = _node(
            0,
            "nn.conv2d",
            ["input", "weight", "bias"],
            ["out"],
            {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "groups": 1, "kernel_shape": [3, 3]},
        )
        generic = _graph(
            [node],
            [_value("input", [1, 3, 32, 32]), _value("out", [])],
            [_value("weight", [16, 3, 3, 3]), _value("bias", [16])],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["shape_inference_status"], "inferred")
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1, 16, 16, 16]))
        self.assertTrue(verifier.verify_generic_graph_ir(inferred).passed)

    def test_add_and_mul_broadcast_shape_inference(self):
        nodes = [
            _node(0, "nn.add", ["input", "bias"], ["add_out"]),
            _node(1, "nn.mul", ["add_out", "scale"], ["mul_out"]),
            _node(2, "nn.sub", ["mul_out", "bias"], ["sub_out"]),
            _node(3, "nn.div", ["sub_out", "scale"], ["div_out"]),
        ]
        generic = _graph(
            nodes,
            [_value("input", [2, 3, 4]), _value("add_out", []), _value("mul_out", []), _value("sub_out", []), _value("div_out", [])],
            [_value("bias", [1, 3, 1]), _value("scale", [4])],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([2, 3, 4]))
        self.assertEqual(inferred["nodes"][1]["inferred_outputs"][0]["shape"], _shape([2, 3, 4]))
        self.assertEqual(inferred["nodes"][2]["inferred_outputs"][0]["shape"], _shape([2, 3, 4]))
        self.assertEqual(inferred["nodes"][3]["inferred_outputs"][0]["shape"], _shape([2, 3, 4]))
        self.assertEqual([n["shape_inference_status"] for n in inferred["nodes"]], ["inferred", "inferred", "inferred", "inferred"])

    def test_maxpool2d_and_conv_transpose2d_shape_inference(self):
        nodes = [
            _node(
                0,
                "nn.maxpool2d",
                ["input"],
                ["pool_out"],
                {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "ceil_mode": 0},
            ),
            _node(
                1,
                "nn.conv_transpose2d",
                ["pool_out", "deconv_weight"],
                ["deconv_out"],
                {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2], "dilations": [1, 1], "groups": 1, "output_padding": [0, 0], "output_shape": []},
            ),
        ]
        generic = _graph(
            nodes,
            [_value("input", [1, 3, 32, 32]), _value("pool_out", []), _value("deconv_out", [])],
            [_value("deconv_weight", [3, 2, 2, 2])],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1, 3, 16, 16]))
        self.assertEqual(inferred["nodes"][1]["inferred_outputs"][0]["shape"], _shape([1, 2, 32, 32]))
        self.assertEqual([n["shape_inference_status"] for n in inferred["nodes"]], ["inferred", "inferred"])

    def test_split_and_slice_shape_inference(self):
        nodes = [
            _node(0, "nn.split", ["input"], ["split0", "split1"], {"axis": 1, "split": [2, 4]}),
            _node(1, "nn.slice", ["split1"], ["slice_out"], {"axes": [2, 3], "starts": [1, 2], "ends": [5, 8], "steps": [1, 2]}),
        ]
        generic = _graph(
            nodes,
            [_value("input", [1, 6, 10, 10]), _value("split0", []), _value("split1", []), _value("slice_out", [])],
            outputs=["slice_out"],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1, 2, 10, 10]))
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][1]["shape"], _shape([1, 4, 10, 10]))
        self.assertEqual(inferred["nodes"][1]["inferred_outputs"][0]["shape"], _shape([1, 4, 4, 3]))
        self.assertEqual([n["shape_inference_status"] for n in inferred["nodes"]], ["inferred", "inferred"])

    def test_split_and_slice_partial_shape_behavior(self):
        nodes = [
            _node(0, "nn.split", ["input"], ["split0", "split1"], {"axis": 1}),
            _node(1, "nn.slice", ["split0"], ["slice_out"], {}),
        ]
        generic = _graph(
            nodes,
            [_value("input", [1, "C", 10, 10]), _value("split0", []), _value("split1", []), _value("slice_out", [])],
            outputs=["slice_out"],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["shape_inference_status"], "partially_inferred")
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1, None, 10, 10]))
        self.assertEqual(inferred["nodes"][1]["shape_inference_status"], "partially_inferred")

    def test_matmul_and_gemm_output_shape_inference(self):
        nodes = [
            _node(0, "nn.matmul", ["input", "rhs"], ["matmul_out"]),
            _node(1, "nn.gemm", ["gemm_lhs", "gemm_rhs", "gemm_bias"], ["gemm_out"], {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1}),
        ]
        generic = _graph(
            nodes,
            [
                _value("input", [2, 3]),
                _value("matmul_out", []),
                _value("gemm_lhs", [5, 7]),
                _value("gemm_out", []),
            ],
            [
                _value("rhs", [3, 4]),
                _value("gemm_rhs", [9, 7]),
                _value("gemm_bias", [9]),
            ],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([2, 4]))
        self.assertEqual(inferred["nodes"][1]["inferred_outputs"][0]["shape"], _shape([5, 9]))

    def test_reshape_transpose_concat_resize_and_unary_inference(self):
        nodes = [
            _node(0, "nn.reshape", ["input"], ["reshape_out"], {"shape": [2, 3, 4]}),
            _node(1, "nn.transpose", ["reshape_out"], ["transpose_out"], {"perm": [0, 2, 1]}),
            _node(2, "nn.concat", ["transpose_out", "transpose_out"], ["concat_out"], {"axis": 1}),
            _node(3, "nn.resize", ["concat_out"], ["resize_out"], {"sizes": [2, 16, 3]}),
            _node(4, "nn.softmax", ["resize_out"], ["softmax_out"], {"axis": -1}),
            _node(5, "nn.sigmoid", ["softmax_out"], ["sigmoid_out"]),
            _node(6, "nn.relu", ["sigmoid_out"], ["relu_out"]),
            _node(7, "nn.identity", ["relu_out"], ["identity_out"]),
        ]
        generic = _graph(
            nodes,
            [_value("input", [24])] + [_value(name, []) for name in [
                "reshape_out", "transpose_out", "concat_out", "resize_out",
                "softmax_out", "sigmoid_out", "relu_out", "identity_out",
            ]],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([2, 3, 4]))
        self.assertEqual(inferred["nodes"][1]["inferred_outputs"][0]["shape"], _shape([2, 4, 3]))
        self.assertEqual(inferred["nodes"][2]["inferred_outputs"][0]["shape"], _shape([2, 8, 3]))
        self.assertEqual(inferred["nodes"][3]["inferred_outputs"][0]["shape"], _shape([2, 16, 3]))
        self.assertEqual(inferred["nodes"][7]["inferred_outputs"][0]["shape"], _shape([2, 16, 3]))
        self.assertTrue(all(node["shape_inference_status"] == "inferred" for node in inferred["nodes"]))

    def test_reshape_resolves_inferred_dimension_before_concat(self):
        nodes = [
            _node(0, "nn.reshape", ["input_a"], ["reshape_a"], {"target_shape": [1, 80, -1], "allowzero": 0}),
            _node(1, "nn.reshape", ["input_b"], ["reshape_b"], {"target_shape": [1, 32, -1], "allowzero": 0}),
            _node(2, "nn.concat", ["reshape_a", "reshape_b"], ["out"], {"axis": 1}),
        ]
        generic = _graph(
            nodes,
            [
                _value("input", [1]),
                _value("input_a", [1, 80, 20, 20]),
                _value("input_b", [1, 32, 20, 20]),
                _value("reshape_a", []),
                _value("reshape_b", []),
                _value("out", []),
            ],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1, 80, 400]))
        self.assertEqual(inferred["nodes"][1]["inferred_outputs"][0]["shape"], _shape([1, 32, 400]))
        self.assertEqual(inferred["nodes"][2]["inferred_outputs"][0]["shape"], _shape([1, 112, 400]))
        self.assertTrue(all(node["shape_inference_status"] == "inferred" for node in inferred["nodes"]))

    def test_reshape_zero_copies_input_dimension(self):
        node = _node(
            0,
            "nn.reshape",
            ["input"],
            ["out"],
            {"target_shape": [0, -1], "allowzero": 0},
        )
        generic = _graph([node], [_value("input", [2, 3, 4]), _value("out", [])])

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([2, 12]))
        self.assertEqual(inferred["nodes"][0]["shape_inference_status"], "inferred")

    def test_resize_partial_shape_behavior(self):
        node = _node(0, "nn.resize", ["input"], ["out"], {"scales": [1.0, 1.0, 2.0, 2.0]})
        generic = _graph([node], [_value("input", [1, "C", None, 32]), _value("out", [])])

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["shape_inference_status"], "partially_inferred")
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1, None, None, 64]))

    def test_unknown_op_remains_unknown(self):
        node = _node(0, "nn.unknown", ["input"], ["out"])
        generic = _graph([node], [_value("input", [1, 2]), _value("out", [])])

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["shape_inference_status"], "unknown")
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"], [{"name": "out", "source_name": "out", "dtype": "float", "shape": []}])

    def test_constant_op_uses_existing_value_metadata(self):
        node = _node(0, "nn.constant", [], ["shape_const"])
        generic = _graph(
            [node],
            [
                _value("input", [1]),
                {
                    "name": "shape_const",
                    "source_name": "shape_const",
                    "dtype": "int64",
                    "shape": _shape([2]),
                    "literal_values": [2, 3],
                },
            ],
            outputs=["shape_const"],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual(inferred["nodes"][0]["shape_inference_status"], "inferred")
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["literal_values"], [2, 3])

    def test_rank_and_dimension_mismatches_produce_error_status(self):
        conv = _node(0, "nn.conv2d", ["bad_input", "weight"], ["conv_out"], {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [3, 3]})
        matmul = _node(1, "nn.matmul", ["lhs", "rhs"], ["matmul_out"])
        add = _node(2, "nn.add", ["a", "b"], ["add_out"])
        generic = _graph(
            [conv, matmul, add],
            [
                _value("input", [1]),
                _value("bad_input", [1, 3, 8]),
                _value("lhs", [2, 3]),
                _value("a", [2, 3]),
                _value("b", [4, 3]),
                _value("conv_out", []),
                _value("matmul_out", []),
                _value("add_out", []),
            ],
            [_value("weight", [4, 3, 3, 3]), _value("rhs", [4, 5])],
            outputs=["add_out"],
        )

        inferred = shape_infer.infer_generic_graph_shapes(generic)

        self.assertEqual([node["shape_inference_status"] for node in inferred["nodes"]], ["error", "error", "error"])
        self.assertTrue(all(node["shape_inference_notes"] for node in inferred["nodes"]))
        self.assertTrue(verifier.verify_generic_graph_ir(inferred).passed)

    def test_no_domain_specific_terms_appear(self):
        node = _node(0, "nn.identity", ["input"], ["out"])
        inferred = shape_infer.infer_generic_graph_shapes(_graph([node], [_value("input", [1]), _value("out", [])]))

        encoded = json.dumps(inferred).lower()
        for term in ["qwen", "llm", "yolo", "cv", "kv", "attention", "backbone", "neck", "head"]:
            self.assertNotIn(term, encoded)

    def test_cli_writes_shape_inferred_graph(self):
        node = _node(0, "nn.matmul", ["input", "rhs"], ["out"])
        generic = _graph([node], [_value("input", [2, 3]), _value("out", [])], [_value("rhs", [3, 4])])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "canonical.json"
            out_path = tmp_path / "inferred.json"
            in_path.write_text(json.dumps(generic), encoding="utf-8")

            old_argv = sys.argv
            sys.argv = ["infer_generic_graph_shapes.py", "--in", str(in_path), "--out", str(out_path)]
            try:
                rc = shape_infer.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["nodes"][0]["inferred_outputs"][0]["shape"], _shape([2, 4]))
            self.assertTrue(verifier.verify_generic_graph_ir(loaded).passed)


if __name__ == "__main__":
    unittest.main()
