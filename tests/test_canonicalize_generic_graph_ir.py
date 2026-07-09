import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import canonicalize_generic_graph_ir as canonicalizer  # noqa: E402
import verify_graph_ir as verifier  # noqa: E402


def _attr(name, attr_type, value):
    return {"name": name, "type": attr_type, "value": value}


def _generic_ir_with_nodes(nodes):
    values = [
        {"name": "input", "source_name": "input", "dtype": "float", "shape": []},
        {"name": "weight", "source_name": "weight", "dtype": "float", "shape": []},
        {"name": "bias", "source_name": "bias", "dtype": "float", "shape": []},
    ]
    for node in nodes:
        for output in node["outputs"]:
            values.append({"name": output, "source_name": output, "dtype": "float", "shape": []})

    return {
        "schema": "generic_graph_ir",
        "schema_version": "0.1.0",
        "graph": {
            "name": "generic_attr_test",
            "source_name": "generic_attr_test",
            "inputs": ["input"],
            "outputs": [nodes[-1]["outputs"][0]],
        },
        "nodes": nodes,
        "values": values,
        "initializers": [
            {"name": "weight", "source_name": "weight", "dtype": "float", "shape": []},
            {"name": "bias", "source_name": "bias", "dtype": "float", "shape": []},
        ],
        "provenance": {
            "source_schema": "imported_graph_ir",
            "source_schema_version": "0.1.0",
            "truth_boundary": "imported_graph_ir_normalized_no_domain_recognition",
        },
    }


def _node(node_id, op, inputs, outputs, attrs=None, source_op_type=None):
    return {
        "id": node_id,
        "name": f"node{node_id}",
        "op": op,
        "inputs": inputs,
        "outputs": outputs,
        "attributes": attrs or [],
        "source_node_id": node_id,
        "source_op_type": source_op_type or op,
        "source_name": f"source_node{node_id}",
    }


class TestCanonicalizeGenericGraphIR(unittest.TestCase):
    def test_conv2d_attributes_normalize_and_source_attributes_preserved(self):
        node = _node(
            0,
            "nn.conv2d",
            ["input", "weight", "bias"],
            ["conv_out"],
            [
                _attr("pads", "ints", [1, 2, 3, 4]),
                _attr("strides", "ints", [2, 2]),
                _attr("dilations", "ints", [1, 2]),
                _attr("group", "int", 4),
                _attr("kernel_shape", "ints", [3, 5]),
            ],
            "Conv",
        )
        generic = _generic_ir_with_nodes([node])

        canonical = canonicalizer.canonicalize_generic_graph_ir(generic)
        out_node = canonical["nodes"][0]

        self.assertTrue(out_node["canonicalized"])
        self.assertEqual(out_node["canonicalization_version"], "0.1.0")
        self.assertEqual(
            out_node["canonical_attrs"],
            {
                "pads": [1, 2, 3, 4],
                "strides": [2, 2],
                "dilations": [1, 2],
                "groups": 4,
                "kernel_shape": [3, 5],
            },
        )
        self.assertEqual(out_node["source_attributes"], node["attributes"])
        self.assertTrue(verifier.verify_generic_graph_ir(canonical).passed)

    def test_transpose_concat_softmax_and_gemm_attributes_normalize(self):
        nodes = [
            _node(0, "nn.transpose", ["input"], ["transpose_out"], [_attr("perm", "ints", [0, 2, 3, 1])], "Transpose"),
            _node(1, "nn.concat", ["transpose_out", "transpose_out"], ["concat_out"], [_attr("axis", "int", 1)], "Concat"),
            _node(2, "nn.softmax", ["concat_out"], ["softmax_out"], [_attr("axis", "int", -1)], "Softmax"),
            _node(3, "nn.gemm", ["softmax_out", "weight", "bias"], ["gemm_out"], [], "Gemm"),
        ]
        generic = _generic_ir_with_nodes(nodes)

        canonical = canonicalizer.canonicalize_generic_graph_ir(generic)

        self.assertEqual(canonical["nodes"][0]["canonical_attrs"], {"perm": [0, 2, 3, 1]})
        self.assertEqual(canonical["nodes"][1]["canonical_attrs"], {"axis": 1})
        self.assertEqual(canonical["nodes"][2]["canonical_attrs"], {"axis": -1})
        self.assertEqual(
            canonical["nodes"][3]["canonical_attrs"],
            {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
        )
        self.assertTrue(verifier.verify_generic_graph_ir(canonical).passed)

    def test_resize_attributes_normalize(self):
        node = _node(
            0,
            "nn.resize",
            ["input"],
            ["resize_out"],
            [
                _attr("mode", "string", "linear"),
                _attr("coordinate_transformation_mode", "string", "align_corners"),
                _attr("nearest_mode", "string", "floor"),
            ],
            "Resize",
        )
        generic = _generic_ir_with_nodes([node])

        canonical = canonicalizer.canonicalize_generic_graph_ir(generic)

        self.assertEqual(
            canonical["nodes"][0]["canonical_attrs"],
            {
                "mode": "linear",
                "coordinate_transformation_mode": "align_corners",
                "nearest_mode": "floor",
            },
        )
        self.assertTrue(verifier.verify_generic_graph_ir(canonical).passed)

    def test_yoloseg_gap_op_attributes_normalize(self):
        nodes = [
            _node(
                0,
                "nn.maxpool2d",
                ["input"],
                ["pool_out"],
                [
                    _attr("kernel_shape", "ints", [3, 3]),
                    _attr("pads", "ints", [1, 1, 1, 1]),
                    _attr("strides", "ints", [2, 2]),
                    _attr("dilations", "ints", [1, 1]),
                    _attr("ceil_mode", "int", 1),
                ],
                "MaxPool",
            ),
            _node(
                1,
                "nn.conv_transpose2d",
                ["pool_out", "weight"],
                ["deconv_out"],
                [
                    _attr("kernel_shape", "ints", [2, 2]),
                    _attr("pads", "ints", [0, 0, 0, 0]),
                    _attr("strides", "ints", [2, 2]),
                    _attr("dilations", "ints", [1, 1]),
                    _attr("group", "int", 2),
                    _attr("output_padding", "ints", [0, 0]),
                    _attr("output_shape", "ints", [16, 16]),
                ],
                "ConvTranspose",
            ),
            _node(2, "nn.split", ["deconv_out"], ["split0", "split1"], [_attr("axis", "int", 1), _attr("split", "ints", [4, 4])], "Split"),
            _node(
                3,
                "nn.slice",
                ["split0"],
                ["slice_out"],
                [
                    _attr("axes", "ints", [2, 3]),
                    _attr("starts", "ints", [1, 2]),
                    _attr("ends", "ints", [5, 8]),
                    _attr("steps", "ints", [1, 2]),
                ],
                "Slice",
            ),
            _node(4, "nn.div", ["slice_out", "bias"], ["div_out"], [], "Div"),
            _node(5, "nn.sub", ["div_out", "bias"], ["sub_out"], [], "Sub"),
        ]
        generic = _generic_ir_with_nodes(nodes)

        canonical = canonicalizer.canonicalize_generic_graph_ir(generic)

        self.assertEqual(
            canonical["nodes"][0]["canonical_attrs"],
            {"kernel_shape": [3, 3], "pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "ceil_mode": 1},
        )
        self.assertEqual(
            canonical["nodes"][1]["canonical_attrs"],
            {
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 2,
                "kernel_shape": [2, 2],
                "output_padding": [0, 0],
                "output_shape": [16, 16],
            },
        )
        self.assertEqual(canonical["nodes"][2]["canonical_attrs"], {"axis": 1, "split": [4, 4]})
        self.assertEqual(canonical["nodes"][3]["canonical_attrs"], {"axes": [2, 3], "starts": [1, 2], "ends": [5, 8], "steps": [1, 2]})
        self.assertEqual(canonical["nodes"][4]["canonical_attrs"], {})
        self.assertEqual(canonical["nodes"][5]["canonical_attrs"], {})
        self.assertTrue(verifier.verify_generic_graph_ir(canonical).passed)

    def test_shape_bearing_inputs_populate_canonical_attrs(self):
        nodes = [
            _node(0, "nn.reshape", ["input", "reshape_shape"], ["reshape_out"], [], "Reshape"),
            _node(1, "nn.slice", ["reshape_out", "slice_starts", "slice_ends", "slice_axes", "slice_steps"], ["slice_out"], [], "Slice"),
            _node(2, "nn.resize", ["slice_out", "", "resize_scales", "resize_sizes"], ["resize_out"], [], "Resize"),
            _node(3, "nn.split", ["resize_out", "split_sizes"], ["split0", "split1"], [_attr("axis", "int", 1)], "Split"),
        ]
        generic = _generic_ir_with_nodes(nodes)
        literal_records = [
            ("reshape_shape", "int64", [2], [1, 6]),
            ("slice_starts", "int64", [1], [1]),
            ("slice_ends", "int64", [1], [5]),
            ("slice_axes", "int64", [1], [1]),
            ("slice_steps", "int64", [1], [1]),
            ("resize_scales", "float", [4], [1.0, 1.0, 2.0, 2.0]),
            ("resize_sizes", "int64", [4], [1, 6, 8, 8]),
            ("split_sizes", "int64", [2], [2, 4]),
        ]
        for name, dtype, dims, values in literal_records:
            record = {
                "name": name,
                "source_name": name,
                "dtype": dtype,
                "shape": [{"kind": "static", "value": dim} for dim in dims],
                "literal_values": values,
            }
            generic["values"].append(record)
            generic["initializers"].append({**record, "raw_data_bytes": len(values) * 8})

        canonical = canonicalizer.canonicalize_generic_graph_ir(generic)

        self.assertEqual(canonical["nodes"][0]["canonical_attrs"], {"allowzero": 0, "target_shape": [1, 6]})
        self.assertEqual(canonical["nodes"][1]["canonical_attrs"], {"starts": [1], "ends": [5], "axes": [1], "steps": [1]})
        self.assertEqual(
            canonical["nodes"][2]["canonical_attrs"],
            {
                "mode": "nearest",
                "coordinate_transformation_mode": "half_pixel",
                "nearest_mode": "round_prefer_floor",
                "scales": [1.0, 1.0, 2.0, 2.0],
                "sizes": [1, 6, 8, 8],
            },
        )
        self.assertEqual(canonical["nodes"][3]["canonical_attrs"], {"axis": 1, "split": [2, 4]})
        self.assertTrue(verifier.verify_generic_graph_ir(canonical).passed)

    def test_unknown_op_is_preserved_with_empty_canonical_attrs(self):
        node = _node(0, "nn.unknown", ["input"], ["unknown_out"], [_attr("custom", "int", 7)], "CustomOp")
        generic = _generic_ir_with_nodes([node])

        canonical = canonicalizer.canonicalize_generic_graph_ir(generic)

        self.assertEqual(canonical["nodes"][0]["op"], "nn.unknown")
        self.assertEqual(canonical["nodes"][0]["canonical_attrs"], {})
        self.assertEqual(canonical["nodes"][0]["source_attributes"], node["attributes"])
        self.assertTrue(verifier.verify_generic_graph_ir(canonical).passed)

    def test_no_domain_specific_terms_appear(self):
        node = _node(0, "nn.relu", ["input"], ["relu_out"], [], "Relu")
        canonical = canonicalizer.canonicalize_generic_graph_ir(_generic_ir_with_nodes([node]))

        encoded = json.dumps(canonical).lower()
        for term in ["qwen", "llm", "yolo", "cv", "kv", "attention", "backbone", "neck", "head"]:
            self.assertNotIn(term, encoded)

    def test_cli_writes_canonicalized_generic_graph_ir(self):
        node = _node(0, "nn.softmax", ["input"], ["softmax_out"], [], "Softmax")
        generic = _generic_ir_with_nodes([node])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "generic.json"
            out_path = tmp_path / "canonical.json"
            in_path.write_text(json.dumps(generic), encoding="utf-8")

            old_argv = sys.argv
            sys.argv = ["canonicalize_generic_graph_ir.py", "--in", str(in_path), "--out", str(out_path)]
            try:
                rc = canonicalizer.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["nodes"][0]["canonical_attrs"], {"axis": -1})
            self.assertTrue(verifier.verify_generic_graph_ir(loaded).passed)

    def test_invalid_input_fails_before_canonicalization(self):
        node = _node(0, "nn.not_supported", ["input"], ["out"], [], "Bad")
        generic = _generic_ir_with_nodes([node])

        with self.assertRaises(canonicalizer.GenericGraphIRCanonicalizationError):
            canonicalizer.canonicalize_generic_graph_ir(copy.deepcopy(generic))


if __name__ == "__main__":
    unittest.main()
