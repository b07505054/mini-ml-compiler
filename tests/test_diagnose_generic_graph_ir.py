import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import canonicalize_generic_graph_ir as canonicalizer  # noqa: E402
import diagnose_generic_graph_ir as diagnostics  # noqa: E402
import imported_graph_ir_to_generic_graph_ir as generic_converter  # noqa: E402
import infer_generic_graph_shapes as shape_infer  # noqa: E402
import onnx_import_to_graph_ir as importer  # noqa: E402
import run_generic_onnx_frontend as pipeline  # noqa: E402
from test_infer_generic_graph_shapes import _graph, _node, _shape, _value  # noqa: E402
from test_onnx_import_to_graph_ir import _build_tiny_conv_add_model  # noqa: E402


def _schema_field_paths(obj, prefix=""):
    paths = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.append(path)
            paths.extend(_schema_field_paths(value, path))
    elif isinstance(obj, list):
        for value in obj:
            paths.extend(_schema_field_paths(value, prefix))
    return paths


def _shape_annotated_tiny_conv_add(tmp_path: Path):
    onnx_path = tmp_path / "tiny_conv_add.onnx"
    onnx.save(_build_tiny_conv_add_model(), str(onnx_path))
    imported = importer.import_onnx_to_graph_ir(onnx_path, onnx)
    generic = generic_converter.convert_imported_graph_ir(imported)
    canonical = canonicalizer.canonicalize_generic_graph_ir(generic)
    return shape_infer.infer_generic_graph_shapes(canonical)


class TestDiagnoseGenericGraphIR(unittest.TestCase):
    def test_conv_add_graph_is_ready_for_generic_lowering(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = diagnostics.diagnose_generic_graph_ir(_shape_annotated_tiny_conv_add(Path(tmp)))

        self.assertEqual(report["frontend_readiness_status"], "ready_for_generic_lowering")
        self.assertEqual(report["op_histogram"], {"nn.conv2d": 1, "nn.add": 1})
        self.assertEqual(report["unknown_op_count"], 0)
        self.assertEqual(report["shape_inference_status_histogram"], {"inferred": 2})
        self.assertEqual(report["shape_error_nodes"], [])
        self.assertEqual(report["shape_unknown_nodes"], [])
        self.assertTrue(report["verifier"]["passed"])

    def test_unknown_op_needs_op_support(self):
        node = _node(0, "nn.unknown", ["input"], ["out"])
        node["source_op_type"] = "CustomOp"
        graph_ir = _graph([node], [_value("input", [1, 2]), _value("out", [])])
        inferred = shape_infer.infer_generic_graph_shapes(graph_ir)

        report = diagnostics.diagnose_generic_graph_ir(inferred)

        self.assertEqual(report["frontend_readiness_status"], "needs_op_support")
        self.assertEqual(report["unknown_op_count"], 1)
        self.assertEqual(report["unknown_source_op_types"], ["CustomOp"])
        self.assertEqual(report["shape_inference_status_histogram"], {"unknown": 1})

    def test_gap_ops_are_supported_not_unknown(self):
        nodes = [
            _node(0, "nn.sub", ["input", "bias"], ["sub_out"]),
            _node(1, "nn.div", ["sub_out", "bias"], ["div_out"]),
            _node(2, "nn.maxpool2d", ["image"], ["pool_out"], {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2], "dilations": [1, 1], "ceil_mode": 0}),
            _node(3, "nn.conv_transpose2d", ["pool_out", "weight"], ["deconv_out"], {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2], "dilations": [1, 1], "groups": 1, "output_padding": [0, 0], "output_shape": []}),
            _node(4, "nn.split", ["deconv_out"], ["split0", "split1"], {"axis": 1, "split": [1, 1]}),
            _node(5, "nn.slice", ["split0"], ["slice_out"], {"axes": [2], "starts": [0], "ends": [4], "steps": [1]}),
        ]
        for node, source_op_type in zip(nodes, ["Sub", "Div", "MaxPool", "ConvTranspose", "Split", "Slice"]):
            node["source_op_type"] = source_op_type
        graph_ir = _graph(
            nodes,
            [
                _value("input", [2, 3]),
                _value("bias", [3]),
                _value("sub_out", []),
                _value("div_out", []),
                _value("image", [1, 3, 8, 8]),
                _value("pool_out", []),
                _value("deconv_out", []),
                _value("split0", []),
                _value("split1", []),
                _value("slice_out", []),
            ],
            [_value("weight", [3, 2, 2, 2])],
            outputs=["slice_out"],
        )
        inferred = shape_infer.infer_generic_graph_shapes(graph_ir)

        report = diagnostics.diagnose_generic_graph_ir(inferred)

        self.assertEqual(report["unknown_op_count"], 0)
        self.assertEqual(report["unknown_source_op_types"], [])
        self.assertEqual(report["frontend_readiness_status"], "ready_for_generic_lowering")

    def test_shape_error_needs_shape_support(self):
        node = _node(0, "nn.add", ["lhs", "rhs"], ["out"])
        graph_ir = _graph(
            [node],
            [_value("input", [1]), _value("lhs", [2, 3]), _value("rhs", [4, 3]), _value("out", [])],
        )
        inferred = shape_infer.infer_generic_graph_shapes(graph_ir)

        report = diagnostics.diagnose_generic_graph_ir(inferred)

        self.assertEqual(report["frontend_readiness_status"], "needs_shape_support")
        self.assertEqual(report["shape_inference_status_histogram"], {"error": 1})
        self.assertEqual(report["shape_error_nodes"][0]["id"], 0)
        self.assertEqual(report["unknown_op_count"], 0)

    def test_invalid_ir_reports_invalid_ir(self):
        node = _node(0, "nn.identity", ["input"], ["out"])
        graph_ir = _graph([node], [_value("input", [1]), _value("out", [])])
        del graph_ir["nodes"][0]["source_node_id"]

        report = diagnostics.diagnose_generic_graph_ir(graph_ir)

        self.assertEqual(report["frontend_readiness_status"], "invalid_ir")
        self.assertFalse(report["verifier"]["passed"])
        self.assertTrue(any("source_node_id" in error for error in report["verifier"]["errors"]))

    def test_metadata_counts_and_initializer_sizes_are_reported(self):
        node = _node(0, "nn.identity", ["input"], ["out"])
        init = _value("weights", [1, None])
        init["raw_data_bytes"] = 16
        graph_ir = _graph(
            [node],
            [_value("input", [1]), _value("out", [])],
            [init],
        )
        inferred = shape_infer.infer_generic_graph_shapes(graph_ir)

        report = diagnostics.diagnose_generic_graph_ir(inferred)

        self.assertEqual(report["metadata_counts"]["unknown_shape"], 1)
        self.assertEqual(report["top_initializers_by_raw_data_bytes"][0]["name"], "weights")
        self.assertEqual(report["top_initializers_by_raw_data_bytes"][0]["raw_data_bytes"], 16)

    def test_pipeline_driver_emits_diagnostics_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = tmp_path / "tiny_conv_add.onnx"
            onnx.save(_build_tiny_conv_add_model(), str(onnx_path))

            report = pipeline.run_pipeline(onnx_path, tmp_path / "out", "tiny")
            diagnostics_path = Path(report["artifact_paths"]["diagnostics"])
            self.assertTrue(diagnostics_path.exists())
            loaded = json.loads(diagnostics_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["frontend_readiness_status"], "ready_for_generic_lowering")
        self.assertEqual(report["diagnostics"]["artifact_path"], str(diagnostics_path))

    def test_report_schema_fields_do_not_use_domain_specific_terms(self):
        node = _node(0, "nn.identity", ["input"], ["out"])
        graph_ir = _graph([node], [_value("input", [1]), _value("out", [])])
        inferred = shape_infer.infer_generic_graph_shapes(graph_ir)

        report = diagnostics.diagnose_generic_graph_ir(inferred)

        for path in _schema_field_paths(report):
            lowered = path.lower()
            for term in ["qwen", "llm", "yolo", "cv", "kv_cache", "attention", "backbone", "neck", "head"]:
                self.assertNotIn(term, lowered)

    def test_cli_writes_diagnostics_report(self):
        node = _node(0, "nn.identity", ["input"], ["out"])
        inferred = shape_infer.infer_generic_graph_shapes(
            _graph([node], [_value("input", [1]), _value("out", [])])
        )
        self.assertEqual(inferred["nodes"][0]["inferred_outputs"][0]["shape"], _shape([1]))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            in_path = tmp_path / "shape_graph.json"
            out_path = tmp_path / "diagnostics.json"
            in_path.write_text(json.dumps(inferred), encoding="utf-8")

            old_argv = sys.argv
            sys.argv = ["diagnose_generic_graph_ir.py", "--in", str(in_path), "--out", str(out_path)]
            try:
                rc = diagnostics.main()
            finally:
                sys.argv = old_argv

            loaded = json.loads(out_path.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(loaded["frontend_readiness_status"], "ready_for_generic_lowering")


if __name__ == "__main__":
    unittest.main()
