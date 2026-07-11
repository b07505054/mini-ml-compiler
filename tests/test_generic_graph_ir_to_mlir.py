import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import generic_graph_ir_to_mlir as emitter  # noqa: E402
from test_infer_generic_graph_shapes import _node, _shape, _value  # noqa: E402


def _mlir_opt() -> str | None:
    return shutil.which("mlir-opt") or (
        "/opt/homebrew/opt/llvm/bin/mlir-opt"
        if Path("/opt/homebrew/opt/llvm/bin/mlir-opt").exists()
        else None
    )


def _filecheck() -> str | None:
    return shutil.which("FileCheck") or (
        "/opt/homebrew/opt/llvm/bin/FileCheck"
        if Path("/opt/homebrew/opt/llvm/bin/FileCheck").exists()
        else None
    )


def _graph(nodes, values, inputs=None, outputs=None, initializers=None):
    initializers = initializers or []
    return {
        "schema": "generic_graph_ir",
        "schema_version": "0.1.0",
        "graph": {
            "name": "emit_test",
            "source_name": "emit_test",
            "inputs": inputs if inputs is not None else ["input"],
            "outputs": outputs if outputs is not None else nodes[-1]["outputs"],
        },
        "nodes": nodes,
        "values": values + initializers,
        "initializers": initializers,
        "provenance": {
            "source_schema": "imported_graph_ir",
            "source_schema_version": "0.1.0",
            "truth_boundary": "test",
        },
    }


def _inferred_node(node):
    node["shape_inference_status"] = "inferred"
    node["inferred_outputs"] = [
        {"name": output, "dtype": "float", "shape": _shape([2, 3])}
        for output in node["outputs"]
    ]
    node["shape_inference_notes"] = []
    return node


def _verify_with_mlir_opt(mlir_text: str) -> str:
    mlir_opt = _mlir_opt()
    if not mlir_opt:
        raise unittest.SkipTest("mlir-opt not available")
    with tempfile.TemporaryDirectory() as tmp:
        mlir_path = Path(tmp) / "emitted.mlir"
        mlir_path.write_text(mlir_text, encoding="utf-8")
        result = subprocess.run(
            [mlir_opt, str(mlir_path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return result.stdout


def _filecheck_mlir(mlir_text: str, checks: str) -> None:
    mlir_opt = _mlir_opt()
    filecheck = _filecheck()
    if not mlir_opt or not filecheck:
        raise unittest.SkipTest("mlir-opt/FileCheck not available")
    with tempfile.TemporaryDirectory() as tmp:
        mlir_path = Path(tmp) / "emitted.mlir"
        check_path = Path(tmp) / "checks.txt"
        mlir_path.write_text(mlir_text, encoding="utf-8")
        check_path.write_text(checks, encoding="utf-8")
        optimized = subprocess.run(
            [mlir_opt, str(mlir_path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            [filecheck, str(check_path)],
            input=optimized.stdout,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


class TestGenericGraphIRToMLIR(unittest.TestCase):
    def _conv_transpose_graph(
        self,
        *,
        input_shape=None,
        weight_shape=None,
        bias_shape=None,
        output_shape=None,
        attrs=None,
        dtype="float",
    ):
        input_shape = input_shape or [1, 2, 3, 4]
        weight_shape = weight_shape or [2, 5, 2, 2]
        output_shape = output_shape or [1, 5, 6, 8]
        attrs = attrs or {
            "pads": [0, 0, 0, 0],
            "strides": [2, 2],
            "dilations": [1, 1],
            "groups": 1,
            "kernel_shape": [2, 2],
            "output_padding": [0, 0],
            "output_shape": [],
        }
        inputs = ["input", "weight"] + (["bias"] if bias_shape is not None else [])
        initializers = [_value("weight", weight_shape, dtype=dtype)]
        if bias_shape is not None:
            initializers.append(_value("bias", bias_shape, dtype=dtype))
        node = _inferred_node(_node(0, "nn.conv_transpose2d", inputs, ["out"], attrs))
        return _graph(
            [node],
            [_value("input", input_shape, dtype=dtype), _value("out", output_shape, dtype=dtype)],
            initializers=initializers,
            inputs=["input"],
            outputs=["out"],
        )

    def test_tiny_elementwise_graph_emits_valid_mlir(self):
        nodes = [
            _inferred_node(_node(0, "nn.add", ["input", "rhs"], ["add_out"])),
            _inferred_node(_node(1, "nn.mul", ["add_out", "rhs"], ["mul_out"])),
        ]
        graph_ir = _graph(
            nodes,
            [_value("input", [2, 3]), _value("rhs", [2, 3]), _value("add_out", [2, 3]), _value("mul_out", [2, 3])],
            inputs=["input", "rhs"],
            outputs=["mul_out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _verify_with_mlir_opt(mlir_text)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK-SAME: tensor<2x3xf32>
CHECK: linalg.generic
CHECK: arith.addf
CHECK: linalg.generic
CHECK: arith.mulf
CHECK: return
""",
        )

    def test_identity_forwards_ssa_value(self):
        node = _inferred_node(_node(0, "nn.identity", ["input"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        optimized = _verify_with_mlir_opt(mlir_text)

        self.assertNotIn("linalg.generic", optimized)
        self.assertIn("return %arg0 : tensor<2x3xf32>", optimized)

    def test_relu_lowering(self):
        node = _inferred_node(_node(0, "nn.relu", ["input"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: linalg.generic
CHECK: arith.constant 0.000000e+00
CHECK: arith.maximumf
CHECK: return
""",
        )

    def test_sigmoid_lowering(self):
        node = _inferred_node(_node(0, "nn.sigmoid", ["input"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: linalg.generic
CHECK: arith.subf
CHECK: math.exp
CHECK: arith.divf
CHECK: return
""",
        )

    def test_constant_lowering(self):
        node = _inferred_node(_node(0, "nn.constant", [], ["out"]))
        graph_ir = _graph(
            [node],
            [
                {
                    "name": "out",
                    "source_name": "out",
                    "dtype": "float",
                    "shape": _shape([2, 3]),
                    "literal_values": [1, 2, 3, 4, 5, 6],
                }
            ],
            inputs=[],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: arith.constant dense<{{.*}}> : tensor<2x3xf32>
CHECK: return
""",
        )

    def test_unsupported_op_fails_clearly(self):
        node = _inferred_node(
            _node(
                0,
                "nn.matmul",
                ["input", "weight"],
                ["out"],
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [2, 4])],
            initializers=[_value("weight", [3, 4])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "does not support op 'nn.matmul'"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_1x1_with_bias(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight", "bias"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 1, 1]), _value("bias", [2])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK: linalg.fill
CHECK: linalg.conv_2d_nchw_fchw
CHECK: linalg.generic
CHECK: arith.addf
CHECK: return
""",
        )

    def test_conv2d_3x3_padded_with_bias(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight", "bias"],
                ["out"],
                {"pads": [1, 1, 1, 1], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [3, 3]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 4, 8, 8])],
            initializers=[_value("weight", [4, 3, 3, 3]), _value("bias", [4])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: arith.constant 0.000000e+00 : f32
CHECK: tensor.pad
CHECK-SAME: low[0, 0, 1, 1] high[0, 0, 1, 1]
CHECK: linalg.conv_2d_nchw_fchw
CHECK: linalg.generic
CHECK: arith.addf
CHECK: return
""",
        )

    def test_conv2d_stride2(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight", "bias"],
                ["out"],
                {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "groups": 1, "kernel_shape": [3, 3]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 4, 4, 4])],
            initializers=[_value("weight", [4, 3, 3, 3]), _value("bias", [4])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: linalg.conv_2d_nchw_fchw
CHECK-SAME: strides = dense<2> : vector<2xi64>
CHECK: return
""",
        )

    def test_conv2d_asymmetric_padding(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight", "bias"],
                ["out"],
                {"pads": [0, 1, 2, 3], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [3, 3]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 5, 5]), _value("out", [1, 4, 5, 7])],
            initializers=[_value("weight", [4, 3, 3, 3]), _value("bias", [4])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.pad
CHECK-SAME: low[0, 0, 0, 1] high[0, 0, 2, 3]
CHECK: linalg.conv_2d_nchw_fchw
CHECK: return
""",
        )

    def test_conv2d_without_bias(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        optimized = _verify_with_mlir_opt(mlir_text)
        self.assertIn("linalg.conv_2d_nchw_fchw", optimized)
        self.assertNotIn("arith.addf", optimized)

    def test_conv2d_representative_yoloseg_configuration(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["images", "model.0.conv.weight", "model.0.conv.bias"],
                ["out"],
                {"pads": [1, 1, 1, 1], "strides": [2, 2], "dilations": [1, 1], "groups": 1, "kernel_shape": [3, 3]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("images", [1, 3, 640, 640]), _value("out", [1, 16, 320, 320])],
            initializers=[_value("model.0.conv.weight", [16, 3, 3, 3]), _value("model.0.conv.bias", [16])],
            inputs=["images"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.pad
CHECK: linalg.conv_2d_nchw_fchw
CHECK-SAME: strides = dense<2> : vector<2xi64>
CHECK: linalg.generic
CHECK: return
""",
        )

    def test_conv2d_unsupported_dtype_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4], dtype="int64"), _value("out", [1, 2, 4, 4], dtype="int64")],
            initializers=[_value("weight", [2, 3, 1, 1], dtype="int64")],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "unsupported dtype"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_invalid_ranks_fail(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "rank-4 NCHW"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_channel_mismatch_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 4, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "channel mismatch"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_kernel_shape_mismatch_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [1, 1, 1, 1], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 3, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "kernel_shape"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_invalid_stride_or_dilation_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [0, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "must be positive"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_invalid_pads_fail(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, -1, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "pads must be non-negative"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_output_shape_mismatch_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 5, 4])],
            initializers=[_value("weight", [2, 3, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "output shape"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_incompatible_bias_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight", "bias"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 1, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 4, 4]), _value("out", [1, 2, 4, 4])],
            initializers=[_value("weight", [2, 3, 1, 1]), _value("bias", [3])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "bias shape"):
            emitter.emit_mlir(graph_ir)

    def test_conv2d_grouped_form_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.conv2d",
                ["input", "weight"],
                ["out"],
                {"pads": [0, 0, 0, 0], "strides": [1, 1], "dilations": [1, 1], "groups": 2, "kernel_shape": [1, 1]},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 4, 4, 4]), _value("out", [1, 4, 4, 4])],
            initializers=[_value("weight", [4, 2, 1, 1])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "groups=1"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_supported_with_bias(self):
        graph_ir = self._conv_transpose_graph(bias_shape=[5])

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-DAG: affine_map<{{.*}}floordiv 2
CHECK-DAG: affine_map<{{.*}}mod 2
CHECK-LABEL: func.func @emit_test
CHECK: linalg.generic
CHECK: linalg.yield
CHECK: linalg.generic
CHECK: arith.mulf
CHECK: arith.addf
CHECK: linalg.yield
CHECK: return
""",
        )

    def test_conv_transpose2d_supported_without_bias(self):
        graph_ir = self._conv_transpose_graph()

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: linalg.fill
CHECK: linalg.generic
CHECK: arith.mulf
CHECK: arith.addf
CHECK: return
""",
        )

    def test_conv_transpose2d_exact_yoloseg_node_regression(self):
        graph_ir = self._conv_transpose_graph(
            input_shape=[1, 64, 80, 80],
            weight_shape=[64, 64, 2, 2],
            bias_shape=[64],
            output_shape=[1, 64, 160, 160],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor<1x64x80x80xf32>
CHECK: tensor<64x64x2x2xf32>
CHECK: tensor<1x64x160x160xf32>
CHECK: linalg.generic
CHECK: arith.mulf
CHECK: arith.addf
CHECK: return
""",
        )

    def test_conv_transpose2d_grouped_form_fails_preflight(self):
        graph_ir = self._conv_transpose_graph(
            input_shape=[1, 4, 3, 4],
            weight_shape=[4, 2, 2, 2],
            output_shape=[1, 4, 6, 8],
            attrs={
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 2,
                "kernel_shape": [2, 2],
                "output_padding": [0, 0],
                "output_shape": [],
            },
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only groups=1 is selected"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_nonzero_padding_fails_preflight(self):
        graph_ir = self._conv_transpose_graph(
            attrs={
                "pads": [1, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 1,
                "kernel_shape": [2, 2],
                "output_padding": [0, 0],
                "output_shape": [],
            }
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only zero padding is selected"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_nonzero_output_padding_fails_preflight(self):
        graph_ir = self._conv_transpose_graph(
            output_shape=[1, 5, 7, 8],
            attrs={
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 1,
                "kernel_shape": [2, 2],
                "output_padding": [1, 0],
                "output_shape": [],
            },
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only zero output_padding is selected"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_nonunit_dilation_fails_preflight(self):
        graph_ir = self._conv_transpose_graph(
            attrs={
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [2, 1],
                "groups": 1,
                "kernel_shape": [2, 2],
                "output_padding": [0, 0],
                "output_shape": [],
            }
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only unit dilation is selected"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_kernel_not_stride_fails_preflight(self):
        graph_ir = self._conv_transpose_graph(
            weight_shape=[2, 5, 3, 3],
            output_shape=[1, 5, 7, 9],
            attrs={
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 1,
                "kernel_shape": [3, 3],
                "output_padding": [0, 0],
                "output_shape": [],
            },
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only kernel_shape equal to strides is selected"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_overlapping_kernel_stride_fails_preflight(self):
        graph_ir = self._conv_transpose_graph(
            weight_shape=[2, 5, 3, 3],
            output_shape=[1, 5, 7, 9],
            attrs={
                "pads": [0, 0, 0, 0],
                "strides": [2, 2],
                "dilations": [1, 1],
                "groups": 1,
                "kernel_shape": [3, 3],
                "output_padding": [0, 0],
                "output_shape": [],
            },
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "kernel_shape equal to strides"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_invalid_weight_channel_layout_fails(self):
        graph_ir = self._conv_transpose_graph(weight_shape=[3, 5, 2, 2])

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "channel mismatch"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_output_shape_mismatch_fails(self):
        graph_ir = self._conv_transpose_graph(output_shape=[1, 5, 6, 9])

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "output shape"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_unsupported_dtype_fails(self):
        graph_ir = self._conv_transpose_graph(dtype="int64")

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "unsupported dtype"):
            emitter.emit_mlir(graph_ir)

    def test_conv_transpose2d_incompatible_bias_fails(self):
        graph_ir = self._conv_transpose_graph(bias_shape=[4])

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "bias shape"):
            emitter.emit_mlir(graph_ir)

    def test_reshape_static_valid_case(self):
        node = _inferred_node(
            _node(0, "nn.reshape", ["input"], ["out"], {"allowzero": 0, "target_shape": [6]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [6])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK: tensor.collapse_shape
CHECK: return
""",
        )

    def test_reshape_multi_dimensional_collapse(self):
        node = _inferred_node(
            _node(0, "nn.reshape", ["input"], ["out"], {"allowzero": 0, "target_shape": [2, 12]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3, 4]), _value("out", [2, 12])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.collapse_shape
CHECK-SAME: [1, 2]
CHECK: return
""",
        )

    def test_reshape_multi_dimensional_expand(self):
        node = _inferred_node(
            _node(0, "nn.reshape", ["input"], ["out"], {"allowzero": 0, "target_shape": [2, 3, 4]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 12]), _value("out", [2, 3, 4])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.expand_shape
CHECK-SAME: [1, 2]
CHECK: return
""",
        )

    def test_yolo_like_two_input_reshape_with_inferred_dimension(self):
        node = _inferred_node(
            _node(0, "nn.reshape", ["input", "shape"], ["out"], {"allowzero": 0, "target_shape": [1, 64, -1]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 64, 80, 80]), _value("out", [1, 64, 6400])],
            initializers=[_value("shape", [3], dtype="int64")],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.collapse_shape
CHECK-SAME: [2, 3]
CHECK: return
""",
        )

    def test_reshape_invalid_element_count_fails(self):
        node = _inferred_node(
            _node(0, "nn.reshape", ["input"], ["out"], {"allowzero": 0, "target_shape": [5]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [5])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "element count mismatch"):
            emitter.emit_mlir(graph_ir)

    def test_reshape_without_valid_reassociation_fails(self):
        node = _inferred_node(
            _node(0, "nn.reshape", ["input"], ["out"], {"allowzero": 0, "target_shape": [3, 2]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [3, 2])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "reassociation/remapping"):
            emitter.emit_mlir(graph_ir)

    def test_transpose_valid_permutation(self):
        node = _inferred_node(_node(0, "nn.transpose", ["input"], ["out"], {"perm": [1, 0]}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [3, 2])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK: linalg.transpose
CHECK-SAME: permutation = [1, 0]
CHECK: return
""",
        )

    def test_transpose_invalid_permutation_fails(self):
        node = _inferred_node(_node(0, "nn.transpose", ["input"], ["out"], {"perm": [0, 0]}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("out", [2, 2])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "invalid permutation"):
            emitter.emit_mlir(graph_ir)

    def test_scalar_broadcast_elementwise(self):
        node = _inferred_node(_node(0, "nn.add", ["input", "scalar"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("scalar", []), _value("out", [2, 3])],
            inputs=["input", "scalar"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: affine_map<(d0, d1) -> ()>
CHECK: linalg.generic
CHECK: arith.addf
CHECK: return
""",
        )

    def test_supported_static_broadcast_elementwise(self):
        node = _inferred_node(_node(0, "nn.add", ["input", "bias"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("bias", [1, 3]), _value("out", [2, 3])],
            inputs=["input", "bias"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: affine_map<(d0, d1) -> (0, d1)>
CHECK: linalg.generic
CHECK: arith.addf
CHECK: return
""",
        )

    def test_unsupported_broadcast_rejection(self):
        node = _inferred_node(_node(0, "nn.add", ["input", "bad"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("bad", [2]), _value("out", [2, 3])],
            inputs=["input", "bad"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "unsupported broadcast"):
            emitter.emit_mlir(graph_ir)

    def test_resize_nearest_asymmetric_floor_2x_emits_valid_mlir(self):
        node = _inferred_node(
            _node(
                0,
                "nn.resize",
                ["input"],
                ["out"],
                {
                    "mode": "nearest",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                    "scales": [1.0, 1.0, 2.0, 2.0],
                },
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 16, 16])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK: tensor.generate
CHECK: arith.divui
CHECK: arith.divui
CHECK: tensor.extract
CHECK: tensor.yield
CHECK: return
""",
        )

    def test_unsupported_resize_mode_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.resize",
                ["input"],
                ["out"],
                {
                    "mode": "linear",
                    "coordinate_transformation_mode": "asymmetric",
                    "nearest_mode": "floor",
                    "scales": [1.0, 1.0, 2.0, 2.0],
                },
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 16, 16])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only nearest mode is selected"):
            emitter.emit_mlir(graph_ir)

    def test_unsupported_resize_scale_fails(self):
        node = _inferred_node(
            _node(
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
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 24, 24])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only static rank-4 2x spatial scale"):
            emitter.emit_mlir(graph_ir)

    def test_unsupported_resize_coordinate_mode_fails(self):
        node = _inferred_node(
            _node(
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
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 3, 8, 8]), _value("out", [1, 3, 16, 16])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "only asymmetric coordinate transformation"):
            emitter.emit_mlir(graph_ir)

    def test_slice_along_one_axis(self):
        node = _inferred_node(
            _node(0, "nn.slice", ["input"], ["out"], {"starts": [1], "ends": [4], "axes": [1], "steps": [1]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK: tensor.extract_slice
CHECK-SAME: [0, 1] [2, 3] [1, 1]
CHECK: return
""",
        )

    def test_slice_normalizes_negative_axis(self):
        node = _inferred_node(
            _node(0, "nn.slice", ["input"], ["out"], {"starts": [2], "ends": [5], "axes": [-1], "steps": [1]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.extract_slice
CHECK-SAME: [0, 2] [2, 3] [1, 1]
CHECK: return
""",
        )

    def test_equal_split_multiple_outputs(self):
        node = _inferred_node(_node(0, "nn.split", ["input"], ["left", "right"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("left", [2, 3]), _value("right", [2, 3])],
            inputs=["input"],
            outputs=["left", "right"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK-LABEL: func.func @emit_test
CHECK: %[[LEFT:.*]] = tensor.extract_slice
CHECK-SAME: [0, 0] [2, 3] [1, 1]
CHECK: %[[RIGHT:.*]] = tensor.extract_slice
CHECK-SAME: [0, 3] [2, 3] [1, 1]
CHECK: return %[[LEFT]], %[[RIGHT]]
""",
        )

    def test_explicit_split_sizes(self):
        node = _inferred_node(_node(0, "nn.split", ["input"], ["a", "b", "c"], {"axis": 1, "split": [1, 2, 3]}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("a", [2, 1]), _value("b", [2, 2]), _value("c", [2, 3])],
            inputs=["input"],
            outputs=["a", "b", "c"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.extract_slice
CHECK-SAME: [0, 0] [2, 1] [1, 1]
CHECK: tensor.extract_slice
CHECK-SAME: [0, 1] [2, 2] [1, 1]
CHECK: tensor.extract_slice
CHECK-SAME: [0, 3] [2, 3] [1, 1]
CHECK: return
""",
        )

    def test_concat_two_inputs(self):
        node = _inferred_node(_node(0, "nn.concat", ["a", "b"], ["out"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("a", [2, 2]), _value("b", [2, 3]), _value("out", [2, 5])],
            inputs=["a", "b"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.empty
CHECK: tensor.insert_slice
CHECK-SAME: [0, 0] [2, 2] [1, 1]
CHECK: tensor.insert_slice
CHECK-SAME: [0, 2] [2, 3] [1, 1]
CHECK: return
""",
        )

    def test_concat_three_inputs(self):
        node = _inferred_node(_node(0, "nn.concat", ["a", "b", "c"], ["out"], {"axis": -1}))
        graph_ir = _graph(
            [node],
            [_value("a", [2, 1]), _value("b", [2, 2]), _value("c", [2, 3]), _value("out", [2, 6])],
            inputs=["a", "b", "c"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.empty
CHECK: tensor.insert_slice
CHECK-SAME: [0, 0] [2, 1] [1, 1]
CHECK: tensor.insert_slice
CHECK-SAME: [0, 1] [2, 2] [1, 1]
CHECK: tensor.insert_slice
CHECK-SAME: [0, 3] [2, 3] [1, 1]
CHECK: return
""",
        )

    def test_split_elementwise_concat_pipeline(self):
        nodes = [
            _inferred_node(_node(0, "nn.split", ["input"], ["left", "right"], {"axis": 1})),
            _inferred_node(_node(1, "nn.add", ["left", "bias"], ["sum"])),
            _inferred_node(_node(2, "nn.concat", ["sum", "right"], ["out"], {"axis": 1})),
        ]
        graph_ir = _graph(
            nodes,
            [
                _value("input", [2, 6]),
                _value("left", [2, 3]),
                _value("right", [2, 3]),
                _value("bias", [2, 3]),
                _value("sum", [2, 3]),
                _value("out", [2, 6]),
            ],
            inputs=["input", "bias"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: tensor.extract_slice
CHECK: tensor.extract_slice
CHECK: linalg.generic
CHECK: arith.addf
CHECK: tensor.insert_slice
CHECK: tensor.insert_slice
CHECK: return
""",
        )

    def test_slice_step_not_one_fails(self):
        node = _inferred_node(
            _node(0, "nn.slice", ["input"], ["out"], {"starts": [0], "ends": [6], "axes": [1], "steps": [2]})
        )
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "unit positive steps"):
            emitter.emit_mlir(graph_ir)

    def test_unresolved_dynamic_slice_parameters_fail(self):
        node = _inferred_node(_node(0, "nn.slice", ["input"], ["out"], {"axes": [1]}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("out", [2, 3])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "missing required canonical attrs"):
            emitter.emit_mlir(graph_ir)

    def test_split_sizes_do_not_sum_fails(self):
        node = _inferred_node(_node(0, "nn.split", ["input"], ["a", "b"], {"axis": 1, "split": [2, 2]}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 5]), _value("a", [2, 2]), _value("b", [2, 2])],
            inputs=["input"],
            outputs=["a", "b"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "sizes do not sum"):
            emitter.emit_mlir(graph_ir)

    def test_split_result_shape_mismatch_fails(self):
        node = _inferred_node(_node(0, "nn.split", ["input"], ["a", "b"], {"axis": 1, "split": [2, 4]}))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 6]), _value("a", [2, 2]), _value("b", [2, 3])],
            inputs=["input"],
            outputs=["a", "b"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "output shape"):
            emitter.emit_mlir(graph_ir)

    def test_concat_rank_mismatch_fails(self):
        node = _inferred_node(_node(0, "nn.concat", ["a", "b"], ["out"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("a", [2, 2]), _value("b", [2, 3, 1]), _value("out", [2, 5])],
            inputs=["a", "b"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "input ranks are incompatible"):
            emitter.emit_mlir(graph_ir)

    def test_concat_non_axis_dimension_mismatch_fails(self):
        node = _inferred_node(_node(0, "nn.concat", ["a", "b"], ["out"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("a", [2, 2]), _value("b", [3, 3]), _value("out", [2, 5])],
            inputs=["a", "b"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "non-axis dimensions"):
            emitter.emit_mlir(graph_ir)

    def test_concat_accumulated_output_size_mismatch_fails(self):
        node = _inferred_node(_node(0, "nn.concat", ["a", "b"], ["out"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("a", [2, 2]), _value("b", [2, 3]), _value("out", [2, 6])],
            inputs=["a", "b"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "accumulated axis size"):
            emitter.emit_mlir(graph_ir)

    def test_maxpool2d_exact_supported_form(self):
        node = _inferred_node(
            _node(
                0,
                "nn.maxpool2d",
                ["input"],
                ["out"],
                {"kernel_shape": [5, 5], "pads": [2, 2, 2, 2], "strides": [1, 1], "dilations": [1, 1], "ceil_mode": 0},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 128, 20, 20]), _value("out", [1, 128, 20, 20])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: arith.constant 0xFF800000 : f32
CHECK: tensor.pad
CHECK: linalg.fill
CHECK: linalg.pooling_nchw_max
CHECK-SAME: dilations = dense<1> : vector<2xi64>
CHECK-SAME: strides = dense<1> : vector<2xi64>
CHECK: return
""",
        )

    def test_maxpool2d_unsupported_rank_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.maxpool2d",
                ["input"],
                ["out"],
                {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2], "dilations": [1, 1], "ceil_mode": 0},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 4, 4]), _value("out", [1, 2, 2])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "rank-4 NCHW"):
            emitter.emit_mlir(graph_ir)

    def test_maxpool2d_unsupported_dtype_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.maxpool2d",
                ["input"],
                ["out"],
                {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2], "dilations": [1, 1], "ceil_mode": 0},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 1, 4, 4], dtype="int64"), _value("out", [1, 1, 2, 2], dtype="int64")],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "unsupported dtype"):
            emitter.emit_mlir(graph_ir)

    def test_maxpool2d_unsupported_attrs_fails(self):
        node = _inferred_node(
            _node(
                0,
                "nn.maxpool2d",
                ["input"],
                ["out"],
                {"kernel_shape": [2, 2], "pads": [0, 0, 0, 0], "strides": [2, 2], "dilations": [1, 1], "ceil_mode": 1},
            )
        )
        graph_ir = _graph(
            [node],
            [_value("input", [1, 1, 4, 4]), _value("out", [1, 1, 2, 2])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "ceil_mode=0"):
            emitter.emit_mlir(graph_ir)

    def test_softmax_supported_axis(self):
        node = _inferred_node(_node(0, "nn.softmax", ["input"], ["out"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("input", [1, 16, 4, 8]), _value("out", [1, 16, 4, 8])],
            inputs=["input"],
            outputs=["out"],
        )

        mlir_text = emitter.emit_mlir(graph_ir)
        _filecheck_mlir(
            mlir_text,
            """
CHECK: arith.constant 0xFF800000 : f32
CHECK: linalg.generic
CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "parallel"]
CHECK: arith.maximumf
CHECK: math.exp
CHECK: linalg.generic
CHECK: arith.addf
CHECK: arith.divf
CHECK: return
""",
        )

    def test_softmax_invalid_axis_fails(self):
        node = _inferred_node(_node(0, "nn.softmax", ["input"], ["out"], {"axis": 4}))
        graph_ir = _graph(
            [node],
            [_value("input", [1, 16, 4, 8]), _value("out", [1, 16, 4, 8])],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "outside rank"):
            emitter.emit_mlir(graph_ir)

    def test_softmax_unsupported_dtype_fails(self):
        node = _inferred_node(_node(0, "nn.softmax", ["input"], ["out"], {"axis": 1}))
        graph_ir = _graph(
            [node],
            [_value("input", [1, 16, 4, 8], dtype="int64"), _value("out", [1, 16, 4, 8], dtype="int64")],
            inputs=["input"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "unsupported dtype"):
            emitter.emit_mlir(graph_ir)

    def test_invalid_axis_fails(self):
        node = _inferred_node(_node(0, "nn.concat", ["a", "b"], ["out"], {"axis": 2}))
        graph_ir = _graph(
            [node],
            [_value("a", [2, 2]), _value("b", [2, 3]), _value("out", [2, 5])],
            inputs=["a", "b"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "outside rank"):
            emitter.emit_mlir(graph_ir)

    def test_missing_dtype_fails_clearly(self):
        node = _inferred_node(_node(0, "nn.add", ["input", "rhs"], ["out"]))
        graph_ir = _graph(
            [node],
            [_value("input", [2, 3]), _value("rhs", [2, 3]), _value("out", [2, 3], dtype="unknown")],
            inputs=["input", "rhs"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "missing required dtypes"):
            emitter.emit_mlir(graph_ir)

    def test_missing_shape_fails_clearly(self):
        node = _inferred_node(_node(0, "nn.add", ["input", "rhs"], ["out"]))
        graph_ir = _graph(
            [node],
            [
                _value("input", [2, 3]),
                _value("rhs", [2, 3]),
                {"name": "out", "source_name": "out", "dtype": "float", "shape": [{"kind": "unknown", "value": None}]},
            ],
            inputs=["input", "rhs"],
            outputs=["out"],
        )

        with self.assertRaisesRegex(emitter.GenericGraphIRToMLIRError, "missing required shapes"):
            emitter.emit_mlir(graph_ir)


if __name__ == "__main__":
    unittest.main()
