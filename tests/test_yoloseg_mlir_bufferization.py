import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _mlir_opt() -> str | None:
    return shutil.which("mlir-opt") or (
        "/opt/homebrew/opt/llvm/bin/mlir-opt"
        if Path("/opt/homebrew/opt/llvm/bin/mlir-opt").exists()
        else None
    )


def _run_bufferize(mlir_text: str) -> str:
    mlir_opt = _mlir_opt()
    if not mlir_opt:
        raise unittest.SkipTest("mlir-opt not available")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.mlir"
        out = Path(tmp) / "bufferized.mlir"
        src.write_text(mlir_text, encoding="utf-8")
        subprocess.run(
            [
                mlir_opt,
                str(src),
                "--pass-pipeline=builtin.module(one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline)",
                "-o",
                str(out),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        subprocess.run(
            [mlir_opt, str(out)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return out.read_text(encoding="utf-8")


class TestYOLOSEGMLIRBufferization(unittest.TestCase):
    def assert_bufferizes_without_tensor_ops(self, mlir_text: str) -> str:
        bufferized = _run_bufferize(mlir_text)
        self.assertNotIn("tensor.", bufferized)
        self.assertIn("memref.", bufferized)
        return bufferized

    def test_tiny_elementwise_emitted_mlir_bufferizes(self):
        self.assert_bufferizes_without_tensor_ops(
            """
module {
  func.func @elementwise(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %empty = tensor.empty() : tensor<2x3xf32>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<2x3xf32>, tensor<2x3xf32>) outs(%empty : tensor<2x3xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
"""
        )

    def test_reshape_and_transpose_bufferize(self):
        bufferized = self.assert_bufferizes_without_tensor_ops(
            """
module {
  func.func @reshape_transpose(%arg0: tensor<2x3x4xf32>) -> tensor<4x6xf32> {
    %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x3x4xf32> into tensor<6x4xf32>
    %empty = tensor.empty() : tensor<4x6xf32>
    %transposed = linalg.transpose ins(%collapsed : tensor<6x4xf32>) outs(%empty : tensor<4x6xf32>) permutation = [1, 0]
    return %transposed : tensor<4x6xf32>
  }
}
"""
        )
        self.assertIn("memref.collapse_shape", bufferized)
        self.assertIn("linalg.transpose", bufferized)

    def test_concat_split_slice_bufferize(self):
        bufferized = self.assert_bufferizes_without_tensor_ops(
            """
module {
  func.func @slice_concat(%arg0: tensor<1x4x4xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<1x4x4xf32> {
    %left = tensor.extract_slice %arg0[0, 0, 0] [1, 2, 4] [1, 1, 1] : tensor<1x4x4xf32> to tensor<1x2x4xf32>
    %right = tensor.extract_slice %arg0[0, 2, 0] [1, 2, 4] [1, 1, 1] : tensor<1x4x4xf32> to tensor<1x2x4xf32>
    %empty = tensor.empty() : tensor<1x4x4xf32>
    %first = tensor.insert_slice %left into %empty[0, 0, 0] [1, 2, 4] [1, 1, 1] : tensor<1x2x4xf32> into tensor<1x4x4xf32>
    %second = tensor.insert_slice %arg1 into %first[0, 2, 0] [1, 2, 4] [1, 1, 1] : tensor<1x2x4xf32> into tensor<1x4x4xf32>
    return %second : tensor<1x4x4xf32>
  }
}
"""
        )
        self.assertIn("memref.subview", bufferized)
        self.assertIn("memref.copy", bufferized)

    def test_conv2d_bufferizes(self):
        bufferized = self.assert_bufferizes_without_tensor_ops(
            """
module {
  func.func @conv(%input: tensor<1x3x5x5xf32>, %weight: tensor<4x3x3x3xf32>) -> tensor<1x4x5x5xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %input low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%n: index, %c: index, %h: index, %w: index):
      tensor.yield %zero : f32
    } : tensor<1x3x5x5xf32> to tensor<1x3x7x7xf32>
    %empty = tensor.empty() : tensor<1x4x5x5xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    %conv = linalg.conv_2d_nchw_fchw {dilations = dense<[1, 1]> : vector<2xi64>, strides = dense<[1, 1]> : vector<2xi64>}
      ins(%padded, %weight : tensor<1x3x7x7xf32>, tensor<4x3x3x3xf32>)
      outs(%init : tensor<1x4x5x5xf32>) -> tensor<1x4x5x5xf32>
    return %conv : tensor<1x4x5x5xf32>
  }
}
"""
        )
        self.assertIn("linalg.conv_2d_nchw_fchw", bufferized)
        self.assertIn("memref.alloc", bufferized)

    def test_resize_prototype_bufferizes(self):
        bufferized = self.assert_bufferizes_without_tensor_ops(
            """
module {
  func.func @resize(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x4xf32> {
    %c2 = arith.constant 2 : index
    %0 = tensor.generate {
    ^bb0(%n: index, %c: index, %oh: index, %ow: index):
      %ih = arith.divui %oh, %c2 : index
      %iw = arith.divui %ow, %c2 : index
      %v = tensor.extract %arg0[%n, %c, %ih, %iw] : tensor<1x1x2x2xf32>
      tensor.yield %v : f32
    } : tensor<1x1x4x4xf32>
    return %0 : tensor<1x1x4x4xf32>
  }
}
"""
        )
        self.assertIn("linalg.map", bufferized)
        self.assertIn("memref.load", bufferized)

    def test_conv_transpose_prototype_bufferizes(self):
        bufferized = self.assert_bufferizes_without_tensor_ops(
            """
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2 floordiv 2, d3 floordiv 2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d2 mod 2, d3 mod 2)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  func.func @conv_transpose(%input: tensor<1x2x2x2xf32>, %weight: tensor<2x3x2x2xf32>, %bias: tensor<3xf32>) -> tensor<1x3x4x4xf32> {
    %bias_empty = tensor.empty() : tensor<1x3x4x4xf32>
    %init = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%bias : tensor<3xf32>) outs(%bias_empty : tensor<1x3x4x4xf32>) {
    ^bb0(%b: f32, %out: f32):
      linalg.yield %b : f32
    } -> tensor<1x3x4x4xf32>
    %0 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
    } ins(%input, %weight : tensor<1x2x2x2xf32>, tensor<2x3x2x2xf32>) outs(%init : tensor<1x3x4x4xf32>) {
    ^bb0(%x: f32, %w: f32, %acc: f32):
      %prod = arith.mulf %x, %w : f32
      %sum = arith.addf %acc, %prod : f32
      linalg.yield %sum : f32
    } -> tensor<1x3x4x4xf32>
    return %0 : tensor<1x3x4x4xf32>
  }
}
"""
        )
        self.assertIn("linalg.generic", bufferized)
        self.assertIn("arith.mulf", bufferized)

    def test_softmax_bufferizes(self):
        bufferized = self.assert_bufferizes_without_tensor_ops(
            """
module {
  func.func @softmax(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %sum_empty = tensor.empty() : tensor<1xf32>
    %sum_init = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<1xf32>) -> tensor<1xf32>
    %sum = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%arg0 : tensor<1x4xf32>) outs(%sum_init : tensor<1xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %exp = math.exp %in : f32
      %next = arith.addf %acc, %exp : f32
      linalg.yield %next : f32
    } -> tensor<1xf32>
    %out_empty = tensor.empty() : tensor<1x4xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0, %sum : tensor<1x4xf32>, tensor<1xf32>) outs(%out_empty : tensor<1x4xf32>) {
    ^bb0(%in: f32, %den: f32, %unused: f32):
      %exp = math.exp %in : f32
      %v = arith.divf %exp, %den : f32
      linalg.yield %v : f32
    } -> tensor<1x4xf32>
    return %out : tensor<1x4xf32>
  }
}
"""
        )
        self.assertIn("math.exp", bufferized)
        self.assertIn("arith.divf", bufferized)

    def test_full_yoloseg_bufferization_regression_if_artifact_present(self):
        generic_mlir = REPO_ROOT / "artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir"
        if not generic_mlir.exists():
            raise unittest.SkipTest("YOLO-Seg generic MLIR artifact not present")
        script = REPO_ROOT / "scripts/lower_yoloseg_mlir_to_bufferized.sh"
        subprocess.run(
            ["bash", str(script)],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        report = json.loads(
            (REPO_ROOT / "artifacts/yoloseg_generic_frontend/yoloseg.bufferization_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["remaining_tensor_ops"], {})
        self.assertEqual(report["mlir_verification_status"], "verified_with_mlir_opt_after_bufferization")
        self.assertEqual(
            report["truth_boundary"],
            "full_graph_bufferization_verified_no_machine_codegen_no_runtime_execution_no_numerical_equivalence_validation_no_execution_plan_generation",
        )
