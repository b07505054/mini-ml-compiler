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


def _plugin() -> str | None:
    candidates = [
        REPO_ROOT / "build-mlir/HIRMatMulBiasReluFusionPass.dylib",
        REPO_ROOT / "build-mlir/HIRMatMulBiasReluFusionPass.so",
        REPO_ROOT / "build-mlir/libHIRMatMulBiasReluFusionPass.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _annotate(mlir_text: str) -> str:
    mlir_opt = _mlir_opt()
    plugin = _plugin()
    if not mlir_opt or not plugin:
        raise unittest.SkipTest("mlir-opt or MLIR pass plugin not available")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.mlir"
        out = Path(tmp) / "annotated.mlir"
        src.write_text(mlir_text, encoding="utf-8")
        subprocess.run(
            [
                mlir_opt,
                str(src),
                f"--load-pass-plugin={plugin}",
                f"--load-dialect-plugin={plugin}",
                "--pass-pipeline=builtin.module(cv-semantic-annotation)",
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


YOLOSEG_LIKE = """
module {
  func.func @semantic_yoloseg_like(
      %boxes: tensor<1x4x8400xf32>,
      %classes: tensor<1x80x8400xf32>,
      %mask_in: tensor<1x32x8400xf32>,
      %proto: tensor<1x32x160x160xf32>)
      -> (tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>) {
    %mask_empty = tensor.empty() : tensor<1x32x8400xf32>
    %masks = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%mask_in : tensor<1x32x8400xf32>)
      outs(%mask_empty : tensor<1x32x8400xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x8400xf32>

    %proto_empty = tensor.empty() : tensor<1x32x160x160xf32>
    %proto_out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%proto : tensor<1x32x160x160xf32>)
      outs(%proto_empty : tensor<1x32x160x160xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x160x160xf32>

    %det_empty = tensor.empty() : tensor<1x116x8400xf32>
    %det0 = tensor.insert_slice %boxes into %det_empty[0, 0, 0] [1, 4, 8400] [1, 1, 1]
      : tensor<1x4x8400xf32> into tensor<1x116x8400xf32>
    %det1 = tensor.insert_slice %classes into %det0[0, 4, 0] [1, 80, 8400] [1, 1, 1]
      : tensor<1x80x8400xf32> into tensor<1x116x8400xf32>
    %det2 = tensor.insert_slice %masks into %det1[0, 84, 0] [1, 32, 8400] [1, 1, 1]
      : tensor<1x32x8400xf32> into tensor<1x116x8400xf32>
    return %det2, %proto_out : tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>
  }
}
"""


class TestCVSemanticAnnotation(unittest.TestCase):
    def test_detection_and_prototype_output_contracts(self):
        annotated = _annotate(YOLOSEG_LIKE)
        self.assertIn('cv.model_family = "yoloseg"', annotated)
        self.assertIn('cv.output_role = "detection"', annotated)
        self.assertIn('cv.output_role = "segmentation_prototype"', annotated)
        self.assertIn('cv.semantic_role = "detection_output"', annotated)
        self.assertIn('cv.semantic_role = "segmentation_prototype"', annotated)

    def test_backward_regions_and_mask_branch(self):
        annotated = _annotate(YOLOSEG_LIKE)
        self.assertIn('cv.region_id = "cv.region.detection_head"', annotated)
        self.assertIn('cv.region_id = "cv.region.segmentation_prototype"', annotated)
        self.assertIn('cv.region_id = "cv.region.mask_coefficient_branch"', annotated)
        self.assertIn('cv.semantic_role = "mask_coefficient_branch"', annotated)
        self.assertIn('cv.contains_mask_coefficients = "true"', annotated)

    def test_multiscale_resize_concat_feature_fusion_recognition(self):
        annotated = _annotate(
            """
module {
  func.func @feature_fusion(%small: tensor<1x8x40x40xf32>,
      %boxes: tensor<1x4x8400xf32>, %classes: tensor<1x80x8400xf32>,
      %masks: tensor<1x32x8400xf32>, %proto: tensor<1x32x160x160xf32>)
      -> (tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>) {
    %c2 = arith.constant 2 : index
    %generated = tensor.generate {
    ^bb0(%n: index, %c: index, %oh: index, %ow: index):
      %ih = arith.divui %oh, %c2 : index
      %iw = arith.divui %ow, %c2 : index
      %v = tensor.extract %small[%n, %c, %ih, %iw] : tensor<1x8x40x40xf32>
      tensor.yield %v : f32
    } : tensor<1x8x80x80xf32>
    %fusion_empty = tensor.empty() : tensor<1x16x80x80xf32>
    %fusion = tensor.insert_slice %generated into %fusion_empty[0, 0, 0, 0] [1, 8, 80, 80] [1, 1, 1, 1]
      : tensor<1x8x80x80xf32> into tensor<1x16x80x80xf32>
    %proto_empty = tensor.empty() : tensor<1x32x160x160xf32>
    %proto_out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%proto : tensor<1x32x160x160xf32>) outs(%proto_empty : tensor<1x32x160x160xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x160x160xf32>
    %det_empty = tensor.empty() : tensor<1x116x8400xf32>
    %det0 = tensor.insert_slice %boxes into %det_empty[0, 0, 0] [1, 4, 8400] [1, 1, 1]
      : tensor<1x4x8400xf32> into tensor<1x116x8400xf32>
    %det1 = tensor.insert_slice %classes into %det0[0, 4, 0] [1, 80, 8400] [1, 1, 1]
      : tensor<1x80x8400xf32> into tensor<1x116x8400xf32>
    %det2 = tensor.insert_slice %masks into %det1[0, 84, 0] [1, 32, 8400] [1, 1, 1]
      : tensor<1x32x8400xf32> into tensor<1x116x8400xf32>
    return %det2, %proto_out : tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>
  }
}
"""
        )
        self.assertIn('cv.semantic_role = "feature_pyramid"', annotated)
        self.assertIn('cv.region_id = "cv.region.feature_pyramid"', annotated)
        self.assertIn('cv.recognition_confidence = "medium"', annotated)

    def test_same_source_names_incompatible_topology_does_not_match(self):
        annotated = _annotate(
            """
module {
  func.func @bad_named(%x: tensor<1x116x8400xf32> {cv.debug_source_name_evidence = "Detect"},
                       %p: tensor<1x32x160x160xf32> {cv.debug_source_name_evidence = "Proto"})
      -> (tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>) {
    return %x, %p : tensor<1x116x8400xf32>, tensor<1x32x160x160xf32>
  }
}
"""
        )
        self.assertNotIn('cv.model_family = "yoloseg"', annotated)
        self.assertNotIn('cv.output_role = "detection"', annotated)
        self.assertNotIn('cv.output_role = "segmentation_prototype"', annotated)

    def test_same_output_shape_incompatible_producer_not_high_confidence(self):
        annotated = _annotate(
            """
module {
  func.func @bad_shape(%x: tensor<1x116x8400xf32>) -> tensor<1x116x8400xf32> {
    %empty = tensor.empty() : tensor<1x116x8400xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%x : tensor<1x116x8400xf32>) outs(%empty : tensor<1x116x8400xf32>) {
    ^bb0(%in: f32, %unused: f32):
      linalg.yield %in : f32
    } -> tensor<1x116x8400xf32>
    return %out : tensor<1x116x8400xf32>
  }
}
"""
        )
        self.assertNotIn('cv.output_role = "detection"', annotated)
        self.assertNotIn('cv.recognition_confidence = "high"', annotated)

    def test_missing_second_output_does_not_invent_prototype(self):
        annotated = _annotate(
            """
module {
  func.func @detection_only(%boxes: tensor<1x4x8400xf32>, %classes: tensor<1x80x8400xf32>, %masks: tensor<1x32x8400xf32>)
      -> tensor<1x116x8400xf32> {
    %det_empty = tensor.empty() : tensor<1x116x8400xf32>
    %det0 = tensor.insert_slice %boxes into %det_empty[0, 0, 0] [1, 4, 8400] [1, 1, 1]
      : tensor<1x4x8400xf32> into tensor<1x116x8400xf32>
    %det1 = tensor.insert_slice %classes into %det0[0, 4, 0] [1, 80, 8400] [1, 1, 1]
      : tensor<1x80x8400xf32> into tensor<1x116x8400xf32>
    %det2 = tensor.insert_slice %masks into %det1[0, 84, 0] [1, 32, 8400] [1, 1, 1]
      : tensor<1x32x8400xf32> into tensor<1x116x8400xf32>
    return %det2 : tensor<1x116x8400xf32>
  }
}
"""
        )
        self.assertIn('cv.output_role = "detection"', annotated)
        self.assertNotIn('cv.output_role = "segmentation_prototype"', annotated)
        self.assertNotIn('cv.model_family = "yoloseg"', annotated)

    def test_unrelated_resize_concat_without_detection_is_not_feature_pyramid(self):
        annotated = _annotate(
            """
module {
  func.func @resize_only(%small: tensor<1x8x40x40xf32>) -> tensor<1x16x80x80xf32> {
    %c2 = arith.constant 2 : index
    %generated = tensor.generate {
    ^bb0(%n: index, %c: index, %oh: index, %ow: index):
      %ih = arith.divui %oh, %c2 : index
      %iw = arith.divui %ow, %c2 : index
      %v = tensor.extract %small[%n, %c, %ih, %iw] : tensor<1x8x40x40xf32>
      tensor.yield %v : f32
    } : tensor<1x8x80x80xf32>
    %empty = tensor.empty() : tensor<1x16x80x80xf32>
    %fusion = tensor.insert_slice %generated into %empty[0, 0, 0, 0] [1, 8, 80, 80] [1, 1, 1, 1]
      : tensor<1x8x80x80xf32> into tensor<1x16x80x80xf32>
    return %fusion : tensor<1x16x80x80xf32>
  }
}
"""
        )
        self.assertNotIn('cv.semantic_role = "feature_pyramid"', annotated)

    def test_ambiguous_mask_coefficients_remain_unresolved(self):
        annotated = _annotate(
            """
module {
  func.func @ambiguous_mask(%boxes: tensor<1x4x8400xf32>, %rest: tensor<1x112x8400xf32>)
      -> tensor<1x116x8400xf32> {
    %det_empty = tensor.empty() : tensor<1x116x8400xf32>
    %det0 = tensor.insert_slice %boxes into %det_empty[0, 0, 0] [1, 4, 8400] [1, 1, 1]
      : tensor<1x4x8400xf32> into tensor<1x116x8400xf32>
    %det1 = tensor.insert_slice %rest into %det0[0, 4, 0] [1, 112, 8400] [1, 1, 1]
      : tensor<1x112x8400xf32> into tensor<1x116x8400xf32>
    return %det1 : tensor<1x116x8400xf32>
  }
}
"""
        )
        self.assertIn("mask_coefficient_branch_unresolved", annotated)
        self.assertNotIn('cv.semantic_role = "mask_coefficient_branch"', annotated)

    def test_real_yoloseg_annotation_regression_if_artifact_present(self):
        artifact = REPO_ROOT / "artifacts/yoloseg_generic_frontend/yoloseg.generic.mlir"
        if not artifact.exists():
            raise unittest.SkipTest("YOLO-Seg generic MLIR artifact not present")
        annotated = _annotate(artifact.read_text(encoding="utf-8"))
        self.assertIn('cv.model_family = "yoloseg"', annotated)
        self.assertIn('cv.semantic_annotation.source_name_dependency = "none"', annotated)
        self.assertIn('cv.semantic_role = "detection_output"', annotated)
        self.assertIn('cv.semantic_role = "segmentation_prototype"', annotated)
        self.assertIn('cv.semantic_role = "mask_coefficient_branch"', annotated)
