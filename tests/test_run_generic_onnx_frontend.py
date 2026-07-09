import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import run_generic_onnx_frontend as pipeline  # noqa: E402
from test_onnx_import_to_graph_ir import _build_tiny_conv_add_model  # noqa: E402


def _write_tiny_model(tmp_path: Path) -> Path:
    model = _build_tiny_conv_add_model()
    onnx_path = tmp_path / "tiny_conv_add.onnx"
    onnx.save(model, str(onnx_path))
    return onnx_path


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


class TestRunGenericOnnxFrontend(unittest.TestCase):
    def test_full_pipeline_emits_all_artifacts_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = _write_tiny_model(tmp_path)
            report = pipeline.run_pipeline(onnx_path, tmp_path / "out", "tiny")

            artifact_paths = {name: Path(path) for name, path in report["artifact_paths"].items()}
            for name in [
                "imported",
                "generic",
                "canonicalized",
                "shapes",
                "diagnostics",
                "report",
            ]:
                self.assertTrue(artifact_paths[name].exists(), name)

            loaded_report = json.loads(artifact_paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(loaded_report["pipeline_status"], "completed")
            self.assertEqual(loaded_report["stages"]["imported"]["op_histogram"], {"Conv": 1, "Add": 1})
            self.assertEqual(loaded_report["stages"]["generic"]["op_histogram"], {"nn.conv2d": 1, "nn.add": 1})
            self.assertEqual(loaded_report["stages"]["generic"]["unknown_op_count"], 0)
            self.assertTrue(loaded_report["stages"]["imported"]["verifier"]["passed"])
            self.assertTrue(loaded_report["stages"]["generic"]["verifier"]["passed"])
            self.assertTrue(loaded_report["stages"]["canonicalized"]["verifier"]["passed"])
            self.assertTrue(loaded_report["stages"]["shapes"]["verifier"]["passed"])
            self.assertIn("shape_inference_summary", loaded_report["stages"]["shapes"])
            self.assertEqual(loaded_report["stages"]["shapes"]["shape_inference_summary"], {"inferred": 2})
            self.assertEqual(
                loaded_report["diagnostics"]["frontend_readiness_status"],
                "ready_for_generic_lowering",
            )

    def test_stop_after_generic_emits_only_prior_artifacts_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = _write_tiny_model(tmp_path)
            report = pipeline.run_pipeline(
                onnx_path,
                tmp_path / "out",
                "tiny",
                stop_after="generic",
            )

            artifact_paths = {name: Path(path) for name, path in report["artifact_paths"].items()}
            self.assertTrue(artifact_paths["imported"].exists())
            self.assertTrue(artifact_paths["generic"].exists())
            self.assertFalse(artifact_paths["canonicalized"].exists())
            self.assertFalse(artifact_paths["shapes"].exists())
            self.assertFalse(artifact_paths["diagnostics"].exists())
            self.assertTrue(artifact_paths["report"].exists())
            self.assertEqual(set(report["stages"]), {"imported", "generic"})

    def test_cli_runs_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = _write_tiny_model(tmp_path)
            out_dir = tmp_path / "out"

            old_argv = sys.argv
            sys.argv = [
                "run_generic_onnx_frontend.py",
                str(onnx_path),
                str(out_dir),
                "--prefix",
                "cli_tiny",
            ]
            try:
                rc = pipeline.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            self.assertTrue((out_dir / "cli_tiny.frontend_report.json").exists())

    def test_report_schema_fields_do_not_use_domain_specific_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = _write_tiny_model(tmp_path)
            report = pipeline.run_pipeline(onnx_path, tmp_path / "out", "tiny")

            for path in _schema_field_paths(report):
                lowered = path.lower()
                for term in ["qwen", "llm", "yolo", "cv", "kv_cache", "attention", "backbone", "neck", "head"]:
                    self.assertNotIn(term, lowered)


if __name__ == "__main__":
    unittest.main()
