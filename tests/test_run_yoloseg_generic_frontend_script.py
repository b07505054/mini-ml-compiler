import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_yoloseg_generic_frontend.sh"


class TestRunYoloSegGenericFrontendScript(unittest.TestCase):
    def test_absent_model_exits_gracefully_without_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_model = tmp_path / "missing-yolo-seg.onnx"
            out_dir = tmp_path / "out"
            env = os.environ.copy()
            env["YOLOSEG_ONNX_PATH"] = str(missing_model)
            env["YOLOSEG_OUT_DIR"] = str(out_dir)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("YOLO-Seg ONNX model not found", result.stdout)
            self.assertIn(str(missing_model), result.stdout)
            self.assertFalse(out_dir.exists())

    @unittest.skipUnless((REPO_ROOT / "models" / "yolo-seg.onnx").exists(), "real model not available")
    def test_present_model_has_complete_op_and_shape_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            env = os.environ.copy()
            env["YOLOSEG_ONNX_PATH"] = str(REPO_ROOT / "models" / "yolo-seg.onnx")
            env["YOLOSEG_OUT_DIR"] = str(out_dir)

            result = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            diagnostics = json.loads(
                (out_dir / "yoloseg.diagnostics_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["unknown_op_count"], 0)
            self.assertEqual(diagnostics["shape_inference_status_histogram"], {"inferred": 268})
            self.assertEqual(diagnostics["frontend_readiness_status"], "ready_for_generic_lowering")


if __name__ == "__main__":
    unittest.main()
