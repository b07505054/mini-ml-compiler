import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "list_project_onnx_models.sh"


class TestListProjectOnnxModelsScript(unittest.TestCase):
    def test_lists_project_models_and_expected_missing_yoloseg(self):
        result = subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("models/bert_tiny.onnx", result.stdout)
        self.assertIn("models/tiny_mlp.onnx", result.stdout)
        self.assertIn("models/matmul_add_relu.onnx", result.stdout)
        self.assertIn("MISSING models/yolo-seg.onnx", result.stdout)
        self.assertIn("MISSING models/yolo-seg.onnx.data", result.stdout)
        self.assertNotIn(".venv/", result.stdout)
        self.assertNotIn("build/", result.stdout)
        self.assertNotIn("site-packages", result.stdout)


if __name__ == "__main__":
    unittest.main()
