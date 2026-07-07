import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import validate_onnx_graph_facts as validator  # noqa: E402


class TestOnnxGraphFactsFixtureRegression(unittest.TestCase):
    """Regression coverage for the existing hand-authored fixture
    (configs/models/qwen_0_5b_onnx_graph_facts.json). Deliberately does not
    import the `onnx` package -- this fixture predates the real ONNX
    protobuf bridge and must keep working as CI-facing regression coverage
    regardless of whether the optional `onnx` toolchain is installed."""

    def test_hand_authored_fixture_still_validates(self):
        fixture_path = REPO_ROOT / "configs" / "models" / "qwen_0_5b_onnx_graph_facts.json"
        facts = json.loads(fixture_path.read_text(encoding="utf-8"))
        results = validator.validate_graph_facts(facts)
        failed = [r for r in results if not r["passed"]]
        self.assertEqual(failed, [], f"unexpected validation failures: {failed}")
        # The fixture is a declared template, not a real parse -- provenance
        # must be absent, and per-layer completeness checks must report as
        # skipped rather than silently passing.
        skipped = [r for r in results if r["name"] == "per_layer_completeness_skipped"]
        self.assertEqual(len(skipped), 1)
        self.assertTrue(skipped[0]["passed"])


if __name__ == "__main__":
    unittest.main()
