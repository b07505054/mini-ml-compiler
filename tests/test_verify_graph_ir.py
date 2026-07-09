import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import imported_graph_ir_to_generic_graph_ir as generic_ir  # noqa: E402
import onnx_import_to_graph_ir as imported_ir  # noqa: E402
import verify_graph_ir as verifier  # noqa: E402
from test_onnx_import_to_graph_ir import _build_tiny_conv_add_model  # noqa: E402


def _valid_imported_graph_ir():
    model = _build_tiny_conv_add_model()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "tiny_conv_add.onnx"
        onnx.save(model, str(onnx_path))
        return imported_ir.import_onnx_to_graph_ir(onnx_path, onnx)


def _valid_generic_graph_ir():
    return generic_ir.convert_imported_graph_ir(_valid_imported_graph_ir())


class TestVerifyGraphIR(unittest.TestCase):
    def test_valid_imported_and_generic_graph_ir_pass(self):
        imported = _valid_imported_graph_ir()
        generic = generic_ir.convert_imported_graph_ir(imported)

        imported_result = verifier.verify_graph_ir(imported)
        generic_result = verifier.verify_graph_ir(generic)

        self.assertTrue(imported_result.passed, imported_result.errors)
        self.assertTrue(generic_result.passed, generic_result.errors)

    def test_imported_missing_required_field_fails(self):
        imported = _valid_imported_graph_ir()
        del imported["graph"]["nodes"]

        result = verifier.verify_imported_graph_ir(imported)

        self.assertFalse(result.passed)
        self.assertTrue(any("graph missing required field 'nodes'" in e for e in result.errors))

    def test_imported_duplicate_node_id_fails(self):
        imported = _valid_imported_graph_ir()
        imported["graph"]["nodes"][1]["id"] = imported["graph"]["nodes"][0]["id"]

        result = verifier.verify_imported_graph_ir(imported)

        self.assertFalse(result.passed)
        self.assertTrue(any("duplicate node id" in e for e in result.errors))

    def test_imported_unresolved_input_fails(self):
        imported = _valid_imported_graph_ir()
        imported["graph"]["nodes"][0]["inputs"][0] = "missing_input"

        result = verifier.verify_imported_graph_ir(imported)

        self.assertFalse(result.passed)
        self.assertTrue(any("input 'missing_input' is unresolved" in e for e in result.errors))

    def test_imported_duplicate_output_fails_without_explicit_allow(self):
        imported = _valid_imported_graph_ir()
        imported["graph"]["nodes"][1]["outputs"] = imported["graph"]["nodes"][0]["outputs"]

        result = verifier.verify_imported_graph_ir(imported)

        self.assertFalse(result.passed)
        self.assertTrue(any("produced more than once" in e for e in result.errors))

    def test_generic_unsupported_op_fails(self):
        generic = _valid_generic_graph_ir()
        generic["nodes"][0]["op"] = "nn.not_supported"

        result = verifier.verify_generic_graph_ir(generic)

        self.assertFalse(result.passed)
        self.assertTrue(any("unsupported" in e for e in result.errors))

    def test_generic_missing_source_mapping_fails(self):
        generic = _valid_generic_graph_ir()
        del generic["nodes"][0]["source_node_id"]

        result = verifier.verify_generic_graph_ir(generic)

        self.assertFalse(result.passed)
        self.assertTrue(any("source_node_id" in e for e in result.errors))

    def test_generic_initializer_without_value_metadata_fails(self):
        generic = _valid_generic_graph_ir()
        generic["values"] = [
            value for value in generic["values"]
            if value["name"] != generic["initializers"][0]["name"]
        ]

        result = verifier.verify_generic_graph_ir(generic)

        self.assertFalse(result.passed)
        self.assertTrue(any("missing corresponding value metadata" in e for e in result.errors))

    def test_cli_validates_imported_and_generic_files(self):
        imported = _valid_imported_graph_ir()
        generic = generic_ir.convert_imported_graph_ir(imported)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            imported_path = tmp_path / "imported.json"
            generic_path = tmp_path / "generic.json"
            imported_path.write_text(json.dumps(imported), encoding="utf-8")
            generic_path.write_text(json.dumps(generic), encoding="utf-8")

            old_argv = sys.argv
            sys.argv = ["verify_graph_ir.py", str(imported_path), str(generic_path)]
            try:
                rc = verifier.main()
            finally:
                sys.argv = old_argv

        self.assertEqual(rc, 0)

    def test_cli_rejects_invalid_file(self):
        imported = _valid_imported_graph_ir()
        broken = copy.deepcopy(imported)
        broken["graph"]["nodes"][0]["inputs"][0] = "missing_input"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text(json.dumps(broken), encoding="utf-8")

            old_argv = sys.argv
            sys.argv = ["verify_graph_ir.py", str(path)]
            try:
                rc = verifier.main()
            finally:
                sys.argv = old_argv

        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
