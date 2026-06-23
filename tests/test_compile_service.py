import hashlib
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deployment"))

import compile_service  # noqa: E402

APPLE_DEMO_DIR = ROOT / "artifacts" / "apple_demo"


class _ServiceFixture:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), compile_service.CompileHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path="/compile"):
        return f"http://127.0.0.1:{self.port}{path}"

    def post(self, payload, path="/compile"):
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        req = urllib.request.Request(
            self.url(path),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _hash_apple_demo_dir():
    digests = {}
    for path in sorted(APPLE_DEMO_DIR.glob("*.json")):
        digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digests


class CompileServiceTest(unittest.TestCase):
    def test_known_graph_returns_compiled_artifact(self):
        fixture = _ServiceFixture()
        try:
            status, data = fixture.post({"graph_name": "tiny_gpt_serving"})
            self.assertEqual(status, 200)
            self.assertEqual(data["result_type"], "compiled_artifact")
            self.assertEqual(data["graph_name"], "tiny_gpt_serving")
            self.assertTrue(data["truth_boundary"]["compiler_logic_executed"])
            self.assertTrue(data["artifact_paths"])
            for path in data["artifact_paths"]:
                self.assertTrue(Path(path).exists(), path)
            self.assertEqual(data["validation"]["status"], "passed")
            self.assertIn("git_commit", data)
        finally:
            fixture.close()

    def test_omitted_graph_name_defaults_to_tiny_gpt_serving(self):
        fixture = _ServiceFixture()
        try:
            status, data = fixture.post({})
            self.assertEqual(status, 200)
            self.assertEqual(data["graph_name"], "tiny_gpt_serving")
            self.assertEqual(data["result_type"], "compiled_artifact")
        finally:
            fixture.close()

    def test_unknown_graph_returns_simulated_compile(self):
        fixture = _ServiceFixture()
        try:
            status, data = fixture.post({"graph_name": "not-a-real-graph"})
            self.assertEqual(status, 200)
            self.assertEqual(data["result_type"], "simulated_compile")
            self.assertEqual(data["artifact_paths"], [])
            self.assertFalse(data["truth_boundary"]["compiler_logic_executed"])
            self.assertEqual(data["validation"]["status"], "not_run")
        finally:
            fixture.close()

    def test_committed_artifacts_remain_unchanged(self):
        before = _hash_apple_demo_dir()
        fixture = _ServiceFixture()
        try:
            status, data = fixture.post({"graph_name": "tiny_gpt_serving"})
            self.assertEqual(status, 200)
        finally:
            fixture.close()
        after = _hash_apple_demo_dir()
        self.assertEqual(before, after)

    def test_selected_passes_matches_fixed_pipeline(self):
        fixture = _ServiceFixture()
        try:
            status, data = fixture.post({"graph_name": "tiny_gpt_serving"})
            self.assertEqual(status, 200)
            self.assertEqual(
                data["selected_passes"],
                [
                    "canonicalize",
                    "matmul_bias_relu_fusion",
                    "hir_lowering",
                    "backend_placement",
                    "memory_planning",
                ],
            )
        finally:
            fixture.close()

    def test_rejects_invalid_json(self):
        fixture = _ServiceFixture()
        try:
            req = urllib.request.Request(
                fixture.url(),
                data=b"{not json",
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                self.fail("expected HTTPError")
            except urllib.error.HTTPError as exc:
                self.assertEqual(exc.code, 400)
                data = json.loads(exc.read().decode("utf-8"))
                self.assertEqual(data["error"], "invalid_json")
        finally:
            fixture.close()

    def test_rejects_unknown_path(self):
        fixture = _ServiceFixture()
        try:
            status, _ = fixture.post({"graph_name": "tiny_gpt_serving"}, path="/not-compile")
            self.assertEqual(status, 404)
        finally:
            fixture.close()

    def test_two_calls_create_isolated_directories(self):
        fixture = _ServiceFixture()
        try:
            _, first = fixture.post({"graph_name": "tiny_gpt_serving"})
            _, second = fixture.post({"graph_name": "tiny_gpt_serving"})
            first_dir = str(Path(first["artifact_paths"][0]).parent)
            second_dir = str(Path(second["artifact_paths"][0]).parent)
            self.assertNotEqual(first_dir, second_dir)
            for path in first["artifact_paths"] + second["artifact_paths"]:
                self.assertTrue(Path(path).exists(), path)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
