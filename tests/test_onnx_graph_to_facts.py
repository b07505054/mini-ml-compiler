import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

onnx = pytest.importorskip("onnx")

import numpy as np  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

import onnx_graph_to_facts as bridge  # noqa: E402
import validate_onnx_graph_facts as validator  # noqa: E402


def _f16_initializer(name: str, shape: list) -> "onnx.TensorProto":
    count = 1
    for dim in shape:
        count *= dim
    return helper.make_tensor(
        name, TensorProto.FLOAT16, shape, np.zeros(count, dtype=np.uint16).tobytes(), raw=True
    )


def _tiny_qwen_initializers(
    num_layers: int,
    hidden: int,
    inter: int,
    vocab: int,
    num_heads: int,
    num_kv_heads: int,
    tie_lm_head: bool,
    add_rope_signal: bool,
) -> list:
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    inits = [_f16_initializer("model.embed_tokens.weight", [vocab, hidden])]
    for i in range(num_layers):
        inits += [
            _f16_initializer(f"model.layers.{i}.input_layernorm.weight", [hidden]),
            _f16_initializer(f"model.layers.{i}.self_attn.q_proj.weight", [hidden, hidden]),
            _f16_initializer(f"model.layers.{i}.self_attn.k_proj.weight", [kv_dim, hidden]),
            _f16_initializer(f"model.layers.{i}.self_attn.v_proj.weight", [kv_dim, hidden]),
            _f16_initializer(f"model.layers.{i}.self_attn.o_proj.weight", [hidden, hidden]),
            _f16_initializer(f"model.layers.{i}.post_attention_layernorm.weight", [hidden]),
            _f16_initializer(f"model.layers.{i}.mlp.gate_proj.weight", [inter, hidden]),
            _f16_initializer(f"model.layers.{i}.mlp.up_proj.weight", [inter, hidden]),
            _f16_initializer(f"model.layers.{i}.mlp.down_proj.weight", [hidden, inter]),
        ]
    inits.append(_f16_initializer("model.norm.weight", [hidden]))
    if not tie_lm_head:
        inits.append(_f16_initializer("lm_head.weight", [vocab, hidden]))
    if add_rope_signal:
        inits.append(_f16_initializer("model.rotary_emb.inv_freq", [head_dim // 2]))
    return inits


def _build_tiny_qwen_model(
    num_layers: int = 2,
    hidden: int = 8,
    inter: int = 16,
    vocab: int = 32,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    tie_lm_head: bool = True,
    add_rope_signal: bool = False,
    drop_role: tuple = None,
):
    """Build a tiny synthetic ONNX graph shaped like Qwen2's HuggingFace
    export (initializer naming convention only -- no real computation).
    `drop_role`, if given, is (layer_index, role_key) to omit, for testing
    the hard-failure path on an incomplete layer."""
    inits = _tiny_qwen_initializers(
        num_layers, hidden, inter, vocab, num_heads, num_kv_heads, tie_lm_head, add_rope_signal
    )
    if drop_role is not None:
        layer_index, role = drop_role
        pattern = bridge.LAYER_ROLE_PATTERNS[role].format(i=layer_index)
        import re

        regex = re.compile(pattern)
        inits = [init for init in inits if not regex.match(init.name)]

    node = helper.make_node("Identity", ["model.embed_tokens.weight"], ["out"])
    graph = helper.make_graph(
        [node],
        "tiny_qwen",
        [],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT16, [vocab, hidden])],
        initializer=inits,
    )
    return helper.make_model(graph, producer_name="test_onnx_graph_to_facts")


class TestOnnxGraphToFactsBridge(unittest.TestCase):
    def test_trivial_non_qwen_graph_fails_hard(self):
        """A minimal 1-node graph with no Qwen-shaped initializers must be
        rejected, not silently treated as a zero-layer model."""
        node = helper.make_node("Add", ["x", "y"], ["z"])
        graph = helper.make_graph(
            [node],
            "trivial",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
                helper.make_tensor_value_info("y", TensorProto.FLOAT, [2]),
            ],
            [helper.make_tensor_value_info("z", TensorProto.FLOAT, [2])],
        )
        model = helper.make_model(graph, producer_name="test")

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "trivial.onnx"
            onnx.save(model, str(onnx_path))
            with self.assertRaises(bridge.OnnxGraphToFactsError):
                bridge.extract_graph_facts(onnx_path, onnx)

    def test_tiny_transformer_block_round_trip(self):
        model = _build_tiny_qwen_model(num_layers=2, tie_lm_head=True)
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "tiny_qwen.onnx"
            onnx.save(model, str(onnx_path))

            extraction = bridge.extract_graph_facts(onnx_path, onnx)
            facts = bridge.build_graph_facts_json(
                onnx_path, "tiny-qwen-test", extraction,
                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
            )

        self.assertEqual(facts["num_layers"], 2)
        self.assertEqual(facts["hidden_size"], 8)
        self.assertEqual(facts["intermediate_size"], 16)
        self.assertEqual(facts["vocab_size"], 32)
        self.assertEqual(facts["dtype"], "fp16")
        self.assertEqual(
            facts["truth_boundary"],
            "onnx_protobuf_parsed_pattern_matched_not_general_graph_interpreter",
        )
        self.assertTrue(facts["provenance"]["lm_head_tied_to_embedding"])
        self.assertNotIn("positional_encoding", facts)

        results = validator.validate_graph_facts(facts)
        failed = [r for r in results if not r["passed"]]
        self.assertEqual(failed, [], f"unexpected validation failures: {failed}")

    def test_rope_detected_and_stamped(self):
        model = _build_tiny_qwen_model(num_layers=1, add_rope_signal=True)
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "rope_qwen.onnx"
            onnx.save(model, str(onnx_path))
            extraction = bridge.extract_graph_facts(onnx_path, onnx)
            facts = bridge.build_graph_facts_json(
                onnx_path, "rope-test", extraction,
                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
            )
        self.assertEqual(facts["positional_encoding"], "rope")

    def test_untied_lm_head_detected(self):
        model = _build_tiny_qwen_model(num_layers=1, tie_lm_head=False)
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "untied_qwen.onnx"
            onnx.save(model, str(onnx_path))
            extraction = bridge.extract_graph_facts(onnx_path, onnx)
        self.assertFalse(extraction.lm_head_tied)
        self.assertEqual(extraction.lm_head_name, "lm_head.weight")

    def test_missing_role_fails_hard_not_silent(self):
        """Dropping v_proj from layer 1 of a 2-layer model must raise, never
        silently produce a GraphFacts document that omits or guesses it."""
        model = _build_tiny_qwen_model(num_layers=2, drop_role=(1, "v_proj"))
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "broken_qwen.onnx"
            onnx.save(model, str(onnx_path))
            with self.assertRaisesRegex(bridge.OnnxGraphToFactsError, "layer 1 missing"):
                bridge.extract_graph_facts(onnx_path, onnx)

    def test_scalar_facts_resolved_from_config_json(self):
        model = _build_tiny_qwen_model(num_layers=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = tmp_path / "model.onnx"
            onnx.save(model, str(onnx_path))
            (tmp_path / "config.json").write_text(
                json.dumps({
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "max_position_embeddings": 512,
                }),
                encoding="utf-8",
            )

            out_path = tmp_path / "graph_facts.json"
            argv = [
                "onnx_graph_to_facts.py",
                "--onnx", str(onnx_path),
                "--out", str(out_path),
            ]
            old_argv = sys.argv
            sys.argv = argv
            try:
                rc = bridge.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 0)
            facts = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(facts["num_attention_heads"], 2)
            self.assertEqual(facts["num_key_value_heads"], 1)
            self.assertEqual(facts["max_position_embeddings"], 512)

    def test_scalar_facts_missing_raises_hard_failure(self):
        model = _build_tiny_qwen_model(num_layers=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = tmp_path / "model.onnx"
            onnx.save(model, str(onnx_path))
            out_path = tmp_path / "graph_facts.json"

            argv = ["onnx_graph_to_facts.py", "--onnx", str(onnx_path), "--out", str(out_path)]
            old_argv = sys.argv
            sys.argv = argv
            try:
                rc = bridge.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(rc, 1)
            self.assertFalse(out_path.exists())

    def test_cli_skips_cleanly_when_onnx_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_path = tmp_path / "graph_facts.json"
            argv = [
                "onnx_graph_to_facts.py",
                "--onnx", str(tmp_path / "does_not_matter.onnx"),
                "--out", str(out_path),
            ]
            old_argv = sys.argv
            old_probe = bridge.PROBE_REPORT_JSON
            sys.argv = argv
            bridge.PROBE_REPORT_JSON = tmp_path / "probe.json"
            try:
                rc = bridge.module_available("onnx_module_that_does_not_exist_at_all")
                self.assertFalse(rc)
                # Simulate the missing-toolchain branch directly rather than
                # uninstalling onnx from this interpreter.
                orig = bridge.module_available
                bridge.module_available = lambda name: False
                try:
                    rc = bridge.main()
                finally:
                    bridge.module_available = orig
            finally:
                sys.argv = old_argv
                bridge.PROBE_REPORT_JSON = old_probe

            self.assertEqual(rc, 0)
            self.assertFalse(out_path.exists())


class TestOnnxGraphToServingMlirEndToEnd(unittest.TestCase):
    """Wires the tiny transformer block through the real C++ importer.
    Skips cleanly if the MLIR toolchain build isn't available locally."""

    def test_bridge_output_flows_through_cpp_importer(self):
        importer = REPO_ROOT / "build-mlir" / "qwen-onnx-to-serving-mlir"
        if not importer.exists():
            pytest.skip(
                f"{importer} not built; run "
                "'cmake --build build-mlir --target qwen-onnx-to-serving-mlir' first"
            )

        model = _build_tiny_qwen_model(num_layers=2, add_rope_signal=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onnx_path = tmp_path / "tiny_qwen.onnx"
            onnx.save(model, str(onnx_path))

            extraction = bridge.extract_graph_facts(onnx_path, onnx)
            facts = bridge.build_graph_facts_json(
                onnx_path, "tiny-qwen-e2e", extraction,
                num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
            )
            graph_facts_path = tmp_path / "graph_facts.json"
            graph_facts_path.write_text(json.dumps(facts), encoding="utf-8")

            mlir_out_path = tmp_path / "out.mlir"
            result = subprocess.run(
                [str(importer), "--graph-facts", str(graph_facts_path), "--out", str(mlir_out_path)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            mlir_text = mlir_out_path.read_text(encoding="utf-8")
            self.assertIn("func.func @qwen_prefill", mlir_text)
            self.assertIn("func.func @qwen_decode", mlir_text)
            self.assertIn('serving.positional_encoding = "rope"', mlir_text)
            self.assertIn("serving.layer_index = 0", mlir_text)
            self.assertIn("serving.layer_index = 1", mlir_text)
            self.assertIn('"llm.q_proj"', mlir_text)
            self.assertIn('"llm.mlp"', mlir_text)


if __name__ == "__main__":
    unittest.main()
