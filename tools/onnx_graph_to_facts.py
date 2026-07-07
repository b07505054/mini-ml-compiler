#!/usr/bin/env python3
"""ONNX protobuf -> GraphFacts JSON frontend adapter (Python edge tooling).

This is Python/onnx edge tooling, not compiler core, per this repo's
"Compiler Core Policy: Zero Python / Zero JSON" (see CLAUDE.md). It is the
frontend adapter seam described in the Phase 2 architecture:

    HF / external model source
      -> ONNX protobuf
      -> Python frontend adapter (this script)
      -> GraphFacts JSON
      -> qwen-onnx-to-serving-mlir (C++, unchanged)
      -> Serving MLIR -> existing compiler passes -> ExecutionPlan

Truth boundary -- be precise about what this does and does not do:

  IMPLEMENTED here:
    - Loads a real .onnx file with the `onnx` Python package and reads real
      protobuf structure: node op types, initializer names, initializer
      shapes/dtypes (metadata only -- tensor values are never materialized).
    - Classifies per-layer roles (q_proj/k_proj/v_proj/o_proj, mlp
      gate/up/down, input/post-attention layernorm) by matching real
      initializer names against Qwen2's HuggingFace parameter-naming
      convention (e.g. "model.layers.{i}.self_attn.q_proj.weight").
    - Derives num_layers, hidden_size, intermediate_size, vocab_size, and
      dtype from real initializer shapes/dtypes.
    - Detects RoPE presence (name/op-type signal) and lm_head/embedding
      weight tying, both from real graph structure.
    - Fails hard (non-zero exit, no output written) when a layer is missing
      an expected role, rather than silently guessing or omitting it.

  NOT IMPLEMENTED here (explicitly, do not assume otherwise):
    - This is NOT a general ONNX importer. It only recognizes Qwen2's
      specific HuggingFace-exported decoder-only architecture (RMSNorm,
      GQA, SwiGLU MLP, separate q/k/v/o Linear layers). Any other model
      family, or a Qwen graph whose parameter names have been renamed or
      fused by an ONNX graph-optimization pass, is out of scope and will
      fail the role-classification checks below.
    - The per-layer *operator sequence* (rmsnorm -> q/k/v_proj ->
      attention_scores -> softmax -> attention_output -> kv_cache_boundary
      -> o_proj -> rmsnorm -> mlp) is a declared Qwen2 architecture
      template, not derived by tracing computational edges through the raw
      ONNX graph. What IS derived from the real graph is: how many layers
      exist, whether each layer's expected weight-bearing roles are
      present, and the model's real dimensions/dtype/RoPE-ness. Recovering
      the op sequence itself from raw graph edges (e.g. proving there is a
      MatMul(Q, Kt) -> Softmax -> MatMul(., V) subgraph) is real future
      engineering, not attempted here.
    - No decode-with-past graph handling (past_key_values inputs/outputs,
      use_cache_branch control flow). This adapter targets a single
      (prefill-shaped) ONNX graph; see docs/future_work.md.
    - No numeric weight-value loading or verification. Only shapes, dtypes,
      and initializer names are read -- never tensor contents. Nothing
      here proves a matched subgraph numerically computes what its role
      name implies.

  truth_boundary emitted in the output JSON:
    "onnx_protobuf_parsed_pattern_matched_not_general_graph_interpreter"

Scalar architecture facts that are not recoverable from ONNX graph
structure alone (num_attention_heads, num_key_value_heads,
max_position_embeddings) are read from an HF `config.json` sitting next to
the .onnx file (the layout HF Optimum's own export already produces), or
from explicit CLI overrides for cases with no config.json (e.g. the tiny
synthetic test fixtures in tests/). This script never guesses these values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
PROBE_REPORT_JSON = ROOT / "trace" / "onnx_graph_to_facts_probe.json"

SCHEMA = "qwen_onnx_graph_facts"
SCHEMA_VERSION = "0.2"
TRUTH_BOUNDARY = "onnx_protobuf_parsed_pattern_matched_not_general_graph_interpreter"

# Fixed Qwen2 per-layer role sequence. This is declared architecture
# knowledge (matching configs/models/qwen_0_5b_onnx_graph_facts.json), not
# derived from raw graph edges -- see module docstring.
EMBEDDING_OPS = ["embed"]
DECODER_LAYER_OPS = [
    "rmsnorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "attention_scores",
    "softmax",
    "attention_output",
    "kv_cache_boundary",
    "o_proj",
    "rmsnorm",
    "mlp",
]
FINAL_NORM_OPS = ["rmsnorm"]
LM_HEAD_OPS = ["lm_head_proj"]

# Per-layer weight-bearing roles this adapter requires to be present in the
# real ONNX graph before it will call a layer "recognized". Values are
# regexes matched against initializer names, with {i} substituted for the
# layer index.
LAYER_ROLE_PATTERNS = {
    "input_layernorm": r"^model\.layers\.{i}\.input_layernorm\.weight$",
    "q_proj": r"^model\.layers\.{i}\.self_attn\.q_proj\.weight$",
    "k_proj": r"^model\.layers\.{i}\.self_attn\.k_proj\.weight$",
    "v_proj": r"^model\.layers\.{i}\.self_attn\.v_proj\.weight$",
    "o_proj": r"^model\.layers\.{i}\.self_attn\.o_proj\.weight$",
    "post_attention_layernorm": r"^model\.layers\.{i}\.post_attention_layernorm\.weight$",
    "gate_proj": r"^model\.layers\.{i}\.mlp\.gate_proj\.weight$",
    "up_proj": r"^model\.layers\.{i}\.mlp\.up_proj\.weight$",
    "down_proj": r"^model\.layers\.{i}\.mlp\.down_proj\.weight$",
}
LAYER_INDEX_RE = re.compile(r"^model\.layers\.(\d+)\.")
EMBED_TOKENS_NAME = "model.embed_tokens.weight"
FINAL_NORM_NAME = "model.norm.weight"
LM_HEAD_NAME = "lm_head.weight"
ROPE_NAME_RE = re.compile(r"rotary|rope|inv_freq", re.IGNORECASE)
ROPE_OP_TYPES = {"Cos", "Sin"}

ONNX_DTYPE_MAP_NAMES = {
    "FLOAT": "fp32",
    "FLOAT16": "fp16",
    "BFLOAT16": "bf16",
}


class OnnxGraphToFactsError(Exception):
    """Raised when the real ONNX graph does not match the expected Qwen2
    pattern. Callers must treat this as a hard failure, never a silent
    fallback to a guessed value."""


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def write_probe_report(payload: dict) -> None:
    PROBE_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROBE_REPORT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class LayerProvenance:
    roles: dict = field(default_factory=dict)  # role -> initializer name

    def missing_roles(self) -> list:
        return [role for role in LAYER_ROLE_PATTERNS if role not in self.roles]


@dataclass
class GraphExtraction:
    layers: dict  # layer_index -> LayerProvenance
    embed_tokens_name: Optional[str]
    final_norm_name: Optional[str]
    lm_head_name: Optional[str]
    lm_head_tied: bool
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    dtype: str
    positional_encoding: Optional[str]


def _dtype_from_onnx_enum(data_type: int, onnx_mod) -> str:
    for enum_name, dtype_str in ONNX_DTYPE_MAP_NAMES.items():
        if data_type == getattr(onnx_mod.TensorProto, enum_name):
            return dtype_str
    raise OnnxGraphToFactsError(
        f"unsupported initializer dtype (onnx TensorProto.data_type={data_type}); "
        "this adapter only recognizes FLOAT, FLOAT16, BFLOAT16 weights"
    )


def _classify_layers(initializer_names: set) -> dict:
    """Group initializer names by declared model.layers.{i} index and
    classify each against LAYER_ROLE_PATTERNS. Returns {layer_index:
    LayerProvenance}. Does not fail here -- missing-role reporting is the
    caller's job so validate_onnx_graph_facts.py can also run this check
    independently against emitted JSON."""
    layer_indices = set()
    for name in initializer_names:
        m = LAYER_INDEX_RE.match(name)
        if m:
            layer_indices.add(int(m.group(1)))

    layers = {i: LayerProvenance() for i in layer_indices}
    for i in layer_indices:
        for role, pattern_template in LAYER_ROLE_PATTERNS.items():
            pattern = re.compile(pattern_template.format(i=i))
            for name in initializer_names:
                if pattern.match(name):
                    layers[i].roles[role] = name
                    break
    return layers


def _detect_lm_head(initializer_names: set) -> tuple:
    if LM_HEAD_NAME in initializer_names:
        return LM_HEAD_NAME, False
    if EMBED_TOKENS_NAME in initializer_names:
        # Qwen2.5-0.5B-Instruct ties lm_head to the input embedding; no
        # separate lm_head.weight initializer exists in that case.
        return EMBED_TOKENS_NAME, True
    return None, False


def _detect_rope(graph, initializer_names: set) -> Optional[str]:
    for name in initializer_names:
        if ROPE_NAME_RE.search(name):
            return "rope"
    for inp in graph.input:
        if ROPE_NAME_RE.search(inp.name):
            return "rope"
    for node in graph.node:
        if node.op_type in ROPE_OP_TYPES:
            return "rope"
    return None


def extract_graph_facts(onnx_path: Path, onnx_mod) -> GraphExtraction:
    model = onnx_mod.load(str(onnx_path))
    graph = model.graph

    initializer_by_name = {init.name: init for init in graph.initializer}
    initializer_names = set(initializer_by_name.keys())

    layers = _classify_layers(initializer_names)
    if not layers:
        raise OnnxGraphToFactsError(
            "no 'model.layers.{i}.*' initializers found; this adapter only "
            "recognizes Qwen2's HuggingFace decoder-layer naming convention"
        )

    max_index = max(layers.keys())
    expected_indices = set(range(max_index + 1))
    missing_indices = expected_indices - set(layers.keys())
    if missing_indices:
        raise OnnxGraphToFactsError(
            f"decoder layer indices are not contiguous from 0..{max_index}; "
            f"missing layer index(es): {sorted(missing_indices)}"
        )

    incomplete = {i: layers[i].missing_roles() for i in sorted(layers)}
    incomplete = {i: roles for i, roles in incomplete.items() if roles}
    if incomplete:
        detail = "; ".join(f"layer {i} missing {roles}" for i, roles in incomplete.items())
        raise OnnxGraphToFactsError(
            f"incomplete Qwen2 decoder layer(s), refusing to guess: {detail}"
        )

    if EMBED_TOKENS_NAME not in initializer_names:
        raise OnnxGraphToFactsError(
            f"embedding initializer '{EMBED_TOKENS_NAME}' not found"
        )
    if FINAL_NORM_NAME not in initializer_names:
        raise OnnxGraphToFactsError(
            f"final norm initializer '{FINAL_NORM_NAME}' not found"
        )

    lm_head_name, lm_head_tied = _detect_lm_head(initializer_names)
    if lm_head_name is None:
        raise OnnxGraphToFactsError(
            f"neither '{LM_HEAD_NAME}' nor a tied embedding "
            f"('{EMBED_TOKENS_NAME}') was found for the lm_head role"
        )

    embed_init = initializer_by_name[EMBED_TOKENS_NAME]
    if len(embed_init.dims) != 2:
        raise OnnxGraphToFactsError(
            f"'{EMBED_TOKENS_NAME}' expected a rank-2 tensor, got dims={list(embed_init.dims)}"
        )
    vocab_size, hidden_size = int(embed_init.dims[0]), int(embed_init.dims[1])

    layer0_gate_name = layers[0].roles.get("gate_proj")
    if layer0_gate_name is None:
        raise OnnxGraphToFactsError(
            "layer 0 is missing 'gate_proj'; cannot derive intermediate_size"
        )
    gate_init = initializer_by_name[layer0_gate_name]
    if len(gate_init.dims) != 2:
        raise OnnxGraphToFactsError(
            f"'{layer0_gate_name}' expected a rank-2 tensor, got dims={list(gate_init.dims)}"
        )
    intermediate_size = int(gate_init.dims[0])

    dtype = _dtype_from_onnx_enum(embed_init.data_type, onnx_mod)

    positional_encoding = _detect_rope(graph, initializer_names)

    return GraphExtraction(
        layers=layers,
        embed_tokens_name=EMBED_TOKENS_NAME,
        final_norm_name=FINAL_NORM_NAME,
        lm_head_name=lm_head_name,
        lm_head_tied=lm_head_tied,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        positional_encoding=positional_encoding,
    )


def _load_hf_config(onnx_path: Path, explicit_config: Optional[Path]) -> dict:
    config_path = explicit_config or (onnx_path.parent / "config.json")
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_graph_facts_json(
    onnx_path: Path,
    model_name: str,
    extraction: GraphExtraction,
    num_attention_heads: int,
    num_key_value_heads: int,
    max_position_embeddings: int,
) -> dict:
    num_layers = len(extraction.layers)

    provenance = {
        "source_onnx_file": onnx_path.name,
        "embedding": extraction.embed_tokens_name,
        "final_norm": extraction.final_norm_name,
        "lm_head": extraction.lm_head_name,
        "lm_head_tied_to_embedding": extraction.lm_head_tied,
        "decoder_layer": {
            str(i): dict(extraction.layers[i].roles)
            for i in sorted(extraction.layers)
        },
    }

    notes = [
        "This file was generated by tools/onnx_graph_to_facts.py from a real",
        f"'{onnx_path.name}' ONNX protobuf file, NOT hand-authored.",
        "Per-layer role PRESENCE (q/k/v/o_proj, mlp gate/up/down, both",
        "layernorms) was verified against real initializer names for every",
        "detected layer. The per-layer OPERATOR SEQUENCE below is a declared",
        "Qwen2 architecture template (matching this repo's existing",
        "hand-authored fixture), not derived by tracing computational edges",
        "in the raw ONNX graph -- see this script's module docstring.",
        "num_attention_heads/num_key_value_heads/max_position_embeddings are",
        "not recoverable from ONNX graph structure alone; they were read from",
        "an HF config.json or an explicit CLI override, not derived here.",
        "'kv_cache_boundary' is a role marker the importer maps to",
        "llm.kv_cache_write when emitting @qwen_prefill and",
        "llm.kv_cache_read when emitting @qwen_decode.",
        "No decode-with-past graph handling: this adapter targets a single",
        "(prefill-shaped) ONNX graph. See docs/future_work.md.",
        "No tensor values were loaded; only shapes, dtypes, and initializer",
        "names were read.",
    ]
    if extraction.positional_encoding:
        notes.append(
            "RoPE was detected (name/op-type signal) and is recorded as the "
            "top-level 'positional_encoding' field. It is absorbed during "
            "pattern recognition, not modeled as a distinct graph op -- see "
            "the Phase 2 design decision in docs/future_work.md."
        )
    if extraction.lm_head_tied:
        notes.append(
            "lm_head is tied to the embedding weight (no separate "
            "lm_head.weight initializer found in this graph)."
        )

    facts = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "model_name": model_name,
        "source": "real_onnx_protobuf_parsed_pattern_matched",
        "truth_boundary": TRUTH_BOUNDARY,
        "num_layers": num_layers,
        "hidden_size": extraction.hidden_size,
        "intermediate_size": extraction.intermediate_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "max_position_embeddings": max_position_embeddings,
        "vocab_size": extraction.vocab_size,
        "dtype": extraction.dtype,
        "graph": {
            "embedding": EMBEDDING_OPS,
            "decoder_layer": DECODER_LAYER_OPS,
            "final_norm": FINAL_NORM_OPS,
            "lm_head": LM_HEAD_OPS,
        },
        "provenance": provenance,
        "notes": notes,
    }
    if extraction.positional_encoding:
        facts["positional_encoding"] = extraction.positional_encoding
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="Path to a real .onnx file")
    parser.add_argument("--out", required=True, type=Path, help="Output GraphFacts JSON path")
    parser.add_argument("--model-name", default="qwen2.5-0.5b")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to an HF config.json (default: config.json next to --onnx)",
    )
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=None)
    args = parser.parse_args()

    if not module_available("onnx"):
        payload = {
            "artifact_type": "onnx_graph_to_facts_probe",
            "source": "tools/onnx_graph_to_facts.py",
            "status": "skipped_missing_toolchain",
            "reason": "Install the 'onnx' package (e.g. into .venv) to run this adapter.",
            "not_claimed": [
                "no ONNX file was parsed",
                f"'{args.out}' was not written",
            ],
        }
        write_probe_report(payload)
        print(payload["status"])
        return 0

    import onnx  # noqa: PLC0415 -- imported lazily, only once availability is confirmed

    if not args.onnx.exists():
        print(f"error: --onnx path does not exist: {args.onnx}", file=sys.stderr)
        return 1

    hf_config = _load_hf_config(args.onnx, args.config)

    def resolve_scalar(cli_value, config_key: str, label: str) -> int:
        if cli_value is not None:
            return cli_value
        if config_key in hf_config:
            return int(hf_config[config_key])
        raise OnnxGraphToFactsError(
            f"{label} is not derivable from ONNX graph structure and was not "
            f"found in config.json (key '{config_key}') or given via CLI override"
        )

    try:
        extraction = extract_graph_facts(args.onnx, onnx)
        num_attention_heads = resolve_scalar(
            args.num_attention_heads, "num_attention_heads", "num_attention_heads"
        )
        num_key_value_heads = resolve_scalar(
            args.num_key_value_heads, "num_key_value_heads", "num_key_value_heads"
        )
        max_position_embeddings = resolve_scalar(
            args.max_position_embeddings,
            "max_position_embeddings",
            "max_position_embeddings",
        )
    except OnnxGraphToFactsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    facts = build_graph_facts_json(
        args.onnx,
        args.model_name,
        extraction,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(f"onnx_graph_to_facts: wrote {args.out}")
    print(f"  num_layers: {facts['num_layers']}")
    print(f"  hidden_size: {facts['hidden_size']}")
    print(f"  dtype: {facts['dtype']}")
    print(f"  positional_encoding: {facts.get('positional_encoding', 'none_detected')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
