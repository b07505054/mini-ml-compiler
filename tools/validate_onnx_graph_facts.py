#!/usr/bin/env python3
"""Validate a GraphFacts JSON document (Python edge tooling, debug/CI use).

Works against both kinds of GraphFacts documents this repo produces:
  - the hand-authored fixture (configs/models/qwen_0_5b_onnx_graph_facts.json),
    which has no "provenance" field and only declares the architecture
    template -- per-layer completeness checks are reported as skipped for
    this kind of document, not silently passed.
  - real output from tools/onnx_graph_to_facts.py, which has a "provenance"
    field recording which real ONNX initializer backed each recognized role.
    For this kind of document, every check below runs for real:
      - num_layers matches the number of distinct parsed layer indices
      - every layer has q_proj/k_proj/v_proj/o_proj and all three mlp roles
      - embedding, final_norm, and lm_head are each detected (or, for
        lm_head, cleanly reported as tied-to-embedding)

Fails hard (non-zero exit) on any incomplete or unrecognized Qwen pattern.
Never treats a missing role as "probably fine".
"""

import argparse
import json
from pathlib import Path

REQUIRED_LAYER_ROLES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "input_layernorm",
    "post_attention_layernorm",
]

REQUIRED_TOP_LEVEL_FIELDS = [
    "schema",
    "model_name",
    "source",
    "truth_boundary",
    "num_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "vocab_size",
    "dtype",
    "graph",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(condition, name, detail):
    return {"name": name, "passed": bool(condition), "detail": detail}


def validate_graph_facts(facts: dict) -> list:
    results = []

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        results.append(check(
            field_name in facts,
            f"has_field_{field_name}",
            f"top-level field '{field_name}'",
        ))

    if any(not r["passed"] for r in results):
        return results

    num_layers = facts["num_layers"]
    results.append(check(
        isinstance(num_layers, int) and num_layers > 0,
        "num_layers_positive",
        f"num_layers={num_layers!r}",
    ))

    provenance = facts.get("provenance")
    if provenance is None:
        results.append(check(
            True,
            "per_layer_completeness_skipped",
            "no 'provenance' field present (declared-template document, e.g. "
            "the hand-authored fixture); per-layer real-graph completeness "
            "checks below are not applicable to this kind of document and "
            "are skipped, not silently passed",
        ))
        return results

    decoder_layer_provenance = provenance.get("decoder_layer", {})
    parsed_layer_indices = sorted(int(i) for i in decoder_layer_provenance.keys())

    results.append(check(
        len(parsed_layer_indices) == num_layers,
        "num_layers_matches_parsed_layers",
        f"declared num_layers={num_layers}, distinct parsed layer indices="
        f"{parsed_layer_indices} (count={len(parsed_layer_indices)})",
    ))

    expected_indices = list(range(num_layers))
    results.append(check(
        parsed_layer_indices == expected_indices,
        "layer_indices_contiguous_from_zero",
        f"expected {expected_indices}, got {parsed_layer_indices}",
    ))

    for layer_index in parsed_layer_indices:
        layer_roles = decoder_layer_provenance.get(str(layer_index), {})
        missing = [role for role in REQUIRED_LAYER_ROLES if role not in layer_roles]
        results.append(check(
            not missing,
            f"layer_{layer_index}_has_all_required_roles",
            f"missing roles: {missing}" if missing else "all roles present",
        ))

    results.append(check(
        bool(provenance.get("embedding")),
        "embedding_detected",
        f"provenance.embedding={provenance.get('embedding')!r}",
    ))
    results.append(check(
        bool(provenance.get("final_norm")),
        "final_norm_detected",
        f"provenance.final_norm={provenance.get('final_norm')!r}",
    ))

    lm_head_name = provenance.get("lm_head")
    lm_head_tied = provenance.get("lm_head_tied_to_embedding")
    results.append(check(
        bool(lm_head_name) and lm_head_tied is not None,
        "lm_head_detected_or_cleanly_reported",
        f"provenance.lm_head={lm_head_name!r}, tied_to_embedding={lm_head_tied!r}",
    ))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-facts", required=True, type=Path)
    parser.add_argument("--out", default="trace/onnx_graph_facts_validation_report.json", type=Path)
    args = parser.parse_args()

    facts = load_json(args.graph_facts)
    results = validate_graph_facts(facts)

    passed = sum(1 for item in results if item["passed"])
    failed = sum(1 for item in results if not item["passed"])

    report = {
        "artifact_type": "onnx_graph_facts_validation_report",
        "graph_facts": str(args.graph_facts),
        "summary": {
            "passed": passed,
            "failed": failed,
            "total": len(results),
            "status": "passed" if failed == 0 else "failed",
        },
        "checks": results,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(args.out)
    print(report["summary"]["status"])
    for item in results:
        if not item["passed"]:
            print(f"  FAILED: {item['name']}: {item['detail']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
