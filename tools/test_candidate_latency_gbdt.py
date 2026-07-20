#!/usr/bin/env python3
"""Validate frozen learned-cost artifacts and candidate semantics."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "configs/cost_model/cortex_a76_fp32_matmul_bias_relu_gbdt_v1"
DATASET = ROOT / "artifacts/cost_model_dataset/cortex_a76_fp32_matmul_bias_relu_v1"
REGISTRY = ROOT / "configs/candidate_registry/cortex_a76_fp32_matmul_bias_relu_v1.json"


def load(path):
    return json.loads(path.read_text())


def main():
    expected_hash = (MODEL / "dataset_hash.txt").read_text().strip()
    assert expected_hash == (DATASET / "dataset_hash.txt").read_text().strip()
    assert expected_hash == (
        "bdb2a835c752426c57ec7499b992f8f7370d5b04886f3dce062348f19e51f3bc")

    schema = load(MODEL / "feature_schema.json")
    order = schema["feature_order"]
    forbidden = {
        "shape_group_id", "split", "benchmark_commit", "binary_hash",
        "object_hash", "checksum", "median_ns", "p95_ns",
        "correctness_pass", "actual_fmla_count", "object_text_bytes",
    }
    assert not forbidden.intersection(order)
    for feature in (
        "tiling_kind==tiled", "vectorization_kind==tiled_vector",
        "vectorized_dimension==multiple", "padding_policy==tile_materialized",
        "requires_full_m_tile", "requires_full_n_tile",
        "requires_full_k_tile", "m_remainder", "n_remainder", "k_remainder",
        "temporary_bytes", "zero_fill_bytes", "copy_bytes",
        "direct_vector_ops",
    ):
        assert feature in order, feature
    assert len(order) == len(set(order)) == 98

    registry = {item["candidate_id"]: item
                for item in load(REGISTRY)["candidates"]}
    direct = registry["tiled_vector_direct_k"]
    assert direct["tiling_kind"] == "tiled"
    assert direct["vectorization_kind"] == "tiled_vector"
    assert direct["padding_policy"] == "none"
    assert direct["requires_full_m_tile"] is True
    assert direct["requires_full_n_tile"] is True
    assert direct["requires_full_k_tile"] is False
    assert direct["k_tail_strategy"] == "direct_vector_cleanup"
    full = registry["tiled_vector_full_tiles"]
    assert all(full[f"requires_full_{d}_tile"] for d in "mnk")
    materialized = registry["tiled_vector_materialized_tail"]
    assert materialized["padding_policy"] == "tile_materialized"
    assert not any(materialized[f"requires_full_{d}_tile"] for d in "mnk")
    assert registry["whole_shape_vector_no_padding"]["padding_policy"] == "none"
    assert (registry["whole_shape_vector_materialized_padding"]
            ["padding_policy"] == "whole_shape_materialized")
    for candidate_id in ("tiled_vector_direct_scalar",
                         "tiled_vector_specialized_tail",
                         "tiled_vector_masked_transfer", "unfused_vector"):
        candidate = registry[candidate_id]
        assert not candidate["lowering_complete"]
        assert not candidate["native_executable"]
        assert candidate["unsupported_reason"]

    metadata = load(MODEL / "metadata.json")
    assert metadata["expected_feature_count"] == 98
    assert metadata["dataset_hash"] == expected_hash
    assert set(metadata["unsupported_candidate_kinds"]) == {
        "unfused_vector", "direct_scalar_cleanup",
        "specialized_microkernel", "masked_transfer",
    }
    model_hash = hashlib.sha256(
        (MODEL / "model.json").read_bytes()
        + (MODEL / "feature_schema.json").read_bytes()
        + (MODEL / "generated_model.h").read_bytes()).hexdigest()
    assert model_hash == (MODEL / "model_hash.txt").read_text().strip()
    print("candidate-latency GBDT artifacts: PASS")


if __name__ == "__main__":
    main()
