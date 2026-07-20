#!/usr/bin/env python3
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dataset", ROOT / "tools/candidate_latency_dataset.py")
dataset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dataset)


class CandidateDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = json.loads((ROOT / (
            "configs/candidate_registry/"
            "cortex_a76_fp32_matmul_bias_relu_v1.json")).read_text())
        config = json.loads((ROOT / (
            "configs/cost_model_dataset/"
            "cortex_a76_fp32_matmul_bias_relu_v1_shapes.json")).read_text())
        cls.shapes = [(v[0], v[2], v[1]) for v in config["shapes"]]

    def test_registry_truthful_and_stable(self):
        ids = [c["candidate_id"] for c in self.registry["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        unsupported = {
            "tiled_vector_direct_scalar",
            "tiled_vector_specialized_tail",
            "tiled_vector_masked_transfer",
        }
        for candidate in self.registry["candidates"]:
            required = {
                "candidate_id", "candidate_kind", "schedule_kind", "fused",
                "vectorized", "padding_policy", "m_tail_strategy",
                "n_tail_strategy", "k_tail_strategy", "lowering_complete",
                "target_legal", "native_executable", "unsupported_reason",
            }
            self.assertTrue(required.issubset(candidate))
            if candidate["candidate_id"] in unsupported:
                self.assertFalse(candidate["lowering_complete"])
                self.assertFalse(candidate["native_executable"])
                self.assertTrue(candidate["unsupported_reason"])

    def test_direct_k_domain(self):
        direct = next(c for c in self.registry["candidates"]
                      if c["candidate_id"] == "tiled_vector_direct_k")
        self.assertEqual(dataset.applicable(direct, 8, 8, 15), (True, ""))
        self.assertFalse(dataset.applicable(direct, 7, 8, 15)[0])
        self.assertFalse(dataset.applicable(direct, 8, 7, 15)[0])
        self.assertFalse(dataset.applicable(direct, 8, 8, 16)[0])

    def test_shape_matrix_and_group_split(self):
        self.assertGreaterEqual(len(self.shapes), 40)
        self.assertEqual(len(self.shapes), len(set(self.shapes)))
        a = dataset.deterministic_splits(self.shapes)
        b = dataset.deterministic_splits(list(reversed(self.shapes)))
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), {"train", "validation", "heldout"})
        executable_rows = 0
        for shape in self.shapes:
            for candidate in self.registry["candidates"]:
                executable_rows += int(dataset.applicable(
                    candidate, *shape)[0])
        self.assertGreaterEqual(executable_rows, 100)

    def test_schema_vocabulary_and_feature_boundary(self):
        self.assertEqual(len(dataset.FIELDS), len(set(dataset.FIELDS)))
        self.assertIn("log_median_ns", dataset.MEASUREMENT)
        self.assertIn("actual_fmla_count", dataset.ANALYSIS_ONLY)
        self.assertNotIn("actual_fmla_count", dataset.PLANNING)
        self.assertIn("direct_vector_ops", dataset.TAIL)

    def test_correctness_gate_duplicate_and_illegal_label(self):
        candidate = next(c for c in self.registry["candidates"]
                         if c["candidate_id"] == "fused_scalar")
        row, legal = dataset.planning_row(
            candidate, (8, 8, 8), "train", "deadbeef")
        self.assertTrue(legal)
        row.update({
            "execution_status": "success", "correctness_pass": True,
            "sentinel_pass": True, "label_valid": True,
            "median_ns": 10.0, "p95_ns": 11.0,
            "binary_hash": "a" * 64, "object_hash": "b" * 64,
        })
        dataset.validate([row], self.registry)
        with self.assertRaisesRegex(ValueError, "duplicate row"):
            dataset.validate([row, dict(row)], self.registry)
        bad = dict(row)
        bad["correctness_pass"] = False
        with self.assertRaisesRegex(ValueError, "correctness gate"):
            dataset.validate([bad], self.registry)

    def test_deterministic_serialization_hash(self):
        candidate = next(c for c in self.registry["candidates"]
                         if c["candidate_id"] == "fused_scalar")
        rows = [dataset.planning_row(candidate, shape, "train", "deadbeef")[0]
                for shape in self.shapes[:3]]
        one = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
        two = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
        self.assertEqual(one, two)


if __name__ == "__main__":
    unittest.main()
