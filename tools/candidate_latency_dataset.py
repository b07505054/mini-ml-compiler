#!/usr/bin/env python3
"""Build and validate the production MatMul-Bias-ReLU latency dataset.

The input JSONL is an append-only stream of native benchmark observations.
This tool never invents latency for missing, illegal, failed, or
correctness-invalid candidates.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA_VERSION = "matmul_bias_relu_latency_row_v1"
TARGET = "raspberry-pi5-cortex-a76"
DTYPE = "f32"
TILE = 8

IDENTITY = [
    "dataset_version", "benchmark_commit", "target_id", "target_cpu",
    "target_features", "dtype", "operator_kind", "shape_group_id",
    "candidate_id", "candidate_kind", "split",
]
SHAPE = [
    "M", "N", "K", "output_elements", "reduction_elements", "total_flops",
    "input_bytes", "output_bytes", "arithmetic_intensity",
]
FUSION = [
    "fused", "bias_enabled", "relu_enabled", "intermediate_tensor_count",
    "estimated_intermediate_bytes", "avoided_intermediate_bytes",
]
SCHEDULE = [
    "schedule_kind", "scalar", "vectorized", "whole_shape_vectorized",
    "tiled", "tile_m", "tile_n", "tile_k", "vector_width",
    "full_tile_count",
]
PADDING = [
    "padding_policy", "padded_m", "padded_n", "padded_k",
    "padded_elements", "padded_flops", "padded_flop_ratio",
    "temporary_bytes", "zero_fill_bytes", "copy_bytes",
]
TAIL = [
    "m_remainder", "n_remainder", "k_remainder", "m_tail_strategy",
    "n_tail_strategy", "k_tail_strategy", "m_tail_invocations",
    "n_tail_invocations", "k_tail_invocations", "direct_vector_ops",
    "masked_lane_waste",
]
PLANNING = [
    "estimated_llvm_ir_bytes", "estimated_object_text_bytes",
    "estimated_static_instruction_count", "branch_cost_feature",
    "register_pressure_estimate", "spill_risk", "lowering_stage_count",
]
LEGALITY = [
    "target_legal", "lowering_complete", "native_executable",
    "legality_reason", "unsupported_reason",
]
MEASUREMENT = [
    "execution_status", "correctness_pass", "sentinel_pass",
    "sanitizer_pass", "label_valid", "warmup_calls", "measured_calls",
    "median_ns", "p95_ns", "mad_ns", "min_ns", "max_ns", "checksum",
    "max_absolute_error", "max_relative_error", "cpu_affinity", "governor",
    "current_frequency_khz", "compiler_flags", "compiler_version",
    "mlir_version", "binary_hash", "object_hash", "actual_llvm_mlir_bytes",
    "actual_llvm_ir_bytes", "actual_object_text_bytes",
    "actual_static_instruction_count", "actual_fmla_count",
    "actual_branch_count", "actual_stack_frame_bytes", "actual_spills",
    "compile_time_ms", "compiler_peak_rss_kib", "log_median_ns",
]
FIELDS = IDENTITY + SHAPE + FUSION + SCHEDULE + PADDING + TAIL + PLANNING + LEGALITY + MEASUREMENT
ANALYSIS_ONLY = {
    "actual_llvm_mlir_bytes", "actual_llvm_ir_bytes",
    "actual_object_text_bytes", "actual_static_instruction_count",
    "actual_fmla_count", "actual_branch_count", "actual_stack_frame_bytes",
    "actual_spills", "compile_time_ms", "compiler_peak_rss_kib",
}
NON_FEATURES = set(IDENTITY + LEGALITY + MEASUREMENT)


def ceil8(x):
    return (x + 7) // 8 * 8


def shape_id(m, n, k):
    return f"m{m}_n{n}_k{k}_f32"


def shape_class(m, n, k):
    if max(m, n, k) <= 8:
        return "small"
    if m % 8 == n % 8 == k % 8 == 0:
        return "aligned"
    if m % 8 == 0 and n % 8 == 0 and k % 8:
        return "k_tail"
    if max(m, n, k) >= 64 and len({m, n, k}) > 1:
        return "rectangular"
    if max(m, n, k) >= 64:
        return "larger"
    return "odd"


def deterministic_splits(shapes):
    buckets = defaultdict(list)
    for s in shapes:
        buckets[shape_class(*s)].append(s)
    result = {}
    for category, values in sorted(buckets.items()):
        values.sort(key=lambda s: hashlib.sha256(
            f"dataset-v1:{category}:{s}".encode()).hexdigest())
        for i, s in enumerate(values):
            slot = i % 5
            result[shape_id(*s)] = "train" if slot < 3 else (
                "validation" if slot == 3 else "heldout")
    return result


def applicable(candidate, m, n, k):
    cid = candidate["candidate_id"]
    if not (candidate["target_legal"] and candidate["lowering_complete"]
            and candidate["native_executable"]):
        return False, candidate["unsupported_reason"]
    if cid.startswith("whole_shape_vector"):
        padded = candidate["padding_policy"] != "none"
        pm, pn, pk = ((ceil8(m), ceil8(n), ceil8(k))
                      if padded else (m, n, k))
        if 512 + pm * pn * pk * 4 > 16384:
            return False, "estimated_object_text_exceeds_profile_limit"
    if cid == "whole_shape_vector_materialized_padding":
        ok = (m % 8 != 0 or n % 8 != 0 or k % 8 != 0)
        return ok, "" if ok else "shape_requires_no_whole_shape_padding"
    if cid == "tiled_vector_full_tiles":
        ok = m % 8 == 0 and n % 8 == 0 and k % 8 == 0
        return ok, "" if ok else "requires_M_N_K_divisible_by_8"
    if cid == "tiled_vector_materialized_tail":
        ok = m % 8 != 0 or n % 8 != 0 or k % 8 != 0
        return ok, "" if ok else "shape_has_no_tail"
    if cid == "tiled_vector_direct_k":
        ok = m % 8 == 0 and n % 8 == 0 and k % 8 != 0
        return ok, "" if ok else "requires_M_N_divisible_by_8_and_K_tail"
    return True, ""


def planning_row(candidate, shape, split, commit):
    m, n, k = shape
    cid = candidate["candidate_id"]
    pm, pn, pk = m, n, k
    if candidate["padding_policy"] == "whole_shape_materialized":
        pm, pn, pk = ceil8(m), ceil8(n), ceil8(k)
    mt, nt, kt = (m + 7) // 8, (n + 7) // 8, (k + 7) // 8
    mr, nr, kr = m % 8, n % 8, k % 8
    tiled = candidate["schedule_kind"] == "tiled_vector"
    full_tiles = (m // 8) * (n // 8) * (k // 8) if tiled else 0
    m_tail_calls = (1 if mr else 0) * nt * kt if tiled else 0
    n_tail_calls = (1 if nr else 0) * mt * kt if tiled else 0
    k_tail_calls = (1 if kr else 0) * mt * nt if tiled else 0
    materialized = candidate["padding_policy"] != "none"
    padded_elements = pm * pk + pk * pn + pm * pn
    original_elements = m * k + k * n + m * n
    temp = 0
    zero = 0
    copy = 0
    if candidate["padding_policy"] == "whole_shape_materialized":
        temp = 4 * padded_elements
        zero = temp
        copy = 4 * original_elements
    elif candidate["padding_policy"] == "tile_materialized":
        temp = 4 * (TILE * TILE * 3)
        tail_calls = m_tail_calls + n_tail_calls + k_tail_calls
        zero = tail_calls * temp
        copy = tail_calls * 4 * TILE * TILE * 2
    common_bytes = 4 * (m * k + k * n + m * n + m * n)
    intermediate_count = 0 if candidate["fused"] else 2
    intermediate_bytes = intermediate_count * m * n * 4
    flops = 2 * m * n * k + 2 * m * n
    padded_flops = 2 * pm * pn * pk + 2 * pm * pn
    direct_ops = 2 * m * n * kr if cid == "tiled_vector_direct_k" else 0
    vector_elements = pm * pn * pk if candidate["vectorized"] else 0
    legal, reason = applicable(candidate, m, n, k)
    row = {f: None for f in FIELDS}
    row.update({
        "dataset_version": SCHEMA_VERSION, "benchmark_commit": commit,
        "target_id": TARGET, "target_cpu": "cortex-a76",
        "target_features": "+neon,+fp-armv8", "dtype": DTYPE,
        "operator_kind": "matmul_bias_relu",
        "shape_group_id": shape_id(m, n, k), "candidate_id": cid,
        "candidate_kind": candidate["candidate_kind"], "split": split,
        "M": m, "N": n, "K": k, "output_elements": m * n,
        "reduction_elements": k, "total_flops": flops,
        "input_bytes": 4 * (m * k + k * n + m * n),
        "output_bytes": 4 * m * n,
        "arithmetic_intensity": flops / max(common_bytes + intermediate_bytes, 1),
        "fused": candidate["fused"], "bias_enabled": True,
        "relu_enabled": True, "intermediate_tensor_count": intermediate_count,
        "estimated_intermediate_bytes": intermediate_bytes,
        "avoided_intermediate_bytes": 2 * m * n * 4 if candidate["fused"] else 0,
        "schedule_kind": candidate["schedule_kind"],
        "scalar": candidate["schedule_kind"] == "scalar",
        "vectorized": candidate["vectorized"],
        "whole_shape_vectorized": candidate["schedule_kind"] == "whole_shape_vector",
        "tiled": tiled, "tile_m": 8 if tiled else 0,
        "tile_n": 8 if tiled else 0, "tile_k": 8 if tiled else 0,
        "vector_width": 4 if candidate["vectorized"] else 1,
        "full_tile_count": full_tiles, "padding_policy": candidate["padding_policy"],
        "padded_m": pm, "padded_n": pn, "padded_k": pk,
        "padded_elements": padded_elements - original_elements if materialized else 0,
        "padded_flops": padded_flops if materialized else flops,
        "padded_flop_ratio": padded_flops / flops if materialized else 1.0,
        "temporary_bytes": temp, "zero_fill_bytes": zero, "copy_bytes": copy,
        "m_remainder": mr, "n_remainder": nr, "k_remainder": kr,
        "m_tail_strategy": candidate["m_tail_strategy"],
        "n_tail_strategy": candidate["n_tail_strategy"],
        "k_tail_strategy": candidate["k_tail_strategy"],
        "m_tail_invocations": m_tail_calls, "n_tail_invocations": n_tail_calls,
        "k_tail_invocations": k_tail_calls, "direct_vector_ops": direct_ops,
        "masked_lane_waste": 0,
        "estimated_llvm_ir_bytes": 512 + vector_elements * 2,
        "estimated_object_text_bytes": 512 + vector_elements * 4,
        "estimated_static_instruction_count": 64 + vector_elements // 4,
        "branch_cost_feature": mt * nt * kt if tiled else 1,
        "register_pressure_estimate": min(3.0, (64 if tiled else max(m*n, 1)) / 64),
        "spill_risk": bool(candidate["schedule_kind"] == "whole_shape_vector"
                           and m * n > 256),
        "lowering_stage_count": 10 if candidate["vectorized"] else 8,
        "target_legal": candidate["target_legal"],
        "lowering_complete": candidate["lowering_complete"],
        "native_executable": candidate["native_executable"],
        "legality_reason": "" if legal else reason,
        "unsupported_reason": candidate["unsupported_reason"],
        "execution_status": "not_benchmarked", "correctness_pass": False,
        "sentinel_pass": False, "sanitizer_pass": None, "label_valid": False,
    })
    return row, legal


def load_raw(path):
    observations = {}
    if not path or not Path(path).exists():
        return observations
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        key = (value["shape_group_id"], value["candidate_id"])
        if key in observations:
            raise ValueError(f"duplicate raw observation {key} at line {number}")
        observations[key] = value
    return observations


def validate(rows, registry):
    errors = []
    keys = set()
    valid_ids = {c["candidate_id"] for c in registry["candidates"]}
    split_by_shape = {}
    for row in rows:
        key = (row["shape_group_id"], row["candidate_id"])
        if key in keys:
            errors.append(f"duplicate row key: {key}")
        keys.add(key)
        if row["candidate_id"] not in valid_ids:
            errors.append(f"candidate absent from registry: {key}")
        old = split_by_shape.setdefault(row["shape_group_id"], row["split"])
        if old != row["split"]:
            errors.append(f"shape leakage: {row['shape_group_id']}")
        if row["label_valid"]:
            if row["execution_status"] != "success" or not row["correctness_pass"]:
                errors.append(f"invalid correctness gate: {key}")
            if not row["sentinel_pass"]:
                errors.append(f"invalid sentinel gate: {key}")
            if not row["binary_hash"] or not row["object_hash"]:
                errors.append(f"missing binary/object hash: {key}")
            if not row["median_ns"] or row["median_ns"] <= 0:
                errors.append(f"non-positive label: {key}")
            if row["p95_ns"] + 1e-9 < row["median_ns"]:
                errors.append(f"median exceeds p95: {key}")
        if (not row["target_legal"] or not row["lowering_complete"]
                or not row["native_executable"]) and row["median_ns"] is not None:
            errors.append(f"unsupported row has latency: {key}")
        if row["candidate_id"] == "tiled_vector_direct_k":
            if row["M"] % 8 or row["N"] % 8 or row["K"] % 8 == 0:
                errors.append(f"direct K row outside legal domain: {key}")
    if errors:
        raise ValueError("\n".join(errors))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--raw-measurements")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--benchmark-commit", required=True)
    args = ap.parse_args()
    registry = json.loads(Path(args.registry).read_text())
    shape_config = json.loads(Path(args.shapes).read_text())
    if shape_config.get("shape_order") != "M,K,N":
        raise SystemExit("shape matrix must explicitly declare shape_order=M,K,N")
    shapes = [(v[0], v[2], v[1]) for v in shape_config["shapes"]]
    if len(set(shapes)) != len(shapes):
        raise SystemExit("duplicate shape in matrix")
    splits = deterministic_splits(shapes)
    raw = load_raw(args.raw_measurements)
    rows, coverage, failures = [], [], []
    for shape in sorted(shapes):
        sid = shape_id(*shape)
        for candidate in registry["candidates"]:
            row, legal = planning_row(candidate, shape, splits[sid],
                                      args.benchmark_commit)
            key = (sid, candidate["candidate_id"])
            status = "illegal"
            if not candidate["lowering_complete"]:
                status = "lowering_incomplete"
            elif legal and candidate["native_executable"]:
                status = "missing_measurement"
                if key in raw:
                    observation = raw[key]
                    for field in MEASUREMENT:
                        if field in observation:
                            row[field] = observation[field]
                    # The first production collection exposed an analysis-only
                    # assembly parser bug: a zero total instruction count is
                    # impossible for a linked kernel and therefore means
                    # unavailable, not measured zero. Never turn it into a
                    # training feature or a false "no FMLA" claim.
                    if row["actual_static_instruction_count"] == 0:
                        row["actual_static_instruction_count"] = None
                        row["actual_fmla_count"] = None
                        row["actual_branch_count"] = None
                    success = row["execution_status"] == "success"
                    row["label_valid"] = bool(
                        success and row["correctness_pass"]
                        and row["sentinel_pass"] and row["median_ns"]
                        and row["median_ns"] > 0)
                    row["log_median_ns"] = (
                        math.log(row["median_ns"]) if row["label_valid"] else None)
                    status = "measured" if row["label_valid"] else (
                        "correctness_failed" if success else row["execution_status"])
                    if not row["label_valid"]:
                        failures.append({
                            "shape_group_id": sid,
                            "candidate_id": candidate["candidate_id"],
                            "execution_status": row["execution_status"],
                            "correctness_pass": row["correctness_pass"],
                        })
                rows.append(row)
            coverage.append({
                "shape_group_id": sid, "candidate_id": candidate["candidate_id"],
                "candidate_kind": candidate["candidate_kind"], "status": status,
            })
    rows.sort(key=lambda r: (r["shape_group_id"], r["candidate_id"]))
    validate(rows, registry)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": SCHEMA_VERSION, "primary_key":
            ["target_id", "dtype", "operator_kind", "shape_group_id", "candidate_id"],
        "fields": [{"name": f, "nullable": f in MEASUREMENT} for f in FIELDS],
        "label": "log_median_ns",
        "label_validity": "execution_status=success AND correctness_pass AND sentinel_pass",
    }
    feature_schema = {
        "schema_version": SCHEMA_VERSION,
        "production_planning_time": [
            f for f in SHAPE + FUSION + SCHEDULE + PADDING + TAIL + PLANNING
            if f not in ANALYSIS_ONLY
        ],
        "analysis_only_post_codegen": sorted(ANALYSIS_ONLY),
        "forbidden_for_production_gbdt": sorted(ANALYSIS_ONLY | NON_FEATURES),
    }
    write_json(out / "candidate_registry.json", registry)
    write_json(out / "dataset_schema.json", schema)
    write_json(out / "feature_schema.json", feature_schema)
    by_split = defaultdict(list)
    for sid, split in sorted(splits.items()):
        by_split[split].append(sid)
    for split in ("train", "validation", "heldout"):
        write_json(out / f"{split}_shapes.json", by_split[split])
    with (out / "measurements.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with (out / "measurements.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "coverage_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "shape_group_id", "candidate_id", "candidate_kind", "status"],
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage)
    write_json(out / "failure_manifest.json", failures)
    valid = [r for r in rows if r["label_valid"]]
    summary = {
        "shape_groups": len(shapes), "rows": len(rows),
        "label_valid_rows": len(valid),
        "executable_candidate_kinds": sorted({
            c["candidate_kind"] for c in registry["candidates"]
            if c["native_executable"] and c["lowering_complete"]}),
        "unsupported_candidate_ids": sorted({
            c["candidate_id"] for c in registry["candidates"]
            if not c["native_executable"]}),
        "candidate_rows": dict(sorted(Counter(
            r["candidate_id"] for r in rows).items())),
        "split_shape_counts": dict(sorted(Counter(splits.values()).items())),
        "split_row_counts": dict(sorted(Counter(r["split"] for r in rows).items())),
        "coverage_status": dict(sorted(Counter(c["status"] for c in coverage).items())),
        "compile_failures": sum(r["execution_status"] == "compile_failed" for r in rows),
        "runtime_failures": sum(r["execution_status"] == "runtime_failed" for r in rows),
        "correctness_failures": sum(
            r["execution_status"] == "success" and not r["correctness_pass"] for r in rows),
        "sanitizer_failures": sum(r["sanitizer_pass"] is False for r in rows),
        "average_valid_candidates_per_shape": len(valid) / len(shapes),
        "latency_ns_range": ([min(r["median_ns"] for r in valid),
                              max(r["median_ns"] for r in valid)] if valid else []),
        "object_text_bytes_range": ([min(r["actual_object_text_bytes"] for r in valid
                                                if r["actual_object_text_bytes"] is not None),
                                     max(r["actual_object_text_bytes"] for r in valid
                                         if r["actual_object_text_bytes"] is not None)]
                                    if any(r["actual_object_text_bytes"] is not None
                                           for r in valid) else []),
        "M_range": [min(s[0] for s in shapes), max(s[0] for s in shapes)],
        "N_range": [min(s[1] for s in shapes), max(s[1] for s in shapes)],
        "K_range": [min(s[2] for s in shapes), max(s[2] for s in shapes)],
    }
    write_json(out / "dataset_summary.json", summary)
    payload = b"".join([
        (out / "measurements.jsonl").read_bytes(),
        (out / "candidate_registry.json").read_bytes(),
        (out / "dataset_schema.json").read_bytes(),
        (out / "feature_schema.json").read_bytes(),
    ])
    (out / "dataset_hash.txt").write_text(hashlib.sha256(payload).hexdigest() + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
