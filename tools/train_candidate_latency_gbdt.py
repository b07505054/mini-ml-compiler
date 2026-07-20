#!/usr/bin/env python3
"""Train a small, dependency-free candidate-latency GBDT.

Configuration is selected exclusively on the frozen validation shape split.
The held-out split is evaluated only after the configuration is frozen.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 1729
SCHEMA_VERSION = "matmul_bias_relu_gbdt_features_v1"
MODEL_VERSION = "cortex_a76_fp32_matmul_bias_relu_gbdt_v1"
CATEGORICAL = {
    "candidate_kind": ["scalar_baseline", "tiled_vector_direct_cleanup",
                       "tiled_vector_full_tiles", "tiled_vector_materialized_tail",
                       "whole_shape_vector_materialized_padding",
                       "whole_shape_vector_no_padding"],
    "schedule_kind": ["scalar", "tiled_vector", "whole_shape_vector"],
    "tiling_kind": ["none", "whole_shape", "tiled"],
    "vectorization_kind": ["none", "whole_shape_vector", "tiled_vector"],
    "vectorized_dimension": ["none", "m", "n", "k", "multiple"],
    "padding_policy": ["none", "tile_materialized", "whole_shape_materialized"],
    "m_tail_strategy": ["none", "materialized_tail", "direct_scalar_cleanup",
                        "direct_vector_cleanup", "specialized_microkernel",
                        "masked_transfer"],
    "n_tail_strategy": ["none", "materialized_tail", "direct_scalar_cleanup",
                        "direct_vector_cleanup", "specialized_microkernel",
                        "masked_transfer"],
    "k_tail_strategy": ["none", "materialized_tail", "direct_scalar_cleanup",
                        "direct_vector_cleanup", "specialized_microkernel",
                        "masked_transfer"],
}
NUMERIC = [
    "M", "N", "K", "log2_m_plus_1", "log2_n_plus_1", "log2_k_plus_1",
    "output_elements", "reduction_elements", "total_flops", "input_bytes",
    "output_bytes", "arithmetic_intensity", "fused", "scalar", "vectorized",
    "whole_shape_vectorized", "tiled", "tile_m", "tile_n", "tile_k",
    "vector_width", "requires_full_m_tile", "requires_full_n_tile",
    "requires_full_k_tile", "full_tile_count", "m_tile_count", "n_tile_count",
    "k_tile_count", "m_remainder", "n_remainder", "k_remainder",
    "m_remainder_ratio", "n_remainder_ratio", "k_remainder_ratio",
    "m_tail_invocations", "n_tail_invocations", "k_tail_invocations",
    "padded_m", "padded_n", "padded_k", "padded_elements", "padded_flops",
    "padded_flop_ratio", "temporary_bytes", "zero_fill_bytes", "copy_bytes",
    "estimated_intermediate_bytes", "avoided_intermediate_bytes",
    "direct_vector_ops", "masked_lane_waste", "estimated_llvm_ir_bytes",
    "estimated_object_text_bytes", "estimated_static_instruction_count",
    "register_pressure_estimate", "spill_risk", "branch_cost_feature",
    "lowering_stage_count",
]
FEATURE_NAMES = NUMERIC + [
    f"{name}=={value}" for name, values in CATEGORICAL.items() for value in values
]


def features(row):
    derived = dict(row)
    kind = row["candidate_kind"]
    derived["tiling_kind"] = (
        "none" if row["schedule_kind"] == "scalar" else
        "whole_shape" if row["schedule_kind"] == "whole_shape_vector" else
        "tiled")
    derived["vectorization_kind"] = (
        "none" if not row["vectorized"] else
        "whole_shape_vector" if row["whole_shape_vectorized"] else
        "tiled_vector")
    derived["vectorized_dimension"] = "none" if not row["vectorized"] else "multiple"
    full = kind == "tiled_vector_full_tiles"
    direct = kind == "tiled_vector_direct_cleanup"
    derived["requires_full_m_tile"] = full or direct
    derived["requires_full_n_tile"] = full or direct
    derived["requires_full_k_tile"] = full
    if kind == "tiled_vector_materialized_tail":
        for dimension in ("m", "n", "k"):
            derived[f"{dimension}_tail_strategy"] = (
                "materialized_tail" if row[f"{dimension}_remainder"] else "none")
    for key, dim in (("m", row["M"]), ("n", row["N"]), ("k", row["K"])):
        derived[f"log2_{key}_plus_1"] = math.log2(dim + 1)
    for key, dim, tile in (("m", row["M"], row["tile_m"]),
                           ("n", row["N"], row["tile_n"]),
                           ("k", row["K"], row["tile_k"])):
        derived[f"{key}_tile_count"] = ((dim + tile - 1) // tile) if tile else 0
        derived[f"{key}_remainder_ratio"] = (
            row[f"{key}_remainder"] / tile if tile else 0.0)
    values = []
    for name in NUMERIC:
        value = derived.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"missing/nonfinite production feature {name}")
        values.append(float(value))
    for name, vocabulary in CATEGORICAL.items():
        value = derived[name]
        if value not in vocabulary:
            raise ValueError(f"unknown category {name}={value}")
        values.extend(1.0 if value == category else 0.0
                      for category in vocabulary)
    return values


def weights(rows, mode):
    if mode == "unweighted":
        return [1.0] * len(rows)
    counts = Counter(r["shape_group_id"] if mode == "per_shape"
                     else r["candidate_kind"] for r in rows)
    return [1.0 / counts[r["shape_group_id"] if mode == "per_shape"
                         else r["candidate_kind"]] for r in rows]


def weighted_mean(values, ws):
    total = sum(ws)
    return sum(v * w for v, w in zip(values, ws)) / total


def build_tree(xs, ys, ws, indices, depth, max_depth, min_leaf, max_features):
    value = weighted_mean([ys[i] for i in indices], [ws[i] for i in indices])
    node = {"value": value, "feature": -1, "threshold": 0.0,
            "left": None, "right": None}
    if depth >= max_depth or len(indices) < 2 * min_leaf:
        return node
    feature_count = max(1, int(len(FEATURE_NAMES) * max_features))
    # Deterministic column sampling: stable stride keyed by depth/node size.
    start = (depth * 17 + len(indices) * 13 + SEED) % len(FEATURE_NAMES)
    columns = [(start + i * 37) % len(FEATURE_NAMES)
               for i in range(feature_count)]
    best = None
    for feature in columns:
        distinct = sorted({xs[i][feature] for i in indices})
        if len(distinct) < 2:
            continue
        thresholds = [(a + b) / 2 for a, b in zip(distinct, distinct[1:])]
        if len(thresholds) > 32:
            thresholds = [thresholds[int(i * (len(thresholds) - 1) / 31)]
                          for i in range(32)]
        for threshold in thresholds:
            left = [i for i in indices if xs[i][feature] <= threshold]
            right = [i for i in indices if xs[i][feature] > threshold]
            if len(left) < min_leaf or len(right) < min_leaf:
                continue
            lv = weighted_mean([ys[i] for i in left], [ws[i] for i in left])
            rv = weighted_mean([ys[i] for i in right], [ws[i] for i in right])
            loss = (sum(ws[i] * (ys[i] - lv) ** 2 for i in left) +
                    sum(ws[i] * (ys[i] - rv) ** 2 for i in right))
            candidate = (loss, feature, threshold, left, right)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best:
        _, feature, threshold, left, right = best
        node.update({
            "feature": feature, "threshold": threshold,
            "left": build_tree(xs, ys, ws, left, depth + 1, max_depth,
                               min_leaf, max_features),
            "right": build_tree(xs, ys, ws, right, depth + 1, max_depth,
                                min_leaf, max_features),
        })
    return node


def tree_predict(node, x):
    while node["feature"] >= 0:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node["value"]


def train_gbdt(rows, config):
    xs = [features(r) for r in rows]
    ys = [r["log_median_ns"] for r in rows]
    ws = weights(rows, config["weighting"])
    base = weighted_mean(ys, ws)
    predictions = [base] * len(rows)
    trees = []
    for _ in range(config["n_estimators"]):
        residual = [y - p for y, p in zip(ys, predictions)]
        tree = build_tree(xs, residual, ws, list(range(len(rows))), 0,
                          config["max_depth"], config["min_samples_leaf"],
                          config["max_features"])
        trees.append(tree)
        for i, x in enumerate(xs):
            predictions[i] += config["learning_rate"] * tree_predict(tree, x)
    return {"base_score": base, "learning_rate": config["learning_rate"],
            "trees": trees, "config": config}


def predict(model, row):
    return predict_vector(model, features(row))


def predict_vector(model, x):
    return model["base_score"] + model["learning_rate"] * sum(
        tree_predict(tree, x) for tree in model["trees"])


def analytical_log(row):
    throughput = {
        "scalar_baseline": 1.0,
        "whole_shape_vector_no_padding": 14.5,
        "whole_shape_vector_materialized_padding": 14.5,
        "tiled_vector_full_tiles": 11.89,
        "tiled_vector_materialized_tail": 11.89,
        "tiled_vector_direct_cleanup": 11.89,
    }[row["candidate_kind"]]
    ns = row["total_flops"] / throughput
    ns += (row["input_bytes"] + row["output_bytes"] +
           row["estimated_intermediate_bytes"]) / 20.0
    ns += row["zero_fill_bytes"] / 20.0 + row["copy_bytes"] / 12.0
    if row["candidate_kind"] == "tiled_vector_direct_cleanup":
        ns += 53.0 + 7.928571 * row["k_remainder"]
    ns += row["branch_cost_feature"] * 1.5
    return math.log(max(ns, 1.0))


LINEAR_FEATURES = [
    "log2_m_plus_1", "log2_n_plus_1", "log2_k_plus_1",
    "arithmetic_intensity", "padded_flop_ratio", "m_remainder_ratio",
    "n_remainder_ratio", "k_remainder_ratio", "register_pressure_estimate",
]


def linear_vector(row):
    full = features(row)
    indices = [FEATURE_NAMES.index(name) for name in LINEAR_FEATURES]
    indices += [i for i, name in enumerate(FEATURE_NAMES)
                if "==" in name and name.startswith(("candidate_kind", "schedule_kind"))]
    return [1.0] + [full[i] for i in indices]


def solve_linear(matrix, rhs):
    n = len(rhs)
    augmented = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        if abs(scale) < 1e-12:
            continue
        augmented[col] = [v / scale for v in augmented[col]]
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [a - factor * b
                                  for a, b in zip(augmented[row], augmented[col])]
    return [augmented[i][-1] for i in range(n)]


def train_ridge(rows, regularization=1e-3):
    xs = [linear_vector(r) for r in rows]
    ys = [r["log_median_ns"] for r in rows]
    n = len(xs[0])
    matrix = [[sum(x[i] * x[j] for x in xs) for j in range(n)]
              for i in range(n)]
    for i in range(1, n):
        matrix[i][i] += regularization
    rhs = [sum(x[i] * y for x, y in zip(xs, ys)) for i in range(n)]
    return solve_linear(matrix, rhs)


def ridge_predict(coefficients, row):
    return sum(a * b for a, b in zip(coefficients, linear_vector(row)))


def hybrid_predictions(rows, gbdt_predictions, margin):
    result = list(gbdt_predictions)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["shape_group_id"]].append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda i: gbdt_predictions[i])
        if len(ordered) < 2:
            continue
        best, second = ordered[:2]
        relative_gap = (math.exp(gbdt_predictions[second]) /
                        math.exp(gbdt_predictions[best]) - 1)
        if relative_gap < margin:
            for index in indices:
                result[index] = analytical_log(rows[index])
    return result


def regression_metrics(rows, predictions):
    errors = [math.exp(p) - r["median_ns"] for r, p in zip(rows, predictions)]
    abs_errors = [abs(e) for e in errors]
    apes = [abs(e) / r["median_ns"] for e, r in zip(errors, rows)]
    log_errors = [p - r["log_median_ns"] for r, p in zip(rows, predictions)]
    mean_y = statistics.mean(r["log_median_ns"] for r in rows)
    denom = sum((r["log_median_ns"] - mean_y) ** 2 for r in rows)
    bias = defaultdict(list)
    for row, error in zip(rows, errors):
        bias[row["candidate_kind"]].append(error)
    return {
        "mae_ns": statistics.mean(abs_errors),
        "median_absolute_error_ns": statistics.median(abs_errors),
        "median_absolute_percentage_error": statistics.median(apes),
        "p90_absolute_percentage_error": sorted(apes)[int(.9 * (len(apes)-1))],
        "log_rmse": math.sqrt(statistics.mean(e * e for e in log_errors)),
        "r2_log": 1 - sum(e * e for e in log_errors) / denom if denom else 0,
        "bias_ns_by_candidate": {k: statistics.mean(v) for k, v in sorted(bias.items())},
    }


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank
    return result


def selection_metrics(rows, predictions):
    groups = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        groups[row["shape_group_id"]].append((row, prediction))
    details, regrets, exact, top2, spearman = [], [], 0, 0, []
    for sid, values in sorted(groups.items()):
        measured = min(values, key=lambda x: x[0]["median_ns"])[0]
        predicted = min(values, key=lambda x: x[1])[0]
        sorted_pred = [x[0]["candidate_id"] for x in sorted(values, key=lambda x: x[1])]
        regret = predicted["median_ns"] / measured["median_ns"] - 1
        regrets.append(regret)
        exact += predicted["candidate_id"] == measured["candidate_id"]
        top2 += measured["candidate_id"] in sorted_pred[:2]
        if len(values) >= 2:
            a = ranks([x[0]["median_ns"] for x in values])
            b = ranks([x[1] for x in values])
            ma, mb = statistics.mean(a), statistics.mean(b)
            denom = math.sqrt(sum((x-ma)**2 for x in a) * sum((x-mb)**2 for x in b))
            spearman.append(sum((x-ma)*(y-mb) for x, y in zip(a, b)) / denom
                            if denom else 1.0)
        details.append({
            "shape_group_id": sid, "measured_winner": measured["candidate_id"],
            "predicted_winner": predicted["candidate_id"],
            "oracle_ns": measured["median_ns"],
            "predicted_winner_measured_ns": predicted["median_ns"],
            "absolute_regret_ns": predicted["median_ns"] - measured["median_ns"],
            "normalized_regret": regret,
        })
    ordered = sorted(regrets)
    percentile = lambda p: ordered[int(p * (len(ordered)-1))]
    return {
        "shape_count": len(groups), "exact_match_rate": exact / len(groups),
        "top2_recall": top2 / len(groups),
        "mean_normalized_regret": statistics.mean(regrets),
        "median_normalized_regret": statistics.median(regrets),
        "p90_normalized_regret": percentile(.9),
        "p95_normalized_regret": percentile(.95),
        "worst_normalized_regret": max(regrets),
        "mean_spearman": statistics.mean(spearman),
    }, details


def flatten_tree(node, nodes):
    index = len(nodes)
    nodes.append(None)
    if node["feature"] < 0:
        nodes[index] = [-1, 0.0, -1, -1, node["value"]]
    else:
        left = flatten_tree(node["left"], nodes)
        right = flatten_tree(node["right"], nodes)
        nodes[index] = [node["feature"], node["threshold"], left, right, node["value"]]
    return index


def export_header(model, uncertainty_margin, path):
    nodes, roots = [], []
    for tree in model["trees"]:
        roots.append(flatten_tree(tree, nodes))
    def nums(values):
        return ", ".join(str(v) for v in values)
    text = f"""#pragma once
#include <array>
#include <cmath>
#include <cstddef>
namespace mlir::hir::gbdt_v1 {{
inline constexpr const char *kModelVersion = "{MODEL_VERSION}";
inline constexpr const char *kSchemaVersion = "{SCHEMA_VERSION}";
inline constexpr size_t kFeatureCount = {len(FEATURE_NAMES)};
inline constexpr double kUncertaintyMargin = {uncertainty_margin:.17g};
struct Node {{ int feature; double threshold; int left; int right; double value; }};
inline constexpr std::array<Node, {len(nodes)}> kNodes = {{{{
"""
    text += "".join(f"  Node{{{n[0]}, {n[1]:.17g}, {n[2]}, {n[3]}, {n[4]:.17g}}},\n"
                    for n in nodes)
    text += f"""}}}};
inline constexpr std::array<int, {len(roots)}> kRoots = {{{{{nums(roots)}}}}};
inline constexpr double kBaseScore = {model['base_score']:.17g};
inline constexpr double kLearningRate = {model['learning_rate']:.17g};
inline bool evaluate(const std::array<double, kFeatureCount> &features,
                     double &predictedLogNs) {{
  for (double value : features) if (!std::isfinite(value)) return false;
  predictedLogNs = kBaseScore;
  for (int root : kRoots) {{
    int index = root;
    while (kNodes[index].feature >= 0) {{
      const Node &node = kNodes[index];
      index = features[node.feature] <= node.threshold ? node.left : node.right;
    }}
    predictedLogNs += kLearningRate * kNodes[index].value;
  }}
  return std::isfinite(predictedLogNs);
}}
}} // namespace mlir::hir::gbdt_v1
"""
    path.write_text(text)


def export_test_vectors(model, rows, path):
    vectors = [features(row) for row in rows]
    predictions = [predict(model, row) for row in rows]
    flat = [value for vector in vectors for value in vector]
    path.write_text(f"""#pragma once
#include <array>
namespace mlir::hir::gbdt_v1::test {{
inline constexpr size_t kRowCount = {len(rows)};
inline constexpr std::array<double, {len(flat)}> kFeatures = {{{{
  {", ".join(f"{v:.17g}" for v in flat)}
}}}};
inline constexpr std::array<double, {len(predictions)}> kPredictions = {{{{
  {", ".join(f"{v:.17g}" for v in predictions)}
}}}};
}} // namespace mlir::hir::gbdt_v1::test
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    data = Path(args.dataset_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected_hash = (data / "dataset_hash.txt").read_text().strip()
    payload = b"".join((data / n).read_bytes() for n in
                       ["measurements.jsonl", "candidate_registry.json",
                        "dataset_schema.json", "feature_schema.json"])
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise SystemExit("dataset hash mismatch")
    rows = [json.loads(line) for line in (data / "measurements.jsonl").read_text().splitlines()]
    valid = [r for r in rows if r["label_valid"]]
    train = [r for r in valid if r["split"] == "train"]
    validation = [r for r in valid if r["split"] == "validation"]
    heldout = [r for r in valid if r["split"] == "heldout"]
    # Materialize every feature before search; leakage/missing failures stop here.
    for row in valid:
        features(row)
    configs = []
    for weighting in ("unweighted", "per_shape", "candidate_balanced"):
        for trees in (25, 50):
            for depth in (2, 3):
                for rate in (.05, .1):
                    for leaf in (4,):
                        configs.append({
                            "weighting": weighting, "n_estimators": trees,
                            "max_depth": depth, "learning_rate": rate,
                            "min_samples_leaf": leaf, "max_features": .9,
                        })
    leaderboard = []
    best = None
    for config in configs:
        model = train_gbdt(train, config)
        preds = [predict(model, r) for r in validation]
        selection, _ = selection_metrics(validation, preds)
        regression = regression_metrics(validation, preds)
        score = (selection["mean_normalized_regret"],
                 selection["p95_normalized_regret"],
                 selection["worst_normalized_regret"],
                 config["n_estimators"], config["max_depth"],
                 regression["log_rmse"])
        leaderboard.append({"config": config, "selection": selection,
                            "regression": regression, "score": score})
        if best is None or score < best[0]:
            best = (score, config)
    frozen_config = best[1]
    model = train_gbdt(train, frozen_config)
    validation_predictions = [predict(model, r) for r in validation]
    validation_selection, validation_details = selection_metrics(
        validation, validation_predictions)
    validation_metrics = {
        "selection": validation_selection,
        "regression": regression_metrics(validation, validation_predictions),
    }
    # Freeze before touching held-out.
    # Confidence margin is chosen on validation only.
    hybrid_options = []
    for margin in (.05, .10, .15, .20):
        hp = hybrid_predictions(validation, validation_predictions, margin)
        metrics = selection_metrics(validation, hp)[0]
        hybrid_options.append((metrics["mean_normalized_regret"],
                               metrics["worst_normalized_regret"], margin, metrics))
    _, _, selected_margin, hybrid_validation_selection = min(hybrid_options)
    freeze = {
        "model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION,
        "features": FEATURE_NAMES, "categorical_vocabulary": CATEGORICAL,
        "hyperparameters": frozen_config, "fixed_seed": SEED,
        "uncertainty_margin": selected_margin,
        "selection_basis": "validation_selection_regret_then_model_size",
        "dataset_hash": expected_hash,
    }
    freeze_hash = hashlib.sha256(
        json.dumps(freeze, sort_keys=True).encode()).hexdigest()
    heldout_predictions = [predict(model, r) for r in heldout]
    heldout_selection, heldout_details = selection_metrics(
        heldout, heldout_predictions)
    heldout_metrics = {
        "configuration_freeze_hash": freeze_hash,
        "selection": heldout_selection,
        "regression": regression_metrics(heldout, heldout_predictions),
        "heldout_used_for_tuning": False,
    }
    baselines = {}
    for name, prediction_fn in (
        ("candidate_kind_mean", None), ("analytical", analytical_log)):
        if prediction_fn:
            vp = [prediction_fn(r) for r in validation]
            hp = [prediction_fn(r) for r in heldout]
        else:
            means = defaultdict(list)
            for row in train:
                means[row["candidate_kind"]].append(row["log_median_ns"])
            vp = [statistics.mean(means[r["candidate_kind"]]) for r in validation]
            hp = [statistics.mean(means[r["candidate_kind"]]) for r in heldout]
        baselines[name] = {
            "validation": {"selection": selection_metrics(validation, vp)[0],
                           "regression": regression_metrics(validation, vp)},
            "heldout": {"selection": selection_metrics(heldout, hp)[0],
                        "regression": regression_metrics(heldout, hp)},
        }
    # A single depth-3 tree and the GBDT serve as tree/GBDT comparisons.
    tree_model = train_gbdt(train, {
        "weighting": "per_shape", "n_estimators": 1, "max_depth": 3,
        "learning_rate": 1.0, "min_samples_leaf": 4, "max_features": 1.0})
    baselines["single_tree"] = {
        split: {"selection": selection_metrics(rs, [predict(tree_model, r) for r in rs])[0],
                "regression": regression_metrics(rs, [predict(tree_model, r) for r in rs])}
        for split, rs in (("validation", validation), ("heldout", heldout))
    }
    ridge = train_ridge(train)
    baselines["ridge"] = {
        split: {"selection": selection_metrics(rs, [ridge_predict(ridge, r) for r in rs])[0],
                "regression": regression_metrics(rs, [ridge_predict(ridge, r) for r in rs])}
        for split, rs in (("validation", validation), ("heldout", heldout))
    }
    hybrid_heldout_predictions = hybrid_predictions(
        heldout, heldout_predictions, selected_margin)
    baselines["hybrid"] = {
        "validation": {
            "selection": hybrid_validation_selection,
            "regression": regression_metrics(
                validation, hybrid_predictions(
                    validation, validation_predictions, selected_margin))},
        "heldout": {
            "selection": selection_metrics(
                heldout, hybrid_heldout_predictions)[0],
            "regression": regression_metrics(
                heldout, hybrid_heldout_predictions)},
    }
    model_json = {
        "model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES), **model,
    }
    model_text = json.dumps(model_json, sort_keys=True, separators=(",", ":"))
    (out / "model.json").write_text(model_text + "\n")
    export_header(model, selected_margin, out / "generated_model.h")
    export_test_vectors(model, valid, out / "generated_model_test_vectors.h")
    feature_schema = {
        "schema_version": SCHEMA_VERSION, "feature_order": FEATURE_NAMES,
        "numeric_features": NUMERIC, "categorical_vocabulary": CATEGORICAL,
        "availability": {
            "production_planning_time": FEATURE_NAMES,
            "analysis_only_rejected": [
                "actual_fmla_count", "actual_instruction_count",
                "actual_branch_count", "actual_stack_frame_bytes",
                "actual_spill_reload_count", "object_text_bytes",
                "binary_hash", "object_hash", "compile_time_ms",
            ],
        },
        "encoding": "stable_one_hot", "normalization": "none",
        "missing_value_policy": "reject_and_fallback_analytical",
        "unknown_category_policy": "reject_and_fallback_analytical",
    }
    domain = {
        "target": "raspberry-pi5-cortex-a76", "dtype": "f32",
        "operator": "matmul_bias_relu", "tile_family": [8, 8, 8],
        "M": [min(r["M"] for r in train), max(r["M"] for r in train)],
        "N": [min(r["N"] for r in train), max(r["N"] for r in train)],
        "K": [min(r["K"] for r in train), max(r["K"] for r in train)],
        "total_flops": [min(r["total_flops"] for r in train),
                        max(r["total_flops"] for r in train)],
        "m_tile_count": [min((r["M"] + r["tile_m"] - 1) // r["tile_m"]
                             if r["tile_m"] else 0 for r in train),
                         max((r["M"] + r["tile_m"] - 1) // r["tile_m"]
                             if r["tile_m"] else 0 for r in train)],
        "n_tile_count": [min((r["N"] + r["tile_n"] - 1) // r["tile_n"]
                             if r["tile_n"] else 0 for r in train),
                         max((r["N"] + r["tile_n"] - 1) // r["tile_n"]
                             if r["tile_n"] else 0 for r in train)],
        "k_tile_count": [min((r["K"] + r["tile_k"] - 1) // r["tile_k"]
                             if r["tile_k"] else 0 for r in train),
                         max((r["K"] + r["tile_k"] - 1) // r["tile_k"]
                             if r["tile_k"] else 0 for r in train)],
        "padded_flop_ratio": [min(r["padded_flop_ratio"] for r in train),
                              max(r["padded_flop_ratio"] for r in train)],
        "temporary_bytes": [min(r["temporary_bytes"] for r in train),
                            max(r["temporary_bytes"] for r in train)],
        "categorical_vocabulary": CATEGORICAL,
    }
    write = lambda name, value: (out / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n")
    write("feature_schema.json", feature_schema)
    write("training_domain.json", domain)
    write("train_config.json", freeze)
    write("validation_metrics.json", validation_metrics)
    write("heldout_metrics.json", heldout_metrics)
    write("model_metrics.json", {"baselines": baselines,
                                  "gbdt": {"validation": validation_metrics,
                                           "heldout": heldout_metrics}})
    # Both measures are analysis-only. Permutation importance rotates one
    # planning-time feature at a time within the frozen validation rows.
    counts = Counter()
    def count_nodes(node):
        if node["feature"] >= 0:
            counts[FEATURE_NAMES[node["feature"]]] += 1
            count_nodes(node["left"]); count_nodes(node["right"])
    for tree in model["trees"]:
        count_nodes(tree)
    baseline_validation_rmse = regression_metrics(
        validation, validation_predictions)["log_rmse"]
    validation_vectors = [features(row) for row in validation]
    permutation = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        permuted_predictions = []
        for row_index, row in enumerate(validation):
            vector = list(validation_vectors[row_index])
            vector[feature_index] = validation_vectors[
                (row_index + 1) % len(validation_vectors)][feature_index]
            permuted_predictions.append(predict_vector(model, vector))
        rmse = regression_metrics(
            validation, permuted_predictions)["log_rmse"]
        permutation.append({
            "feature": feature_name,
            "log_rmse_increase": rmse - baseline_validation_rmse,
        })
    permutation.sort(key=lambda item: item["log_rmse_increase"], reverse=True)
    write("feature_importance.json", {
        "interpretation": "predictive association, not causality",
        "split_count": [{"feature": k, "split_count": v}
                        for k, v in counts.most_common()],
        "validation_permutation": permutation,
    })
    metadata = {
        "model_version": MODEL_VERSION, "schema_version": SCHEMA_VERSION,
        "target": "raspberry-pi5-cortex-a76", "dtype": "f32",
        "operator": "matmul_bias_relu", "tile_family": [8, 8, 8],
        "supported_candidate_kinds": CATEGORICAL["candidate_kind"],
        "unsupported_candidate_kinds": ["unfused_vector",
            "direct_scalar_cleanup", "specialized_microkernel", "masked_transfer"],
        "repository_commit": "907b29eea29e392cc014d738376ea04720cfba07",
        "dataset_commit": "907b29eea29e392cc014d738376ea04720cfba07",
        "dataset_hash": expected_hash, "training_library": "python_stdlib_gbdt_v1",
        "python_version": platform.python_version(), "fixed_seed": SEED,
        "hyperparameters": frozen_config, "weighting_method": frozen_config["weighting"],
        "uncertainty_margin": selected_margin,
        "expected_feature_count": len(FEATURE_NAMES),
        "configuration_freeze_hash": freeze_hash,
    }
    write("metadata.json", metadata)
    with (out / "per_shape_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_details[0]) + ["split"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows([{**r, "split": "validation"} for r in validation_details])
        writer.writerows([{**r, "split": "heldout"} for r in heldout_details])
    (out / "dataset_hash.txt").write_text(expected_hash + "\n")
    model_hash = hashlib.sha256(
        (out / "model.json").read_bytes() +
        (out / "feature_schema.json").read_bytes() +
        (out / "generated_model.h").read_bytes()).hexdigest()
    (out / "model_hash.txt").write_text(model_hash + "\n")
    comparison_names = ["candidate_kind_mean", "analytical", "ridge",
                        "single_tree", "hybrid"]
    report = [
        "# Candidate-latency model regret",
        "",
        "The held-out split was evaluated only after the model configuration "
        "and uncertainty policy were frozen.",
        "",
        "| Model | Split | Exact match | Mean regret | P95 regret | Worst regret |",
        "|---|---|---:|---:|---:|---:|",
    ]
    comparison = {name: baselines[name] for name in comparison_names}
    comparison["gbdt"] = {
        "validation": validation_metrics,
        "heldout": heldout_metrics,
    }
    for name, results in comparison.items():
        for split in ("validation", "heldout"):
            selection = results[split]["selection"]
            report.append(
                f"| {name} | {split} | {selection['exact_match_rate']:.6f} | "
                f"{selection['mean_normalized_regret']:.6f} | "
                f"{selection['p95_normalized_regret']:.6f} | "
                f"{selection['worst_normalized_regret']:.6f} |")
    (out / "regret_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"config": frozen_config,
                      "validation": validation_selection,
                      "heldout": heldout_selection,
                      "model_hash": model_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
