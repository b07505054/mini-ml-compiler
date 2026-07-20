#!/usr/bin/env python3
"""Dependency-free dataset readability and grouped baseline sanity check."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines()]
    valid = [r for r in rows if r["label_valid"]]
    if not valid:
        raise SystemExit("dataset contains no correctness-filtered labels")
    train = [r for r in valid if r["split"] == "train"]
    means = defaultdict(list)
    for row in train:
        means[row["candidate_kind"]].append(row["log_median_ns"])
    fitted = {key: sum(values) / len(values) for key, values in means.items()}
    metrics = {}
    for split in ("train", "validation", "heldout"):
        subset = [r for r in valid if r["split"] == split
                  and r["candidate_kind"] in fitted]
        mae = (sum(abs(r["log_median_ns"] - fitted[r["candidate_kind"]])
                   for r in subset) / len(subset)) if subset else None
        metrics[split] = {"rows": len(subset), "log_mae": mae}
    print(json.dumps({
        "model": "candidate_kind_mean_dependency_free_readability_baseline",
        "production_deployment": False,
        "fitted_candidate_kinds": sorted(fitted),
        "metrics": metrics,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
