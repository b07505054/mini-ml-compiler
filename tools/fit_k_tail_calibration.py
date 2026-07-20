#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

RESULT = re.compile(
    r"RESULT K=(?P<k>\d+) strategy=(?P<strategy>\S+) "
    r"m_tiles=(?P<m_tiles>\d+) n_tiles=(?P<n_tiles>\d+).*"
    r"median_ns=(?P<median>[0-9.eE+-]+) p95_ns=(?P<p95>[0-9.eE+-]+)"
)


def linear_fit(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / denom
    return ybar - slope * xbar, slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--csv-output", required=True)
    ap.add_argument("--base-config", required=True)
    args = ap.parse_args()

    rows = []
    for line in Path(args.input).read_text().splitlines():
        match = RESULT.search(line)
        if match and 1 <= int(match["k"]) <= 7:
            rows.append({
                "k_remainder": int(match["k"]),
                "strategy": match["strategy"],
                "m_tiles": int(match["m_tiles"]),
                "n_tiles": int(match["n_tiles"]),
                "median_ns": float(match["median"]),
                "p95_ns": float(match["p95"]),
            })
    direct = [(r["k_remainder"], r["median_ns"]) for r in rows
              if r["strategy"] == "direct_vector" and
              r["m_tiles"] == 1 and r["n_tiles"] == 1]
    materialized = [r["median_ns"] for r in rows
                    if r["strategy"] == "materialized" and
                    r["m_tiles"] == 1 and r["n_tiles"] == 1]
    if len(direct) != 7 or len(materialized) != 7:
        raise SystemExit("expected K=1..7 materialized and direct_vector rows")

    base, per_k = linear_fit(direct)
    config = json.loads(Path(args.base_config).read_text())
    config["direct_vector_k_tail_base_ns"] = round(base, 6)
    config["direct_vector_k_tail_per_k_ns"] = round(per_k, 6)
    config["materialized_tail_base_ns"] = round(
        statistics.median(materialized), 6)
    config["fit_method"] = "ordinary_least_squares_direct_vector_and_robust_median_materialized"
    config["fit_sample_count"] = len(rows)
    Path(args.output).write_text(json.dumps(config, indent=2) + "\n")
    with Path(args.csv_output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "direct_vector_k_tail_base_ns": base,
        "direct_vector_k_tail_per_k_ns": per_k,
        "materialized_tail_base_ns": statistics.median(materialized),
        "samples": len(rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
