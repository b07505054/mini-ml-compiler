import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_PATH = Path("trace/onnxruntime_cuda_benchmark.json")
OUTPUT_PATH = Path("trace/onnxruntime_cuda_benchmark.png")


with open(REPORT_PATH, "r") as f:
    results = json.load(f)


rows = []

for r in results:
    rows.append({
        "requested": r["requested_provider"].replace("ExecutionProvider", ""),
        "actual": r["actual_provider"].replace("ExecutionProvider", ""),
        "avg_ms": r["avg_ms"],
        "p50_ms": r["p50_ms"],
        "p95_ms": r["p95_ms"],
        "p99_ms": r["p99_ms"],
    })

df = pd.DataFrame(rows)

print(df)

fig, ax = plt.subplots(figsize=(10, 5))

labels = df["requested"].tolist()
avg = df["avg_ms"].tolist()
p95 = df["p95_ms"].tolist()

x = range(len(labels))

ax.bar(
    x,
    avg,
    label="avg latency"
)

ax.scatter(
    x,
    p95,
    label="p95 latency",
    marker="o"
)

for i, row in df.iterrows():
    fallback = ""

    if row["requested"] != row["actual"]:
        fallback = f"\nfallback → {row['actual']}"

    ax.text(
        i,
        row["avg_ms"],
        f"{row['avg_ms']:.3f} ms{fallback}",
        ha="center",
        va="bottom",
        fontsize=8
    )

ax.set_xticks(list(x))
ax.set_xticklabels(labels)

ax.set_ylabel("Latency (ms)")
ax.set_title("ONNX Runtime Execution Provider Benchmark")

ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(OUTPUT_PATH)

print(f"Saved visualization to {OUTPUT_PATH}")