import json
import matplotlib.pyplot as plt


TRACE_PATH = "trace/runtime_trace.json"


with open(TRACE_PATH, "r") as f:
    trace = json.load(f)

ops = []
latencies = []
backends = []

for item in trace:
    ops.append(item["op_name"])
    latencies.append(item["latency_ms"])
    backends.append(item["backend"])

plt.figure(figsize=(10, 5))

bars = plt.bar(
    range(len(ops)),
    latencies
)

for i, backend in enumerate(backends):
    plt.text(
        i,
        latencies[i],
        backend,
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.xticks(
    range(len(ops)),
    ops,
    rotation=20
)

plt.ylabel("Latency (ms)")
plt.xlabel("Operator")
plt.title("Runtime Execution Timeline")

plt.tight_layout()

output_path = "trace/runtime_timeline.png"

plt.savefig(output_path)

print(f"Saved visualization to {output_path}")