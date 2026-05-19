import json
import matplotlib.pyplot as plt

TRACE_PATH = "trace/runtime_trace.json"

with open(TRACE_PATH, "r") as f:
    trace = json.load(f)

base_time = min(
    e["start_ms"]
    for e in trace
)

colors = {
    "Metal": "tab:blue",
    "MockGPU": "tab:purple",
    "CPU": "tab:green",
}

plt.figure(figsize=(11, 4))

for idx, event in enumerate(trace):
    op = event["op_name"]
    backend = event["backend"]
    mem = event["memory_offset"]

    start = event["start_ms"] - base_time
    duration = event["latency_ms"]

    label = (
        f"{op}\n"
        f"{backend}\n"
        f"{duration:.2f} ms\n"
        f"mem={mem}"
    )

    plt.barh(
        y=idx,
        width=duration,
        left=start,
        height=0.45,
        color=colors.get(backend, "gray"),
        alpha=0.85
    )

    plt.text(
        start + duration / 2,
        idx,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="white"
    )

plt.yticks(
    range(len(trace)),
    [e["op_name"] for e in trace]
)

plt.xlabel("Relative Runtime Time (ms)")
plt.ylabel("Operator")

plt.title("Runtime Execution Trace")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=color, label=backend)
    for backend, color in colors.items()
]

plt.legend(
    handles=legend_handles,
    loc="upper right"
)

plt.grid(axis="x", alpha=0.25)

plt.tight_layout()

output_path = "trace/runtime_execution_trace.png"

plt.savefig(output_path)

print(f"Saved runtime execution trace to {output_path}")