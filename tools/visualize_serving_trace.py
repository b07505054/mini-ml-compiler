import json
from collections import defaultdict

import matplotlib.pyplot as plt

TRACE_PATH = "trace/serving_trace.json"

with open(TRACE_PATH, "r") as f:
    trace = json.load(f)

colors = {
    "prefill": "tab:blue",
    "decode": "tab:green",
    "finish": "tab:red",
}

grouped = defaultdict(list)

for event in trace:
    grouped[event["request_id"]].append(event)

for request_id in grouped:
    grouped[request_id].sort(
        key=lambda e: (
            e["step"],
            e["timestamp_ms"]
        )
    )

plt.figure(figsize=(12, 5))

bar_height = 0.28
bar_width = 0.75

for request_id, events in grouped.items():
    for idx, event in enumerate(events):
        phase = event["phase"]
        step = event["step"]

        x = idx

        label = phase

        if phase == "decode":
            label = f"decode {step}"

        plt.barh(
            y=request_id,
            width=bar_width,
            left=x,
            height=bar_height,
            color=colors.get(phase, "gray"),
            alpha=0.85
        )

        plt.text(
            x + bar_width / 2,
            request_id,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="white"
        )

plt.xlabel("Serving Event Order")
plt.ylabel("Request ID")
plt.title("LLM Serving Timeline")

plt.yticks(sorted(grouped.keys()))
plt.xticks(range(0, max(len(v) for v in grouped.values()) + 1))

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors["prefill"], label="Prefill"),
    plt.Rectangle((0, 0), 1, 1, color=colors["decode"], label="Decode"),
    plt.Rectangle((0, 0), 1, 1, color=colors["finish"], label="Finish"),
]

plt.legend(handles=legend_handles, loc="upper right")

plt.grid(axis="x", alpha=0.25)

plt.tight_layout()

output_path = "trace/serving_timeline.png"

plt.savefig(output_path)

print(f"Saved visualization to {output_path}")