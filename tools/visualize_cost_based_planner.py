import json
import matplotlib.pyplot as plt
import numpy as np

PATH = "trace/cv_cost_based_planner.json"
OUT = "trace/cv_cost_based_planner.png"

with open(PATH, "r") as f:
    candidates = json.load(f)

names = [c["name"] for c in candidates]
latency = [c["total_latency_ms"] for c in candidates]
switch = [c["switch_cost_ms"] for c in candidates]
occupancy = [c["gpu_occupancy"] for c in candidates]
memory = [c["memory_pressure_mb"] for c in candidates]
chosen = [c["chosen"] for c in candidates]

x = np.arange(len(names))

fig, ax1 = plt.subplots(figsize=(12, 5))

bar_colors = [
    "#F5B041" if is_chosen else "#7FB3D5"
    for is_chosen in chosen
]

bars = ax1.bar(
    x,
    latency,
    color=bar_colors,
    edgecolor="black",
    label="total latency ms",
)

ax1.scatter(
    x,
    switch,
    marker="o",
    s=80,
    label="switch cost ms",
)

ax1.set_ylabel("Latency / switch cost (ms)")
ax1.set_xticks(x)
ax1.set_xticklabels(names)
ax1.grid(axis="y", alpha=0.3)

ax2 = ax1.twinx()

ax2.plot(
    x,
    occupancy,
    marker="x",
    linewidth=2,
    label="GPU occupancy proxy",
)

ax2.set_ylabel("GPU occupancy proxy")

for i, b in enumerate(bars):
    label = f"{latency[i]:.2f} ms\nmem={memory[i]:.1f} MB"

    if chosen[i]:
        label += "\nBEST"

    ax1.text(
        b.get_x() + b.get_width() / 2,
        b.get_height(),
        label,
        ha="center",
        va="bottom",
        fontsize=8,
    )

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
)

plt.title(
    "Cost-Based Backend Planner\n"
    "Candidate latency, switch cost, memory pressure, and GPU occupancy"
)

plt.tight_layout()
plt.savefig(OUT)

print(f"Saved {OUT}")
