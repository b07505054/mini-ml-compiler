import json
import matplotlib.pyplot as plt

PATH = "trace/cv_runtime_timeline.json"
OUT = "trace/cv_runtime_timeline.png"

with open(PATH, "r") as f:
    ops = json.load(f)

colors = {
    "Metal": "#4C97D8",
    "CPU": "#58B957",
}

fig, ax = plt.subplots(figsize=(12, 3.5))

y_map = {
    "Metal": 1,
    "CPU": 0,
}

for op in ops:
    y = y_map[op["backend"]]

    ax.barh(
        y=y,
        width=op["duration_ms"],
        left=op["start_ms"],
        height=0.35,
        color=colors[op["backend"]],
        edgecolor="black",
    )

    ax.text(
        op["start_ms"] + op["duration_ms"] / 2,
        y,
        op["op"],
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )

for i in range(len(ops) - 1):
    a = ops[i]
    b = ops[i + 1]

    gap = b["start_ms"] - (
        a["start_ms"] + a["duration_ms"]
    )

    if gap > 0:
        ax.text(
            a["start_ms"] + a["duration_ms"] + gap / 2,
            1.35,
            f"stall {gap:.2f} ms",
            fontsize=8,
            ha="center",
            color="red",
        )

ax.set_yticks([0, 1])
ax.set_yticklabels(["CPU", "Metal"])

ax.set_xlabel("Time (ms)")
ax.set_title(
    "CV Runtime Execution Timeline\n"
    "Heterogeneous runtime scheduling and backend transition stalls",
    fontsize=14,
)

ax.grid(True, axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(OUT)

print(f"Saved {OUT}")