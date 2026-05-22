import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

PATH = "trace/cv_runtime_replan.json"
OUT = "trace/cv_runtime_replan.png"

with open(PATH, "r") as f:
    data = json.load(f)

before = data["before"]
after = data["after"]
obs = data["observations"]

colors = {
    "Metal": "#4C97D8",
    "CPU": "#58B957",
}

def parse_assignment(a):
    op, backend = a.split(" -> ")
    return op.strip(), backend.strip()

before_ops = [parse_assignment(a) for a in before["assignments"]]
after_ops = [parse_assignment(a) for a in after["assignments"]]

fig, ax = plt.subplots(figsize=(13, 5))
ax.axis("off")

x_before = 0
x_after = 7
y_start = 2.0
y_gap = 0.7

ax.text(
    x_before,
    2.9,
    f"Initial Plan: {before['name']}\n"
    f"latency={before['total_latency_ms']:.2f} ms | "
    f"GPU occ={before['gpu_occupancy']:.2f}",
    ha="center",
    fontsize=11,
    fontweight="bold",
)

ax.text(
    x_after,
    2.9,
    f"Replanned: {after['name']}\n"
    f"latency={after['total_latency_ms']:.2f} ms | "
    f"GPU occ={after['gpu_occupancy']:.2f}",
    ha="center",
    fontsize=11,
    fontweight="bold",
)

for i, (op, backend) in enumerate(before_ops):
    y = y_start - i * y_gap

    ax.text(
        x_before,
        y,
        f"{op}\n{backend}",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=colors.get(backend, "gray"),
            edgecolor="black",
        ),
    )

for i, (op, backend) in enumerate(after_ops):
    y = y_start - i * y_gap

    ax.text(
        x_after,
        y,
        f"{op}\n{backend}",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=colors.get(backend, "gray"),
            edgecolor="black",
        ),
    )

for i in range(len(before_ops)):
    y = y_start - i * y_gap

    arrow = FancyArrowPatch(
        (x_before + 1.0, y),
        (x_after - 1.0, y),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.2,
        color="black",
    )

    ax.add_patch(arrow)

obs_text = "\n".join(
    [
        f"{o['backend']} observed {o['observed_latency_ms']:.2f} ms"
        + (" overload" if o["overloaded"] else "")
        for o in obs
    ]
)

ax.text(
    3.5,
    2.65,
    "Runtime Feedback Trigger\n" + obs_text,
    ha="center",
    va="center",
    fontsize=10,
    color="black",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#FADBD8",
        edgecolor="red",
    ),
)

ax.set_xlim(-1.5, 8.5)
ax.set_ylim(-1.2, 3.5)

plt.title(
    "Runtime Adaptive Replanning\n"
    "Runtime feedback changes backend assignment after overload detection",
    fontsize=14,
    pad=16,
)

plt.tight_layout()
plt.savefig(OUT)

print(f"Saved {OUT}")