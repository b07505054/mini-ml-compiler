import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

PATH = "trace/cv_subgraph_partition.json"
OUT = "trace/cv_subgraph_partition.png"

with open(PATH, "r") as f:
    parts = json.load(f)

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.axis("off")

x_gap = 3.3
y = 0

colors = {
    "Metal": "#4C97D8",
    "CPU": "#58B957",
}

positions = []

for i, part in enumerate(parts):
    x = i * x_gap
    positions.append((x, y))

    backend = part["backend"]
    ops = "\n".join(part["ops"])

    label = (
        f"Subgraph {part['subgraph_id']}\n"
        f"{backend}\n\n"
        f"{ops}"
    )

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=colors.get(backend, "gray"),
            edgecolor="black",
            linewidth=1.2,
        ),
    )

for i in range(len(positions) - 1):
    x1, y1 = positions[i]
    x2, y2 = positions[i + 1]

    arrow = FancyArrowPatch(
        (x1 + 1.0, y1),
        (x2 - 1.0, y2),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.4,
        color="black",
    )

    ax.add_patch(arrow)

ax.set_xlim(-1.5, (len(parts) - 1) * x_gap + 1.5)
ax.set_ylim(-1.2, 1.2)

plt.title(
    "CV Subgraph Partitioning\nMetal / CPU / Metal heterogeneous execution plan",
    fontsize=14,
    pad=18,
)

plt.tight_layout()
plt.savefig(OUT)

print(f"Saved {OUT}")