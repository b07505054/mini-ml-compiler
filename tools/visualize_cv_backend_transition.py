import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

PARTITION_PATH = "trace/cv_subgraph_partition.json"
COST_PATH = "trace/cv_cost_report.json"
OUT = "trace/cv_backend_transition.png"

with open(PARTITION_PATH, "r") as f:
    parts = json.load(f)

with open(COST_PATH, "r") as f:
    costs = json.load(f)

cost_by_op = {
    c["op_name"]: c
    for c in costs
}

colors = {
    "Metal": "#4C97D8",
    "CPU": "#58B957",
}

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.axis("off")

x_gap = 3.8
y = 0

positions = []

for i, part in enumerate(parts):
    x = i * x_gap
    positions.append((x, y))

    backend = part["backend"]
    ops = part["ops"]

    total_flops = 0
    total_read = 0
    total_write = 0

    for op in ops:
        c = cost_by_op.get(op, {})
        total_flops += c.get("estimated_flops", 0)
        total_read += c.get("estimated_read_bytes", 0)
        total_write += c.get("estimated_write_bytes", 0)

    total_mb = (total_read + total_write) / 1e6
    total_flops_m = total_flops / 1e6

    label = (
        f"Subgraph {part['subgraph_id']}\n"
        f"{backend}\n\n"
        f"{', '.join(ops)}\n\n"
        f"FLOPs: {total_flops_m:.1f}M\n"
        f"Mem: {total_mb:.2f} MB"
    )

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.65",
            facecolor=colors.get(backend, "gray"),
            edgecolor="black",
            linewidth=1.2,
        ),
    )

for i in range(len(parts) - 1):
    x1, y1 = positions[i]
    x2, y2 = positions[i + 1]

    src = parts[i]
    dst = parts[i + 1]

    switch_cost = 0.0

    if src["backend"] != dst["backend"]:
        first_dst_op = dst["ops"][0]
        switch_cost = cost_by_op.get(
            first_dst_op,
            {}
        ).get(
            "estimated_backend_switch_cost",
            0.0
        )

    arrow = FancyArrowPatch(
        (x1 + 1.25, y1),
        (x2 - 1.25, y2),
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.5,
        color="black",
    )

    ax.add_patch(arrow)

    mid_x = (x1 + x2) / 2

    ax.text(
        mid_x,
        y + 0.35,
        f"backend switch\ncost={switch_cost:.3f} ms",
        ha="center",
        va="center",
        fontsize=8,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

ax.set_xlim(-1.8, (len(parts) - 1) * x_gap + 1.8)
ax.set_ylim(-1.5, 1.5)

plt.title(
    "CV Backend Transition Cost\nHeterogeneous execution planning with estimated switch overhead",
    fontsize=14,
    pad=18,
)

plt.tight_layout()
plt.savefig(OUT)

print(f"Saved {OUT}")