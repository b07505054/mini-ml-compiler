import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

states = [
    ("NORMAL", "#4C97D8"),
    ("OVERLOAD\nDETECTED", "#E67E22"),
    ("REPLANNING", "#F4D03F"),
    ("CPU\nFALLBACK", "#58B957"),
    ("RECOVERY\nCHECK", "#AF7AC5"),
    ("RESTORE\nGPU PLAN", "#5DADE2"),
]

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

x_positions = [0, 2.2, 4.4, 6.6, 8.8, 11.0]
y = 0

for i, ((label, color), x) in enumerate(zip(states, x_positions)):
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.7",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        ),
    )

    if i < len(states) - 1:
        next_x = x_positions[i + 1]

        arrow = FancyArrowPatch(
            (x + 0.8, y),
            (next_x - 0.8, y),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.8,
            color="black",
        )

        ax.add_patch(arrow)

transition_labels = [
    "Metal latency spike\n> 2.5 ms",
    "planner invoked",
    "backend migration",
    "GPU health probe",
    "latency normalized",
]

for i in range(len(transition_labels)):
    mid_x = (x_positions[i] + x_positions[i + 1]) / 2

    ax.text(
        mid_x,
        0.52,
        transition_labels[i],
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#F8F9F9",
            edgecolor="gray",
        ),
    )

ax.text(
    5.5,
    -1.0,
    (
        "Runtime orchestration loop for adaptive heterogeneous inference.\n"
        "The runtime monitors backend latency, triggers replanning, "
        "migrates execution, and restores GPU execution after recovery."
    ),
    ha="center",
    fontsize=10,
)

ax.set_xlim(-1, 12)
ax.set_ylim(-1.5, 1.2)

plt.title(
    "Adaptive Runtime State Machine\n"
    "Dynamic backend orchestration and runtime recovery pipeline",
    fontsize=15,
    pad=20,
)

plt.tight_layout()

OUT = "trace/cv_runtime_state_machine.png"

plt.savefig(OUT)

print(f"Saved {OUT}")