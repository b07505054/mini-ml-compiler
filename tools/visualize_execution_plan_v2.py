import json
import pandas as pd
import matplotlib.pyplot as plt

PLAN_PATH = "trace/execution_plan_v2.json"

with open(PLAN_PATH, "r") as f:
    plan = json.load(f)

rows = []

for step in plan:
    rows.append({
        "step": step["step_id"],
        "lowered_op": step["lowered_op_type"],
        "backend": step["backend"],
        "deps": str(step["dependency_steps"]),
        "inputs": str(step["inputs"]),
        "outputs": str(step["outputs"]),
        "mem": step["memory_offset"],
        "launch": step["launch_config"],
    })

df = pd.DataFrame(rows)

print(df)

fig, ax = plt.subplots(
    figsize=(16, 2.8)
)

ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.15, 1.8)

for i in range(len(df)):
    backend = df.iloc[i]["backend"]

    color = (
        "#4C97D8"
        if backend == "Metal"
        else "#58B957"
    )

    for j in range(len(df.columns)):
        table[(i + 1, j)].set_facecolor(color)

plt.title(
    "ExecutionPlan IR v2",
    fontsize=14,
    pad=12
)

plt.tight_layout()

output_path = "trace/execution_plan_v2.png"

plt.savefig(output_path)

print(f"Saved ExecutionPlan visualization to {output_path}")