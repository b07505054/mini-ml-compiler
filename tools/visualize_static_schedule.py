import json
import pandas as pd
import matplotlib.pyplot as plt

SCHEDULE_PATH = "trace/static_schedule.json"

with open(SCHEDULE_PATH, "r") as f:
    sched = json.load(f)

rows = []

for e in sched:
    rows.append({
        "order": e["start_order"],
        "op": e["op_name"],
        "type": e["op_type"],
        "backend": e["backend"],
        "inputs": str(e["inputs"]),
        "outputs": str(e["outputs"]),
        "mem_offset": e["memory_offset"],
    })

df = pd.DataFrame(rows)

print(df)

fig, ax = plt.subplots(
    figsize=(12, 2.8)
)

ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center"
)

table.auto_set_font_size(False)

table.set_fontsize(9)

table.scale(1.2, 1.8)

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
    "Static Execution Schedule",
    fontsize=14,
    pad=12
)

plt.tight_layout()

output_path = (
    "trace/static_schedule_table.png"
)

plt.savefig(output_path)

print(
    f"Saved schedule table to "
    f"{output_path}"
)