import json
import pandas as pd
import matplotlib.pyplot as plt

LOWERED_PATH = "trace/lowered_graph.json"

with open(LOWERED_PATH, "r") as f:
    lowered = json.load(f)

rows = []

for op in lowered:
    rows.append({
        "op_id": op["op_id"],
        "source": op["source_op_name"],
        "lowered_type": op["lowered_op_type"],
        "backend": op["backend"],
        "inputs": str(op["inputs"]),
        "outputs": str(op["outputs"]),
        "mem_offset": op["memory_offset"],
    })

df = pd.DataFrame(rows)

print(df)

fig, ax = plt.subplots(
    figsize=(14, 2.8)
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
    "Lowered Graph IR",
    fontsize=14,
    pad=12
)

plt.tight_layout()

output_path = "trace/lowered_graph.png"

plt.savefig(output_path)

print(f"Saved lowered graph visualization to {output_path}")