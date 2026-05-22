import json
import pandas as pd
import matplotlib.pyplot as plt

PATH = "trace/cv_lowered_graph.json"
OUT = "trace/cv_lowered_graph.png"

with open(PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame([
    {
        "op_id": x["op_id"],
        "source": x["source_op_name"],
        "lowered": x["lowered_op_type"],
        "backend": x["backend"],
        "inputs": str(x["inputs"]),
        "outputs": str(x["outputs"]),
        "mem": x["memory_offset"],
    }
    for x in data
])

print(df)

fig, ax = plt.subplots(figsize=(16, 3.2))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.15, 1.8)

for i in range(len(df)):
    color = "#4C97D8" if df.iloc[i]["backend"] == "Metal" else "#58B957"
    for j in range(len(df.columns)):
        table[(i + 1, j)].set_facecolor(color)

plt.title("CV Lowered Graph IR", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(OUT)
print(f"Saved {OUT}")