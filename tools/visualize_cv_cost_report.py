import json
import pandas as pd
import matplotlib.pyplot as plt

PATH = "trace/cv_cost_report.json"
OUT = "trace/cv_cost_report.png"

with open(PATH, "r") as f:
    data = json.load(f)

df = pd.DataFrame([
    {
        "op": x["op_name"],
        "type": x["op_type"],
        "backend": x["backend"],
        "read_MB": round(x["estimated_read_bytes"] / 1e6, 3),
        "write_MB": round(x["estimated_write_bytes"] / 1e6, 3),
        "FLOPs_M": round(x["estimated_flops"] / 1e6, 3),
        "intensity": round(x["arithmetic_intensity"], 3),
        "switch": x["estimated_backend_switch_cost"],
        "fusion": x["fusion_note"],
    }
    for x in data
])

print(df)

fig, ax = plt.subplots(figsize=(18, 3.5))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc="center",
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 1.8)

for i in range(len(df)):
    color = "#4C97D8" if df.iloc[i]["backend"] == "Metal" else "#58B957"
    for j in range(len(df.columns)):
        table[(i + 1, j)].set_facecolor(color)

plt.title("CV Compiler Cost Report", fontsize=14, pad=12)
plt.tight_layout()
plt.savefig(OUT)
print(f"Saved {OUT}")