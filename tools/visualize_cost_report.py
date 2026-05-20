import json
import pandas as pd
import matplotlib.pyplot as plt

REPORT_PATH = "trace/compiler_cost_report.json"

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

rows = []

for e in report:
    rows.append({
        "op": e["op_name"],
        "type": e["op_type"],
        "backend": e["backend"],
        "read_B": e["estimated_read_bytes"],
        "write_B": e["estimated_write_bytes"],
        "FLOPs": e["estimated_flops"],
        "intensity": round(e["arithmetic_intensity"], 2),
        "launch": e["estimated_kernel_launch_cost"],
        "switch": e["estimated_backend_switch_cost"],
        "actual_backend": e["actual_backend"],
        "actual_ms": round(e["actual_latency_ms"], 4),
        "fusion": e["fusion_note"],
    })

df = pd.DataFrame(rows)

print(df)

fig, ax = plt.subplots(
    figsize=(18, 3.0)
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
    "Compiler Cost Report",
    fontsize=14,
    pad=12
)

plt.tight_layout()

output_path = "trace/compiler_cost_report.png"

plt.savefig(output_path)

print(f"Saved cost report visualization to {output_path}")