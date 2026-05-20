import json
import matplotlib.pyplot as plt

with open("trace/metal_kernel_profile.json", "r") as f:
    profile = json.load(f)

labels = ["avg", "p50", "p95", "p99"]

values = [
    profile["avg_ms"],
    profile["p50_ms"],
    profile["p95_ms"],
    profile["p99_ms"],
]

fig, ax = plt.subplots(figsize=(6, 4))

ax.bar(labels, values)

ax.set_ylabel("Latency (ms)")

ax.set_title(
    f"Metal Kernel Profile\n{profile['device']}"
)

plt.tight_layout()

plt.savefig(
    "trace/metal_kernel_profile.png"
)

print(
    "Saved visualization to trace/metal_kernel_profile.png"
)