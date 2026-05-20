import json
import matplotlib.pyplot as plt

PROFILE_PATH = "trace/metal_vector_add_profile.json"

with open(PROFILE_PATH, "r") as f:
    profile = json.load(f)

labels = ["avg", "p50", "p95", "p99"]

values = [
    profile["avg_ms"],
    profile["p50_ms"],
    profile["p95_ms"],
    profile["p99_ms"],
]

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(labels, values)

ax.set_ylabel("Latency (ms)")

ax.set_title(
    f"Metal {profile['kernel']} Profile\n"
    f"{profile['device']} | N={profile['elements']} | correct={profile['correct']}"
)

plt.tight_layout()

output_path = "trace/metal_vector_add_profile.png"

plt.savefig(output_path)

print(f"Saved visualization to {output_path}")