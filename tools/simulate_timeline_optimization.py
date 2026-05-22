import json
import matplotlib.pyplot as plt

COST_PATH = "trace/cv_cost_report.json"
OUT_JSON = "trace/cv_timeline_optimization.json"
OUT_PNG = "trace/cv_timeline_optimization.png"

with open(COST_PATH, "r") as f:
    costs = json.load(f)

cost_by_op = {c["op_name"]: c for c in costs}

# Simple estimated runtime model.
# This is not real measured latency.
# It is a planner-side what-if simulation.
BACKEND_SPEED = {
    "Metal": 1.0,
    "CPU": 4.0,
}

BASE_LATENCY_MS = {
    "conv1": 0.32,
    "pool1": 0.18,
    "flatten": 0.05,
    "linear": 0.21,
}

SWITCH_COST_MS = 0.02

plans = {
    "current": [
        ("conv1", "Metal"),
        ("pool1", "CPU"),
        ("flatten", "CPU"),
        ("linear", "Metal"),
    ],
    "alt_metal_pool": [
        ("conv1", "Metal"),
        ("pool1", "Metal"),
        ("flatten", "CPU"),
        ("linear", "Metal"),
    ],
    "alt_all_metal_except_flatten": [
        ("conv1", "Metal"),
        ("pool1", "Metal"),
        ("flatten", "CPU"),
        ("linear", "Metal"),
    ],
    "alt_cpu_middle_fused": [
        ("conv1", "Metal"),
        ("pool1+flatten", "CPU"),
        ("linear", "Metal"),
    ],
}

def op_latency(op_name, backend):
    if "+" in op_name:
        return sum(
            op_latency(x, backend)
            for x in op_name.split("+")
        )

    base = BASE_LATENCY_MS.get(op_name, 0.05)
    return base * BACKEND_SPEED.get(backend, 1.0)

def memory_pressure(plan):
    # approximate memory traffic from cost report
    total = 0

    for op_name, _ in plan:
        if "+" in op_name:
            for x in op_name.split("+"):
                c = cost_by_op.get(x)
                if c:
                    total += c["estimated_read_bytes"]
                    total += c["estimated_write_bytes"]
        else:
            c = cost_by_op.get(op_name)
            if c:
                total += c["estimated_read_bytes"]
                total += c["estimated_write_bytes"]

    return total / 1e6

def gpu_occupancy_proxy(plan, total_ms):
    gpu_ms = 0

    for op_name, backend in plan:
        if backend == "Metal":
            gpu_ms += op_latency(op_name, backend)

    if total_ms == 0:
        return 0

    return gpu_ms / total_ms

def simulate(plan):
    t = 0
    timeline = []
    switch_total = 0

    prev_backend = None

    for op_name, backend in plan:
        if prev_backend is not None and backend != prev_backend:
            t += SWITCH_COST_MS
            switch_total += SWITCH_COST_MS

        duration = op_latency(op_name, backend)

        timeline.append({
            "op": op_name,
            "backend": backend,
            "start_ms": t,
            "duration_ms": duration,
        })

        t += duration
        prev_backend = backend

    return {
        "timeline": timeline,
        "total_latency_ms": t,
        "switch_cost_ms": switch_total,
        "memory_pressure_MB": memory_pressure(plan),
        "gpu_occupancy_proxy": gpu_occupancy_proxy(plan, t),
    }

results = {}

for name, plan in plans.items():
    results[name] = simulate(plan)

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))

labels = list(results.keys())
latencies = [results[x]["total_latency_ms"] for x in labels]
switches = [results[x]["switch_cost_ms"] for x in labels]
memories = [results[x]["memory_pressure_MB"] for x in labels]
occupancy = [results[x]["gpu_occupancy_proxy"] for x in labels]

fig, ax = plt.subplots(figsize=(12, 5))

x = range(len(labels))

ax.bar(x, latencies, label="total latency ms")
ax.scatter(x, switches, label="switch cost ms", marker="o")
ax.scatter(x, occupancy, label="GPU occupancy proxy", marker="x")

for i, name in enumerate(labels):
    ax.text(
        i,
        latencies[i],
        f"{latencies[i]:.2f} ms\nmem={memories[i]:.1f} MB",
        ha="center",
        va="bottom",
        fontsize=8,
    )

ax.set_xticks(list(x))
ax.set_xticklabels(labels, rotation=15)

ax.set_ylabel("Cost")
ax.set_title(
    "Timeline Optimization Simulation\n"
    "What-if runtime planner for heterogeneous execution"
)

ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PNG)

print(f"Saved {OUT_JSON}")
print(f"Saved {OUT_PNG}")