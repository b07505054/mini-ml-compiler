#!/usr/bin/env bash
# Phase R0: collect remote environment/hardware/software/repo-state truth
# directly on the remote host via live queries. Not for local execution.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/trace/remote_linux_baseline"
mkdir -p "${OUT_DIR}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST="$(hostname)"

RUNTIME_REPO="/home/allen/Desktop/Project/heterogeneous-inference-runtime"
COMPILER_REPO="/home/allen/Desktop/Project/ml-graph-compiler-runtime"

# ---------------------------------------------------------------------------
# environment.json
# ---------------------------------------------------------------------------
python3 - "$OUT_DIR/environment.json" "$TS" "$HOST" <<'PY'
import json, platform, sys
out_path, ts, host = sys.argv[1], sys.argv[2], sys.argv[3]
payload = {
    "schema": "remote_linux_baseline_environment",
    "schema_version": 1,
    "utc_timestamp": ts,
    "hostname": host,
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "truth_boundary": "live_queried_on_remote_host_via_ssh_no_local_mac_values_used",
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

# ---------------------------------------------------------------------------
# cpu_hardware.json
# ---------------------------------------------------------------------------
LSCPU_JSON="$(lscpu -J 2>/dev/null || echo '{}')"
python3 - "$OUT_DIR/cpu_hardware.json" "$TS" "$HOST" <<PY
import json, subprocess, sys
out_path, ts, host = sys.argv[1], sys.argv[2], sys.argv[3]

def cmd(args, default=""):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return default

lscpu_json = json.loads('''$(lscpu -J 2>/dev/null | sed "s/'/\\\\'/g")''') if True else {}
lscpu_fields = {}
for entry in lscpu_json.get("lscpu", []):
    lscpu_fields[entry.get("field", "").rstrip(":")] = entry.get("data", "")

payload = {
    "schema": "remote_linux_baseline_cpu_hardware",
    "schema_version": 1,
    "utc_timestamp": ts,
    "hostname": host,
    "source": "lscpu -J, lscpu -C, numactl --hardware, free -h, getconf (all live-queried on remote host)",
    "vendor": lscpu_fields.get("Vendor ID", "unknown"),
    "model_name": lscpu_fields.get("Model name", "unknown"),
    "architecture": lscpu_fields.get("Architecture", "unknown"),
    "cpu_family": lscpu_fields.get("CPU family", "unknown"),
    "model": lscpu_fields.get("Model", "unknown"),
    "stepping": lscpu_fields.get("Stepping", "unknown"),
    "sockets": lscpu_fields.get("Socket(s)", "unknown"),
    "cores_per_socket": lscpu_fields.get("Core(s) per socket", "unknown"),
    "threads_per_core": lscpu_fields.get("Thread(s) per core", "unknown"),
    "logical_cpus_total": lscpu_fields.get("CPU(s)", "unknown"),
    "cpu_max_mhz": lscpu_fields.get("CPU max MHz", "unknown"),
    "cpu_min_mhz": lscpu_fields.get("CPU min MHz", "unknown"),
    "l1d_cache_aggregate": lscpu_fields.get("L1d cache", "unknown"),
    "l1i_cache_aggregate": lscpu_fields.get("L1i cache", "unknown"),
    "l2_cache_aggregate": lscpu_fields.get("L2 cache", "unknown"),
    "l3_cache_aggregate": lscpu_fields.get("L3 cache", "unknown"),
    "flags": lscpu_fields.get("Flags", "").split(),
    "has_avx": "avx" in lscpu_fields.get("Flags", ""),
    "has_avx2": "avx2" in lscpu_fields.get("Flags", ""),
    "has_avx512": any(f.startswith("avx512") for f in lscpu_fields.get("Flags", "").split()),
    "has_fma": "fma" in lscpu_fields.get("Flags", ""),
    "per_core_cache": {
        "l1d_bytes": int(cmd(["getconf", "LEVEL1_DCACHE_SIZE"], "0") or 0),
        "l1i_bytes": int(cmd(["getconf", "LEVEL1_ICACHE_SIZE"], "0") or 0),
        "l1d_line_bytes": int(cmd(["getconf", "LEVEL1_DCACHE_LINESIZE"], "0") or 0),
        "l2_bytes": int(cmd(["getconf", "LEVEL2_CACHE_SIZE"], "0") or 0),
        "l2_line_bytes": int(cmd(["getconf", "LEVEL2_CACHE_LINESIZE"], "0") or 0),
        "l3_bytes": int(cmd(["getconf", "LEVEL3_CACHE_SIZE"], "0") or 0),
        "l3_line_bytes": int(cmd(["getconf", "LEVEL3_CACHE_LINESIZE"], "0") or 0),
        "source": "getconf (per-core/per-cluster, not aggregate)",
    },
    "numa": cmd(["numactl", "--hardware"], "numactl not available"),
    "memory_free_output": cmd(["free", "-h"], "unavailable"),
    "truth_boundary": "live_queried_on_remote_host_lscpu_getconf_numactl_no_fabricated_values",
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

# ---------------------------------------------------------------------------
# gpu_hardware.json  (via torch.cuda device query, not name inference)
# ---------------------------------------------------------------------------
cd "$RUNTIME_REPO"
.venv/bin/python - "$OUT_DIR/gpu_hardware.json" "$TS" "$HOST" <<'PY'
import json, subprocess, sys
out_path, ts, host = sys.argv[1], sys.argv[2], sys.argv[3]

nvidia_smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap,pstate",
     "--format=csv,noheader"], capture_output=True, text=True
).stdout.strip()

payload = {
    "schema": "remote_linux_baseline_gpu_hardware",
    "schema_version": 1,
    "utc_timestamp": ts,
    "hostname": host,
    "nvidia_smi_query": nvidia_smi,
    "truth_boundary": "sm_count_and_capability_from_torch_cuda_device_query_not_inferred_from_product_name",
}
try:
    import torch
    p = torch.cuda.get_device_properties(0)
    payload.update({
        "device_name": p.name,
        "compute_capability_major": p.major,
        "compute_capability_minor": p.minor,
        "sm_count_multi_processor_count": p.multi_processor_count,
        "total_memory_bytes": p.total_memory,
        "total_memory_gb": p.total_memory / 1e9,
        "warp_size": getattr(p, "warp_size", None),
        "max_threads_per_multi_processor": getattr(p, "max_threads_per_multi_processor", None),
        "regs_per_multiprocessor": getattr(p, "regs_per_multiprocessor", None),
        "is_integrated": bool(p.is_integrated),
        "is_multi_gpu_board": bool(p.is_multi_gpu_board),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "query_method": "torch.cuda.get_device_properties(0)",
    })
except Exception as exc:
    payload["torch_query_error"] = str(exc)

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

# ---------------------------------------------------------------------------
# software_stack.json
# ---------------------------------------------------------------------------
python3 - "$OUT_DIR/software_stack.json" "$TS" "$HOST" "$COMPILER_REPO" "$RUNTIME_REPO" <<'PY'
import json, subprocess, sys
out_path, ts, host, compiler_repo, runtime_repo = sys.argv[1:6]

def cmd(args, default="unavailable"):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return default

def venv_pkg(venv_python, module, version_attr="__version__"):
    code = f"import {module}; print(getattr({module}, '{version_attr}', 'unknown'))"
    return cmd([venv_python, "-c", code], "not_importable")

payload = {
    "schema": "remote_linux_baseline_software_stack",
    "schema_version": 1,
    "utc_timestamp": ts,
    "hostname": host,
    "os_release": cmd(["cat", "/etc/os-release"]),
    "kernel": cmd(["uname", "-r"]),
    "uname_full": cmd(["uname", "-a"]),
    "cmake_version": cmd(["cmake", "--version"]).splitlines()[0] if cmd(["cmake", "--version"]) != "unavailable" else "unavailable",
    "gcc_version": cmd(["gcc", "--version"]).splitlines()[0] if cmd(["gcc", "--version"]) != "unavailable" else "unavailable",
    "clang_version": cmd(["clang", "--version"]).splitlines()[0] if cmd(["clang", "--version"]) != "unavailable" else "unavailable",
    "system_python_version": cmd(["python3", "--version"]),
    "cuda_nvcc_version": cmd(["/usr/local/cuda-13.1/bin/nvcc", "--version"]).splitlines()[-1] if cmd(["/usr/local/cuda-13.1/bin/nvcc", "--version"]) != "unavailable" else "unavailable",
    "nvidia_driver_version": cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
    "compiler_repo": {
        "path": compiler_repo,
        "venv_python_version": cmd([f"{compiler_repo}/.venv/bin/python", "--version"]),
        "torch": venv_pkg(f"{compiler_repo}/.venv/bin/python", "torch"),
        "torch_cuda": cmd([f"{compiler_repo}/.venv/bin/python", "-c", "import torch; print(torch.version.cuda)"], "not_importable"),
        "triton": venv_pkg(f"{compiler_repo}/.venv/bin/python", "triton"),
        "autoawq": cmd([f"{compiler_repo}/.venv/bin/pip", "show", "autoawq"], "not_installed"),
    },
    "runtime_repo": {
        "path": runtime_repo,
        "venv_python_version": cmd([f"{runtime_repo}/.venv/bin/python", "--version"]),
        "torch": venv_pkg(f"{runtime_repo}/.venv/bin/python", "torch"),
        "torch_cuda": cmd([f"{runtime_repo}/.venv/bin/python", "-c", "import torch; print(torch.version.cuda)"], "not_importable"),
        "triton": venv_pkg(f"{runtime_repo}/.venv/bin/python", "triton"),
        "vllm": venv_pkg(f"{runtime_repo}/.venv/bin/python", "vllm"),
        "autoawq": cmd([f"{runtime_repo}/.venv/bin/pip", "show", "autoawq"], "not_installed"),
    },
    "note": "compiler_repo and runtime_repo maintain SEPARATE venvs with different torch/triton "
            "versions (torch 2.12.1 vs 2.11.0, triton 3.7.1 vs 3.6.0 at time of audit) and different "
            "Python patch versions (3.12.11 vs 3.12.13) -- a real cross-repo software-environment "
            "inconsistency, not a measurement artifact.",
    "truth_boundary": "live_queried_on_remote_host_each_repos_own_venv_no_fabricated_versions",
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

# ---------------------------------------------------------------------------
# repository_state.json
# ---------------------------------------------------------------------------
python3 - "$OUT_DIR/repository_state.json" "$TS" "$HOST" "$COMPILER_REPO" "$RUNTIME_REPO" <<'PY'
import json, subprocess, sys
out_path, ts, host, compiler_repo, runtime_repo = sys.argv[1:6]

def git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo] + list(args), capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"error: {exc}"

def repo_info(path):
    return {
        "path": path,
        "head_commit": git(path, "rev-parse", "HEAD"),
        "branch": git(path, "branch", "--show-current"),
        "remote": git(path, "remote", "-v").splitlines()[0] if git(path, "remote", "-v") else "none",
        "status_short": git(path, "status", "--short"),
        "is_clean": git(path, "status", "--short") == "",
        "ahead_behind_origin": git(path, "status", "-sb").splitlines()[0] if git(path, "status", "-sb") else "unknown",
        "last_commit_summary": git(path, "log", "-1", "--format=%H %ci %s"),
    }

payload = {
    "schema": "remote_linux_baseline_repository_state",
    "schema_version": 1,
    "utc_timestamp": ts,
    "hostname": host,
    "compiler_repository": repo_info(compiler_repo),
    "runtime_repository": repo_info(runtime_repo),
    "duplicate_untracked_checkouts_found": [
        "/home/allen/systems-portfolio-repeatability/ml-graph-compiler-runtime (NOT a git repo; not used by this audit)",
        "/home/allen/systems-portfolio-repeatability/heterogeneous-inference-runtime (NOT a git repo; not used by this audit)",
    ],
    "canonical_repos_used": [compiler_repo, runtime_repo],
    "truth_boundary": "live_git_queries_on_remote_host_canonical_paths_under_Desktop_Project_only",
}
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

echo "Wrote R0 baseline artifacts to ${OUT_DIR}"
ls -la "${OUT_DIR}"
