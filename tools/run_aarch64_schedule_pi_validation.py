#!/usr/bin/env python3
"""run_aarch64_schedule_pi_validation.py

Stage 13: Raspberry Pi correctness and controlled runtime validation of the
Stage 12 machine-scheduling backend classifications. Runs on the DEV HOST
(this repo checkout) and drives the real Raspberry Pi over SSH -- no
network calls happen except to the Pi host given by --pi-host.

Reuses, unmodified:
  - mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh (compilation)
  - mlir_passes/tools/generate_schedule_harness.sh + the new
    aarch64_matmul_bias_relu_schedule_harness.cpp.template (Stage 13, added
    alongside this script -- see that template's header for why a new
    per-candidate harness was necessary rather than extending the existing
    5-way tiled_harness: same-shape different-schedule-unroll-k objects
    export the identical symbol name and cannot be linked into one binary)

For each candidate:
  1. Compile on the dev host, hash the object.
  2. Generate the per-shape harness .cpp, transfer .cpp + .o to the Pi.
  3. Build on the Pi with g++, re-hash the transferred .o on the Pi side to
     confirm byte-for-byte transfer integrity (stale/corrupt-binary guard).
  4. Run the harness (correctness + benchmark in one process), parse its
     JSON output.

Group A candidates get the full required shape matrix (correctness); the
two Class-A-confirmed shapes (primary 32x32x32, cube64 64x64x64) addition-
ally get multiple INTERLEAVED baseline/scheduled measurement groups for
statistical comparison. Group B (Stage 12 Class D diagnostics) gets a
single measurement group each, on the primary shape only, matched against
same-tile baselines.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

COMPILE_SCRIPT = os.path.join(REPO_ROOT, "mlir_passes", "tools", "compile_hir_matmul_bias_relu_aarch64.sh")
GENERATE_HARNESS = os.path.join(REPO_ROOT, "mlir_passes", "tools", "generate_schedule_harness.sh")
FIXTURE_DIR = os.path.join(REPO_ROOT, "mlir_passes", "test", "backend_codegen")

TARGET_TRIPLE = "aarch64-linux-gnu"
TARGET_CPU = "cortex-a76"


class MismatchedComparisonError(ValueError):
    """Raised when two candidates intended for a matched-baseline
    comparison differ in anything other than schedule_unroll_k."""


def sh(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(str(c) for c in cmd)}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def ssh_run(pi_host, remote_cmd, check=True, timeout=600):
    proc = subprocess.run(["ssh", pi_host, remote_cmd], capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"ssh command failed on {pi_host}: {remote_cmd}\n{proc.stdout}\n{proc.stderr}")
    return proc


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pi environment capture (task section 2)
# ---------------------------------------------------------------------------

def capture_pi_environment(pi_host, out_dir):
    commands = {
        "hostname": "hostname",
        "model": "cat /proc/device-tree/model; echo",
        "uname": "uname -a",
        "os_release": "cat /etc/os-release",
        "lscpu": "lscpu",
        "governor": "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        "cur_freq_khz": "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq",
        "max_freq_khz": "cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
        "temp_before": "vcgencmd measure_temp",
        "throttled_before": "vcgencmd get_throttled",
        "meminfo": "free -h",
        "load": "uptime",
        "gcc_version": "g++ --version | head -1",
        "perf_available": "which perf || echo NOT_FOUND",
        "taskset_available": "which taskset || echo NOT_FOUND",
        "disk": "df -h /tmp",
    }
    env = {}
    raw_lines = []
    for key, cmd in commands.items():
        proc = ssh_run(pi_host, cmd, check=False)
        env[key] = proc.stdout.strip()
        raw_lines.append(f"$ {cmd}\n{proc.stdout}{proc.stderr}")

    env["perf_installed"] = "NOT_FOUND" not in env["perf_available"]
    env["taskset_installed"] = "NOT_FOUND" not in env["taskset_available"]
    env["perf_note"] = (
        "perf is NOT installed on this Pi. Not installed by this script -- installing new "
        "system packages on shared hardware without explicit authorization is out of scope "
        "for this slice. Hardware-counter evidence (section 8/10 of the task brief) is "
        "therefore unavailable; this is stated explicitly rather than silently omitted."
        if not env["perf_installed"] else "perf is installed; hardware counters were collected."
    )

    git_commit = sh(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).strip()
    git_dirty = sh(["git", "status", "--short"], cwd=REPO_ROOT).strip()
    env["git_commit"] = git_commit
    env["git_working_tree_dirty"] = bool(git_dirty)
    env["git_dirty_files"] = git_dirty.splitlines() if git_dirty else []
    env["target_triple"] = TARGET_TRIPLE
    env["target_cpu"] = TARGET_CPU

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    with open(os.path.join(out_dir, "environment_raw.txt"), "w") as f:
        f.write("\n\n".join(raw_lines) + "\n")
    return env


def capture_thermal_snapshot(pi_host, label, out_dir):
    temp = ssh_run(pi_host, "vcgencmd measure_temp", check=False).stdout.strip()
    throttled = ssh_run(pi_host, "vcgencmd get_throttled", check=False).stdout.strip()
    snap = {"label": label, "timestamp": time.time(), "temp": temp, "throttled": throttled}
    path = os.path.join(out_dir, "thermal_snapshots.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(snap) + "\n")
    return snap


# ---------------------------------------------------------------------------
# Candidate compilation + Pi transfer/build (task section 3)
# ---------------------------------------------------------------------------

def compile_candidate(shape, tile_m, tile_n, tile_k, unroll_k, name, out_dir):
    fixture = os.path.join(FIXTURE_DIR, f"matmul_bias_relu_tiled_{shape}.mlir")
    if not os.path.isfile(fixture):
        raise RuntimeError(f"fixture not found: {fixture}")
    cmd = [
        "bash", COMPILE_SCRIPT, "--variant", "tiled-scheduled",
        "--tile-m", str(tile_m), "--tile-n", str(tile_n), "--tile-k", str(tile_k),
        "--schedule-unroll-k", str(unroll_k),
        fixture, out_dir, name,
    ]
    sh(cmd)
    obj_path = os.path.join(out_dir, f"{name}.o")
    return {
        "command": " ".join(cmd),
        "obj_path": obj_path,
        "obj_sha256": sha256_file(obj_path),
        "obj_bytes": os.path.getsize(obj_path),
        "asm_path": os.path.join(out_dir, f"{name}.s"),
        "ll_path": os.path.join(out_dir, f"{name}.ll"),
        "llvm_mlir_path": os.path.join(out_dir, f"{name}_llvm.mlir"),
    }


class StaleArtifactError(RuntimeError):
    """Raised when a transferred/compiled object does not match the hash
    of the object that was intended to be benchmarked -- guards against
    silently benchmarking a stale or corrupted binary (task section 3)."""


def verify_object_identity(candidate_name, local_sha256, remote_sha256):
    if not remote_sha256:
        raise StaleArtifactError(f"object identity check FAILED for {candidate_name}: remote object missing or unreadable (empty checksum)")
    if local_sha256 != remote_sha256:
        raise StaleArtifactError(
            f"object transfer integrity check FAILED for {candidate_name}: "
            f"local sha256={local_sha256} remote sha256={remote_sha256}"
        )
    return True


def build_on_pi(pi_host, pi_dir, shape, name, obj_path, obj_sha256, local_out_dir):
    harness_cpp = os.path.join(local_out_dir, f"{name}_harness.cpp")
    sh(["bash", GENERATE_HARNESS, "--shape", shape, "--output", harness_cpp])

    ssh_run(pi_host, f"mkdir -p '{pi_dir}'")
    subprocess.run(["scp", "-q", harness_cpp, f"{pi_host}:{pi_dir}/{name}_harness.cpp"], check=True)
    subprocess.run(["scp", "-q", obj_path, f"{pi_host}:{pi_dir}/{name}.o"], check=True)

    remote_sha = ssh_run(pi_host, f"sha256sum '{pi_dir}/{name}.o' 2>/dev/null | cut -d' ' -f1").stdout.strip()
    verify_object_identity(name, obj_sha256, remote_sha)

    build_cmd = (
        f"cd '{pi_dir}' && g++ -O2 -std=c++17 -c {name}_harness.cpp -o {name}_harness.o && "
        f"g++ -O2 {name}_harness.o {name}.o -o {name}_bin"
    )
    ssh_run(pi_host, build_cmd)
    return {
        "harness_cpp_local": harness_cpp,
        "remote_binary": f"{pi_dir}/{name}_bin",
        "remote_obj_sha256_verified": remote_sha == obj_sha256,
        "build_command": build_cmd,
    }


def run_harness(pi_host, remote_binary, shape_m, shape_n, shape_k, tile_m, tile_n, tile_k,
                 unroll_k, candidate_label, iterations, warmup, repeated_calls,
                 seed_a, seed_b, dump_output_path=None, taskset_core=None, timeout=300):
    args = (
        f"--shape-m {shape_m} --shape-n {shape_n} --shape-k {shape_k} "
        f"--tile-m {tile_m} --tile-n {tile_n} --tile-k {tile_k} --schedule-unroll-k {unroll_k} "
        f"--candidate-label {candidate_label} --iterations {iterations} --warmup {warmup} "
        f"--repeated-calls {repeated_calls} --seed-a {seed_a} --seed-b {seed_b}"
    )
    if dump_output_path:
        args += f" --dump-output {dump_output_path}"
    prefix = f"taskset -c {taskset_core} " if taskset_core is not None else ""
    cmd = f"{prefix}{remote_binary} {args}"
    proc = ssh_run(pi_host, cmd, check=False, timeout=timeout)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"harness output was not valid JSON for {candidate_label}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n{e}")
    result["_exit_code"] = proc.returncode
    result["_command"] = cmd
    result["_stderr"] = proc.stderr[-2000:] if proc.stderr else ""
    return result


# ---------------------------------------------------------------------------
# Matched-comparison guard (task section 7)
# ---------------------------------------------------------------------------

def assert_matched_comparison(baseline_cfg, scheduled_cfg):
    """baseline_cfg/scheduled_cfg are dicts with keys: shape, tile, target_cpu,
    opt_level, iterations, warmup, git_commit. Everything must match except
    schedule_unroll_k."""
    fields_that_must_match = ["shape", "tile", "target_cpu", "opt_level", "iterations", "warmup", "git_commit"]
    for field in fields_that_must_match:
        if baseline_cfg.get(field) != scheduled_cfg.get(field):
            raise MismatchedComparisonError(
                f"refusing matched comparison: field '{field}' differs "
                f"(baseline={baseline_cfg.get(field)!r} scheduled={scheduled_cfg.get(field)!r})"
            )
    if baseline_cfg.get("schedule_unroll_k") == scheduled_cfg.get("schedule_unroll_k"):
        raise MismatchedComparisonError(
            "refusing matched comparison: schedule_unroll_k is identical on both sides "
            "(nothing to compare)"
        )


# ---------------------------------------------------------------------------
# Benchmark distribution / runtime classification (task sections 6, 10)
# ---------------------------------------------------------------------------

def aggregate_measurement_groups(group_results):
    """group_results: list of harness JSON 'benchmark' dicts from repeated
    interleaved invocations of the SAME candidate. Aggregates medians across
    groups (a group-of-medians summary is more robust to any single noisy
    group than pooling every raw sample)."""
    medians = [g["median_ms"] for g in group_results]
    mins = [g["min_ms"] for g in group_results]
    return {
        "group_count": len(group_results),
        "median_of_medians_ms": statistics.median(medians),
        "min_of_mins_ms": min(mins),
        "mean_of_medians_ms": statistics.mean(medians),
        "stddev_of_medians_ms": statistics.stdev(medians) if len(medians) > 1 else 0.0,
        "cv_of_medians": (statistics.stdev(medians) / statistics.mean(medians)) if len(medians) > 1 and statistics.mean(medians) > 0 else 0.0,
        "per_group_medians_ms": medians,
    }


RUNTIME_CLASS_MEANING = {
    "A": "Confirmed runtime win -- statistically stable improvement, correctness passes, no harmful backend regression",
    "B": "Runtime neutral -- difference within noise, correctness passes",
    "C": "Trade-off -- some shapes improve and others regress, or speed improves with meaningful code-size cost",
    "D": "Runtime regression -- stable slowdown, correctness may still pass",
    "E": "Incorrect -- numerical or execution failure",
}


def classify_runtime(baseline_agg, scheduled_agg, baseline_correct, scheduled_correct,
                      baseline_spills, scheduled_spills, noise_threshold_pct=3.0):
    if not (baseline_correct and scheduled_correct):
        return "E", "one or both candidates failed correctness"
    if scheduled_spills > baseline_spills:
        return "D", f"new spills relative to Stage 12 evidence (baseline={baseline_spills}, scheduled={scheduled_spills})"

    b_med = baseline_agg["median_of_medians_ms"]
    s_med = scheduled_agg["median_of_medians_ms"]
    pct_change = ((b_med - s_med) / b_med) * 100.0 if b_med else 0.0  # positive == scheduled is faster

    # "not a sub-1% win" + must clear noise: use max(3%, 2x the larger CV) as
    # the effective noise floor, so a run with high variance can't produce a
    # spurious "A" classification from a single lucky measurement.
    noise_floor = max(noise_threshold_pct, 2 * 100 * max(baseline_agg["cv_of_medians"], scheduled_agg["cv_of_medians"]))

    if pct_change > noise_floor:
        return "A", f"median-of-medians improved {pct_change:.2f}% (baseline={b_med:.5f}ms, scheduled={s_med:.5f}ms), clears noise floor {noise_floor:.2f}%"
    elif pct_change < -noise_floor:
        return "D", f"median-of-medians regressed {-pct_change:.2f}% (baseline={b_med:.5f}ms, scheduled={s_med:.5f}ms), clears noise floor {noise_floor:.2f}%"
    else:
        return "B", f"median-of-medians change {pct_change:.2f}% is within the {noise_floor:.2f}% noise floor (baseline={b_med:.5f}ms, scheduled={s_med:.5f}ms)"


# ---------------------------------------------------------------------------
# Spill-prediction validation (task section 9)
# ---------------------------------------------------------------------------

def validate_spill_prediction(candidate_label, stage12_spills, stage12_reloads, stage12_stack_bytes,
                               baseline_median_ms, scheduled_median_ms, correctness_pass):
    if not correctness_pass:
        return "inconclusive", "correctness failure prevents a defensible latency comparison"
    pct_change = ((baseline_median_ms - scheduled_median_ms) / baseline_median_ms) * 100.0 if baseline_median_ms else 0.0
    regressed = pct_change < -1.0  # any measurable slowdown, not just a "large" one, for this diagnostic table
    if stage12_spills > 0:
        if regressed:
            return "confirmed", f"Stage 12 predicted spills ({stage12_spills} stores/{stage12_reloads} reloads); measured a {-pct_change:.2f}% latency regression"
        elif pct_change > 3.0:
            return "contradicted", f"Stage 12 predicted spills ({stage12_spills} stores/{stage12_reloads} reloads) but measured a {pct_change:.2f}% latency IMPROVEMENT"
        else:
            return "partially confirmed", f"Stage 12 predicted spills ({stage12_spills} stores/{stage12_reloads} reloads); measured latency change {pct_change:.2f}% is small/noisy, not a clear regression"
    else:
        return "not applicable", "Stage 12 reported zero spills for this candidate (not a Group B diagnostic)"


# ---------------------------------------------------------------------------
# Candidate matrix (task sections 1, 4, 9)
# ---------------------------------------------------------------------------
# (label, shape, tile_m, tile_n, tile_k, unroll_k, group, full_rigor,
#  iterations, warmup, repeated_calls, stage12_key)
# stage12_key indexes into Stage 12's schedule_comparison_results.json
# "candidates" dict, for cross-referencing spill/register evidence -- None
# where Stage 12 never analyzed this exact (shape, tile, unroll) triple.
CANDIDATES = [
    ("small_control_uk1", "8x8x8", 4, 8, 8, 1, "A", False, 500, 100, 200, "small_control_collapsed"),
    ("cube16_uk1", "16x16x16", 8, 8, 8, 1, "A", False, 1000, 200, 200, None),
    ("cube16_uk2", "16x16x16", 8, 8, 8, 2, "A", False, 1000, 200, 200, None),
    ("primary_uk1", "32x32x32", 8, 8, 8, 1, "A", True, 2000, 200, 500, "primary_unroll1"),
    ("primary_uk2", "32x32x32", 8, 8, 8, 2, "A", True, 2000, 200, 500, "primary_unroll2"),
    ("cube64_uk1", "64x64x64", 8, 8, 8, 1, "A", True, 500, 100, 200, "cube64_unroll1"),
    ("cube64_uk2", "64x64x64", 8, 8, 8, 2, "A", True, 500, 100, 200, "cube64_unroll2"),
    ("rect_uk1", "32x64x32", 8, 8, 8, 1, "A", False, 1000, 200, 200, None),
    ("rect_uk2", "32x64x32", 8, 8, 8, 2, "A", False, 1000, 200, 200, None),
    ("large_uk1", "128x128x128", 8, 8, 8, 1, "A", False, 200, 50, 100, None),
    ("large_uk2", "128x128x128", 8, 8, 8, 2, "A", False, 200, 50, 100, None),
    ("diag_full_unroll_uk4", "32x32x32", 8, 8, 8, 4, "B", False, 2000, 200, 200, "primary_full_unroll"),
    ("diag_alt_ktile_uk1", "32x32x32", 8, 8, 4, 1, "B", False, 2000, 200, 200, "alt_k_tile_unroll1"),
    ("diag_alt_ktile_uk2", "32x32x32", 8, 8, 4, 2, "B", False, 2000, 200, 200, "alt_k_tile_unroll2"),
]

FULL_RIGOR_GROUPS = 5  # interleaved measurement groups for primary/cube64

# (baseline_label, scheduled_label, matched_baseline_for_diagnostic)
COMPARISON_PAIRS = [
    ("small_control_uk1", None),
    ("cube16_uk1", "cube16_uk2"),
    ("primary_uk1", "primary_uk2"),
    ("cube64_uk1", "cube64_uk2"),
    ("rect_uk1", "rect_uk2"),
    ("large_uk1", "large_uk2"),
]
DIAGNOSTIC_PAIRS = [
    ("primary_uk1", "diag_full_unroll_uk4"),
    ("diag_alt_ktile_uk1", "diag_alt_ktile_uk2"),
]


def shape_dims(shape_str):
    m, n, k = (int(x) for x in shape_str.split("x"))
    return m, n, k


def load_stage12(stage12_json_path):
    if not stage12_json_path or not os.path.isfile(stage12_json_path):
        return None
    with open(stage12_json_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--pi-host", default="allen@100.110.37.6")
    ap.add_argument("--stage12-json", default=os.path.join(
        REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_scheduling",
        "schedule_comparison_results.json"))
    ap.add_argument("--taskset-core", type=int, default=3)
    ap.add_argument("--only", help="comma-separated candidate labels to run (default: all)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    compile_dir = os.path.join(args.output_dir, "compiled")
    candidates_dir = os.path.join(args.output_dir, "candidates")
    os.makedirs(compile_dir, exist_ok=True)
    os.makedirs(candidates_dir, exist_ok=True)

    pi_dir = f"/tmp/stage13_schedule_pi_{int(time.time())}"
    stage12 = load_stage12(args.stage12_json)

    print(f"[env] capturing Pi environment ({args.pi_host}) ...", file=sys.stderr)
    env_before = capture_pi_environment(args.pi_host, args.output_dir)
    capture_thermal_snapshot(args.pi_host, "before_all", args.output_dir)

    wanted = set(args.only.split(",")) if args.only else None
    candidates_to_run = [c for c in CANDIDATES if wanted is None or c[0] in wanted]
    by_label = {c[0]: c for c in candidates_to_run}

    # ---- Phase 1: compile + transfer + build every candidate binary
    # first, WITHOUT running any measurements yet. This is required for
    # true interleaving below: a matched pair's measurement groups must be
    # able to alternate baseline/scheduled invocations, which is impossible
    # if each candidate's full compile-build-run pipeline runs to
    # completion before the next candidate starts. ----
    built = {}
    errors = {}
    for (label, shape, tm, tn, tk, uk, group, full_rigor, iterations, warmup, repeated_calls, s12_key) in candidates_to_run:
        print(f"[build {label}] shape={shape} tile={tm}x{tn}x{tk} uk={uk} ...", file=sys.stderr)
        try:
            cand_dir = os.path.join(candidates_dir, label)
            os.makedirs(cand_dir, exist_ok=True)
            compiled = compile_candidate(shape, tm, tn, tk, uk, label, compile_dir)
            build_info = build_on_pi(args.pi_host, pi_dir, shape, label, compiled["obj_path"], compiled["obj_sha256"], cand_dir)
            built[label] = {
                "label": label, "shape": shape, "tile": {"m": tm, "n": tn, "k": tk},
                "schedule_unroll_k": uk, "group": group, "full_rigor": full_rigor,
                "iterations": iterations, "warmup": warmup, "repeated_calls": repeated_calls,
                "cand_dir": cand_dir, "compiled": compiled, "build": build_info, "s12_key": s12_key,
            }
        except Exception as e:
            errors[label] = str(e)
            print(f"  BUILD FAILED: {e}", file=sys.stderr)

    # ---- Phase 2: run measurement groups. Matched comparison pairs
    # (COMPARISON_PAIRS, DIAGNOSTIC_PAIRS) are executed with their
    # measurement groups INTERLEAVED (baseline group 0, scheduled group 0,
    # baseline group 1, scheduled group 1, ...) -- task section 6/7:
    # candidates run "in an interleaved or counterbalanced order when
    # practical". Any candidate not part of a pair still in `built` (there
    # are none in the current matrix, but kept for robustness) runs its
    # single group standalone afterward. ----
    all_group_results = {label: [] for label in built}
    paired_labels = set()
    interleave_groups = [(b, s) for b, s in COMPARISON_PAIRS if s is not None] + list(DIAGNOSTIC_PAIRS)
    for baseline_label, scheduled_label in interleave_groups:
        if baseline_label not in built or scheduled_label not in built:
            continue
        paired_labels.add(baseline_label)
        paired_labels.add(scheduled_label)
        b_cfg, s_cfg = built[baseline_label], built[scheduled_label]
        n_groups = max(
            FULL_RIGOR_GROUPS if b_cfg["full_rigor"] or s_cfg["full_rigor"] else 1, 1
        )
        print(f"[interleave] {baseline_label} <-> {scheduled_label}: {n_groups} group(s) each", file=sys.stderr)
        for g in range(n_groups):
            for cfg, cfg_label in ((b_cfg, baseline_label), (s_cfg, scheduled_label)):
                if len(all_group_results[cfg_label]) > g:
                    continue  # already ran this group for this label (label appears in >1 pair)
                shape_m, shape_n, shape_k = shape_dims(cfg["shape"])
                dump_path = f"{pi_dir}/{cfg_label}_out_g{g}.bin" if g == 0 else None
                try:
                    r = run_harness(
                        args.pi_host, cfg["build"]["remote_binary"], shape_m, shape_n, shape_k,
                        cfg["tile"]["m"], cfg["tile"]["n"], cfg["tile"]["k"], cfg["schedule_unroll_k"],
                        cfg_label, cfg["iterations"], cfg["warmup"], cfg["repeated_calls"],
                        "0x1234", "0x5eed", dump_output_path=dump_path, taskset_core=args.taskset_core,
                    )
                    all_group_results[cfg_label].append(r)
                except Exception as e:
                    errors[cfg_label] = str(e)
                    print(f"  RUN FAILED [{cfg_label} group {g}]: {e}", file=sys.stderr)

    for label, cfg in built.items():
        if label in paired_labels or all_group_results[label]:
            continue
        print(f"[run standalone] {label} ...", file=sys.stderr)
        shape_m, shape_n, shape_k = shape_dims(cfg["shape"])
        try:
            r = run_harness(
                args.pi_host, cfg["build"]["remote_binary"], shape_m, shape_n, shape_k,
                cfg["tile"]["m"], cfg["tile"]["n"], cfg["tile"]["k"], cfg["schedule_unroll_k"],
                label, cfg["iterations"], cfg["warmup"], cfg["repeated_calls"],
                "0x1234", "0x5eed", dump_output_path=f"{pi_dir}/{label}_out_g0.bin", taskset_core=args.taskset_core,
            )
            all_group_results[label].append(r)
        except Exception as e:
            errors[label] = str(e)
            print(f"  RUN FAILED [{label}]: {e}", file=sys.stderr)

    capture_thermal_snapshot(args.pi_host, "after_all", args.output_dir)

    results = {}
    for label, cfg in built.items():
        group_results = all_group_results[label]
        if not group_results:
            continue
        record = {
            "label": label, "shape": cfg["shape"], "tile": cfg["tile"],
            "schedule_unroll_k": cfg["schedule_unroll_k"], "group": cfg["group"], "full_rigor": cfg["full_rigor"],
            "compiled": {k: v for k, v in cfg["compiled"].items() if k != "command"},
            "compile_command": cfg["compiled"]["command"],
            "build": cfg["build"],
            "measurement_groups": group_results,
            "correctness_pass": all(g["correctness"]["overall_pass"] for g in group_results),
            "benchmark_aggregate": aggregate_measurement_groups([g["benchmark"] for g in group_results]),
            "stage12_key": cfg["s12_key"],
            "stage12_evidence": (stage12["candidates"].get(cfg["s12_key"]) if stage12 and cfg["s12_key"] else None),
            "config_for_matching": {
                "shape": cfg["shape"], "tile": cfg["tile"], "target_cpu": TARGET_CPU,
                "opt_level": "O2", "iterations": cfg["iterations"], "warmup": cfg["warmup"],
                "git_commit": env_before["git_commit"], "schedule_unroll_k": cfg["schedule_unroll_k"],
            },
        }
        with open(os.path.join(cfg["cand_dir"], "manifest.json"), "w") as f:
            json.dump(record, f, indent=2)
        results[label] = record

    # ---- Output-order comparison for the primary pair (task section 5) ----
    output_order_comparison = None
    if "primary_uk1" in results and "primary_uk2" in results:
        p1 = ssh_run(args.pi_host, f"sha256sum '{pi_dir}/primary_uk1_out_g0.bin' 2>/dev/null | cut -d' ' -f1", check=False).stdout.strip()
        p2 = ssh_run(args.pi_host, f"sha256sum '{pi_dir}/primary_uk2_out_g0.bin' 2>/dev/null | cut -d' ' -f1", check=False).stdout.strip()
        output_order_comparison = {
            "baseline_output_sha256": p1, "scheduled_output_sha256": p2,
            "bitwise_identical": bool(p1) and (p1 == p2),
            "note": (
                "bitwise-identical output buffers, given both candidates share the identical "
                "deterministic input seed and Stage 12 found no fast-math flags anywhere in the "
                "matrix, would corroborate that both execute the same reduction order at runtime, "
                "not just that both happen to be within the 1e-3 correctness tolerance of the "
                "scalar reference"
            ),
        }

    # ---- Comparisons + classification (task sections 7, 10) ----
    comparisons = {}
    for baseline_label, scheduled_label in COMPARISON_PAIRS:
        if scheduled_label is None:
            continue  # single-candidate correctness-only entry (small_control), no comparison
        if baseline_label not in results or scheduled_label not in results:
            continue
        b, s = results[baseline_label], results[scheduled_label]
        try:
            assert_matched_comparison(b["config_for_matching"], s["config_for_matching"])
            match_ok, match_reason = True, "configuration matched"
        except MismatchedComparisonError as e:
            match_ok, match_reason = False, str(e)

        b_spills = (b["stage12_evidence"]["register_allocation"]["comparison"]["spill_stores_inserted_by_ra"]
                    if b["stage12_evidence"] else 0)
        s_spills = (s["stage12_evidence"]["register_allocation"]["comparison"]["spill_stores_inserted_by_ra"]
                    if s["stage12_evidence"] else 0)

        cls, cls_reason = (None, "comparison rejected: " + match_reason) if not match_ok else classify_runtime(
            b["benchmark_aggregate"], s["benchmark_aggregate"],
            b["correctness_pass"], s["correctness_pass"], b_spills, s_spills,
        )
        comparisons[f"{baseline_label}_vs_{scheduled_label}"] = {
            "baseline": baseline_label, "scheduled": scheduled_label,
            "matched_comparison": match_ok, "match_reason": match_reason,
            "runtime_classification": cls, "runtime_classification_meaning": RUNTIME_CLASS_MEANING.get(cls),
            "classification_reason": cls_reason,
            "speedup_median_of_medians": (
                b["benchmark_aggregate"]["median_of_medians_ms"] / s["benchmark_aggregate"]["median_of_medians_ms"]
                if match_ok and s["benchmark_aggregate"]["median_of_medians_ms"] else None
            ),
        }

    # ---- Spill-prediction validation table (task section 9) ----
    spill_validation = {}
    for baseline_label, diag_label in DIAGNOSTIC_PAIRS:
        if baseline_label not in results or diag_label not in results:
            continue
        d = results[diag_label]
        s12 = d["stage12_evidence"]
        if not s12:
            continue
        try:
            assert_matched_comparison(results[baseline_label]["config_for_matching"], d["config_for_matching"])
        except MismatchedComparisonError as e:
            spill_validation[diag_label] = {"matched_baseline": baseline_label, "prediction_outcome": "rejected", "reason": str(e)}
            continue
        spills = s12["register_allocation"]["comparison"]["spill_stores_inserted_by_ra"]
        reloads = s12["register_allocation"]["comparison"]["reload_loads_inserted_by_ra"]
        stack_bytes = s12["register_allocation"]["final_stack_frame_bytes"]
        b_med = results[baseline_label]["benchmark_aggregate"]["median_of_medians_ms"]
        d_med = d["benchmark_aggregate"]["median_of_medians_ms"]
        outcome, reason = validate_spill_prediction(diag_label, spills, reloads, stack_bytes, b_med, d_med, d["correctness_pass"])
        spill_validation[diag_label] = {
            "matched_baseline": baseline_label,
            "stage12_spill_stores": spills, "stage12_reload_loads": reloads, "stage12_stack_frame_bytes": stack_bytes,
            "baseline_median_ms": b_med, "diagnostic_median_ms": d_med,
            "relative_latency_pct": ((b_med - d_med) / b_med * 100.0) if b_med else None,
            "correctness_pass": d["correctness_pass"],
            "prediction_outcome": outcome, "reason": reason,
        }

    summary = {
        "stage": "Stage 13: Raspberry Pi correctness and controlled runtime validation",
        "pi_environment": env_before,
        "candidates": results,
        "candidate_errors": errors,
        "output_order_comparison_primary": output_order_comparison,
        "comparisons": comparisons,
        "spill_prediction_validation": spill_validation,
    }
    out_json = os.path.join(args.output_dir, "pi_validation_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} candidate(s) FAILED: {list(errors.keys())}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
