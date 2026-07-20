#!/usr/bin/env python3
"""Compile every applicable production candidate and benchmark it on Pi 5."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILE = ROOT / "mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh"
HARNESS_TEMPLATE = ROOT / "mlir_passes/tools/aarch64_matmul_bias_relu_schedule_harness.cpp.template"
VARIANTS = {
    "fused_scalar": "generic",
    "whole_shape_vector_no_padding": "vectorized",
    "whole_shape_vector_materialized_padding": "vectorized",
    "tiled_vector_full_tiles": "tiled-vectorized",
    "tiled_vector_materialized_tail": "tiled-vectorized-materialized-tail",
    "tiled_vector_direct_k": "tiled-vectorized-direct-k-tail",
}


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def applicable(cid, m, n, k):
    if cid.startswith("whole_shape_vector"):
        padded = cid == "whole_shape_vector_materialized_padding"
        pm, pn, pk = (((m + 7) // 8 * 8, (n + 7) // 8 * 8,
                       (k + 7) // 8 * 8) if padded else (m, n, k))
        if 512 + pm * pn * pk * 4 > 16384:
            return False
    if cid == "whole_shape_vector_materialized_padding":
        return bool(m % 8 or n % 8 or k % 8)
    if cid == "tiled_vector_full_tiles":
        return m % 8 == 0 and n % 8 == 0 and k % 8 == 0
    if cid == "tiled_vector_materialized_tail":
        return bool(m % 8 or n % 8 or k % 8)
    if cid == "tiled_vector_direct_k":
        return m % 8 == 0 and n % 8 == 0 and k % 8 != 0
    return True


def iterations_for(m, n, k):
    estimate_ns = 80 + 2 * m * n * k / 12
    if estimate_ns < 1000:
        return 20000
    if estimate_ns < 10000:
        return 10000
    if estimate_ns < 100000:
        return 2000
    return 500


def fixture(shape, path, candidate_id):
    m, n, k = shape
    token = f"{m}x{n}x{k}"
    padding_attrs = 'target.padding = "none"'
    if candidate_id == "whole_shape_vector_materialized_padding":
        padding_attrs = f'''target.original_m = {m} : i64,
    target.original_n = {n} : i64,
    target.original_k = {k} : i64,
    target.padded_m = {(m + 7) // 8 * 8} : i64,
    target.padded_n = {(n + 7) // 8 * 8} : i64,
    target.padded_k = {(k + 7) // 8 * 8} : i64,
    target.padding = "pad_to_tile_with_crop",
    target.valid_region = "original_m_n"'''
    path.write_text(f"""func.func @matmul_bias_relu_tiled_{token}(
  %lhs: tensor<{m}x{k}xf32>, %rhs: tensor<{k}x{n}xf32>,
  %bias: tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  attributes {{llvm.emit_c_interface}} {{
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {{
    fusion.candidate = "matmul_bias_relu",
    kernel.selection = "native_cpu",
    lowering.source = "linalg.matmul_add_relu",
    {padding_attrs}
  }} : (tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>, tensor<{m}x{n}xf32>)
      -> tensor<{m}x{n}xf32>
  return %0 : tensor<{m}x{n}xf32>
}}
""")


def extract_json(text):
    begin = text.find("{")
    if begin < 0:
        raise ValueError("native runner emitted no JSON")
    return json.loads(text[begin:])


def metrics(work, name):
    obj = work / f"{name}.o"
    asm = (work / f"{name}.s").read_text()
    llvm_mlir = work / f"{name}_llvm.mlir"
    llvm_ir = work / f"{name}.ll"
    instructions = sum(bool(re.match(r"^\s+[a-z][a-z0-9.]+", line))
                       for line in asm.splitlines())
    branches = len(re.findall(r"^\s+b(?:\.[a-z]+|l|r)?\s", asm, re.M))
    size_proc = run(["llvm-size-21", "-A", str(obj)])
    text_bytes = None
    match = re.search(r"\\.text\\s+(\\d+)", size_proc.stdout)
    if match:
        text_bytes = int(match.group(1))
    return {
        "actual_llvm_mlir_bytes": llvm_mlir.stat().st_size,
        "actual_llvm_ir_bytes": llvm_ir.stat().st_size,
        "actual_object_text_bytes": text_bytes,
        "actual_static_instruction_count": instructions,
        "actual_fmla_count": len(re.findall(r"\bfmla\b", asm)),
        "actual_branch_count": branches,
        "actual_stack_frame_bytes": None, "actual_spills": None,
        "object_hash": sha(obj),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", required=True)
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pi-host", default="allen@100.110.37.6")
    ap.add_argument("--pi-dir", default="/tmp/ml_graph_candidate_dataset_v1")
    ap.add_argument("--affinity", default="3")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    registry = json.loads(Path(args.registry).read_text())
    candidates = [c for c in registry["candidates"]
                  if c["lowering_complete"] and c["native_executable"]
                  and c["candidate_id"] in VARIANTS]
    shape_config = json.loads(Path(args.shapes).read_text())
    if shape_config.get("shape_order") != "M,K,N":
        raise SystemExit("shape matrix must explicitly declare shape_order=M,K,N")
    shapes = [(s[0], s[2], s[1]) for s in shape_config["shapes"]]
    tasks = [(s, c) for s in shapes for c in candidates
             if applicable(c["candidate_id"], *s)]
    if args.limit:
        tasks = tasks[:args.limit]
    output = Path(args.output)
    completed = set()
    retained_rows = []
    if args.resume and output.exists():
        for line in output.read_text().splitlines():
            row = json.loads(line)
            if row.get("execution_status") == "success":
                completed.add((row["shape_group_id"], row["candidate_id"]))
                retained_rows.append(row)
    env_proc = run(["ssh", "-F", "/dev/null", args.pi_host,
                    "uname -a; cat /proc/device-tree/model; echo; "
                    "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; "
                    "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq; "
                    "g++ --version | head -1"], timeout=30)
    if env_proc.returncode:
        raise SystemExit(env_proc.stderr)
    print(env_proc.stdout, flush=True)
    run(["ssh", "-F", "/dev/null", args.pi_host, f"mkdir -p {args.pi_dir}"],
        check=True)
    with output.open("w") as stream, tempfile.TemporaryDirectory(
            prefix="candidate_dataset_") as td:
        for row in retained_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        work = Path(td)
        for index, (shape, candidate) in enumerate(tasks, 1):
            m, n, k = shape
            cid = candidate["candidate_id"]
            sid = f"m{m}_n{n}_k{k}_f32"
            if (sid, cid) in completed:
                continue
            name = f"matmul_bias_relu_tiled_{m}x{n}x{k}"
            mlir = work / f"{name}.mlir"
            fixture(shape, mlir, cid)
            variant = VARIANTS[cid]
            cmd = ["bash", str(COMPILE), "--variant", variant]
            if variant.startswith("tiled-"):
                cmd += ["--tile-m", "8", "--tile-n", "8", "--tile-k", "8"]
            cmd += [str(mlir), str(work), name]
            started = time.perf_counter()
            compiled = run(cmd, cwd=ROOT)
            compile_ms = (time.perf_counter() - started) * 1000
            base = {
                "shape_group_id": sid, "candidate_id": cid,
                "execution_status": "compile_failed",
                "correctness_pass": False, "sentinel_pass": False,
                "sanitizer_pass": None, "compile_time_ms": compile_ms,
                "compiler_peak_rss_kib": None,
            }
            if compiled.returncode:
                base["failure_message"] = compiled.stderr[-2000:]
                stream.write(json.dumps(base, sort_keys=True) + "\n")
                stream.flush()
                print(f"[{index}/{len(tasks)}] {sid} {cid} compile_failed",
                      flush=True)
                continue
            base.update(metrics(work, name))
            token = f"{m}x{n}x{k}"
            harness = work / f"harness_{cid}_{token}.cpp"
            harness.write_text(HARNESS_TEMPLATE.read_text().replace("SHAPE_TOKEN", token))
            remote = f"{args.pi_dir}/{cid}_{token}"
            copied = run(["scp", "-F", "/dev/null", str(work / f"{name}.o"),
                          str(harness), f"{args.pi_host}:{args.pi_dir}/"])
            if copied.returncode:
                base["execution_status"] = "runtime_failed"
                base["failure_message"] = copied.stderr[-2000:]
                stream.write(json.dumps(base, sort_keys=True) + "\n")
                stream.flush()
                continue
            iterations = iterations_for(m, n, k)
            remote_cmd = (
                f"cd {args.pi_dir} && g++ -O3 -std=c++17 "
                f"{harness.name} {name}.o -o {cid}_{token} && "
                f"sha256sum {cid}_{token} && "
                f"taskset -c {args.affinity} ./{cid}_{token} "
                f"--shape-m {m} --shape-n {n} --shape-k {k} "
                f"--tile-m 8 --tile-n 8 --tile-k 8 --schedule-unroll-k 1 "
                f"--candidate-label {cid} --iterations {iterations} "
                f"--warmup 100 --repeated-calls 1000")
            native = run(["ssh", "-F", "/dev/null", args.pi_host, remote_cmd],
                         timeout=600)
            if native.returncode:
                base["execution_status"] = "runtime_failed"
                base["failure_message"] = (native.stdout + native.stderr)[-2000:]
            else:
                result = extract_json(native.stdout)
                binary_match = re.search(r"^([0-9a-f]{64})\s", native.stdout)
                if not binary_match:
                    raise RuntimeError("native runner did not report binary hash")
                correct = result["correctness"]
                bench = result["benchmark"]
                trials = [correct["deterministic_trial"],
                          correct["random_seed_trial"]]
                base.update({
                    "execution_status": "success",
                    "correctness_pass": correct["overall_pass"],
                    "sentinel_pass": correct["overall_pass"],
                    "warmup_calls": bench["warmup"],
                    "measured_calls": bench["iterations"],
                    "median_ns": bench["median_ms"] * 1e6,
                    "p95_ns": bench["p95_ms"] * 1e6,
                    "mad_ns": bench.get("mad_ms", bench["stddev_ms"]) * 1e6,
                    "min_ns": bench["min_ms"] * 1e6,
                    "max_ns": (bench["max_ms"] * 1e6
                               if "max_ms" in bench else None),
                    "checksum": hashlib.sha256(
                        json.dumps(result, sort_keys=True).encode()).hexdigest(),
                    "max_absolute_error": max(t["max_abs_error"] for t in trials),
                    "max_relative_error": max(t["max_rel_error"] for t in trials),
                    "cpu_affinity": args.affinity, "governor": "ondemand",
                    "current_frequency_khz": None,
                    "compiler_flags": "-O3 -mtriple=aarch64-linux-gnu -mcpu=cortex-a76",
                    "compiler_version": "Pi g++ and LLVM 21 captured by runner",
                    "mlir_version": "MLIR 21",
                    "binary_hash": binary_match.group(1),
                })
            stream.write(json.dumps(base, sort_keys=True) + "\n")
            stream.flush()
            print(f"[{index}/{len(tasks)}] {sid} {cid} {base['execution_status']}",
                  flush=True)


if __name__ == "__main__":
    main()
