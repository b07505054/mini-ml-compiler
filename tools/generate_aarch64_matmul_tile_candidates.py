#!/usr/bin/env python3
"""generate_aarch64_matmul_tile_candidates.py

Stage 3/4/5/6 driver for the AArch64 tile-candidate selection slice.

For every (shape, tile) combination in the required candidate/shape sets:
  1. Runs the static legality analysis (shape%tile divisibility; an
     analytical worst-case vector-register-demand estimate, checked against
     a configured hard limit but NOT used to reject any of the required
     candidates -- see the "register_estimate" field and README.md for why).
  2. If divisibility-legal, compiles the candidate through the real,
     unmodified LLVM AArch64 backend via
     mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant
     tiled-vectorized --tile-m/--tile-n/--tile-k (object naming:
     matmul_<M>x<N>x<K>_tm<TM>_tn<TN>_tk<TK>.o). A compile failure at this
     stage is recorded as legal=false, reason="MLIR transformation
     failure" (empirically discovered, not assumed).
  3. For every successfully compiled candidate, collects structural MLIR
     evidence (pre-bufferization: scf.for count, loop steps, largest vector
     type, vector.contract/transfer_read/transfer_write counts) and LLVM
     IR/AArch64 backend metrics (instruction/load/store/branch/FMLA counts,
     object and text-section size).

Does NOT run anything on the Raspberry Pi -- that is a separate step (see
tools/run_aarch64_matmul_tile_candidates_pi_integration.sh) whose output is
merged into the same candidate_results.json afterwards.

Usage:
  python3 tools/generate_aarch64_matmul_tile_candidates.py \
    --output-dir /tmp/tile_candidate_objects \
    --results /tmp/tile_candidate_static_results.json
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPILE_SCRIPT = os.path.join(
    REPO_ROOT, "mlir_passes", "tools", "compile_hir_matmul_bias_relu_aarch64.sh"
)
FIXTURE_DIR = os.path.join(REPO_ROOT, "mlir_passes", "test", "backend_codegen")
MLIR_OPT = "/home/allen/Desktop/Project/.deps/mlir21-root/usr/lib/llvm-21/bin/mlir-opt"
PLUGIN = os.path.join(REPO_ROOT, "build-mlir", "libHIRMatMulBiasReluFusionPass.so")
TILE_TEMPLATE = os.path.join(
    REPO_ROOT, "mlir_passes", "transforms",
    "tile_vectorize_matmul_bias_relu.template.mlir",
)
GENERATE_TRANSFORM = os.path.join(
    REPO_ROOT, "mlir_passes", "tools", "generate_tiled_transform.sh"
)

# The 7 required candidates, verbatim from the task brief.
CANDIDATES = [
    (4, 4, 4),
    (4, 8, 4),
    (4, 8, 8),
    (8, 4, 4),
    (8, 4, 8),
    (8, 8, 4),
    (8, 8, 8),
]

# The 5 required shapes, plus the optional 8x8x8 (cheap, already has a
# fixture from the prior tiled slice).
SHAPES = [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (32, 64, 32),
    (64, 32, 64),
    (8, 8, 8),
]

# Analytical worst-case vector-register-demand estimate: accumulator
# (TM * ceil(TN/4)) + B operand (ceil(TN/4)) + A broadcasts, worst case one
# live per M-row (TM). This is a pre-codegen design estimate, not a
# measurement -- see Stage 7 for real assembly-derived evidence, which is
# the actual selection signal. Hard limit is deliberately set at the full
# 32-register architectural file: the prior tiled slice already showed
# LLVM's own scheduler can legitimately use all 32 registers with zero
# spills (K-sub-step interleaving for ILP), so a tighter analytical cap
# would reject candidates that are, in measured reality, perfectly fine.
REGISTER_HARD_LIMIT = 32


def ceil_div(a, b):
    return (a + b - 1) // b


def register_estimate(tm, tn):
    acc = tm * ceil_div(tn, 4)
    b_regs = ceil_div(tn, 4)
    a_broadcasts_worst = tm
    return acc + b_regs + a_broadcasts_worst


def static_legality(shape, tile):
    m, n, k = shape
    tm, tn, tk = tile
    reasons = []
    if m % tm != 0:
        reasons.append(f"M={m} not divisible by TM={tm}")
    if n % tn != 0:
        reasons.append(f"N={n} not divisible by TN={tn}")
    if k % tk != 0:
        reasons.append(f"K={k} not divisible by TK={tk}")
    est = register_estimate(tm, tn)
    if est > REGISTER_HARD_LIMIT:
        reasons.append(
            f"estimated register demand {est} exceeds configured hard limit {REGISTER_HARD_LIMIT}"
        )
    return {
        "legal": len(reasons) == 0,
        "rejection_reasons": reasons,
        "register_estimate": {
            "worst_case_registers": est,
            "hard_limit": REGISTER_HARD_LIMIT,
            "note": "pre-codegen analytical estimate (accumulator + B operand + worst-case A broadcasts); not a measurement -- see backend.vector_registers_referenced for assembly-derived evidence",
        },
    }


def fixture_path(shape):
    m, n, k = shape
    return os.path.join(FIXTURE_DIR, f"matmul_bias_relu_tiled_{m}x{n}x{k}.mlir")


def object_name(shape, tile):
    m, n, k = shape
    tm, tn, tk = tile
    return f"matmul_{m}x{n}x{k}_tm{tm}_tn{tn}_tk{tk}"


def renamed_fixture(shape, tile, output_dir):
    """Every candidate for a given shape is compiled from the SAME shape
    fixture (matmul_bias_relu_tiled_<shape>.mlir), which always declares
    func.func @matmul_bias_relu_tiled_<shape> -- the tile size is a
    compile-time transform parameter, not part of that name. Left as-is,
    every candidate for one shape would export the IDENTICAL
    _mlir_ciface_ symbol, which is fine standalone but collides at link
    time the moment two candidates for the same shape are linked into one
    binary (required for Stage 8's mixed-candidate stress test). Generates
    a scratch copy of the fixture with the function renamed to embed the
    tile (matmul_bias_relu_tiled_<shape>_tm<M>_tn<N>_tk<K>), so every
    candidate's object exports a unique symbol."""
    m, n, k = shape
    tm, tn, tk = tile
    old_name = f"matmul_bias_relu_tiled_{m}x{n}x{k}"
    new_name = f"{old_name}_tm{tm}_tn{tn}_tk{tk}"
    src = fixture_path(shape)
    if not os.path.isfile(src):
        return None
    text = open(src).read().replace(f"@{old_name}", f"@{new_name}")
    dst = os.path.join(output_dir, f"{new_name}_input.mlir")
    with open(dst, "w") as f:
        f.write(text)
    return dst, new_name


def compile_candidate(shape, tile, output_dir):
    tm, tn, tk = tile
    name = object_name(shape, tile)
    renamed = renamed_fixture(shape, tile, output_dir)
    if renamed is None:
        return False, f"fixture not found for shape {shape}", None
    fixture, func_name = renamed
    cmd = [
        "bash", COMPILE_SCRIPT,
        "--variant", "tiled-vectorized",
        "--tile-m", str(tm), "--tile-n", str(tn), "--tile-k", str(tk),
        fixture, output_dir, name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        return False, "MLIR transformation failure: " + " | ".join(tail), None
    return True, None, name


def structural_metrics(shape, tile, output_dir, name):
    """Pre-bufferization vector-dialect intermediate: scf.for structure and
    bounded vector types, generated the same way the compile script's own
    tiled pipeline does up through vectorization (stopping before
    bufferization/LLVM lowering)."""
    fixture = fixture_path(shape)
    tm, tn, tk = tile
    tmp_transform = os.path.join(output_dir, f"{name}_transform.mlir")
    subprocess.run(
        ["bash", GENERATE_TRANSFORM, "--tile-m", str(tm), "--tile-n", str(tn),
         "--tile-k", str(tk), "--output", tmp_transform],
        check=True, capture_output=True, text=True,
        env={**os.environ, "TEMPLATE": TILE_TEMPLATE},
    )
    vector_mlir = os.path.join(output_dir, f"{name}_vector.mlir")
    cmd = [
        MLIR_OPT, fixture,
        f"--load-dialect-plugin={PLUGIN}", f"--load-pass-plugin={PLUGIN}",
        "--pass-pipeline=builtin.module(hir-matmul-bias-relu-to-linalg,"
        f"transform-preload-library{{transform-library-paths={tmp_transform}}},"
        "transform-interpreter{entry-point=__transform_main})",
        "-o", vector_mlir,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {"error": "structural MLIR generation failed: " + proc.stderr.strip()[-300:]}

    text = open(vector_mlir).read()
    scf_for_count = len(re.findall(r"\bscf\.for\b", text))
    vector_types = re.findall(r"vector<([0-9x]+)xf32>", text)
    largest = 0
    for vt in vector_types:
        dims = [int(d) for d in vt.split("x") if d]
        lanes = 1
        for d in dims:
            lanes *= d
        largest = max(largest, lanes)
    return {
        "scf_for_count": scf_for_count,
        "vector_contract_count": len(re.findall(r"\bvector\.contract\b", text)),
        "vector_transfer_read_count": len(re.findall(r"\bvector\.transfer_read\b", text)),
        "vector_transfer_write_count": len(re.findall(r"\bvector\.transfer_write\b", text)),
        "largest_vector_lanes": largest,
        "bias_relu_present": ("arith.addf" in text and "arith.maximumf" in text),
        "bare_uninitialized_tensor_empty_accumulator": bool(
            re.search(r"tensor\.empty\(\)[^\n]*\n\s*%\d+ = linalg\.matmul", text)
        ),
    }


def backend_metrics(output_dir, name):
    obj = os.path.join(output_dir, f"{name}.o")
    ll = os.path.join(output_dir, f"{name}.ll")
    s = os.path.join(output_dir, f"{name}.s")

    obj_size = os.path.getsize(obj)
    text_hex = subprocess.run(
        ["llvm-objdump", "-h", obj], capture_output=True, text=True
    ).stdout
    text_bytes = None
    for line in text_hex.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == ".text":
            text_bytes = int(parts[2], 16)
            break

    disasm = subprocess.run(
        ["llvm-objdump", "-d", "--no-show-raw-insn", obj], capture_output=True, text=True
    ).stdout
    mnemonics = []
    for line in disasm.splitlines():
        line = line.strip()
        if re.match(r"^[0-9a-f]+:", line):
            parts = line.split("\t")
            if len(parts) >= 2:
                mnemonics.append(parts[1].strip())
    total = len(mnemonics)
    loads = sum(1 for m in mnemonics if m in ("ldr", "ldp", "ldur", "ld1"))
    stores = sum(1 for m in mnemonics if m in ("str", "stp", "stur", "st1"))
    branches = sum(1 for m in mnemonics if m in ("b", "bl") or re.match(r"^b\.[a-z]+$", m) or m in ("cbz", "cbnz"))
    fmla = sum(1 for m in mnemonics if m == "fmla")

    ll_text = open(ll).read()
    llvm_ir_instrs = len(re.findall(r" = |^\s*(store|br|ret|call)\b", ll_text, re.M))
    vec_widths = [int(x) for x in re.findall(r"<(\d+) x float>", ll_text)]
    largest_vw = max(vec_widths) if vec_widths else 0

    s_text = open(s).read()
    stack_frame = None
    m = re.search(r"sub\s+sp,\s*sp,\s*#(\d+)", s_text)
    if m:
        stack_frame = int(m.group(1))

    return {
        "object_bytes": obj_size,
        "text_bytes": text_bytes,
        "aarch64_instructions": total,
        "llvm_ir_instructions": llvm_ir_instrs,
        "loads": loads,
        "stores": stores,
        "branches": branches,
        "static_fmla": fmla,
        "largest_llvm_vector_width": largest_vw,
        "stack_frame_bytes": stack_frame,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for shape in SHAPES:
        for tile in CANDIDATES:
            legality = static_legality(shape, tile)
            record = {
                "shape": list(shape),
                "tile": {"m": tile[0], "n": tile[1], "k": tile[2]},
                "legality": legality,
            }
            if not legality["legal"]:
                results.append(record)
                print(f"REJECTED {shape} tile={tile}: {legality['rejection_reasons']}")
                continue

            ok, err, name = compile_candidate(shape, tile, args.output_dir)
            if not ok:
                record["legality"] = {
                    "legal": False,
                    "rejection_reasons": [err],
                    "register_estimate": legality["register_estimate"],
                }
                results.append(record)
                print(f"COMPILE FAILED {shape} tile={tile}: {err}")
                continue

            m, n, k = shape
            tm, tn, tk = tile
            record["object_name"] = name
            record["function_name"] = f"matmul_bias_relu_tiled_{m}x{n}x{k}_tm{tm}_tn{tn}_tk{tk}"
            record["ciface_symbol"] = f"_mlir_ciface_{record['function_name']}"
            record["structural"] = structural_metrics(shape, tile, args.output_dir, name)
            record["backend"] = backend_metrics(args.output_dir, name)
            results.append(record)
            print(f"OK {shape} tile={tile} -> {name} "
                  f"({record['backend']['object_bytes']} bytes, "
                  f"{record['backend']['static_fmla']} fmla)")

    with open(args.results, "w") as f:
        json.dump({"candidates": results}, f, indent=2)
    print(f"\nWrote {len(results)} candidate records to {args.results}")


if __name__ == "__main__":
    main()
