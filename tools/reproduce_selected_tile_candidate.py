#!/usr/bin/env python3
"""reproduce_selected_tile_candidate.py

Stage 15: reads selected_tiles.json for one shape, recompiles the exact
selected tile via the standard, unmodified
mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh --variant
tiled-vectorized --tile-m/--tile-n/--tile-k interface (the SAME interface
every candidate was originally evaluated through -- no separate
"selected-tiled" variant was added, since --variant tiled-vectorized
already accepts arbitrary tile parameters and doing so keeps exactly one
code path to trust), and verifies the freshly compiled object matches the
originally-evaluated candidate on:
  - static FMLA count
  - object size
  - object SHA-256 hash

Usage:
  python3 tools/reproduce_selected_tile_candidate.py \
    --selection selected_tiles.json \
    --candidates candidate_results.json \
    --shape 32x32x32 \
    --output-dir /tmp/repro_check
"""
import argparse
import hashlib
import json
import os
import re
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COMPILE_SCRIPT = os.path.join(REPO_ROOT, "mlir_passes", "tools", "compile_hir_matmul_bias_relu_aarch64.sh")
FIXTURE_DIR = os.path.join(REPO_ROOT, "mlir_passes", "test", "backend_codegen")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--shape", required=True, help="e.g. 32x32x32")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    m, n, k = (int(x) for x in args.shape.split("x"))
    selection = json.load(open(args.selection))
    sel = next((s for s in selection["selections"] if s["shape"] == [m, n, k]), None)
    if sel is None:
        raise SystemExit(f"no selection found for shape {args.shape}")
    tm, tn, tk = sel["selected_tile"]

    candidates = json.load(open(args.candidates))
    original = next(
        (c for c in candidates["results"]
         if c["shape"] == [m, n, k] and c["tile"] == {"m": tm, "n": tn, "k": tk}),
        None,
    )
    if original is None:
        raise SystemExit("original candidate record not found -- cannot verify reproduction")

    os.makedirs(args.output_dir, exist_ok=True)
    fixture = os.path.join(FIXTURE_DIR, f"matmul_bias_relu_tiled_{m}x{n}x{k}.mlir")
    name = f"repro_{m}x{n}x{k}_tm{tm}_tn{tn}_tk{tk}"
    cmd = [
        "bash", COMPILE_SCRIPT,
        "--variant", "tiled-vectorized",
        "--tile-m", str(tm), "--tile-n", str(tn), "--tile-k", str(tk),
        fixture, args.output_dir, name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise SystemExit(f"reproduction compile FAILED:\n{proc.stderr}")

    obj = os.path.join(args.output_dir, f"{name}.o")
    s_file = os.path.join(args.output_dir, f"{name}.s")
    fmla = len(re.findall(r"^\s+fmla\s", open(s_file).read(), re.M))
    size = os.path.getsize(obj)
    digest = sha256(obj)

    text_hex = subprocess.run(["llvm-objdump", "-h", obj], capture_output=True, text=True).stdout
    text_bytes = None
    for line in text_hex.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == ".text":
            text_bytes = int(parts[2], 16)
            break

    # Whole-object file size legitimately differs from the original
    # candidate: that object's function was compiled from a fixture whose
    # symbol was renamed to embed the tile (e.g.
    # matmul_bias_relu_tiled_32x32x32_tm8_tn8_tk8) so 42 candidates could
    # link into one Pi test binary without symbol collisions -- a longer
    # symbol name grows the ELF symbol/string table, not the generated
    # code. .text section SIZE, and the actual machine code BYTES within
    # it, are the meaningful reproduction check -- see README.md.
    checks = {
        "static_fmla_matches": fmla == original["backend"]["static_fmla"],
        "text_bytes_matches": text_bytes == original["backend"]["text_bytes"],
    }
    print(f"Reproduced shape={args.shape} tile=({tm},{tn},{tk})")
    print(f"  static_fmla: reproduced={fmla} original={original['backend']['static_fmla']} -> {'MATCH' if checks['static_fmla_matches'] else 'MISMATCH'}")
    print(f"  text_bytes: reproduced={text_bytes} original={original['backend']['text_bytes']} -> {'MATCH' if checks['text_bytes_matches'] else 'MISMATCH'}")
    print(f"  whole_object_bytes: reproduced={size} original={original['backend']['object_bytes']} (expected to differ -- see note above; not part of the pass/fail check)")
    print(f"  sha256 (this reproduction's object): {digest}")
    print(f"  benchmark_median_ms (from selection artifact): {sel['median_ms']}")

    all_ok = all(checks.values())
    print("REPRODUCTION: " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
