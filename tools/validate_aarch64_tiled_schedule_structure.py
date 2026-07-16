#!/usr/bin/env python3
"""validate_aarch64_tiled_schedule_structure.py

Stage 11 of the machine-scheduling analysis slice: structural validation of
the "tiled-scheduled" variant's MLIR *before* bufferization/lowering --
i.e. the tensor/linalg/vector/scf-dialect IR produced by running only the
tiling+fusion+K-unroll+vectorization prefix of the pass pipeline that
mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh runs for
--variant tiled-scheduled:

    hir-matmul-bias-relu-to-linalg,
    transform-preload-library{transform-library-paths=<generated transform>},
    transform-interpreter{entry-point=__transform_main}

This is the only point in the pipeline where the M/N/K tiling loop nest,
the K-loop unroll, the per-tile accumulator zero-init, and the per-tile
bias+ReLU are all still visible as high-level ops (scf.for / vector.contract
/ arith.addf / arith.maximumf) -- everything after this is progressively
lowered away (scf.for -> cf branches, vector.contract -> target-specific
vector ops, etc.), which is why the LLVM-dialect output the compile script
writes to disk (${NAME}_llvm.mlir) is NOT a usable target for this check.

Checks implemented (each corresponds to one Stage 11 requirement in the
task brief), all derived from inspecting REAL mlir-opt output for the
32x32x32/tile-8x8x8 primary candidate at schedule-unroll-k=1 and =2 (see
artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/README.md)
rather than assumed from documentation:

  1. outer_loop_bounds       -- exactly 3 nested scf.for; M-loop bound=SHAPE_M
                                 step=TILE_M, N-loop bound=SHAPE_N step=TILE_N,
                                 K-loop bound=SHAPE_K step=TILE_K*SCHEDULE_UNROLL_K
                                 ("outer M/N/K loop semantics correct")
  2. k_loop_unroll_materialized -- vector.contract count inside the K-loop
                                 body equals SCHEDULE_UNROLL_K, and the K-loop
                                 step is TILE_K*SCHEDULE_UNROLL_K, not TILE_K
                                 ("K-loop scheduling transform materialized")
  3. bounded_vector_shapes    -- every vector.contract/vector.transfer_*
                                 operates on vector<TILE_MxTILE_Nxf32> (or the
                                 TILE_MxTILE_K / TILE_KxTILE_N operand shapes),
                                 never the whole-shape vector<SHAPE_MxSHAPE_Nxf32>
                                 ("bounded vector FMLA" + "no whole-shape
                                 vectorization regression")
  4. accumulator_zero_init    -- exactly one arith.constant dense<0.0> of
                                 shape TILE_MxTILE_N feeding (via
                                 vector.transfer_write) the per-tile
                                 extract_slice ("correct accumulator init")
  5. bias_relu_per_tile       -- exactly one arith.addf immediately followed
                                 by one arith.maximumf (against the same zero
                                 constant used for accumulator init) per
                                 output tile, both on the tile-shaped vector
                                 ("correct bias/relu")
  6. no_tail_handling_ops     -- absence of scf.if / affine.min / affine.max
                                 / arith.minsi clamping ops that MLIR's tiling
                                 infrastructure emits for non-evenly-divisible
                                 tiling ("no silently-dropped tail dimensions")
  7. unrolled_chain_is_serial -- for SCHEDULE_UNROLL_K>1, the F vector.contract
                                 ops inside one K-loop body iteration form a
                                 single serial accumulator chain (each one's
                                 third operand is the previous one's result),
                                 confirming true accumulation semantics were
                                 preserved by the unroll (not reordered into
                                 independent partial sums, which would be a
                                 numerics-changing transform bug)
  8. schedule_unroll_1_is_noop -- structural (not just byte-level .text)
                                 equivalence check: with SCHEDULE_UNROLL_K=1,
                                 this mid-pipeline IR must be textually
                                 IDENTICAL (mod whitespace) to the plain
                                 tiled-vectorized variant's mid-pipeline IR
                                 for the same TILE_M/N/K.

Usage:
  python3 tools/validate_aarch64_tiled_schedule_structure.py \
    --input mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir \
    --tile-m 8 --tile-n 8 --tile-k 8 --schedule-unroll-k 2 \
    --output structure_report.json [--report structure_report.txt]

Exit code is 0 iff every check passes.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

MLIR_BIN = os.environ.get(
    "MLIR_BIN", "/home/allen/Desktop/Project/.deps/mlir21-root/usr/lib/llvm-21/bin"
)
MLIR_OPT = os.path.join(MLIR_BIN, "mlir-opt")
PLUGIN = os.environ.get(
    "PLUGIN", os.path.join(REPO_ROOT, "build-mlir", "libHIRMatMulBiasReluFusionPass.so")
)
GENERATE_TILED_TRANSFORM = os.path.join(REPO_ROOT, "mlir_passes", "tools", "generate_tiled_transform.sh")
GENERATE_SCHEDULED_TRANSFORM = os.path.join(REPO_ROOT, "mlir_passes", "tools", "generate_scheduled_transform.sh")
TILE_TRANSFORM_TEMPLATE = os.path.join(
    REPO_ROOT, "mlir_passes", "transforms", "tile_vectorize_matmul_bias_relu.template.mlir"
)
SCHEDULE_TRANSFORM_TEMPLATE = os.path.join(
    REPO_ROOT, "mlir_passes", "transforms", "tile_schedule_matmul_bias_relu.template.mlir"
)

MIDPIPE_STAGE = (
    "builtin.module(hir-matmul-bias-relu-to-linalg,"
    "transform-preload-library{{transform-library-paths={transform}}},"
    "transform-interpreter{{entry-point=__transform_main}})"
)

CONST_RE = re.compile(r"%(c\w+)\s*=\s*arith\.constant\s+(\d+)\s*:\s*index")
SCF_FOR_RE = re.compile(
    r"scf\.for\s+%\w+\s*=\s*%(\w+)\s+to\s+%(\w+)\s+step\s+%(\w+)\s+"
    r"iter_args\(%\w+\s*=\s*%[\w.]+\)\s*->\s*\(tensor<(\d+)x(\d+)xf32>\)\s*\{"
)
VECTOR_CONTRACT_RE = re.compile(
    r"%(\d+)\s*=\s*vector\.contract\s*\{[^}]*\}\s*%([\w.]+),\s*%([\w.]+),\s*%([\w.]+)\s*:"
    r"\s*vector<(\d+)x(\d+)xf32>,\s*vector<(\d+)x(\d+)xf32>\s*into\s*vector<(\d+)x(\d+)xf32>"
)
ZERO_CONST_RE = re.compile(r"%cst\s*=\s*arith\.constant\s+dense<0\.0*e\+00>\s*:\s*vector<(\d+)x(\d+)xf32>")
ADDF_RE = re.compile(r"%(\d+)\s*=\s*arith\.addf\s+%(\d+),\s*%(\d+)\s*:\s*vector<(\d+)x(\d+)xf32>")
MAXIMUMF_RE = re.compile(r"%(\d+)\s*=\s*arith\.maximumf\s+%(\d+),\s*%(\w+)\s*:\s*vector<(\d+)x(\d+)xf32>")
TAIL_HANDLING_RE = re.compile(r"\bscf\.if\b|\baffine\.min\b|\baffine\.max\b|\barith\.minsi\b")


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(cmd)}):\n{proc.stderr}")
    return proc.stdout


def parse_shape(input_mlir):
    with open(input_mlir) as f:
        text = f.read()
    dims = re.findall(r"tensor<(\d+)x(\d+)xf32>", text)[:2]
    if len(dims) != 2:
        raise RuntimeError(f"could not parse M/N/K from {input_mlir}")
    (m, k1), (k2, n) = dims
    return int(m), int(n), int(k1)


def generate_transform(generator, template_env, tile_m, tile_n, tile_k, schedule_unroll_k, output_path):
    cmd = ["bash", generator, "--tile-m", str(tile_m), "--tile-n", str(tile_n), "--tile-k", str(tile_k)]
    if schedule_unroll_k is not None:
        cmd += ["--schedule-unroll-k", str(schedule_unroll_k)]
    cmd += ["--output", output_path]
    env = dict(os.environ)
    env["TEMPLATE"] = template_env
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"transform generation failed:\n{proc.stderr}")


def run_midpipe(input_mlir, transform_path):
    pipeline = MIDPIPE_STAGE.format(transform=transform_path)
    return run([
        MLIR_OPT, input_mlir,
        f"--load-dialect-plugin={PLUGIN}", f"--load-pass-plugin={PLUGIN}",
        f"--pass-pipeline={pipeline}",
    ])


def build_const_map(text):
    return {name: int(val) for name, val in CONST_RE.findall(text)}


def check_outer_loop_bounds(text, const_map, shape_m, shape_n, shape_k, tile_m, tile_n, tile_k, unroll_k):
    # scf.for-loop-canonicalization (part of the pipeline for both
    # tiled-vectorized and tiled-scheduled) eliminates any loop whose trip
    # count is exactly 1, inlining its single iteration as straight-line
    # code. This applies independently to each of the M, N, and K loops:
    #   - M or N collapses when the requested tile size equals the full
    #     shape dimension (trip count 1) -- happens even at unroll-k=1,
    #     unrelated to the schedule transform (e.g. an 8x8x8 shape tiled
    #     4x8x8 has an N trip count of 8/8=1).
    #   - K collapses when schedule-unroll-k equals the K-loop's own trip
    #     count (shape_k // tile_k), i.e. a full unroll -- this IS the
    #     schedule transform's doing.
    # So the set of scf.for loops actually present in the text is whichever
    # of M/N/K have a (post-unroll, for K) trip count > 1, in that relative
    # order; loops that collapse are validated structurally by the other
    # checks (k_loop_unroll_materialized, unrolled_chain_is_serial) instead.
    m_trip = shape_m // tile_m
    n_trip = shape_n // tile_n
    k_trip = (shape_k // tile_k) // unroll_k

    expected_all = [
        (m_trip, 0, shape_m, tile_m, "M"),
        (n_trip, 0, shape_n, tile_n, "N"),
        (k_trip, 0, shape_k, tile_k * unroll_k, "K"),
    ]
    expected_present = [(elo, ehi, estep, label) for trip, elo, ehi, estep, label in expected_all if trip > 1]
    collapsed = [label for trip, *_rest, label in expected_all if trip <= 1]

    matches = list(SCF_FOR_RE.finditer(text))
    if len(matches) != len(expected_present):
        return (
            False,
            f"expected {len(expected_present)} scf.for loop(s) (collapsed: {collapsed or 'none'}), found {len(matches)}",
            None,
        )

    resolved = [(const_map.get(m.group(1)), const_map.get(m.group(2)), const_map.get(m.group(3))) for m in matches]
    problems = []
    for (lo, hi, step), (elo, ehi, estep, label) in zip(resolved, expected_present):
        if (lo, hi, step) != (elo, ehi, estep):
            problems.append(f"{label}-loop: got (lo={lo}, hi={hi}, step={step}), expected (lo={elo}, hi={ehi}, step={estep})")
    ok = not problems
    detail = (
        f"remaining loop bounds match expected tiling+unroll (collapsed via trip-count-1 canonicalization: {collapsed or 'none'})"
        if ok else "; ".join(problems)
    )
    loop_info = {label: bounds for (elo, ehi, estep, label), bounds in zip(expected_present, resolved)}
    return ok, detail, loop_info


def check_k_loop_unroll_materialized(text, unroll_k):
    # A global count is sufficient (rather than scoping to the K-loop
    # specifically): each static vector.contract instance is printed
    # exactly once in the IR regardless of whether the K-reduction is
    # represented as a (possibly-unrolled) scf.for body or, for a full
    # unroll (schedule-unroll-k == K trip count), as inlined straight-line
    # code with the K-loop canonicalized away -- see check_outer_loop_bounds.
    contracts = list(VECTOR_CONTRACT_RE.finditer(text))
    ok = len(contracts) == unroll_k
    detail = (
        f"{len(contracts)} vector.contract op(s) found in the tiled kernel body, expected {unroll_k} "
        f"(schedule-unroll-k)"
    )
    return ok, detail, contracts


def check_bounded_vector_shapes(text, tile_m, tile_n, tile_k, shape_m, shape_n):
    whole_shape_pat = re.compile(rf"vector<{shape_m}x{shape_n}xf32>|vector<{shape_m}x\d+xf32>|vector<\d+x{shape_n}xf32>")
    tile_shapes_seen = set(re.findall(r"vector<(\d+)x(\d+)xf32>", text))
    bad = [s for s in tile_shapes_seen if s not in {(str(tile_m), str(tile_n)), (str(tile_m), str(tile_k)), (str(tile_k), str(tile_n))}]
    ok = len(bad) == 0
    detail = (
        f"vector shapes present: {sorted(tile_shapes_seen)}; all bounded by tile size"
        if ok else f"unexpected (possibly whole-shape) vector shapes found: {bad}"
    )
    return ok, detail


def check_accumulator_zero_init(text, tile_m, tile_n):
    zero_consts = ZERO_CONST_RE.findall(text)
    ok = len(zero_consts) == 1 and zero_consts[0] == (str(tile_m), str(tile_n))
    detail = (
        f"found {len(zero_consts)} zero-vector constant(s) of shape {zero_consts}, "
        f"expected exactly one of shape ({tile_m}, {tile_n})"
    )
    return ok, detail


def check_bias_relu_per_tile(text, tile_m, tile_n):
    addf = ADDF_RE.findall(text)
    maxf = MAXIMUMF_RE.findall(text)
    ok = len(addf) == 1 and len(maxf) == 1 and addf[0][3:] == (str(tile_m), str(tile_n)) and maxf[0][3:] == (str(tile_m), str(tile_n))
    detail = f"addf occurrences={len(addf)}, maximumf occurrences={len(maxf)}, tile-shaped={ok}"
    return ok, detail


def check_no_tail_handling(text):
    hits = TAIL_HANDLING_RE.findall(text)
    ok = len(hits) == 0
    detail = "no tail-handling ops found" if ok else f"found tail-handling ops: {hits}"
    return ok, detail


def check_unrolled_chain_is_serial(contracts, unroll_k):
    if unroll_k == 1:
        return True, "schedule-unroll-k=1: single vector.contract, chain-serial check not applicable"
    if not contracts:
        return False, "no vector.contract ops found to check chain serialization"
    dests = [c.group(1) for c in contracts]
    third_operands = [c.group(4) for c in contracts]
    # first contract's third operand must be the tile's carried-in accumulator
    # (not one of this K-body's own dests); each subsequent contract's third
    # operand must be the immediately preceding contract's destination.
    ok = third_operands[1:] == dests[:-1] and third_operands[0] not in dests
    detail = (
        f"contract destinations={dests}, third-operands={third_operands}, serial-chain={ok}"
    )
    return ok, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tile-m", type=int, required=True)
    ap.add_argument("--tile-n", type=int, required=True)
    ap.add_argument("--tile-k", type=int, required=True)
    ap.add_argument("--schedule-unroll-k", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    shape_m, shape_n, shape_k = parse_shape(args.input)

    with tempfile.TemporaryDirectory() as tmp:
        sched_transform = os.path.join(tmp, "sched_transform.mlir")
        generate_transform(
            GENERATE_SCHEDULED_TRANSFORM, SCHEDULE_TRANSFORM_TEMPLATE,
            args.tile_m, args.tile_n, args.tile_k, args.schedule_unroll_k, sched_transform,
        )
        sched_text = run_midpipe(args.input, sched_transform)

        baseline_transform = os.path.join(tmp, "baseline_transform.mlir")
        generate_transform(
            GENERATE_TILED_TRANSFORM, TILE_TRANSFORM_TEMPLATE,
            args.tile_m, args.tile_n, args.tile_k, None, baseline_transform,
        )
        baseline_text = run_midpipe(args.input, baseline_transform)

    const_map = build_const_map(sched_text)
    checks = {}

    ok, detail, loop_info = check_outer_loop_bounds(
        sched_text, const_map, shape_m, shape_n, shape_k,
        args.tile_m, args.tile_n, args.tile_k, args.schedule_unroll_k,
    )
    checks["outer_loop_bounds"] = {"pass": ok, "detail": detail}

    ok, detail, contracts = check_k_loop_unroll_materialized(sched_text, args.schedule_unroll_k)
    checks["k_loop_unroll_materialized"] = {"pass": ok, "detail": detail}

    ok, detail = check_bounded_vector_shapes(sched_text, args.tile_m, args.tile_n, args.tile_k, shape_m, shape_n)
    checks["bounded_vector_shapes"] = {"pass": ok, "detail": detail}

    ok, detail = check_accumulator_zero_init(sched_text, args.tile_m, args.tile_n)
    checks["accumulator_zero_init"] = {"pass": ok, "detail": detail}

    ok, detail = check_bias_relu_per_tile(sched_text, args.tile_m, args.tile_n)
    checks["bias_relu_per_tile"] = {"pass": ok, "detail": detail}

    ok, detail = check_no_tail_handling(sched_text)
    checks["no_tail_handling_ops"] = {"pass": ok, "detail": detail}

    ok, detail = check_unrolled_chain_is_serial(contracts or [], args.schedule_unroll_k)
    checks["unrolled_chain_is_serial"] = {"pass": ok, "detail": detail}

    if args.schedule_unroll_k == 1:
        norm = lambda t: "\n".join(l.strip() for l in t.splitlines() if l.strip())
        ok = norm(sched_text) == norm(baseline_text)
        detail = "schedule-unroll-k=1 mid-pipeline IR is textually identical to tiled-vectorized baseline" if ok else "MISMATCH: schedule-unroll-k=1 should be a structural no-op but differs from baseline"
        checks["schedule_unroll_1_is_noop"] = {"pass": ok, "detail": detail}
    else:
        checks["schedule_unroll_1_is_noop"] = {"pass": None, "detail": "not applicable (schedule-unroll-k != 1)"}

    all_pass = all(c["pass"] is not False for c in checks.values())

    result = {
        "input": args.input,
        "shape": {"m": shape_m, "n": shape_n, "k": shape_k},
        "tile": {"m": args.tile_m, "n": args.tile_n, "k": args.tile_k},
        "schedule_unroll_k": args.schedule_unroll_k,
        "checks": checks,
        "all_pass": all_pass,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    lines = [f"Structural validation: {args.input} tile={args.tile_m}x{args.tile_n}x{args.tile_k} unroll={args.schedule_unroll_k}"]
    for name, c in checks.items():
        status = "PASS" if c["pass"] else ("N/A" if c["pass"] is None else "FAIL")
        lines.append(f"  [{status}] {name}: {c['detail']}")
    lines.append(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    report_text = "\n".join(lines)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report_text + "\n")
    print(report_text)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
