#!/usr/bin/env python3
"""compare_aarch64_schedule_variants.py

Stage 12 of the machine-scheduling analysis slice: LLVM scheduling
evidence comparison between the "tiled-scheduled" variant at a baseline
schedule-unroll-k=1 (verified in Stage 11 to be a structural AND
byte-for-byte .text no-op vs. the plain tiled-vectorized variant -- see
tools/validate_aarch64_tiled_schedule_structure.py's schedule_unroll_1_is_noop
check) and a nontrivial schedule-unroll-k. Because both sides go through
the SAME --variant tiled-scheduled code path, the only thing that differs
between "baseline" and "scheduled" is the schedule-unroll-k value itself --
this is what makes it a controlled comparison (task brief section 2).

REUSED, UNMODIFIED, existing Stage 3/5/6 tooling (this script does not
duplicate their logic, only orchestrates them and adds what they do not
cover):
  - mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh
      HIR -> LLVM-dialect MLIR -> LLVM IR -> assembly/object, via
      --variant tiled-scheduled. This is the Stage 10 compile-interface
      extension; Stage 12 does not touch it further.
  - tools/extract_aarch64_candidate_mir.py
      MIR extraction at 5 real llc pass boundaries (post_isel,
      pre_scheduler, pre_ra/post_scheduler, post_ra, post_prologue_epilogue)
      -- see that tool's own header for how each boundary was discovered
      and verified on this exact LLVM 21.1.8 build.
  - tools/analyze_aarch64_candidate_mir.py
      Register-class distribution, frame objects, and the spill/reload
      classifier that separates allocator spill-slots
      (type: spill-slot, no callee-saved-register field) from ABI
      callee-saved-register preservation and ordinary fixedStack ABI
      argument slots.
  - tools/analyze_aarch64_machine_schedule.py
      FMLA accumulator-chain / load-to-use-distance / same-accumulator
      scheduling-distance metrics, parsed from MIR instruction order (NOT
      cycle-accurate latency -- labeled as such throughout).

NEW in this script (nothing upstream already does these):
  - check_fp_reduction_order(): parses the LLVM IR (.ll) for
    @llvm.fmuladd.* calls, checks for fast-math-flag tokens on each call,
    and reconstructs accumulator chains by following each call's
    accumulator (3rd) operand back to an earlier call's destination --
    the same "self-referencing chain" methodology
    analyze_aarch64_machine_schedule.py already uses at the MIR level,
    applied here one level higher (LLVM IR) as an independent
    cross-check. Directly answers Stage 12 section 6: does LLVM preserve
    the MLIR-level serial reduction order, or does it reassociate into
    parallel partial sums? (Only legal under 'reassoc'/'fast' FMF, which
    this function explicitly checks for and did not find on this
    project's inputs -- MLIR's vector-to-llvm lowering emits
    @llvm.fmuladd directly, whose fusion is unconditional by the
    intrinsic's own definition, not FMF-gated.)
  - extract_hot_loop_region() + run_llvm_mca(): locates the innermost
    ("This Inner Loop Header") loop in the final assembly via its LLVM
    block-comment annotations, wraps it in
    '# LLVM-MCA-BEGIN'/'# LLVM-MCA-END' markers (llvm-mca's own supported
    region-scoping mechanism), and runs llvm-mca against Cortex-A76's
    scheduling model. Explicitly labeled a STATIC MACHINE-MODEL ESTIMATE,
    never conflated with measured Raspberry Pi latency (no Pi execution
    happens anywhere in this script).
  - assembly_counts(): FMLA / vector-load / vector-store / scalar-load /
    scalar-store / branch counts directly from the final .s text, for the
    assembly-level comparison (task section 11).
  - classify_pair(): applies the task's A/B/C/D evidence classification
    (section 12) from the assembled metrics -- never from expectation.

Usage:
  python3 tools/compare_aarch64_schedule_variants.py \
    --output-dir artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling \
    [--candidates-json candidates.json]   # else uses the default mandatory matrix

Every artifact JSON records: git commit/dirty state, target triple/cpu,
llc/mlir-opt versions, the full compiler command, and the candidate's
shape/tile/schedule-unroll-k/variant -- per task section 3.
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

COMPILE_SCRIPT = os.path.join(REPO_ROOT, "mlir_passes", "tools", "compile_hir_matmul_bias_relu_aarch64.sh")
EXTRACT_TOOL = os.path.join(REPO_ROOT, "tools", "extract_aarch64_candidate_mir.py")
REGISTER_TOOL = os.path.join(REPO_ROOT, "tools", "analyze_aarch64_candidate_mir.py")
SCHEDULE_TOOL = os.path.join(REPO_ROOT, "tools", "analyze_aarch64_machine_schedule.py")
FIXTURE_DIR = os.path.join(REPO_ROOT, "mlir_passes", "test", "backend_codegen")

TARGET_TRIPLE = "aarch64-linux-gnu"
TARGET_CPU = "cortex-a76"

OPERAND_RE = r"(?:%(\S+?)|(zeroinitializer|undef|poison))"
FMULADD_RE = re.compile(
    r"%(\d+)\s*=\s*call\s+(fast\s+|reassoc\s+|contract\s+|afn\s+|nnan\s+|ninf\s+|nsz\s+|arcp\s+)*"
    r"<(\d+)\s*x\s*float>\s*@llvm\.fmuladd\.v\d+f32\("
    rf"<\d+\s*x\s*float>\s*{OPERAND_RE},\s*"
    rf"<\d+\s*x\s*float>\s*{OPERAND_RE},\s*<\d+\s*x\s*float>\s*{OPERAND_RE}\)"
)
FADD_RE = re.compile(r"%(\d+)\s*=\s*fadd\s+(fast\s+|reassoc\s+|contract\s+)*<\d+\s*x\s*float>")
FMUL_RE = re.compile(r"%(\d+)\s*=\s*fmul\s+(fast\s+|reassoc\s+|contract\s+)*<\d+\s*x\s*float>")

LOOP_HEADER_COMMENT_RE = re.compile(r"This (?:Inner )?Loop Header:\s*Depth=(\d+)")
LABEL_RE = re.compile(r"^(\.LBB\d+_\d+):")


def sh(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(str(c) for c in cmd)}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def get_environment_metadata():
    llc_version = subprocess.run(["llc", "--version"], capture_output=True, text=True).stdout.splitlines()
    mlir_bin = os.environ.get("MLIR_BIN", "/home/allen/Desktop/Project/.deps/mlir21-root/usr/lib/llvm-21/bin")
    mlir_opt = os.path.join(mlir_bin, "mlir-opt")
    mlir_version = subprocess.run([mlir_opt, "--version"], capture_output=True, text=True).stdout.splitlines()
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    git_dirty = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    return {
        "git_commit": git_commit,
        "git_working_tree_dirty": bool(git_dirty),
        "git_dirty_files": git_dirty.splitlines() if git_dirty else [],
        "llc_version": next((l for l in llc_version if "LLVM version" in l), llc_version[0] if llc_version else None),
        "mlir_opt_version": next((l for l in mlir_version if "LLVM version" in l), mlir_version[0] if mlir_version else None),
        "target_triple": TARGET_TRIPLE,
        "target_cpu": TARGET_CPU,
        "target_features_note": "no explicit -mattr flags used anywhere in this pipeline; feature resolution is llc's default for -mcpu=cortex-a76 on this Release-build toolchain (no -debug-only output available to print the resolved feature string -- see analyze_aarch64_candidate_mir.py's own module docstring for the same toolchain limitation)",
    }


def compile_variant(shape, tile_m, tile_n, tile_k, unroll_k, out_dir, name):
    fixture = os.path.join(FIXTURE_DIR, f"matmul_bias_relu_tiled_{shape}.mlir")
    if not os.path.isfile(fixture):
        raise RuntimeError(f"fixture not found: {fixture}")
    cmd = [
        "bash", COMPILE_SCRIPT,
        "--variant", "tiled-scheduled",
        "--tile-m", str(tile_m), "--tile-n", str(tile_n), "--tile-k", str(tile_k),
        "--schedule-unroll-k", str(unroll_k),
        fixture, out_dir, name,
    ]
    sh(cmd)
    return {
        "command": " ".join(cmd),
        "llvm_ir": os.path.join(out_dir, f"{name}.ll"),
        "llvm_dialect_mlir": os.path.join(out_dir, f"{name}_llvm.mlir"),
        "asm": os.path.join(out_dir, f"{name}.s"),
        "obj": os.path.join(out_dir, f"{name}.o"),
    }


def extract_mir(llvm_ir, shape, tile_m, tile_n, tile_k, unroll_k, out_dir):
    cmd = [
        "python3", EXTRACT_TOOL,
        "--llvm-ir", llvm_ir, "--cpu", TARGET_CPU, "--shape", shape,
        "--tile-m", str(tile_m), "--tile-n", str(tile_n), "--tile-k", str(tile_k),
        "--schedule-unroll-k", str(unroll_k),
        "--output-dir", out_dir,
    ]
    sh(cmd)
    prefix = f"{shape}_tm{tile_m}_tn{tile_n}_tk{tile_k}_uk{unroll_k}_greedy_misched-default"
    return {
        "command": " ".join(cmd),
        "post_isel": os.path.join(out_dir, f"{prefix}_post_isel.mir"),
        "pre_scheduler": os.path.join(out_dir, f"{prefix}_pre_scheduler.mir"),
        "pre_ra": os.path.join(out_dir, f"{prefix}_pre_ra.mir"),
        "post_ra": os.path.join(out_dir, f"{prefix}_post_ra.mir"),
        "post_prologue_epilogue": os.path.join(out_dir, f"{prefix}_post_prologue_epilogue.mir"),
        "final_asm": os.path.join(out_dir, f"{prefix}.s"),
        "final_obj": os.path.join(out_dir, f"{prefix}.o"),
    }


def analyze_registers(mir_paths, out_json):
    cmd = [
        "python3", REGISTER_TOOL,
        "--post-isel", mir_paths["post_isel"],
        "--pre-ra", mir_paths["pre_ra"],
        "--post-ra", mir_paths["post_ra"],
        "--post-prologue-epilogue", mir_paths["post_prologue_epilogue"],
        "--output", out_json,
    ]
    sh(cmd)
    with open(out_json) as f:
        return json.load(f), " ".join(cmd)


def analyze_schedule(mir_path, asm_path, out_json):
    cmd = ["python3", SCHEDULE_TOOL, "--mir", mir_path, "--asm", asm_path, "--output", out_json]
    sh(cmd)
    with open(out_json) as f:
        return json.load(f), " ".join(cmd)


def check_fp_reduction_order(llvm_ir_path):
    with open(llvm_ir_path) as f:
        text = f.read()

    fmuladd_calls = []
    for m in FMULADD_RE.finditer(text):
        dest, fmf, width, op1_named, op1_bare, op2_named, op2_bare, op3_named, op3_bare = m.groups()
        # Each operand matches either a numbered/named SSA value (%foo) or
        # a bareword literal (zeroinitializer/undef/poison, no leading %)
        # -- e.g. the first fmuladd in a fully-unrolled reduction chain
        # accumulates directly into `zeroinitializer` rather than a named
        # %cst, which an earlier version of this regex (requiring a
        # leading %) silently failed to match, undercounting fmuladd calls
        # for full-unroll candidates specifically (256 real calls in the
        # 32x32x32/tile-8x8x8/schedule-unroll-k=4 candidate's LLVM IR,
        # only 248 matched) -- verified against a plain `grep -c` count.
        op1 = op1_named or op1_bare
        op2 = op2_named or op2_bare
        op3 = op3_named or op3_bare
        fmuladd_calls.append({
            "dest": dest, "fmf": fmf.strip() if fmf else None, "vector_width": int(width),
            "mul_operand_1": op1, "mul_operand_2": op2, "accumulator_operand": op3,
        })

    fmf_flags_found = sorted({c["fmf"] for c in fmuladd_calls if c["fmf"]})
    fadd_with_fmf = len(FADD_RE.findall(text))
    fmul_with_fmf = len(FMUL_RE.findall(text))

    dest_index = {c["dest"]: i for i, c in enumerate(fmuladd_calls)}
    visited = set()
    chains = []
    for i, c in enumerate(fmuladd_calls):
        if i in visited or c["accumulator_operand"] in dest_index:
            continue  # not a chain root: either already covered or fed by an earlier call
        chain = [i]
        visited.add(i)
        cur_dest = c["dest"]
        while True:
            nxt = next((j for j, cc in enumerate(fmuladd_calls) if cc["accumulator_operand"] == cur_dest and j not in visited), None)
            if nxt is None:
                break
            chain.append(nxt)
            visited.add(nxt)
            cur_dest = fmuladd_calls[nxt]["dest"]
        chains.append(chain)

    chain_lengths = [len(c) for c in chains]
    total_in_chains = sum(chain_lengths)
    return {
        "fmuladd_call_count": len(fmuladd_calls),
        "fmf_flags_found_on_fmuladd_calls": fmf_flags_found,
        "fmf_flags_found_note": (
            "empty list means no fast-math-flag tokens (fast/reassoc/contract/afn/nnan/ninf/nsz/arcp) "
            "were found on any @llvm.fmuladd call -- fusion here comes from MLIR's vector-to-llvm lowering "
            "emitting the fmuladd intrinsic directly (unconditionally fused by the intrinsic's own definition), "
            "not from FMF-gated reassociation"
        ),
        "fadd_call_count": fadd_with_fmf,
        "fmul_call_count": fmul_with_fmf,
        "accumulator_chain_count": len(chains),
        "accumulator_chain_lengths": sorted(chain_lengths, reverse=True),
        "max_accumulator_chain_length": max(chain_lengths) if chain_lengths else None,
        "all_fmuladd_calls_accounted_for": total_in_chains == len(fmuladd_calls),
        "reduction_order_note": (
            "chains reconstructed by following each fmuladd call's accumulator (3rd) operand back to an "
            "earlier call's destination in the same function, mirroring "
            "analyze_aarch64_machine_schedule.py's MIR-level accumulator-chain methodology one level higher "
            "(LLVM IR); a chain count/length that matches the MIR-level accumulator_chains/"
            "max_accumulator_chain_length is direct cross-level evidence that LLVM preserved the MLIR "
            "transform's serial reduction order rather than reassociating into independent parallel partial sums"
        ),
    }


def find_innermost_loop_region(asm_lines):
    # Pick the loop-header comment with the HIGHEST Depth=N, not the first
    # match: "This (?:Inner )?Loop Header" matches the OUTER loop's "This
    # Loop Header: Depth=1" comment too (Inner is optional in the regex),
    # so taking the first match previously locked onto the M-loop instead
    # of the true innermost (K) loop -- verified concretely: for the
    # primary 32x32x32/tile-8x8x8/unroll-2 candidate the assembly has
    # "This Loop Header: Depth=1" at line 49 and the real target, "This
    # Inner Loop Header: Depth=3", at line 170; the first-match version
    # returned the wrong (outer, non-representative) region every time.
    best_idx, best_depth = None, -1
    for i, line in enumerate(asm_lines):
        m = LOOP_HEADER_COMMENT_RE.search(line)
        if m and int(m.group(1)) > best_depth:
            best_idx, best_depth = i, int(m.group(1))
    if best_idx is None:
        return None, None, None
    header_idx = None
    label = None
    for j in range(best_idx, -1, -1):
        m = LABEL_RE.match(asm_lines[j])
        if m:
            label = m.group(1)
            header_idx = j
            break
    if header_idx is None:
        return None, None, None
    back_idx = None
    for i in range(header_idx + 1, len(asm_lines)):
        if re.search(rf"\bb(?:\.\w+)?\s+{re.escape(label)}\b", asm_lines[i]):
            back_idx = i
    return label, header_idx, back_idx


def extract_hot_loop_region(asm_path, out_path):
    with open(asm_path) as f:
        lines = f.read().splitlines()
    label, header_idx, back_idx = find_innermost_loop_region(lines)
    if label is None or back_idx is None:
        return None
    region = lines[header_idx:back_idx + 1]
    marked = [f"# LLVM-MCA-BEGIN {label}"] + region + ["# LLVM-MCA-END"]
    with open(out_path, "w") as f:
        f.write("\n".join(marked) + "\n")
    return {"label": label, "line_count": len(region), "path": out_path}


MCA_SUMMARY_RE = {
    "iterations": re.compile(r"Iterations:\s*(\d+)"),
    "instructions": re.compile(r"Instructions:\s*(\d+)"),
    "total_cycles": re.compile(r"Total Cycles:\s*(\d+)"),
    "total_uops": re.compile(r"Total uOps:\s*(\d+)"),
    "dispatch_width": re.compile(r"Dispatch Width:\s*(\d+)"),
    "uops_per_cycle": re.compile(r"uOps Per Cycle:\s*([\d.]+)"),
    "ipc": re.compile(r"IPC:\s*([\d.]+)"),
    "block_rthroughput": re.compile(r"Block RThroughput:\s*([\d.]+)"),
}


def run_llvm_mca(marked_asm_path, iterations=100):
    cmd = ["llvm-mca", f"-mtriple={TARGET_TRIPLE}", f"-mcpu={TARGET_CPU}", f"-iterations={iterations}", marked_asm_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"ok": False, "command": " ".join(cmd), "stderr": proc.stderr[-1000:]}
    out = proc.stdout
    summary = {}
    for key, pat in MCA_SUMMARY_RE.items():
        m = pat.search(out)
        summary[key] = (int(m.group(1)) if key not in ("uops_per_cycle", "ipc", "block_rthroughput") else float(m.group(1))) if m else None
    return {
        "ok": True,
        "command": " ".join(cmd),
        "cpu_model": TARGET_CPU,
        "label": "STATIC MACHINE-MODEL ESTIMATE, not measured Raspberry Pi latency",
        "summary": summary,
        "raw_output_excerpt": "\n".join(out.splitlines()[:25]),
    }


ASM_COUNT_PATTERNS = {
    "fmla": re.compile(r"^\s+fmla\s", re.M),
    "vector_load": re.compile(r"^\s+(?:ldr|ldp)\s+q\d+", re.M),
    "vector_store": re.compile(r"^\s+(?:str|stp)\s+q\d+", re.M),
    "scalar_load": re.compile(r"^\s+ldr\s+[xw]\d+", re.M),
    "scalar_store": re.compile(r"^\s+str\s+[xw]\d+", re.M),
    "branch": re.compile(r"^\s+(?:b|b\.\w+|cbz|cbnz)\s+\.LBB", re.M),
}


def assembly_counts(asm_path):
    with open(asm_path) as f:
        text = f.read()
    counts = {name: len(pat.findall(text)) for name, pat in ASM_COUNT_PATTERNS.items()}
    counts["object_bytes_note"] = "see object_bytes field alongside this record for the compiled .o size"
    return counts


def run_candidate(shape, tile_m, tile_n, tile_k, unroll_k, base_dir):
    name = f"{shape}_tm{tile_m}_tn{tile_n}_tk{tile_k}_uk{unroll_k}"
    cand_dir = os.path.join(base_dir, name)
    os.makedirs(cand_dir, exist_ok=True)

    compiled = compile_variant(shape, tile_m, tile_n, tile_k, unroll_k, cand_dir, name)
    mir_paths = extract_mir(compiled["llvm_ir"], shape, tile_m, tile_n, tile_k, unroll_k, cand_dir)

    reg_metrics, reg_cmd = analyze_registers(mir_paths, os.path.join(cand_dir, "register_metrics.json"))
    sched_pre, sched_pre_cmd = analyze_schedule(mir_paths["pre_scheduler"], mir_paths["final_asm"], os.path.join(cand_dir, "schedule_pre_scheduler_metrics.json"))
    sched_post, sched_post_cmd = analyze_schedule(mir_paths["pre_ra"], mir_paths["final_asm"], os.path.join(cand_dir, "schedule_post_scheduler_metrics.json"))

    fp_order = check_fp_reduction_order(compiled["llvm_ir"])
    asm_counts = assembly_counts(mir_paths["final_asm"])
    obj_bytes = os.path.getsize(mir_paths["final_obj"])

    record = {
        "shape": shape,
        "tile": {"m": tile_m, "n": tile_n, "k": tile_k},
        "schedule_unroll_k": unroll_k,
        "variant": "tiled-scheduled",
        "candidate_dir": cand_dir,
        "commands": {
            "compile": compiled["command"],
            "extract_mir": mir_paths["command"] if "command" in mir_paths else None,
            "register_analysis": reg_cmd,
            "schedule_analysis_pre_scheduler": sched_pre_cmd,
            "schedule_analysis_post_scheduler": sched_post_cmd,
        },
        "artifact_paths": {**compiled, **{f"mir_{k}": v for k, v in mir_paths.items() if k != "command"}},
        "pass_boundaries": {
            "instruction_selection_output": "finalize-isel",
            "machine_scheduler_input": "machine-scheduler (--stop-before)",
            "machine_scheduler_output": "machine-scheduler (--stop-after, == pre_ra)",
            "register_allocator_input": "machine-scheduler (--stop-after)",
            "register_allocator_output": "virtregrewriter",
            "post_ra_scheduler": "not enabled/inspected in this slice (task brief explicitly excludes custom/post-RA scheduling work)",
            "final_emission_input": "prologepilog",
        },
        "register_allocation": reg_metrics,
        "schedule": {
            "pre_scheduler": sched_pre,
            "post_scheduler": sched_post,
        },
        "fp_reduction_order": fp_order,
        "assembly_counts": asm_counts,
        "object_bytes": obj_bytes,
    }
    return record


def build_hot_loop_mca(record, cand_dir):
    marked_path = os.path.join(cand_dir, "hot_loop_region.s")
    region = extract_hot_loop_region(record["artifact_paths"]["mir_final_asm"], marked_path)
    if region is None:
        return {"ok": False, "reason": "could not locate an innermost-loop-header region in the final assembly"}
    mca = run_llvm_mca(marked_path)
    mca["hot_loop_region"] = region
    return mca


AARCH64_VECTOR_REGISTER_BUDGET = 32  # v0-v31 / q0-q31, architectural limit


class MismatchedComparisonError(ValueError):
    """Raised when two candidates differ in shape or tile -- comparing
    them would attribute tile-shape or problem-size differences to the
    schedule transform, which is exactly what the task brief forbids
    ("Do not compare mismatched tile shapes and call the difference
    scheduling evidence")."""


def classify_pair(baseline, scheduled):
    if baseline["shape"] != scheduled["shape"] or baseline["tile"] != scheduled["tile"]:
        raise MismatchedComparisonError(
            f"refusing to compare mismatched configurations: "
            f"baseline shape={baseline['shape']} tile={baseline['tile']} vs "
            f"scheduled shape={scheduled['shape']} tile={scheduled['tile']} "
            f"-- only schedule_unroll_k may differ for a controlled comparison"
        )
    reasons = []
    b_reg, s_reg = baseline["register_allocation"]["comparison"], scheduled["register_allocation"]["comparison"]
    spill_delta = s_reg.get("spill_stores_inserted_by_ra", 0) - b_reg.get("spill_stores_inserted_by_ra", 0)
    reload_delta = s_reg.get("reload_loads_inserted_by_ra", 0) - b_reg.get("reload_loads_inserted_by_ra", 0)
    obj_ratio = scheduled["object_bytes"] / baseline["object_bytes"] if baseline["object_bytes"] else None

    b_same_acc = baseline["schedule"]["post_scheduler"]["same_accumulator_distance"].get("median")
    s_same_acc = scheduled["schedule"]["post_scheduler"]["same_accumulator_distance"].get("median")
    b_load_use = baseline["schedule"]["post_scheduler"]["load_to_use_distance"].get("median")
    s_load_use = scheduled["schedule"]["post_scheduler"]["load_to_use_distance"].get("median")
    b_chains = baseline["schedule"]["post_scheduler"].get("accumulator_chains")
    s_chains = scheduled["schedule"]["post_scheduler"].get("accumulator_chains")

    # Authoritative pressure evidence, not raw SSA-def counts (which grow
    # roughly linearly with static unroll factor regardless of actual live
    # range overlap -- e.g. the primary candidate's virtual_vector_registers
    # count goes 242 -> 402 at unroll 1 -> 2, purely from more static FMLA
    # defs, while BOTH allocate to the identical 28 physical vector
    # registers with zero spills; using the raw SSA count here would
    # misclassify a genuinely harmless result as a pressure regression).
    b_physical_vec = baseline["register_allocation"]["stages"]["post_ra"]["physical_vector_registers_referenced"]
    s_physical_vec = scheduled["register_allocation"]["stages"]["post_ra"]["physical_vector_registers_referenced"]
    b_approx_peak = baseline["register_allocation"]["stages"]["pre_ra"]["approx_peak_live_vector_registers"]
    s_approx_peak = scheduled["register_allocation"]["stages"]["pre_ra"]["approx_peak_live_vector_registers"]

    has_new_spills = spill_delta > 0 or reload_delta > 0
    if has_new_spills:
        reasons.append(f"spill_stores delta={spill_delta}, reload_loads delta={reload_delta} (scheduled variant introduces new allocator spills)")
        classification = "D"
    else:
        code_growth_high = obj_ratio is not None and obj_ratio > 1.5
        # The authoritative pressure-cost signal is whether RA actually
        # needed more physical registers (post-RA evidence). The pre-RA
        # approx_peak_live_vector_registers heuristic is a linear scan with
        # no loop-back-edge modeling (see analyze_aarch64_candidate_mir.py's
        # own docstring) -- on this project's loop-bodied MIR it reports
        # absolute values (e.g. 113, 145) far above the real 32-register
        # architectural budget even when actual allocation is spill-free at
        # 28 physical registers, so it is NOT used to gate classification
        # here, only reported as supplementary context below.
        physical_pressure_increase = s_physical_vec > b_physical_vec
        # An "A" classification additionally requires that an unroll
        # actually happened (schedule_unroll_k strictly increased): the
        # primary source of any expected benefit here is fewer DYNAMIC
        # K-loop iterations (fewer loop-control instructions executed at
        # runtime), a structural fact whenever unroll_k increases with no
        # new spills. Without this gate, two identical schedule_unroll_k=1
        # candidates (a true no-op-vs-no-op comparison) could still satisfy
        # the non-regression checks below (>= is true for equal values) and
        # be misreported as a "win" -- caught by
        # test_identical_metrics_at_unroll_1_is_neutral_not_a_or_d.
        unroll_increased = scheduled["schedule_unroll_k"] > baseline["schedule_unroll_k"]
        schedule_not_worse = (s_chains or 0) >= (b_chains or 0) and (
            s_load_use is None or b_load_use is None or s_load_use >= b_load_use
        )
        overlap_improved = unroll_increased and schedule_not_worse
        if code_growth_high or physical_pressure_increase:
            classification = "C"
            reasons.append(
                f"object_bytes ratio={obj_ratio}; post_ra physical vector registers: baseline={b_physical_vec} scheduled={s_physical_vec}; "
                f"pre_ra approx_peak_live_vector_registers (MIR-derived estimate, NOT gating, budget={AARCH64_VECTOR_REGISTER_BUDGET}): baseline={b_approx_peak} scheduled={s_approx_peak}"
            )
        elif overlap_improved:
            classification = "A"
            reasons.append(f"schedule_unroll_k increased ({baseline['schedule_unroll_k']} -> {scheduled['schedule_unroll_k']}): fewer dynamic K-loop iterations at runtime")
            reasons.append(f"accumulator_chains: baseline={b_chains} scheduled={s_chains}; same_accumulator_distance median: baseline={b_same_acc} scheduled={s_same_acc}; load_to_use median: baseline={b_load_use} scheduled={s_load_use} (not worse)")
            reasons.append(f"no new spills; post_ra physical vector registers unchanged ({b_physical_vec} -> {s_physical_vec}, both at or under the {AARCH64_VECTOR_REGISTER_BUDGET}-register budget); pre_ra approx_peak_live_vector_registers (MIR-derived estimate, not gating): baseline={b_approx_peak} scheduled={s_approx_peak}")
        else:
            classification = "B"
            reasons.append(f"no new spills, no material change in overlap/pressure/code-size (object ratio={obj_ratio}, physical vector registers baseline={b_physical_vec} scheduled={s_physical_vec})")

    return {
        "classification": classification,
        "classification_meaning": {
            "A": "Scheduling win likely -- better overlap or shorter modeled throughput, no harmful spill increase",
            "B": "Neutral -- structure survives, backend schedule materially similar, no clear static benefit",
            "C": "Trade-off -- fewer branches/more overlap but higher pressure, code size, or spills",
            "D": "Regression risk -- spills, worse modeled throughput, broken FMLA generation, or excessive code growth",
        }[classification],
        "reasons": reasons,
        "spill_stores_delta": spill_delta,
        "reload_loads_delta": reload_delta,
        "object_bytes_ratio": obj_ratio,
    }


DEFAULT_CANDIDATES = [
    # (label, shape, tile_m, tile_n, tile_k, unroll_k)
    ("primary_unroll1", "32x32x32", 8, 8, 8, 1),
    ("primary_unroll2", "32x32x32", 8, 8, 8, 2),
    ("primary_full_unroll", "32x32x32", 8, 8, 8, 4),
    ("alt_k_tile_unroll1", "32x32x32", 8, 8, 4, 1),
    ("alt_k_tile_unroll2", "32x32x32", 8, 8, 4, 2),
    ("cube64_unroll1", "64x64x64", 8, 8, 8, 1),
    ("cube64_unroll2", "64x64x64", 8, 8, 8, 2),
    ("small_control_collapsed", "8x8x8", 4, 8, 8, 1),
]

COMPARISON_GROUPS = [
    ("primary", "primary_unroll1", "primary_unroll2"),
    ("primary_full_unroll_edge_case", "primary_unroll1", "primary_full_unroll"),
    ("alt_k_tile", "alt_k_tile_unroll1", "alt_k_tile_unroll2"),
    ("cube64", "cube64_unroll1", "cube64_unroll2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--candidates-json", help="override DEFAULT_CANDIDATES with a JSON list of [label, shape, tm, tn, tk, unroll_k]")
    ap.add_argument("--mca-iterations", type=int, default=100)
    args = ap.parse_args()

    candidates = DEFAULT_CANDIDATES
    if args.candidates_json:
        with open(args.candidates_json) as f:
            candidates = [tuple(c) for c in json.load(f)]

    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, "candidates")
    os.makedirs(work_dir, exist_ok=True)

    env_meta = get_environment_metadata()

    records = {}
    errors = {}
    for label, shape, tm, tn, tk, uk in candidates:
        print(f"[{label}] shape={shape} tile={tm}x{tn}x{tk} unroll={uk} ...", file=sys.stderr)
        try:
            records[label] = run_candidate(shape, tm, tn, tk, uk, work_dir)
        except Exception as e:
            errors[label] = str(e)
            print(f"  FAILED: {e}", file=sys.stderr)

    # llvm-mca only for the primary comparison pair (task section 10: "the
    # main baseline/scheduled comparison"), not the full matrix.
    mca_results = {}
    for label in ("primary_unroll1", "primary_unroll2"):
        if label in records:
            mca_results[label] = build_hot_loop_mca(records[label], records[label]["candidate_dir"])

    comparisons = {}
    for group_name, baseline_label, scheduled_label in COMPARISON_GROUPS:
        if baseline_label in records and scheduled_label in records:
            comparisons[group_name] = {
                "baseline": baseline_label,
                "scheduled": scheduled_label,
                **classify_pair(records[baseline_label], records[scheduled_label]),
            }

    summary = {
        "stage": "Stage 12: LLVM scheduling evidence comparison",
        "environment": env_meta,
        "candidates": records,
        "candidate_errors": errors,
        "llvm_mca_primary_comparison": mca_results,
        "comparisons": comparisons,
    }

    out_json = os.path.join(args.output_dir, "schedule_comparison_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")

    lines = ["# Stage 12 Schedule Comparison Summary\n"]
    for group_name, cmp in comparisons.items():
        lines.append(f"## {group_name}: {cmp['baseline']} vs {cmp['scheduled']}")
        lines.append(f"Classification: **{cmp['classification']}** -- {cmp['classification_meaning']}")
        for r in cmp["reasons"]:
            lines.append(f"- {r}")
        lines.append("")
    report_path = os.path.join(args.output_dir, "schedule_comparison_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")

    if errors:
        print(f"\n{len(errors)} candidate(s) FAILED: {list(errors.keys())}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
