#!/usr/bin/env python3
"""analyze_aarch64_candidate_mir.py

Stage 5/6/7/14: parses real LLVM MIR text (as produced by
tools/extract_aarch64_candidate_mir.py's --stop-after boundaries) and
emits structured metrics -- virtual-register counts, register-class
distribution, frame objects, allocator-inserted spill/reload counts (vs.
ABI/callee-saved traffic), and copy counts.

MIR files here are LLVM's own "--- |"-delimited YAML-per-machine-function
text format. Each file contains one or more machine functions (the actual
compute kernel PLUS its _mlir_ciface_ wrapper, for this project's ABI --
see mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh). This tool
analyzes the KERNEL function (name not containing "_mlir_ciface_") by
default, since that is where the tiled loop nest and FMLA microkernel
live; the wrapper is reported separately, not merged in, to avoid diluting
kernel-specific register pressure with the thin marshalling wrapper's own
(much smaller) footprint.

Spill/reload classification (Stage 7): a stack access is counted as an
ALLOCATOR SPILL/RELOAD only if it references a frame index whose `stack:`
YAML entry has `type: spill-slot`. This explicitly excludes:
  - `fixedStack:` entries (ABI incoming-argument stack slots, caller-owned)
  - `stack:` entries that are `callee-saved-register` slots (ABI
    prologue/epilogue preservation, not allocator register pressure)
  - any MLIR-generated local buffer that might appear as a `stack:` entry
    with a different `type` (none observed in this project's candidates,
    but the classification would correctly exclude it if present)

Register-pressure (Stage 6): LLVM 21's `-print-regusage`/pressure-tracker
debug output was evaluated and found to require `-debug-only=regalloc` on
a debug build of LLVM (this project's LLVM 21.1.8 is an Optimized/Release
build with assertions/debug logging compiled out -- `-debug-only` produces
no output). No LLVM-reported exact peak register pressure is available on
this toolchain. This tool therefore computes a clearly labeled
MIR-DERIVED APPROXIMATION: for the pre-RA body, a linear scan tracks how
many distinct virtual registers of each class are "live" between their
last def-or-use and are not yet dead, using operand `killed`/`dead` flags
already present in the MIR text (not a true dataflow liveness analysis --
no CFG join/loop back-edge handling). This is reported under the key
`approx_peak_live_vector_registers` and is NEVER conflated with an exact
LLVM allocator metric.

Usage:
  python3 tools/analyze_aarch64_candidate_mir.py \
    --pre-ra pre_ra.mir --post-ra post_ra.mir \
    --post-prologue-epilogue post_pe.mir \
    --output metrics.json --report report.txt
"""
import argparse
import json
import re
import sys


def split_machine_functions(text):
    """Splits MIR text into per-machine-function YAML documents. The first
    '--- |' document is the embedded LLVM IR module, not a machine
    function -- skipped."""
    docs = re.split(r"^---\s*$", text, flags=re.M)
    functions = []
    for doc in docs:
        m = re.search(r"^name:\s*(\S+)", doc, re.M)
        if m:
            functions.append((m.group(1), doc))
    return functions


def pick_kernel_function(functions):
    for name, doc in functions:
        if "_mlir_ciface_" not in name:
            return name, doc
    return functions[0] if functions else (None, None)


def parse_registers(doc):
    """Returns {vreg_id: register_class} from the `registers:` YAML list."""
    regs = {}
    m = re.search(r"^registers:\n((?:  - .*\n)*)", doc, re.M)
    if not m:
        return regs
    for line in m.group(1).splitlines():
        rm = re.search(r"id:\s*(\d+),\s*class:\s*(\S+),", line)
        if rm:
            regs[int(rm.group(1))] = rm.group(2)
    return regs


def parse_stack_objects(doc, key="stack"):
    """Returns list of dicts for `stack:` or `fixedStack:` YAML entries.

    Each entry is a flow-mapping `- { field: value, field: value, ... }`
    that LLVM's YAML emitter WRAPS across two or more physical lines once
    it gets long (e.g. `callee-saved-register` routinely lands on a
    continuation line, not the same line as `id:`/`type:`). An earlier
    version of this parser matched fields line-by-line and silently missed
    every wrapped field -- verified concretely: it reported 0 real
    allocator spill slots AND 0 callee-saved slots for a candidate whose
    raw MIR visibly contains 12 callee-saved entries, because the
    `callee-saved-register:` field was always on line 2 of each entry.
    Fixed by first collecting each `- { ... }` block (balanced braces,
    joining continuation lines) into one logical string before extracting
    fields -- this is why field extraction happens on `entry_text` below,
    not on individual `line`s.
    """
    m = re.search(rf"^{key}:\s*\n((?:.*\n)*?)(?=^\S|\Z)", doc, re.M)
    objs = []
    if not m:
        return objs
    block = m.group(1)
    # Split into individual `- { ... }` entries by tracking brace balance,
    # since a single entry's `{ ... }` may itself span multiple lines.
    entries = []
    current = []
    depth = 0
    for line in block.splitlines():
        if line.strip().startswith("- {") and depth == 0:
            current = [line]
            depth = line.count("{") - line.count("}")
            if depth == 0:
                entries.append(" ".join(current))
                current = []
        elif current:
            current.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                entries.append(" ".join(current))
                current = []
                depth = 0
    for entry_text in entries:
        entry = {}
        for field in ("id", "type", "size", "alignment", "callee-saved-register"):
            fm = re.search(rf"{field}:\s*'?([^,'}}]*?)'?\s*(?:,|}})", entry_text)
            if fm:
                entry[field] = fm.group(1).strip()
        if entry:
            objs.append(entry)
    return objs


def parse_frame_info(doc):
    info = {}
    m = re.search(r"^frameInfo:\n((?:  \S.*\n)*)", doc, re.M)
    if not m:
        return info
    for field in ("stackSize", "maxAlignment", "hasCalls", "adjustsStack"):
        fm = re.search(rf"{field}:\s*(\S+)", m.group(1))
        if fm:
            val = fm.group(1)
            info[field] = int(val) if val.isdigit() else val
    return info


def extract_body(doc):
    m = re.search(r"^body:\s*\|\n(.*)", doc, re.S | re.M)
    return m.group(1) if m else ""


def analyze_body(body_text, spill_slot_ids, vreg_classes=None):
    lines = [l for l in body_text.splitlines() if l.strip()]
    mbb_count = sum(1 for l in lines if re.match(r"^\s*bb\.\d+", l))
    instr_lines = [l for l in lines
                   if not re.match(r"^\s*bb\.\d+", l)
                   and not l.strip().startswith(("successors:", "liveins:", ";"))]
    instr_count = len(instr_lines)

    copies = sum(1 for l in instr_lines if re.search(r"\bCOPY\b", l))

    spill_stores = 0
    spill_reloads = 0
    for l in instr_lines:
        m = re.search(r"%stack\.(\d+)", l)
        if not m:
            continue
        sid = int(m.group(1))
        if sid not in spill_slot_ids:
            continue  # references a non-spill-slot stack object (shouldn't occur, but be safe)
        # The opcode is NOT always at line start -- an instruction with a
        # destination register is written `%N:class = OPCODE ...` (or
        # `$q0 = OPCODE ...` post-RA), so the mnemonic can appear well
        # after the start of the line. An earlier version anchored this
        # match to `^\s*`, which correctly found stores (STRQui has no
        # destination operand, so it IS line-initial) but silently missed
        # every reload (LDRQui always assigns a destination, so it never
        # matched) -- verified concretely against the 8x8x8/tile-8x8x8
        # candidate's known single spill+reload pair. Fixed by searching
        # for the opcode token anywhere on the line via word boundaries.
        stripped = l.strip()
        is_store = bool(re.search(r"(?:^|=\s*)(ST[RUP]\w*)\b", stripped))
        is_load = bool(re.search(r"(?:^|=\s*)(LD[RUP]\w*)\b", stripped))
        if is_store:
            spill_stores += 1
        elif is_load:
            spill_reloads += 1

    # MIR-derived approximate peak live vector registers: linear scan over
    # def/use order using `killed`/explicit vreg mentions -- NOT a true
    # dataflow analysis (see module docstring). Only meaningful for pre-RA
    # text where vregs with register-class annotations are still present.
    approx_peak = None
    if vreg_classes:
        live = set()
        peak = 0
        for l in instr_lines:
            defs = re.findall(r"%(\d+):(\w+)\s*=", l)
            uses = re.findall(r"(?:killed\s+)?%(\d+)\b", l)
            for vid_s, vclass in defs:
                vid = int(vid_s)
                if vreg_classes.get(vid, "").startswith("fpr"):
                    live.add(vid)
            peak = max(peak, len(live))
            for vid_s in uses:
                vid = int(vid_s)
                if "killed %" + vid_s in l and vid in live:
                    live.discard(vid)
        approx_peak = peak

    return {
        "machine_basic_blocks": mbb_count,
        "machine_instructions": instr_count,
        "copies": copies,
        "spill_stores": spill_stores,
        "spill_reloads": spill_reloads,
        "approx_peak_live_vector_registers": approx_peak,
    }


def analyze_stage(path, stage_label):
    with open(path) as f:
        text = f.read()
    functions = split_machine_functions(text)
    name, doc = pick_kernel_function(functions)
    if doc is None:
        return {"error": f"no machine function found in {path}"}

    registers = parse_registers(doc)
    stack_objs = parse_stack_objects(doc, "stack")
    fixed_stack_objs = parse_stack_objects(doc, "fixedStack")
    frame_info = parse_frame_info(doc)
    body = extract_body(doc)

    # A TRUE allocator register-pressure spill slot has type == 'spill-slot'
    # AND an EMPTY callee-saved-register field. LLVM reuses the identical
    # `type: spill-slot` tag for callee-saved ABI preservation slots too
    # (confirmed empirically: every one of a candidate's 12 prologue
    # save slots for x19-x25/lr/d8-d11 is tagged type: spill-slot with a
    # non-empty callee-saved-register) -- without this second condition,
    # every ordinary callee-saved-heavy function would be misreported as
    # having dozens of allocator spills.
    callee_saved_slots = [o for o in stack_objs if o.get("callee-saved-register")]
    real_spill_objs = [o for o in stack_objs
                        if o.get("type") == "spill-slot" and not o.get("callee-saved-register")]
    spill_slot_ids = {int(o["id"]) for o in real_spill_objs}
    spill_slot_bytes = sum(int(o.get("size", 0)) for o in real_spill_objs)

    class_counts = {}
    vector_vregs = 0
    gpr_vregs = 0
    for vid, vclass in registers.items():
        class_counts[vclass] = class_counts.get(vclass, 0) + 1
        if vclass.startswith("fpr"):
            vector_vregs += 1
        elif vclass.startswith("gpr"):
            gpr_vregs += 1

    body_metrics = analyze_body(body, spill_slot_ids, vreg_classes=registers if registers else None)

    # Physical registers actually referenced in the body (post-RA stages
    # have zero virtual registers; this counts real $q*/$x*/$w* mentions).
    physical_vector = sorted(set(int(m) for m in re.findall(r"\$q(\d+)\b", body)))
    physical_gpr = sorted(set(re.findall(r"\$([wx]\d+)\b", body)))

    return {
        "stage": stage_label,
        "function_name": name,
        "virtual_registers_total": len(registers),
        "virtual_vector_registers": vector_vregs,
        "virtual_gpr_registers": gpr_vregs,
        "register_class_distribution": class_counts,
        "frame_objects_total": len(stack_objs),
        "fixed_stack_objects": len(fixed_stack_objs),
        "spill_slot_count": len(spill_slot_ids),
        "spill_slot_bytes": spill_slot_bytes,
        "callee_saved_stack_slots": len(callee_saved_slots),
        "frame_info": frame_info,
        "physical_vector_registers_referenced": len(physical_vector),
        "physical_vector_register_ids": physical_vector,
        "physical_gpr_registers_referenced": len(physical_gpr),
        **body_metrics,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--post-isel")
    ap.add_argument("--pre-ra")
    ap.add_argument("--post-ra")
    ap.add_argument("--post-prologue-epilogue")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    stages = {}
    if args.post_isel:
        stages["post_isel"] = analyze_stage(args.post_isel, "post_isel")
    if args.pre_ra:
        stages["pre_ra"] = analyze_stage(args.pre_ra, "pre_ra")
    if args.post_ra:
        stages["post_ra"] = analyze_stage(args.post_ra, "post_ra")
    if args.post_prologue_epilogue:
        stages["post_prologue_epilogue"] = analyze_stage(args.post_prologue_epilogue, "post_prologue_epilogue")

    # Cross-stage comparison.
    comparison = {}
    if "pre_ra" in stages and "post_ra" in stages:
        pre = stages["pre_ra"]
        post = stages["post_ra"]
        comparison = {
            "virtual_registers_before_ra": pre["virtual_registers_total"],
            "virtual_vector_registers_before_ra": pre["virtual_vector_registers"],
            "physical_vector_registers_after_ra": post["physical_vector_registers_referenced"],
            "spill_stores_inserted_by_ra": post["spill_stores"],
            "reload_loads_inserted_by_ra": post["spill_reloads"],
            "spill_slot_count": post["spill_slot_count"],
            "spill_slot_bytes": post["spill_slot_bytes"],
            "copies_pre_ra": pre["copies"],
            "copies_post_ra": post["copies"],
            "copies_removed_by_ra": pre["copies"] - post["copies"],
            "approx_peak_live_vector_registers_note": "MIR-derived approximate peak live registers (linear scan over def/kill order, not a full dataflow analysis) -- see pre_ra.approx_peak_live_vector_registers",
        }

    stack_frame_bytes = None
    if "post_prologue_epilogue" in stages:
        stack_frame_bytes = stages["post_prologue_epilogue"]["frame_info"].get("stackSize")

    out = {
        "stages": stages,
        "comparison": comparison,
        "final_stack_frame_bytes": stack_frame_bytes,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    lines = []
    lines.append("MIR analysis report")
    for key in ("post_isel", "pre_ra", "post_ra", "post_prologue_epilogue"):
        if key not in stages:
            continue
        s = stages[key]
        lines.append(f"\n[{key}] function={s.get('function_name')}")
        lines.append(f"  vregs total={s.get('virtual_registers_total')} (vector={s.get('virtual_vector_registers')}, gpr={s.get('virtual_gpr_registers')})")
        lines.append(f"  register classes: {s.get('register_class_distribution')}")
        lines.append(f"  MBBs={s.get('machine_basic_blocks')} instrs={s.get('machine_instructions')} copies={s.get('copies')}")
        lines.append(f"  frame objects={s.get('frame_objects_total')} fixed_stack={s.get('fixed_stack_objects')} spill_slots={s.get('spill_slot_count')} ({s.get('spill_slot_bytes')} bytes)")
        lines.append(f"  spill_stores={s.get('spill_stores')} spill_reloads={s.get('spill_reloads')}")
        lines.append(f"  physical vector regs referenced={s.get('physical_vector_registers_referenced')}")
        if s.get("approx_peak_live_vector_registers") is not None:
            lines.append(f"  approx peak live vector vregs (MIR-derived, not exact)={s.get('approx_peak_live_vector_registers')}")
    if comparison:
        lines.append(f"\n[comparison pre_ra -> post_ra]")
        for k, v in comparison.items():
            lines.append(f"  {k}: {v}")
    if stack_frame_bytes is not None:
        lines.append(f"\nfinal stack frame bytes (post prologue/epilogue): {stack_frame_bytes}")

    report_text = "\n".join(lines)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report_text + "\n")
    print(report_text)


if __name__ == "__main__":
    sys.exit(main())
