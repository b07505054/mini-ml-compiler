#!/usr/bin/env python3
"""analyze_register_pressure.py

Stage 7 (register-pressure analysis) for the AArch64 tile-candidate slice.
Assembly-derived evidence only -- no custom register allocator, no LLVM MIR
pass added. Method (documented per the task's "preferred methods" list,
option 3: "parse final disassembly for stack accesses inside the identified
hot-loop address range"):

  1. Split each candidate's .s file into basic blocks by '.LBBn_m:' labels.
  2. The "hot loop" block is the fmla-containing block with the deepest
     loop-nest comment LLVM emits ("Depth=N"); ties broken by highest fmla
     count. For shapes/tiles where the loop collapses entirely (tile ==
     shape in every dim, so trip count is 1 and canonicalization removes
     the scf.for -- see generate_aarch64_matmul_tile_candidates.py's
     structural.scf_for_count field), the whole function body is used
     instead and is reported as such.
  3. Within that block only: count distinct v0-v31 registers referenced
     (assembly-derived vector-register-use evidence, NOT exact LLVM
     register-pressure/liveness analysis -- labeled as such throughout).
  4. Count stack-relative loads/stores (`[sp`) within that block, split
     into vector (q/d/s register) vs. integer (x/w register) spills
     (stores) and reloads (loads). This explicitly excludes the function
     prologue/epilogue (outside the labeled loop blocks), where
     AAPCS64 callee-saved register save/restore always appears regardless
     of loop register pressure -- that is ABI overhead, not a microkernel
     spill, and is reported separately for transparency.

Usage:
  python3 tools/analyze_register_pressure.py \
    --objects-dir /tmp/tile_candidate_objects \
    --candidates /tmp/tile_candidate_static_results.json \
    --output /tmp/tile_candidate_register_pressure.json
"""
import argparse
import json
import os
import re


def parse_blocks(s_text):
    """Returns list of (label, depth_or_none, lines) in file order."""
    blocks = []
    cur_label = "PROLOGUE"
    cur_depth = None
    cur_lines = []
    for line in s_text.splitlines():
        m = re.match(r"^(\.LBB\d+_\d+):", line)
        if m:
            blocks.append((cur_label, cur_depth, cur_lines))
            cur_label = m.group(1)
            cur_depth = None
            cur_lines = []
        depth_m = re.search(r"Depth=(\d+)", line)
        if depth_m:
            cur_depth = max(cur_depth or 0, int(depth_m.group(1)))
        cur_lines.append(line)
    blocks.append((cur_label, cur_depth, cur_lines))
    return blocks


def find_hot_loop(blocks):
    fmla_blocks = [(label, depth, lines, sum(1 for l in lines if re.search(r"\bfmla\b", l)))
                   for (label, depth, lines) in blocks]
    fmla_blocks = [b for b in fmla_blocks if b[3] > 0]
    if not fmla_blocks:
        return None, None
    # Prefer deepest loop nest; tie-break by most fmla instructions.
    fmla_blocks.sort(key=lambda b: (b[1] or 0, b[3]), reverse=True)
    label, depth, lines, fmla_count = fmla_blocks[0]
    return label, lines


def classify_stack_access(line):
    """Returns ('vector'|'integer', 'spill'|'reload') or None.

    Lines explicitly marked "Folded Spill"/"Folded Reload" by LLVM are
    ALWAYS AAPCS64 callee-saved register preservation (prologue/epilogue),
    never hot-loop register-pressure spills -- excluded here regardless of
    which block they happen to fall in, which matters for the degenerate
    case where a function has no labeled loop blocks at all (tile == shape
    in every dim, fully collapsed to one basic block) and the fallback
    scope is the whole function including its own prologue.
    """
    if "Folded Spill" in line or "Folded Reload" in line:
        return None
    m = re.search(r"\b(str|stp|ldr|ldp|stur|ldur)\s+([a-z][0-9]+)", line)
    if not m or "[sp" not in line:
        return None
    mnemonic, reg = m.group(1), m.group(2)
    kind = "spill" if mnemonic.startswith(("st",)) else "reload"
    regclass = "vector" if reg[0] in ("q", "d", "s") else "integer"
    return regclass, kind


def analyze_one(s_path):
    text = open(s_path).read()
    blocks = parse_blocks(text)
    hot_label, hot_lines = find_hot_loop(blocks)

    degenerate = hot_label is None or hot_label == "PROLOGUE"
    scope_lines = hot_lines if hot_lines is not None else [l for (_, _, ls) in blocks for l in ls]

    vector_regs = set()
    for line in scope_lines:
        for m in re.finditer(r"\bv(\d+)\.", line):
            vector_regs.add(int(m.group(1)))

    hot_vec_spills = hot_vec_reloads = hot_int_spills = hot_int_reloads = 0
    for line in scope_lines:
        c = classify_stack_access(line)
        if c is None:
            continue
        regclass, kind = c
        if regclass == "vector" and kind == "spill":
            hot_vec_spills += 1
        elif regclass == "vector" and kind == "reload":
            hot_vec_reloads += 1
        elif regclass == "integer" and kind == "spill":
            hot_int_spills += 1
        elif regclass == "integer" and kind == "reload":
            hot_int_reloads += 1

    # ABI save/restore: callee-saved d8-d15 / x19-x30 folded spill/reload,
    # identified by LLVM's own "Folded Spill"/"Folded Reload" comments,
    # which only appear in the prologue/epilogue, never in a loop body.
    abi_saves = len(re.findall(r"Folded Spill", text))
    abi_restores = len(re.findall(r"Folded Reload", text))

    stack_frame = None
    m = re.search(r"sub\s+sp,\s*sp,\s*#(\d+)", text)
    if m:
        stack_frame = int(m.group(1))

    return {
        "hot_loop_label": hot_label,
        "hot_loop_is_whole_function": degenerate,
        "vector_registers_referenced": sorted(vector_regs),
        "vector_registers_referenced_count": len(vector_regs),
        "hot_loop_vector_spills": hot_vec_spills,
        "hot_loop_vector_reloads": hot_vec_reloads,
        "hot_loop_integer_spills": hot_int_spills,
        "hot_loop_integer_reloads": hot_int_reloads,
        "abi_callee_saved_folded_spills": abi_saves,
        "abi_callee_saved_folded_reloads": abi_restores,
        "stack_frame_bytes": stack_frame,
        "evidence_kind": "assembly-derived register-use evidence (not exact LLVM register-pressure analysis)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.load(open(args.candidates))
    out = []
    for c in data["candidates"]:
        if not c["legality"]["legal"] or "object_name" not in c:
            continue
        name = c["object_name"]
        s_path = os.path.join(args.objects_dir, f"{name}.s")
        if not os.path.isfile(s_path):
            continue
        rp = analyze_one(s_path)
        out.append({"shape": c["shape"], "tile": c["tile"], "object_name": name,
                     "register_pressure": rp})
        print(f"{name}: vregs={rp['vector_registers_referenced_count']} "
              f"vspills={rp['hot_loop_vector_spills']} vreloads={rp['hot_loop_vector_reloads']} "
              f"hot_loop={rp['hot_loop_label']}")

    with open(args.output, "w") as f:
        json.dump({"register_pressure": out}, f, indent=2)
    print(f"\nWrote {len(out)} register-pressure records to {args.output}")


if __name__ == "__main__":
    main()
