#!/usr/bin/env python3
"""analyze_aarch64_machine_schedule.py

Stage 5 of the scheduling-analysis slice: parses MIR (and, for the FMLA/
load/store/branch counts, final assembly) to report machine-scheduling
metrics -- FMLA accumulator dependency chains, load-to-use distance, and
independent-instruction-distance between consecutive writes to the same
accumulator. All are MIR-INSTRUCTION-ORDER metrics (position in the
static instruction stream), NOT cycle-accurate latency -- labeled as such
throughout ("scheduling-distance metric, not cycle-accurate latency", per
the task brief).

Accumulator identification: an FMLA-family instruction in this project's
generated MIR takes the form `%N:fpr128 = FMLAv4i32_indexed %N, %A, %B,
idx` -- i.e. destination and first source operand are the SAME virtual
register (an in-place accumulator update; MIR is not strict SSA once a
loop body reuses one vreg across iterations of unrolled reduction steps).
An "accumulator chain" is the set of FMLA instructions, in program order,
that share this same self-referencing vreg.

Reuses tools/analyze_aarch64_candidate_mir.py's MIR-document parsing
(split_machine_functions / pick_kernel_function / extract_body) rather
than re-implementing it.

Usage:
  python3 tools/analyze_aarch64_machine_schedule.py \
    --mir some_stage.mir \
    --asm some_stage.s \
    --output metrics.json --report report.txt
"""
import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_aarch64_candidate_mir as mirlib  # noqa: E402

FMLA_RE = re.compile(r"^\s*%(\d+):fpr128\s*=\s*(?:nofpexcept\s+)?(FMLA\S*)\s+(?:killed\s+)?%(\d+)(?::\w+)?,")
LOAD_DEF_RE = re.compile(r"^\s*%(\d+):fpr128\s*=\s*(LDR\S*)\b")
ANY_VREG_USE_RE = re.compile(r"%(\d+)\b")


def load_kernel_body_lines(mir_path):
    with open(mir_path) as f:
        text = f.read()
    functions = mirlib.split_machine_functions(text)
    name, doc = mirlib.pick_kernel_function(functions)
    body = mirlib.extract_body(doc)
    lines = [l for l in body.splitlines() if l.strip()]
    instr_lines = [l for l in lines
                   if not re.match(r"^\s*bb\.\d+", l)
                   and not l.strip().startswith(("successors:", "liveins:", ";"))]
    return name, instr_lines


def accumulator_chain_metrics(instr_lines):
    """Returns (chains: {accum_vreg: [positions]}, per-chain gap lists)."""
    chains = {}
    for pos, line in enumerate(instr_lines):
        m = FMLA_RE.match(line)
        if not m:
            continue
        dest, opcode, src1 = m.group(1), m.group(2), m.group(3)
        if dest == src1:
            chains.setdefault(dest, []).append(pos)

    chain_lengths = [len(v) for v in chains.values()]
    gaps = []  # instructions between consecutive writes to the SAME accumulator
    for positions in chains.values():
        for a, b in zip(positions, positions[1:]):
            gaps.append(b - a - 1)  # intervening instruction count

    return {
        "accumulator_chains": len(chains),
        "max_accumulator_chain_length": max(chain_lengths) if chain_lengths else None,
        "min_accumulator_chain_length": min(chain_lengths) if chain_lengths else None,
        "average_accumulator_chain_length": (
            round(statistics.mean(chain_lengths), 2) if chain_lengths else None
        ),
        "same_accumulator_distance": {
            "note": "intervening (independent-of-that-chain) instruction count between consecutive FMLA writes to the same accumulator vreg -- a scheduling-distance metric derived from MIR instruction order, NOT cycle-accurate latency",
            "min": min(gaps) if gaps else None,
            "median": statistics.median(gaps) if gaps else None,
            "p95": (sorted(gaps)[int(0.95 * (len(gaps) - 1))] if len(gaps) > 1 else (gaps[0] if gaps else None)),
            "max": max(gaps) if gaps else None,
            "sample_count": len(gaps),
        },
    }


def load_to_use_metrics(instr_lines):
    """For each vector load, distance (in instructions) to its first
    subsequent use as an FMLA operand."""
    distances = []
    for pos, line in enumerate(instr_lines):
        m = LOAD_DEF_RE.match(line)
        if not m:
            continue
        dest_reg = m.group(1)
        for later_pos in range(pos + 1, len(instr_lines)):
            later_line = instr_lines[later_pos]
            fmla_m = FMLA_RE.match(later_line)
            if fmla_m and dest_reg in (fmla_m.group(1), fmla_m.group(3)):
                # dest_reg used as an FMLA operand (accumulator OR operand)
                if re.search(rf"%{dest_reg}\b", later_line):
                    distances.append(later_pos - pos - 1)
                    break
            elif re.search(rf"%{dest_reg}\b", later_line) and not FMLA_RE.match(later_line):
                # used by something else first (e.g. another load's address
                # computation) -- still record first use, not FMLA-specific,
                # for completeness; but only count toward the FMLA-specific
                # metric if an FMLA use is what we found above. Stop scanning
                # this load once ANY use is found to avoid re-counting.
                break

    if not distances:
        return {
            "min": None, "median": None, "p95": None, "max": None,
            "sample_count": 0,
            "note": "no vector loads found feeding an FMLA directly in this stage's kernel body",
        }
    sorted_d = sorted(distances)
    return {
        "min": sorted_d[0],
        "median": statistics.median(sorted_d),
        "p95": sorted_d[int(0.95 * (len(sorted_d) - 1))] if len(sorted_d) > 1 else sorted_d[0],
        "max": sorted_d[-1],
        "sample_count": len(distances),
        "note": "instructions between a vector load and its first use as an FMLA operand (accumulator-in or multiplicand) -- MIR instruction-order distance, not cycle-accurate latency",
    }


def instruction_counts(instr_lines):
    fmla = sum(1 for l in instr_lines if re.search(r"\bFMLA\S*\b", l))
    loads = sum(1 for l in instr_lines if re.search(r"(?:^|=\s*)(LDR\S*)\b", l.strip()))
    stores = sum(1 for l in instr_lines if re.search(r"(?:^|=\s*)(STR\S*)\b", l.strip()) or re.match(r"^\s*STR", l.strip()))
    branches = sum(1 for l in instr_lines if re.search(r"\b(B|Bcc|CBZ|CBNZ|TBZ|TBNZ)\b", l))
    copies = sum(1 for l in instr_lines if re.search(r"\bCOPY\b", l))
    return {"fmla_count": fmla, "load_count": loads, "store_count": stores,
            "branch_count": branches, "copy_count": copies,
            "hot_loop_instruction_count": len(instr_lines)}


def analyze_mir(mir_path):
    name, instr_lines = load_kernel_body_lines(mir_path)
    result = {"function_name": name}
    result.update(instruction_counts(instr_lines))
    result.update(accumulator_chain_metrics(instr_lines))
    result["load_to_use_distance"] = load_to_use_metrics(instr_lines)
    return result


def analyze_asm_fmla(asm_path):
    with open(asm_path) as f:
        text = f.read()
    fmla = len(re.findall(r"^\s+fmla\s", text, re.M))
    branches = len(re.findall(r"^\s+(b\.[a-z]+|cbz|cbnz|b)\s", text, re.M))
    return {"asm_fmla_count": fmla, "asm_branch_count": branches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mir", required=True, help="MIR file to analyze (e.g. pre_scheduler or pre_ra/post_scheduler)")
    ap.add_argument("--asm", help="optional final assembly, for FMLA/branch cross-check")
    ap.add_argument("--output", required=True)
    ap.add_argument("--report")
    args = ap.parse_args()

    result = analyze_mir(args.mir)
    if args.asm:
        result["assembly_cross_check"] = analyze_asm_fmla(args.asm)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    lines = [
        f"Machine-scheduling analysis: {args.mir}",
        f"function={result['function_name']}",
        f"hot_loop_instructions={result['hot_loop_instruction_count']} fmla={result['fmla_count']} "
        f"loads={result['load_count']} stores={result['store_count']} branches={result['branch_count']} copies={result['copy_count']}",
        f"accumulator_chains={result['accumulator_chains']} "
        f"max_chain_length={result['max_accumulator_chain_length']} "
        f"avg_chain_length={result['average_accumulator_chain_length']}",
        f"same_accumulator_distance: {result['same_accumulator_distance']}",
        f"load_to_use_distance: {result['load_to_use_distance']}",
    ]
    if args.asm:
        lines.append(f"assembly cross-check: {result['assembly_cross_check']}")
    report_text = "\n".join(lines)
    if args.report:
        with open(args.report, "w") as f:
            f.write(report_text + "\n")
    print(report_text)


if __name__ == "__main__":
    main()
