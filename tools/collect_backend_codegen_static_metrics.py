#!/usr/bin/env python3
"""Collect reproducible static backend metrics for each compiled shape:
LLVM IR instruction counts (raw kernel + ciface wrapper, split), AArch64
instruction counts per mnemonic class (loads/stores/branches/FP arithmetic),
and object file size.

All counts are derived from the actual .ll / .o files produced by
compile_hir_matmul_bias_relu_aarch64.sh -- nothing here is a manual estimate.

Methodology (documented so it is reproducible without reading this file):
  - LLVM IR instruction count: lines within a `define ... { ... }` body that
    either contain " = " (a value-producing instruction) or start with one
    of store/br/ret/call void (void instructions with no result value).
  - AArch64 instruction count: `llvm-objdump -d --no-show-raw-insn <obj>`,
    counting lines that start with an address followed by ':'. Split per
    function using the `<symbol>:` disassembly headers.
  - Loads/stores/branches/FP arithmetic: classified from the AArch64
    mnemonic in each disassembled instruction line (2nd tab-separated
    field). Load mnemonics: ldr, ldp, ldur (and w/h/b/sw variants). Store
    mnemonics: str, stp, stur. Branch mnemonics: b, b.*, bl, cbz, cbnz, ret.
    FP arithmetic mnemonics: fadd, fsub, fmul, fdiv, fmax, fmin, fmadd,
    fmaximum, fminimum (this kernel's real assembly only needs
    fadd/fmul/fmax).
  - Object file size: byte size of the .o file on disk.

Usage:
  collect_backend_codegen_static_metrics.py --output-dir <dir> --shapes 8x8x8 16x16x16 32x32x32
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LOAD_MNEMONICS = {"ldr", "ldp", "ldur", "ldrb", "ldrh", "ldrsw", "ldrsb", "ldrsh"}
STORE_MNEMONICS = {"str", "stp", "stur", "strb", "strh"}
BRANCH_MNEMONICS_PREFIXES = ("b.", "cbz", "cbnz", "tbz", "tbnz")
BRANCH_MNEMONICS_EXACT = {"b", "bl", "blr", "br", "ret"}
FP_ARITH_MNEMONICS = {"fadd", "fsub", "fmul", "fdiv", "fmax", "fmin",
                      "fmadd", "fmsub", "fnmadd", "fnmsub",
                      "fmaximum", "fminimum"}


def count_ir_instructions(ll_path: Path, symbol: str) -> int:
    text = ll_path.read_text()
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("define") and f"@{symbol}(" in line:
            start = i
            break
    if start is None:
        raise RuntimeError(f"symbol {symbol} not found in {ll_path}")
    count = 0
    for line in lines[start + 1:]:
        if line.startswith("}"):
            break
        if " = " in line or re.match(r"^\s*(store|br|ret|call void)\b", line):
            count += 1
    return count


def disassemble(obj_path: Path) -> str:
    result = subprocess.run(
        ["llvm-objdump", "-d", "--no-show-raw-insn", str(obj_path)],
        check=True, capture_output=True, text=True)
    return result.stdout


def classify_instructions(disasm: str, symbol: str):
    lines = disasm.splitlines()
    in_symbol = False
    mnemonics = []
    for line in lines:
        header_match = re.match(r"^[0-9a-f]+ <([^>]+)>:$", line)
        if header_match:
            in_symbol = (header_match.group(1) == symbol)
            continue
        if not in_symbol:
            continue
        m = re.match(r"^\s*[0-9a-f]+:\s*\t(\S+)", line)
        if m:
            mnemonics.append(m.group(1))
    loads = sum(1 for mn in mnemonics if mn in LOAD_MNEMONICS)
    stores = sum(1 for mn in mnemonics if mn in STORE_MNEMONICS)
    branches = sum(1 for mn in mnemonics
                   if mn in BRANCH_MNEMONICS_EXACT or mn.startswith(BRANCH_MNEMONICS_PREFIXES))
    fp_arith = sum(1 for mn in mnemonics if mn in FP_ARITH_MNEMONICS)
    return {
        "total": len(mnemonics),
        "loads": loads,
        "stores": stores,
        "branches": branches,
        "fp_arithmetic": fp_arith,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shapes", nargs="+", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    result = {"methodology": {
        "llvm_ir_instructions": "lines in the define{} body containing ' = ' or starting with store/br/ret/call void",
        "aarch64_instructions": "llvm-objdump -d --no-show-raw-insn, address-prefixed lines, split per <symbol>: header",
        "object_size_bytes": "stat size of the .o file",
    }, "shapes": {}}

    for shape in args.shapes:
        ll_path = out_dir / f"matmul_bias_relu_{shape}.ll"
        obj_path = out_dir / f"matmul_bias_relu_{shape}.o"
        raw_symbol = f"matmul_bias_relu_{shape}"
        ciface_symbol = f"_mlir_ciface_matmul_bias_relu_{shape}"

        ir_raw = count_ir_instructions(ll_path, raw_symbol)
        ir_ciface = count_ir_instructions(ll_path, ciface_symbol)

        disasm = disassemble(obj_path)
        asm_raw = classify_instructions(disasm, raw_symbol)
        asm_ciface = classify_instructions(disasm, ciface_symbol)

        result["shapes"][shape] = {
            "llvm_ir_instructions": {
                "raw_kernel": ir_raw,
                "ciface_wrapper": ir_ciface,
                "total": ir_raw + ir_ciface,
            },
            "aarch64_instructions": {
                "raw_kernel": asm_raw,
                "ciface_wrapper": asm_ciface,
                "total": asm_raw["total"] + asm_ciface["total"],
                "loads_total": asm_raw["loads"] + asm_ciface["loads"],
                "stores_total": asm_raw["stores"] + asm_ciface["stores"],
                "branches_total": asm_raw["branches"] + asm_ciface["branches"],
                "fp_arithmetic_total": asm_raw["fp_arithmetic"] + asm_ciface["fp_arithmetic"],
            },
            "object_size_bytes": obj_path.stat().st_size,
        }

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
