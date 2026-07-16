#!/usr/bin/env python3
"""extract_aarch64_candidate_mir.py

Stage 3 of the MIR-analysis slice: extracts real LLVM Machine IR at
several pipeline boundaries for one AArch64 tile candidate, using ONLY
llc's own --stop-after=<pass> flags -- no custom passes, no LLVM patches.

Pass boundaries used (discovered from `llc -debug-pass=Structure` on LLVM
21.1.8 for this exact aarch64-linux-gnu/cortex-a76 target; NOT assumed from
another LLVM version -- see the extraction pipeline's own header comment
and artifacts/.../aarch64_matmul_bias_relu_mir_analysis/README.md
"LLVM Machine-Pass Discovery" section):

  post_isel            --stop-after=finalize-isel
    Right after instruction selection completes. Virtual registers
    present, register classes attached, AArch64 machine opcodes
    (post-isel form) present. NOT yet scheduled/coalesced/allocated.

  pre_scheduler         --stop-before=machine-scheduler
    ADDED for the scheduling-analysis slice: stops immediately BEFORE the
    pre-RA machine scheduler pass runs (after two-address-instruction
    conversion, live-interval computation, and register coalescing, which
    all precede -machine-scheduler in the pipeline -- see
    -debug-pass=Structure output). Virtual registers still 100% present.
    Verified empirically to differ in real instruction ORDER from pre_ra
    below (e.g. loads are grouped in address-adjacent bursts here, vs.
    interleaved with compute after scheduling) -- this is the "before" half
    of the pre/post-scheduler comparison this slice adds.

  pre_ra               --stop-after=machine-scheduler
    Also usable as "post_scheduler" for the pre/post-scheduler comparison
    above -- this IS the machine-scheduler pass's own output boundary.
    The LAST pass boundary before the register allocator (-greedy) runs.
    Verified empirically: virtual registers are still 100% present here
    (185 distinct vregs for the 32x32x32/tm8_tn8_tk8 candidate), zero
    physical vector-register mentions. This is the honest "pre-RA" MIR --
    coalescing and pre-RA scheduling have already run, which is standard
    LLVM convention for what "pre-RA" means (immediately before the
    allocator pass itself, not immediately after isel).

  post_ra              --stop-after=virtregrewriter
    IMPORTANT, verified empirically: --stop-after=greedy alone does NOT
    produce physical registers in the printed MIR -- the greedy allocator
    populates an internal VirtRegMap, but the actual MachineOperands are
    not rewritten from virtual to physical registers until the SEPARATE
    -virtregrewriter pass runs afterward. Confirmed: --stop-after=greedy
    on the 32x32x32/tm8_tn8_tk8 candidate still shows 185 distinct vregs,
    0 physical vector-register mentions -- indistinguishable from pre_ra
    in that respect. --stop-after=virtregrewriter shows 0 vregs, 307
    physical-register mentions, and any allocator-inserted spill/reload
    code (real stack slots appear in this stage's `stack:` YAML list with
    `type: spill-slot`, distinct from `%fixed-stack.N` ABI argument
    slots). This is the ONLY pipeline point that satisfies the task's
    explicit requirement: "Do not call an artifact post-RA unless
    physical registers have been assigned and allocator effects are
    visible."

  post_prologue_epilogue   --stop-after=prologepilog
    Stack frame finalized (callee-saved register save/restore inserted,
    real stack-frame size fixed). Used to distinguish ABI/callee-saved
    save-restore from allocator spill-slot traffic (both are visible by
    this stage, but the `stack:` YAML section keeps them as separate
    entries: `type: spill-slot` for allocator spills vs. objects tagged
    with `callee-saved-register: '$dXX'` for ABI preservation).

  final_asm             (no --stop-after; the ordinary full pipeline)
    Real AArch64 assembly, for cross-checking FMLA presence and semantic
    equivalence against the object actually used for correctness/Pi
    testing.

Usage:
  python3 tools/extract_aarch64_candidate_mir.py \
    --llvm-ir generated.ll \
    --cpu cortex-a76 \
    --shape 32x32x32 \
    --tile-m 8 --tile-n 8 --tile-k 8 \
    --output-dir /tmp/mir_32x32x32_tm8_tn8_tk8 \
    [--regalloc greedy|fast] [--misched default|disabled]

--misched disabled passes -enable-misched=false to llc, which was
verified (this slice) to produce a genuinely different object
(different .text, different SHA-256) from the default -- i.e. it has
real effect on this pipeline, not a no-op flag. Note LLVM's own
-debug-pass=Structure output still LISTS the -machine-scheduler pass in
the pipeline even with this flag set; the flag changes the pass's
internal behavior rather than removing it from the pipeline.
"""
import argparse
import os
import subprocess
import sys

STAGE_PASSES = {
    "post_isel": "finalize-isel",
    "pre_scheduler": "machine-scheduler",  # --stop-BEFORE
    "pre_ra": "machine-scheduler",  # --stop-AFTER (== post_scheduler)
    "post_ra": "virtregrewriter",
    "post_prologue_epilogue": "prologepilog",
}
STOP_BEFORE_STAGES = {"pre_scheduler"}


def run_llc(llvm_ir, cpu, args_extra, output):
    cmd = ["llc", "-mtriple=aarch64-linux-gnu", f"-mcpu={cpu}", "-O2"] + args_extra + [llvm_ir, "-o", output]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return proc, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-ir", required=True)
    ap.add_argument("--cpu", default="cortex-a76")
    ap.add_argument("--shape", required=True, help="e.g. 32x32x32")
    ap.add_argument("--tile-m", type=int, required=True)
    ap.add_argument("--tile-n", type=int, required=True)
    ap.add_argument("--tile-k", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--regalloc", default="greedy", choices=["greedy", "fast"],
                     help="Register allocator to use (prior MIR-analysis slice's greedy-vs-fast experiment). Default: greedy (LLVM's normal optimized default for this target).")
    ap.add_argument("--misched", default="default", choices=["default", "disabled"],
                     help="default: LLVM's normal pre-RA machine scheduler. disabled: -enable-misched=false (Stage 4 scheduler on/off experiment).")
    args = ap.parse_args()

    if not os.path.isfile(args.llvm_ir):
        print(f"error: LLVM IR input not found: {args.llvm_ir}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.shape}_tm{args.tile_m}_tn{args.tile_n}_tk{args.tile_k}_{args.regalloc}_misched-{args.misched}"

    regalloc_flag = [] if args.regalloc == "greedy" else [f"-regalloc={args.regalloc}"]
    misched_flag = [] if args.misched == "default" else ["-enable-misched=false"]
    common_flags = regalloc_flag + misched_flag

    results = {}
    for stage, pass_name in STAGE_PASSES.items():
        out_path = os.path.join(args.output_dir, f"{prefix}_{stage}.mir")
        boundary_flag = f"--stop-before={pass_name}" if stage in STOP_BEFORE_STAGES else f"--stop-after={pass_name}"
        proc, cmd = run_llc(args.llvm_ir, args.cpu, common_flags + [boundary_flag], out_path)
        if proc.returncode != 0:
            print(f"error extracting {stage} (pass={pass_name}):", file=sys.stderr)
            print(" ".join(cmd), file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            results[stage] = {"ok": False, "pass": pass_name, "stderr": proc.stderr[-2000:]}
            continue
        results[stage] = {"ok": True, "pass": pass_name, "path": out_path}
        print(f"OK  {stage:24s} ({boundary_flag}) -> {out_path}")

    # Final assembly + object, full pipeline, same regalloc/misched choice.
    asm_path = os.path.join(args.output_dir, f"{prefix}.s")
    obj_path = os.path.join(args.output_dir, f"{prefix}.o")
    proc_s, _ = run_llc(args.llvm_ir, args.cpu, common_flags + ["-filetype=asm"], asm_path)
    proc_o, _ = run_llc(args.llvm_ir, args.cpu, common_flags + ["-filetype=obj"], obj_path)
    if proc_s.returncode != 0 or proc_o.returncode != 0:
        print("error: final assembly/object generation failed", file=sys.stderr)
        print(proc_s.stderr, file=sys.stderr)
        print(proc_o.stderr, file=sys.stderr)
        return 1
    results["final_asm"] = {"ok": True, "path": asm_path}
    results["final_obj"] = {"ok": True, "path": obj_path}
    print(f"OK  final_asm                  -> {asm_path}")
    print(f"OK  final_obj                  -> {obj_path}")

    all_ok = all(r.get("ok") for r in results.values())
    print("\nEXTRACTION: " + ("PASS" if all_ok else "FAIL (see errors above)"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
