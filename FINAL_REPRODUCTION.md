# FINAL_REPRODUCTION.md — AArch64 Schedule-Unroll Slice

Exact reproduction workflow for Stages 10-18. Each stage's own
`commands.txt` (inside its artifact directory) is the authoritative,
complete command list with every flag; this document gives the top-level
sequence and points to those files rather than duplicating them in full.

Environment: dev host `ssh allen@100.87.220.5`
(`~/Desktop/Project/ml-graph-compiler-runtime`, LLVM 21.1.8
Optimized/Release), validation hardware `ssh allen@100.110.37.6`
(Raspberry Pi 5, Cortex-A76). Pi-facing scripts other than
`tools/run_aarch64_schedule_pi_validation.py` hardcode this Pi's address
(see `FINAL_AUDIT.md` §5) — reproducing on different hardware currently
requires editing `PI_HOST` in `tools/run_boundary_pi.py` and
`tools/run_multidomain_pi.py`.

## Host-Side Only (no SSH/Pi/toolchain dependency)

```
python3 -m unittest tests.test_aarch64_schedule_comparison \
  tests.test_aarch64_schedule_pi_validation \
  tests.test_aarch64_schedule_candidate_model \
  tests.test_aarch64_schedule_selection \
  tests.test_aarch64_schedule_multidomain \
  tests.test_aarch64_schedule_boundary
```
159 tests, all synthetic-fixture-based, run in well under a second.

## Stage 11 — Structural Validation

```
python3 tools/validate_aarch64_tiled_schedule_structure.py \
  --input mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x32.mlir \
  --tile-m 8 --tile-n 8 --tile-k 8 --schedule-unroll-k <1|2|4> \
  --output <out>.json
```

## Stage 12 — LLVM MIR Backend Evidence

```
python3 tools/compare_aarch64_schedule_variants.py \
  --output-dir artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling
```
Requires the AArch64 LLVM toolchain (`llc`, `mlir-translate`, `llvm-mca`)
on the dev host. Full per-candidate command breakdown (compile → extract
MIR at 5 pass boundaries → analyze → llvm-mca) is in that directory's
`commands.txt`.

## Stage 13 — Raspberry Pi Correctness + Benchmark

```
python3 tools/run_aarch64_schedule_pi_validation.py \
  --output-dir artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling
```
Requires SSH access to the Pi and a working `g++` toolchain there. Runs
14 candidates (11 Group A + 3 Group B), matched baseline/scheduled pairs
interleaved, with environment capture and thermal snapshots before/after.

## Stage 14 — Cost Model

```
python3 tools/run_aarch64_schedule_cost_model.py \
  --output-dir artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model \
  --schedule-candidate-mode calibrated \
  --emit-schedule-cost-breakdown <out>/production_cost_breakdown_calibrated.json
```
Defaults `--schedule-profile` to Stage 13's `pi_validation_results.json`
and `--stage12-json` to Stage 12's `schedule_comparison_results.json` —
no live hardware needed at this stage, only the prior stages' JSON
evidence.

## Stage 15 — Opt-In Selection and Materialization

```
python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode manual --schedule-unroll-k 1 \
  --tile-m 8 --tile-n 8 --tile-k 8 ...          # today's unchanged default path

python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode static --tile-m 8 --tile-n 8 --tile-k 8 ...

python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode calibrated \
  --schedule-profile artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/pi_validation_results.json \
  --tile-m 8 --tile-n 8 --tile-k 8 ...
```
Four scenarios (A-D, including a real incompatible-target fixture) are
fully spelled out in that directory's `commands.txt` and
`fixtures/`.

## Stage 16 — Multi-Domain Calibration

```
python3 tools/run_multidomain_pi.py          # Pi hardware required
python3 tools/run_multidomain_analysis.py    # builds multi_domain_profile.json etc.

python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode calibrated \
  --schedule-profile <out>/multi_domain_profile.json \
  --tile-m 8 --tile-n 8 --tile-k <8|4> ...
```

## Stage 17 — Boundary Search

```
python3 tools/run_boundary_pi.py             # Pi hardware required
python3 tools/run_boundary_analysis.py       # timing quality, winner/policy analysis

python3 tools/select_and_compile_aarch64_matmul_schedule.py \
  --schedule-candidate-mode calibrated \
  --schedule-profile <out>/updated_multidomain_profile.json \
  --tile-m 8 --tile-n 8 --tile-k <4|8> ...
```
Timer-overhead probe (optional, used to derive the ~37ns threshold, not
required for a normal rerun): `tools/timer_overhead_probe.cpp`, compiled
and run directly on the Pi.

## Stage 18 — Final Artifact Curation

```
python3 /tmp/build_final_manifest.py   # regenerates artifact_manifest.json + checksums.txt
```
This script is not checked into the repository (it is a one-shot
finalization tool); its logic and `STAGE_DIRS` mapping are documented in
`FINAL_AUDIT.md` §2 and `artifacts/backend_codegen/aarch64_schedule_final/commands.txt`.
Re-run it any time an upstream stage artifact directory changes, to keep
`checksums.txt` current.

## End-to-End Sanity Check

After any change to `tools/aarch64_schedule_candidate_model.py` or any
`select_and_compile_aarch64_matmul_schedule.py` dependency, re-run the
full 159-test suite (see "Host-Side Only" above) plus
`git diff --check`. This is exactly the Stage 19B regression procedure
documented in `FINAL_AUDIT.md` §7.
