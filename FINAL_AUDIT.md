# FINAL_AUDIT.md — AArch64 Schedule-Unroll Slice: Stage 19 Repository Final Audit

Scope: the AArch64 machine-scheduling evidence + opt-in schedule-unroll
selection slice added across Stages 10-18 (fused MatMul-Bias-ReLU only).
This audit does not cover the rest of the repository, and does not
re-validate any other subsystem (e.g. `ServingCostModel`, vLLM
`max_num_seqs` selection) beyond confirming those subsystems were left
untouched.

## 1. Repository State

38 paths changed relative to `main` (`git status --short`):

- 4 modified (`M`):
  - `ARCHITECTURE_STATUS.md`, `KNOWN_GAPS.md` — Stage 18 documentation additions (one new maturity row, three new gap rows), append-only.
  - `mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh`, `tools/extract_aarch64_candidate_mir.py` — pre-existing modifications from earlier stages of this same engagement (Stage 3-6, candidate-MIR extraction support), not touched during Stages 18-20.
- 34 untracked (`??`): new tools, tests, MLIR fixtures/templates, `DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md`, and 7 artifact directories under `artifacts/backend_codegen/`.
- `git diff --check` — clean, no whitespace errors, exit 0.
- Two files outside this slice appear untracked (`tools/select_vllm_max_num_seqs.py`, `tests/test_vllm_max_num_seqs_selection.py`). These are pre-existing, unrelated working-tree content and were not created, modified, or inspected beyond confirming they exist — preserved per instruction.
- No files were deleted or renamed.

## 2. Evidence Integrity

- 152 JSON files across all 7 artifact directories (Stages 12-18) parsed successfully; 0 invalid.
- `artifacts/backend_codegen/aarch64_schedule_final/artifact_manifest.json` + `checksums.txt` cover all 6 upstream stage directories (Stage 12-17) plus the final summary itself; classification is Tier 1/2/3 by content type, not file relocation.
- Two additional genuinely-unused imports were found and removed during this audit (see §6); neither is hashed by the manifest (`tools/` is out of its scope), so `checksums.txt` did not need regeneration.
- Measured results (spill/reload counts, winners, timings) were not touched by this audit — the two import removals are dead-code cleanup in library/analysis modules with no behavioral effect, confirmed by full regression (§7).

## 3. Documentation Integrity

- Documentation-link audit checked 83 bare-filename references (in backticks, `.py`/`.sh`/`.json`/`.md`/`.mlir`/`.cpp`/`.txt`) across the 6 stage-artifact READMEs, `aarch64_schedule_final/summary.md`, and `DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md`.
- 68 resolved directly (repo-root-relative or sibling-to-the-README).
- 15 did not resolve under either rule. All 15 were manually confirmed (via `find`) to be real, existing files referenced by bare filename rather than full path — e.g. `select_and_compile_aarch64_matmul_schedule.py` (exists at `tools/select_and_compile_aarch64_matmul_schedule.py`), `generate_scheduled_transform.sh` (at `mlir_passes/tools/`), three `fixture_*.json` files (at `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/fixtures/`), `aarch64_matmul_bias_relu_tiled_harness.cpp` / `aarch64_matmul_bias_relu_repeated_call_test.cpp` (at `mlir_passes/tools/`), `tile_schedule_matmul_bias_relu.template.mlir` (at `mlir_passes/transforms/`), `pi_validation_results.json` (at `artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/`), and one generic mention of "each stage's own `commands.txt`" (a prose pattern reference, not a single-file link).
- **Result: no genuinely broken navigational links.** The 15 flagged items are a documentation-style imprecision (bare filename instead of full relative path), not a defect.

## 4. Schema Integrity

- `aarch64_schedule_candidate_model_v1` — candidate/evidence schema, used consistently by Stages 14-17.
- `stage16_multidomain_profile_v1` — multi-domain profile schema, correctly recognized by `select_and_compile_aarch64_matmul_schedule.py`'s profile-pool loader (third schema branch, added Stage 16, regression-tested).
- `stage13_pi5_harness_v1` — `BENCHMARK_METHODOLOGY_VERSION`; stale-version profiles are rejected wholesale at load time (regression-tested).
- `aarch64_schedule_final_summary_v1` — Stage 18 canonical summary schema.
- No schema-version collisions or silent-fallback paths found; all cross-schema loads are explicit and tested.

## 5. Portability Audit

- `MLIR_BIN` environment-variable convention (default path, overridable) — established pattern from earlier stages, unchanged.
- `tools/run_aarch64_schedule_pi_validation.py` exposes `--pi-host` as an overridable CLI default (`allen@100.110.37.6`).
- **Finding**: `tools/run_boundary_pi.py` and `tools/run_multidomain_pi.py` hardcode `PI_HOST = "allen@100.110.37.6"` as a bare module-level constant with **no CLI override**, unlike the Stage 13 script they both reuse functions from. This is a real, minor portability gap — reproducing Stage 16/17's Pi runs against different hardware would require editing these two files directly rather than passing a flag. Not fixed here (a behavior change is out of scope for a finalization-only pass); recorded as a limitation.
- No other hardcoded absolute paths, usernames, or IPs found outside the two files above and their already-documented, overridable Stage 13 counterpart.

## 6. Additional Dead-Code Findings (this audit)

Beyond the two unused imports fixed and reported in the Stage 18/19 work (`import statistics`, unused `field` import in `tools/aarch64_schedule_candidate_model.py`), an AST-based unused-import scan of all 8 Stage 14-17 tool modules found two more genuinely unused imports, fixed here:

- `tools/aarch64_schedule_candidate_model.py`: unused `import os` (removed).
- `tools/run_multidomain_analysis.py`: unused `import select_and_compile_aarch64_matmul_schedule as sel` (removed — the module was imported but never referenced).

No other unused imports, dead functions, or stale comments were found in the scanned files.

## 7. Full Regression (Stage 19B)

- `python3 -m unittest tests.test_aarch64_schedule_comparison tests.test_aarch64_schedule_pi_validation tests.test_aarch64_schedule_candidate_model tests.test_aarch64_schedule_selection tests.test_aarch64_schedule_multidomain tests.test_aarch64_schedule_boundary` — **159 tests, all pass**, re-run after the two additional import removals in §6.
- `git diff --check` — clean.
- JSON schema/parse validation — 152/152 valid (§2).
- Documentation link validation — see §3, no genuine breaks.
- Cross-domain evidence rejection — already covered by Stage 16/17's own tests (30 pairs, all correctly rejected); re-confirmed passing as part of the 159-test run above, not re-derived independently in this audit.
- No accidental label-matching — covered by existing regression tests (`test_no_universal_uk4_rule_in_code`, `test_no_universal_uk4_rule_in_ranking_code`, and the Stage 17 field-based winner check); re-confirmed passing.
- Deterministic iteration/fallback — covered by `test_fallback_remains_deterministic` and per-stage reruns during Stages 13-17; re-confirmed passing.
- Duplicated logic — the per-domain `build_static_evidence_from_metrics`-style helper functions are intentionally near-duplicated across Stage 16/17 analysis scripts (independent, self-contained domain analysis is a deliberate tradeoff to avoid coupling counterexample-search logic to the calibration-domain logic it is meant to independently stress-test); acknowledged, not treated as a defect.

## 8. Remaining Limitations (unchanged from Stage 18C truth-boundary audit, restated for completeness)

- Measured only on Raspberry Pi 5 / Cortex-A76 / FP32 / this one tiled NEON microkernel family.
- `schedule-unroll-k=4` won in all 6 tested domains but is **not** claimed universal; calibrated/uk4 selection is opt-in only (default mode remains `manual`).
- Static backend evidence alone correctly predicts the measured winner in exactly 2 of 6 domains (both spill-free winners) — real Pi calibration is required for reliable selection, not claimed to be solved by static scoring.
- Supported tiles are limited to `{(8,8,8), (8,8,4), (4,8,8)}`; supported unroll factors to `{1,2,4}` (must evenly divide the K-loop trip count).
- No hardware performance-counter evidence (cycles, cache misses) was collected — only wall-clock timing with an empirically validated overhead floor (~37ns).
- `tools/run_boundary_pi.py` / `tools/run_multidomain_pi.py` hardcode the validation Pi's Tailscale address with no CLI override (§5) — a reproducibility inconvenience, not a correctness issue.
- Two of the 15 flagged documentation references remain stylistically imprecise (bare filename vs. full path) even though they resolve to real files (§3) — cosmetic only.

## Result: PASS

No genuine defects block finalization. Two additional dead-code items were found and fixed during this audit (§6); one real but minor portability gap was found and documented, not fixed, as out of scope for a finalization-only pass (§5). All other checklist items pass cleanly.
