# FINAL_CHANGELOG.md — AArch64 Schedule-Unroll Slice

Entries are grouped by stage. All stages are additive to the repository;
no existing default compiler behavior was changed at any point (verified
by regression in every stage and re-verified in Stage 19).

## Stage 10-11 — Tiled-Scheduled Variant + Structural Validation
- Added `mlir_passes/transforms/tile_schedule_matmul_bias_relu.template.mlir` (Transform-dialect script applying stock `transform.loop.unroll` to the K-reduction loop).
- Added `mlir_passes/tools/generate_scheduled_transform.sh`.
- Structurally validated tile set `{8x8x8, 8x8x4, 4x8x8}` and unroll-factor set `{1, 2, 4}`.

## Stage 12 — LLVM MIR Backend Evidence
- Added `tools/analyze_aarch64_machine_schedule.py`, `tools/analyze_aarch64_candidate_mir.py`.
- Extracted and analyzed real LLVM 21.1.8 MIR at 5 pass boundaries for the `primary` domain (32x32x32, tile 8x8x8); correctly detected 11 spills / 12 reloads at uk4.
- Artifacts: `artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/`.

## Stage 13 — Raspberry Pi 5 Correctness + Benchmark Validation
- Added `mlir_passes/tools/aarch64_matmul_bias_relu_schedule_harness.cpp.template`, `mlir_passes/tools/generate_schedule_harness.sh`.
- Added `tools/run_aarch64_schedule_pi_validation.py`.
- Established `BENCHMARK_METHODOLOGY_VERSION = "stage13_pi5_harness_v1"`.
- Found (real result, not revised by later stages): spill counts alone were not reliable runtime predictors — uk4 measured fastest on real hardware despite having the most spills of the tested candidates in this domain.
- Artifacts: `artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/`.
- Tests: `tests/test_aarch64_schedule_pi_validation.py` (24 tests).

## Stage 14 — Schedule Attribution and Cost-Model Integration
- Added `tools/aarch64_schedule_candidate_model.py` (`CandidateKey`, evidence categories, classification, compatibility, cost model, ranking).
- Added `tools/run_aarch64_schedule_cost_model.py`.
- Corrected ranking policy: spill is a soft penalty, never an automatic hard veto (`RANKING_MODE_STATIC_SOFT_PENALTY` as the corrected default; `RANKING_MODE_STATIC_HARD_REJECT` retained only for legacy comparison).
- Tests: `tests/test_aarch64_schedule_candidate_model.py` (34 tests).
- Artifacts: `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model/`.

## Stage 15 — Opt-In Compiler-Side Candidate Selection
- Added `tools/select_and_compile_aarch64_matmul_schedule.py` — `--schedule-candidate-mode=manual|static|calibrated` (default `manual`, unchanged existing behavior); hard identity guard `verify_no_mismatch()`; deterministic 4-tier fallback.
- Added `mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_128x128x128.mlir` fixture.
- Tests: `tests/test_aarch64_schedule_selection.py` (34 tests).
- Artifacts: `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/` (including real incompatible-target fixtures).

## Stage 16 — Multi-Domain Calibration
- Added `tools/run_multidomain_pi.py`, `tools/run_multidomain_analysis.py`.
- Introduced 3 new independent domains: `cube64`, `altk`, `rect`; all measured uk4 as winner.
- Extended `select_and_compile_aarch64_matmul_schedule.py`'s profile-pool loader with a third schema branch (`stage16_multidomain_profile_v1`) to recognize multi-domain profiles.
- Fixed: cross-domain evidence isolation verified for 6 new domain pairs; `test_serving_cost_model_untouched` false-positive fixed (was substring-matching legitimate prose).
- Tests: `tests/test_aarch64_schedule_multidomain.py` (29 tests).
- Artifacts: `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_multidomain/`.

## Stage 17 — Boundary Search and Counterexample-Oriented Validation
- Added `tools/timer_overhead_probe.cpp` — empirical clock-read overhead measurement (~37ns on real Pi).
- Added `mlir_passes/test/backend_codegen/matmul_bias_relu_tiled_32x32x128.mlir` fixture.
- Added `tools/run_boundary_pi.py`, `tools/run_boundary_analysis.py`.
- Introduced 2 new counterexample-oriented domains: `smallA`, `highK`; both measured uk4 as winner — no counterexample found despite deliberately stressing uk4's known costs.
- Fixed two real bugs found via manual cross-checking before reporting: (1) evidence-merge condition that silently left blank-placeholder records incomplete for the `rect` domain, producing a false "static predicts 3/6" finding, corrected to "static predicts 2/6"; (2) `.endswith("uk4")` label-string matching replaced with a semantic `schedule_unroll_k == 4` field check.
- Tests: `tests/test_aarch64_schedule_boundary.py` (26 tests).
- Artifacts: `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/`.

## Stage 18 — Final Artifact Curation and Documentation Consolidation
- Generated canonical summary: `artifacts/backend_codegen/aarch64_schedule_final/` (`summary.json`, `summary.md`, `commands.txt`, `artifact_manifest.json`, `checksums.txt`) — 3-tier classification (Tier1 reviewer-facing, Tier2 reproducible JSON evidence, Tier3 raw MIR/IR/assembly) across all 6 stage directories, 449 files, ~17.1MB, via references + checksums (no file reorganization).
- Added `DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md` — consolidated Stage 10-17 report; explicitly preserves Stage 12/13 findings as separate, non-conflated facts.
- Added `artifacts/backend_codegen/aarch64_schedule_final/TRUTH_BOUNDARY_AUDIT.md` — confirmed no forbidden universal/production/custom-scheduler claims anywhere in documentation.
- Updated `ARCHITECTURE_STATUS.md` (one new maturity row) and `KNOWN_GAPS.md` (three new gap rows), append-only.
- Dead-code fix: removed unused `import statistics` and unused `field` import from `tools/aarch64_schedule_candidate_model.py`.

## Stage 19 — Repository Final Audit + Regression
- Audited: repository state, evidence integrity (152/152 JSON valid), documentation integrity (83 references checked, 0 genuine broken links), schema integrity, portability (found and documented a hardcoded Pi-host limitation in 2 driver scripts, not fixed as out of scope), dead code.
- Dead-code fix: removed two additional genuinely unused imports (`import os` in `tools/aarch64_schedule_candidate_model.py`; unused `sel` module import in `tools/run_multidomain_analysis.py`).
- Full regression: 159 tests across all 6 schedule-related test modules, all pass; `git diff --check` clean.
- Added `FINAL_AUDIT.md`.

## Stage 20 — Release Readiness
- Added `FINAL_RELEASE_NOTES.md`, `FINAL_CHANGELOG.md` (this file), `FINAL_TEST_SUMMARY.md`, `FINAL_REPRODUCTION.md`, `FINAL_ARTIFACT_INDEX.md`.
- No functionality changed in this stage.
