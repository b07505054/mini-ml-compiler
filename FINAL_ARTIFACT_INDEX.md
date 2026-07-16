# FINAL_ARTIFACT_INDEX.md — AArch64 Schedule-Unroll Slice

Top-level index of every artifact directory, documentation file, and
top-level tool for this slice. Full 3-tier classification, per-file
checksums, and regeneration commands live in
`artifacts/backend_codegen/aarch64_schedule_final/artifact_manifest.json`
and `checksums.txt` — this file is a human-oriented map, not a
replacement for those.

## Start Here

- **`artifacts/backend_codegen/aarch64_schedule_final/summary.md`** (and `summary.json`) — the single canonical, reviewer-facing summary of the entire slice: pipeline, transformation, supported tiles/unroll factors, all 6 measured domains, correctness, backend/calibrated evidence, truth boundary, limitations.
- **`DOC/result/AARCH64_SCHEDULE_UNROLL_FINAL_REPORT.md`** — the consolidated Stage 10-17 narrative report (motivation through exact reproduction workflow).
- **`artifacts/backend_codegen/aarch64_schedule_final/TRUTH_BOUNDARY_AUDIT.md`** — confirms no forbidden universal/production-autotuning/custom-scheduler claims exist anywhere in the documentation.
- **`FINAL_AUDIT.md`** (repo root) — Stage 19 repository-finalization audit (this stage's own integrity check).
- **`FINAL_RELEASE_NOTES.md`, `FINAL_CHANGELOG.md`, `FINAL_TEST_SUMMARY.md`, `FINAL_REPRODUCTION.md`** (repo root) — the other 4 Stage 20 documents.

## Stage Artifact Directories

| Directory | Stage | Contents |
|---|---|---|
| `artifacts/backend_codegen/aarch64_matmul_bias_relu_scheduling/` | 12 | LLVM MIR backend evidence: `schedule_comparison_results.json`, per-candidate MIR/IR/asm/object under `candidates/`. |
| `artifacts/backend_codegen/aarch64_matmul_bias_relu_pi_scheduling/` | 13 | Real Pi correctness + benchmark: `pi_validation_results.json`, `environment.json`/`environment_raw.txt`, `thermal_snapshots.jsonl`, compiled objects under `compiled/`. |
| `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_cost_model/` | 14 | Evidence/cost model: `candidate_schema.json`, `evidence.json`, `static_ranking.json`, `calibrated_ranking.json`, `attribution_summary.json`, `shape_aware_findings.json`, `classification_summary.json`, both `production_cost_breakdown_*.json`. |
| `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_selection/` | 15 | Opt-in selection: 4 scenario selection reports (manual/static/calibrated/incompatible-fallback), real incompatible-target `fixtures/`, `pi_correctness_and_sanity_result.json`. |
| `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_multidomain/` | 16 | 3 new domains (cube64, altk, rect): `multi_domain_profile.json`, `domain_summary.json`, `compatibility_matrix.json`, `cross_domain_rejection_report.json`, per-domain selection reports, `pi_multidomain_results.json`. |
| `artifacts/backend_codegen/aarch64_matmul_bias_relu_schedule_boundary/` | 17 | 2 new domains (smallA, highK): `domain_design.json`, `timing_quality.json`, `winner_summary.json`, `static_model_evaluation.json`, `policy_comparison.json`, `boundary_analysis.json`, `integrated_selection_examples/`. |
| `artifacts/backend_codegen/aarch64_schedule_final/` | 18 | Canonical summary (`summary.json`/`.md`), `commands.txt`, `artifact_manifest.json`, `checksums.txt`, `TRUTH_BOUNDARY_AUDIT.md`. |

Approximate sizes: scheduling 6.8M, pi_scheduling 3.3M, cost_model 412K,
selection 1.2M, multidomain 3.4M, boundary 2.5M, schedule_final 240K
(~17.9M total across 7 directories).

## Source (Tools, Not Artifacts)

- `tools/aarch64_schedule_candidate_model.py` — candidate identity, evidence, classification, cost model, ranking (Stage 14 core library).
- `tools/run_aarch64_schedule_cost_model.py` — Stage 14 CLI driver.
- `tools/select_and_compile_aarch64_matmul_schedule.py` — Stage 15 opt-in selection + materialization driver.
- `tools/run_aarch64_schedule_pi_validation.py` — Stage 13 Pi orchestration (also reused directly by Stage 16/17).
- `tools/run_multidomain_pi.py`, `tools/run_multidomain_analysis.py` — Stage 16.
- `tools/run_boundary_pi.py`, `tools/run_boundary_analysis.py` — Stage 17.
- `tools/timer_overhead_probe.cpp` — Stage 17 clock-overhead probe.
- `tools/analyze_aarch64_machine_schedule.py`, `tools/analyze_aarch64_candidate_mir.py`, `tools/compare_aarch64_schedule_variants.py`, `tools/validate_aarch64_tiled_schedule_structure.py` — Stage 3-11 tools, reused unmodified.
- `mlir_passes/transforms/tile_schedule_matmul_bias_relu.template.mlir` — the project-owned Transform-dialect script.
- `mlir_passes/tools/generate_scheduled_transform.sh`, `mlir_passes/tools/generate_schedule_harness.sh`, `mlir_passes/tools/aarch64_matmul_bias_relu_schedule_harness.cpp.template` — Stage 10/13 generators and harness template.

## Tests

`tests/test_aarch64_schedule_comparison.py`,
`tests/test_aarch64_schedule_pi_validation.py`,
`tests/test_aarch64_schedule_candidate_model.py`,
`tests/test_aarch64_schedule_selection.py`,
`tests/test_aarch64_schedule_multidomain.py`,
`tests/test_aarch64_schedule_boundary.py` — 159 tests total, see
`FINAL_TEST_SUMMARY.md`.

## Documentation

- `ARCHITECTURE_STATUS.md` — one new maturity row (this slice).
- `KNOWN_GAPS.md` — three new gap rows (production integration scope, evidence scope, hardware-counter evidence).
- Each stage artifact directory's own `README.md` — stage-specific detail; not duplicated here or in the canonical summary.
