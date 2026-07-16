# FINAL_TEST_SUMMARY.md — AArch64 Schedule-Unroll Slice

## Test Modules and Counts

| Module | Tests | Stage | Focus |
|---|---|---|---|
| `tests/test_aarch64_schedule_comparison.py` | 12 | 11-12 | Structural tile/unroll validation, LLVM MIR spill/reload extraction correctness |
| `tests/test_aarch64_schedule_pi_validation.py` | 24 | 13 | Real Pi harness generation, correctness comparison, methodology versioning |
| `tests/test_aarch64_schedule_candidate_model.py` | 34 | 14 | Candidate identity, evidence/provenance, classification, compatibility, cost model, ranking |
| `tests/test_aarch64_schedule_selection.py` | 34 | 15 | Manual/static/calibrated mode selection, identity guard, deterministic fallback |
| `tests/test_aarch64_schedule_multidomain.py` | 29 | 16 | Multi-domain schema, cross-domain rejection, regression (manual default, calibrated opt-in, no universal uk4 rule, ServingCostModel untouched) |
| `tests/test_aarch64_schedule_boundary.py` | 26 | 17 | Timing-quality classification, winner classification, stress-domain identity, selection-policy regression |
| **Total** | **159** | | |

## Result

```
python3 -m unittest tests.test_aarch64_schedule_comparison \
  tests.test_aarch64_schedule_pi_validation \
  tests.test_aarch64_schedule_candidate_model \
  tests.test_aarch64_schedule_selection \
  tests.test_aarch64_schedule_multidomain \
  tests.test_aarch64_schedule_boundary

Ran 159 tests in 0.048s

OK
```

Run on the dev host (`ssh allen@100.87.220.5`, repo root
`~/Desktop/Project/ml-graph-compiler-runtime`, `.venv` activated), as the
final Stage 19B regression pass, after the two additional dead-code fixes
described in `FINAL_AUDIT.md` §6. All 159 tests pass; zero failures, zero
errors, zero skips.

## What Is NOT Covered by This Suite

- No test in this suite runs real hardware — Pi-hardware execution is
  exercised by the `tools/run_*_pi.py` orchestration scripts directly
  (documented and reproducible via `FINAL_REPRODUCTION.md`), not by
  `unittest`. The unit tests validate the harness-generation, evidence,
  selection, and classification logic around that hardware execution.
- No lit/FileCheck-style MLIR test exists specifically named for this
  slice beyond the structural fixtures under
  `mlir_passes/test/backend_codegen/`, which are exercised through the
  compile driver script and `test_aarch64_schedule_comparison.py`, not a
  separate lit test target.
- Regression tests for other repository subsystems (`ServingCostModel`,
  vLLM `max_num_seqs` selection, etc.) are out of scope for this slice and
  were not run as part of this summary — only confirmed untouched via
  `git status` (see `FINAL_AUDIT.md` §1).

## Key Regression Guarantees Asserted by This Suite

- `test_default_remains_manual` (Stage 16 + 17): default `--schedule-candidate-mode` is `manual` everywhere.
- `test_calibrated_remains_opt_in` (Stage 16 + 17): calibrated mode never activates without an explicit flag + profile.
- `test_no_universal_uk4_rule_in_code` / `test_no_universal_uk4_rule_in_ranking_code` (Stage 16 + 17): no code path hardcodes uk4 as a universal winner.
- `test_serving_cost_model_untouched` (Stage 16): real `git status` check confirming the separate `ServingCostModel` subsystem was not imported, called, or modified.
- `test_spill_hard_rejection_does_not_return_as_default` (Stage 17): the legacy hard-reject ranking mode is never the active default.
- `test_universal_uk4_policy_marked_unsafe_not_exposed` (Stage 17): the "always pick uk4" policy variant evaluated during the Stage 17 policy comparison is explicitly marked unsafe and not exposed as a real selectable mode.
- `test_fallback_remains_deterministic` (Stage 17): repeated fallback selection on identical inputs is byte-identical.
