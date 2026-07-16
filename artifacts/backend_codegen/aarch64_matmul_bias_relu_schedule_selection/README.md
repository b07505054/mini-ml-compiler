# Opt-In Compiler-Driver Schedule Candidate Selection

**The AArch64 matmul compiler driver can optionally rank restricted
schedule candidates using provenance-checked backend and Raspberry Pi
evidence, then materialize the selected K-unroll schedule into executable
MLIR.**

This is **not**: general production autotuning, arbitrary loop
interchange, universal uk2 optimality, cross-target calibration, or a
default automatic schedule selection. Default behavior is unchanged
(manual mode, exactly today's `--schedule-unroll-k` flag). Calibrated
mode is opt-in only and is never enabled by default anywhere in this repo.

## 1. Integration point (audit finding)

`--schedule-unroll-k` was already a real CLI flag on
`mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh` (`--variant
tiled-scheduled`), flowing unchanged into
`generate_scheduled_transform.sh` and from there into the Transform-
dialect script materialized by `mlir-opt`. **What was missing**: nothing
verified that a caller's *intended* selection actually matched what got
passed to the compile script -- there was no automated guard against, say,
a selector computing "uk2" while a typo or stale variable passed "uk1" to
the actual compile invocation.

**Chosen integration**: a new Python driver,
`tools/select_and_compile_aarch64_matmul_schedule.py`, sits one layer
above the existing compile script and calls it, unmodified, as a
subprocess with the selected `--tile-m/--tile-n/--tile-k/--schedule-unroll-k`
values. This reuses the existing materialization mechanism entirely --
**no second pass pipeline, no second Transform-dialect script, no second
MLIR lowering path was built**. Accurate description: *"Opt-in
compiler-driver schedule candidate selection controlling a real MLIR
schedule materialization pass."* This is explicitly not a native C++
compiler pass -- selection happens in Python, materialization happens in
the existing, unmodified shell script + MLIR/LLVM toolchain.

The gap is closed by `verify_no_mismatch()`: after compilation, the
literal `--tile-m/--tile-n/--tile-k/--schedule-unroll-k` values passed to
the compile script are used to reconstruct a `CandidateKey`, which is
asserted equal to the selected candidate's key. A mismatch raises
`ArtifactIdentityMismatchError` and aborts -- never a report that claims
one candidate while a different one was compiled.

## 2. Experimental interface

```
--schedule-candidate-mode=manual|static|calibrated   (default: manual)
--schedule-profile=<path>                             (required for calibrated)
--schedule-unroll-k=<integer>                          (required for manual)
--emit-schedule-selection=<path>
--tile-m / --tile-n / --tile-k                         (default 8/8/8)
```

- **manual** (default): today's existing behavior. The compiler never
  overrides a user-supplied `--schedule-unroll-k`.
- **static**: generates the legal candidate set, ranks with
  `RANKING_MODE_STATIC_SOFT_PENALTY` (Stage 14's corrected policy --
  spills are penalties, never automatic vetoes), selects deterministically,
  emits the full score breakdown.
- **calibrated**: requires `--schedule-profile`. Ranks using compatible
  measured Raspberry Pi evidence where available (`RANKING_MODE_CALIBRATED_PI`),
  falls back deterministically to static (and, if static has no evidence
  either, to the conservative `schedule-unroll-k=1` baseline) when
  measured evidence is absent or incompatible. Every accepted/rejected
  evidence item is recorded with a reason.

## 3. Candidate generation scope

Deliberately restricted (`generate_supported_candidates()`):

- Tile must be one Stage-11-structurally-validated configuration:
  `{(8,8,8), (8,8,4), (4,8,8)}`. Anything else is rejected before scoring,
  never silently generated.
- Shape must divide the tile evenly (the same rule the compile script's
  own legality check already enforces).
- `schedule-unroll-k` candidates: 1 and 2 always considered; 4 is
  additionally considered only when the K-loop's pre-unroll trip count
  (`shape_k / tile_k`) is itself divisible by 4. No arbitrary integer
  factors are ever generated.

For the primary validated configuration (32x32x32, tile 8x8x8, K trip
count 4), the candidate set is exactly `{uk1, uk2, uk4}` -- matching every
candidate this project has ever structurally validated (Stage 11) or
measured (Stage 12/13) for this shape/tile.

## 4. Compatibility and fallback policy

`check_compatibility()` (reused unmodified from Stage 14) requires
`target_arch`/`target_cpu`/`target_features`/`dtype`/`microkernel_id`/
`vector_width`/`loop_order_id` to match exactly (fails closed on any
mismatch) plus a matching `benchmark_methodology_version`
(`stage13_pi5_harness_v1` -- checked explicitly at profile-load time in
Stage 15's own loader, since Stage 14's pool-search logic has no way to
see a caller-supplied profile's declared version). Four levels: exact
match, cross-shape-same-schedule, shape-bucket, incompatible.

Fallback policy (deterministic, never silent):
1. Exact compatible measured evidence, if available.
2. Compatible declared evidence bucket (cross-shape-same-schedule or
   shape-bucket), if available.
3. Static scoring, if any real backend evidence exists for the requested
   shape/tile.
4. Conservative baseline (`schedule-unroll-k=1`), if nothing else is
   available.

Every selection report records `requested_mode`, `effective_mode`,
`fallback_reason`, and `confidence` explicitly -- the mode is never
silently switched without a visible reason.

## 5. Selection-to-materialization data flow

```
HIR fixture (parsed for M/N/K)
  -> generate_supported_candidates()        [legality-gated candidate set]
  -> load_available_evidence() / load_profile_pool()  [real Stage 12/13/fixture evidence only]
  -> select_candidate()                     [cm.rank_candidates(), Stage 14 unmodified]
  -> compile_selected_candidate()           [subprocess: existing compile_hir_matmul_bias_relu_aarch64.sh, unmodified]
  -> verify_no_mismatch()                   [hard guard: compiled artifact key == selected key, or abort]
  -> selection report + manifest JSON
```

## 6-10. Results (all four modes compiled and verified through the real pipeline)

| Mode | Selected | Object bytes | Object SHA-256 (first 12) |
|---|---|---|---|
| manual uk1 | uk1 | 2608 | `dbbeb28bebaf` |
| manual uk2 | uk2 | 3248 | `5edd623a20c3` |
| static | uk2 | 3248 | `5edd623a20c3` (byte-identical to manual uk2) |
| calibrated (real Pi profile) | **uk4** | 4232 | `aec26e5acb2e` |
| calibrated (incompatible x86_64 profile) | uk2 (fallback) | 3248 | `5edd623a20c3` (byte-identical to manual uk2) |

- **Manual uk1**: Stage 11 validator confirms the structural no-op baseline (`schedule_unroll_1_is_noop: PASS`).
- **Manual uk2**: Stage 11 validator confirms the materialized 2-way unroll (`unrolled_chain_is_serial: PASS`, serial accumulator chain of length 2 per K-loop body).
- **Static mode** selected uk2 -- the same limitation Stage 14 already reported honestly: static evidence alone cannot see that uk4 is real-hardware-faster (it has real spills, so static scoring ranks it last). This is expected and correct, not a bug.
- **Calibrated mode** selected uk4, matching the real Stage 13 measured winner for this exact shape/tile -- `evidence_accepted` shows all 3 candidates (uk1/uk2/uk4) found `exact_match` compatible evidence in the real Pi profile.
- **Incompatible-profile fallback**: the x86_64 fixture is correctly rejected for every candidate (`evidence_rejected` populated, `evidence_accepted` empty), `effective_mode` becomes `fallback_static`, and the resulting object is byte-identical to both manual uk2 and the static-mode output -- the fallback path materializes correctly, not just reports correctly.

Every one of these five compiled objects passed the hard identity guard
(`compiled_artifact_key_matches_selection: true` in every selection
report).

## 11. Real incompatible-target fixtures (`fixtures/`)

All marked `_fixture_kind: "COMPATIBILITY_TEST_FIXTURE"` with an explicit
`_provenance` field -- none claim to be measured benchmark data.

- `fixture_incompatible_x86_64.json` -- `target_arch=x86_64`,
  `target_cpu=skylake`: real values, directly observed via `llc --version`
  on this project's own dev host during this session.
- `fixture_incompatible_cortex_a72.json` -- `target_cpu=cortex-a72`: a
  real ARM CPU (Raspberry Pi 4 generation), distinct from this project's
  actual target (`cortex-a76`, Raspberry Pi 5). No Cortex-A72 hardware was
  ever used in this project.
- `fixture_incompatible_feature_set.json` -- same real `target_cpu` but a
  synthetic `target_features` string (`+dotprod`), since this project's
  actual pipeline never passes `-mattr` flags (always `"none"`) -- the one
  intentionally-synthetic field, clearly labeled.
- `fixture_stale_methodology.json` -- otherwise-compatible profile with an
  out-of-date `benchmark_methodology_version`.
- `fixture_malformed_provenance.json` -- missing required schema fields
  entirely.

## 12. Pi correctness and focused performance sanity

All 5 newly-Stage-15-compiled objects (manual uk1, manual uk2, static uk2,
calibrated uk4, fallback uk2) were transferred to the real Raspberry Pi 5,
built with the exact Stage 13 harness template, and run with the Stage 13
methodology (2000 iterations, 200 warmup, 500 repeated calls, `taskset -c 3`).
Full detail: `pi_correctness_and_sanity_result.json`.

- **Correctness**: 5/5 candidates bit-exact (`max_abs_error = 0`) against
  the scalar reference, zero repeated-call failures, zero guard-buffer
  corruption.
- **Performance sanity** (not a new benchmark campaign -- a focused
  consistency check): manual uk2 (0.00237ms) is ~8.6% faster than manual
  uk1 (0.002593ms), matching Stage 13's original result. Calibrated-
  selected uk4 (0.002074ms) is the fastest of all five, exactly matching
  what Stage 13 originally measured for this candidate. The incompatible-
  profile fallback output is latency-identical to manual uk2 (same
  object) -- confirming the fallback path did not alter any other
  compiler option.

## 13. Compiler integration architecture note

Selection happens in this Python driver, not in a native MLIR/C++ pass.
This is explicitly acceptable for this stage (task brief section 9) --
the driver performs selection deterministically before constructing the
MLIR pass pipeline, semantic candidate identity is validated (reusing
Stage 14's `CandidateKey` directly, no shell/C++ reimplementation), the
selected value controls real compiler behavior (proven above via 5 real
compiled objects with matching hashes), and every artifact makes the
selection boundary explicit via the pre-compile manifest and the hard
identity guard. A future native C++ MLIR pass integration remains a
separate roadmap item, not attempted here.

## Files in this directory

- `README.md` -- this file
- `commands.txt` -- every command used
- `fixtures/` -- 5 compatibility-test fixture profiles (see section 11)
- `manual_uk1_selection.json` / `manual_uk2_selection.json` /
  `static_selection.json` / `calibrated_selection.json` /
  `incompatible_fallback_selection.json` -- full selection reports for
  each of the 5 required scenarios (task section 10 schema: requested/
  effective mode, candidate set, rejected candidates, evidence accepted/
  rejected, cost breakdown, selected key, confidence, fallback reason,
  compile command, output artifact paths + checksums)
- `compiled/` -- the 5 actual compiled artifacts (LLVM-dialect MLIR, LLVM
  IR, assembly, object) plus each run's `*_pre_compile_manifest.json`
- `pi_correctness_and_sanity_result.json` -- Pi correctness + focused
  performance sanity result (section 12 above)

This directory references Stage 12/13/14 artifacts (via
`stage12_key`/`stage12_evidence` fields inside the evidence loaded for
each selection, and via `--schedule-profile` pointing directly at the
real Stage 13 `pi_validation_results.json`) rather than duplicating their
raw MIR/register/Pi-execution evidence.

## Truth boundary (unchanged from Stage 14, restated for this stage)

- Calibrated mode is calibrated only for the tested Raspberry Pi 5
  Cortex-A76 path. It is never used for, or extrapolated to, other
  targets -- proven by the incompatible-fixture rejection tests.
- Spill count remains a cost signal, never a universal rejection
  condition, at every layer of this stage.
- This is not general autotuning: the candidate set is restricted to
  `{uk1, uk2, uk4}` for one structurally-validated tile family; no loop
  interchange, no arbitrary unroll factors, no other kernel families.
- Calibrated selection is never the default. `--schedule-candidate-mode`
  defaults to `manual` everywhere in this driver and nowhere else in the
  repository invokes this driver automatically.
