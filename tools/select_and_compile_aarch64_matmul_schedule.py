#!/usr/bin/env python3
"""select_and_compile_aarch64_matmul_schedule.py

Stage 15: OPT-IN compiler-driver schedule candidate selection for the
AArch64 tiled-scheduled matmul microkernel. This is NOT a native C++
compiler pass -- selection happens in this Python driver, which then
invokes the existing, unmodified
mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh (--variant
tiled-scheduled) to perform the actual MLIR/LLVM materialization. Accurate
description: "Opt-in compiler-driver schedule candidate selection
controlling a real MLIR schedule materialization pass."

INTEGRATION POINT (Stage 15 section 1 audit finding): schedule-unroll-k is
currently supplied as a CLI flag directly to the compile script
(--schedule-unroll-k), which flows unchanged into
generate_scheduled_transform.sh and from there into the Transform-dialect
script materialized by mlir-opt. There was previously NOTHING that
verified a caller's *intended* selection actually matched what got passed
to the compile script -- this driver's hard guard (verify_no_mismatch())
closes that gap by recomputing the compiled artifact's semantic key from
the literal --tile-m/--tile-n/--tile-k/--schedule-unroll-k values passed
to the compile script and asserting it equals the selected candidate's
key before declaring success.

DEFAULT BEHAVIOR IS UNCHANGED: --schedule-candidate-mode defaults to
"manual", which is exactly today's existing --schedule-unroll-k behavior
(the compiler does not silently override a manually-specified value).
Static and calibrated modes are opt-in only.

SCOPE RESTRICTION: static/calibrated ranking is only supported for
candidates already covered by real Stage 12 (static/backend) and Stage 13
(measured) evidence -- currently the primary validated configuration
(32x32x32, tile 8x8x8). A shape/tile outside that coverage falls back
deterministically to the conservative baseline (schedule-unroll-k=1) with
an explicit fallback reason, rather than fabricating evidence for an
unanalyzed configuration. This matches the task brief's explicit
instruction to keep the first compiler-side candidate set restricted and
to reject candidates that have never passed structural validation.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import aarch64_schedule_candidate_model as cm  # noqa: E402

COMPILE_SCRIPT = os.path.join(REPO_ROOT, "mlir_passes", "tools", "compile_hir_matmul_bias_relu_aarch64.sh")
FIXTURE_DIR = os.path.join(REPO_ROOT, "mlir_passes", "test", "backend_codegen")
DEFAULT_STAGE12_JSON = os.path.join(
    REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_scheduling",
    "schedule_comparison_results.json")
DEFAULT_STAGE13_JSON = os.path.join(
    REPO_ROOT, "artifacts", "backend_codegen", "aarch64_matmul_bias_relu_pi_scheduling",
    "pi_validation_results.json")

# Candidate generation scope (task section 3): only these tile
# configurations have passed Stage 11 structural validation. A candidate
# whose tile is not in this set is rejected before scoring, regardless of
# mode -- never generated as a legal candidate at all.
STRUCTURALLY_VALIDATED_TILES = {(8, 8, 8), (8, 8, 4), (4, 8, 8)}

MODE_MANUAL = "manual"
MODE_STATIC = "static"
MODE_CALIBRATED = "calibrated"
VALID_MODES = (MODE_MANUAL, MODE_STATIC, MODE_CALIBRATED)

CONSERVATIVE_BASELINE_UNROLL_K = 1


class ScheduleSelectionError(RuntimeError):
    pass


class ArtifactIdentityMismatchError(RuntimeError):
    """Hard guard failure: the compiled artifact's manifest key does not
    equal the selected candidate's key. Must never happen if the driver
    is correct; aborting loudly is strictly preferable to silently
    shipping a report that claims one candidate while a different one was
    actually compiled."""


def sh(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(str(c) for c in cmd)}):\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def parse_shape_from_fixture(fixture_path):
    """Same M/N/K parsing convention as compile_hir_matmul_bias_relu_aarch64.sh's
    own legality check: first two tensor<AxBxf32> matches in the raw HIR
    text, lhs=MxK, rhs=KxN."""
    import re
    with open(fixture_path) as f:
        text = f.read()
    dims = re.findall(r"tensor<(\d+)x(\d+)xf32>", text)[:2]
    if len(dims) != 2:
        raise ScheduleSelectionError(f"could not parse M/N/K from {fixture_path}")
    (m, k1), (k2, n) = dims
    return int(m), int(n), int(k1)


# ---------------------------------------------------------------------------
# Candidate generation (task section 3)
# ---------------------------------------------------------------------------

def generate_supported_candidates(shape_m, shape_n, shape_k, tile_m, tile_n, tile_k):
    """Returns a deterministically-ordered list of legal CandidateKeys.
    Legality: tile must be structurally validated; shape must divide the
    tile evenly (same rule the compile script itself enforces); the
    unroll factor must be >=1 and must evenly divide the K-loop's
    pre-unroll trip count (K/tile_k) -- never an arbitrary integer.
    Duplicates (impossible here since unroll_k differs per generated
    candidate, but checked defensively) are removed."""
    rejected = []
    if (tile_m, tile_n, tile_k) not in STRUCTURALLY_VALIDATED_TILES:
        rejected.append({"tile": [tile_m, tile_n, tile_k], "reason": f"tile ({tile_m},{tile_n},{tile_k}) has not passed Stage 11 structural validation; only {sorted(STRUCTURALLY_VALIDATED_TILES)} are supported"})
        return [], rejected
    if shape_m % tile_m or shape_n % tile_n or shape_k % tile_k:
        rejected.append({"tile": [tile_m, tile_n, tile_k], "reason": f"shape {shape_m}x{shape_n}x{shape_k} is not evenly divisible by tile {tile_m}x{tile_n}x{tile_k} (no tail handling)"})
        return [], rejected

    k_trip = shape_k // tile_k
    candidates = []
    seen_keys = set()
    for factor in (1, 2, 4):
        if factor > 1 and k_trip % factor != 0:
            rejected.append({"schedule_unroll_k": factor, "reason": f"factor {factor} does not evenly divide the K-loop trip count ({k_trip}) -- transform.loop.unroll requires this (see Stage 10/11 legality rule)"})
            continue
        key = cm.CandidateKey(shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
                               tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, schedule_unroll_k=factor)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(key)
    return candidates, rejected


# ---------------------------------------------------------------------------
# Evidence loading (restricted to real, already-analyzed candidates)
# ---------------------------------------------------------------------------

def load_available_evidence(stage12_json, stage13_json):
    """Returns {CandidateKey: CandidateEvidenceRecord} for every candidate
    Stage 12/13 actually analyzed -- real evidence only, nothing
    fabricated for candidates outside that set."""
    records = cm.load_stage12_records(stage12_json)
    if stage13_json and os.path.isfile(stage13_json):
        cm.load_stage13_measured(stage13_json, records)
    return {r.key: r for r in records.values()}


def load_profile_pool(profile_path):
    """Loads a --schedule-profile file into a list of (CandidateKey,
    MeasuredHardwareEvidence) tuples, tolerant of two shapes:

    1. A real Stage 13 pi_validation_results.json (has "pi_environment"
       and "candidates" with the full harness output schema).
    2. A minimal Stage 15 compatibility-test fixture:
         {"schema_version": "stage15_schedule_profile_v1",
          "profile_kind": "compatibility_test_fixture",
          "target_arch": ..., "target_cpu": ..., "target_features": ...,
          "dtype": ..., "benchmark_methodology_version": ...,
          "candidates": {label: {"shape_m":.., "shape_n":.., "shape_k":..,
                                  "tile_m":.., "tile_n":.., "tile_k":..,
                                  "schedule_unroll_k":..,
                                  "median_latency_ms": .. or null,
                                  "correctness_pass": .. or null,
                                  "cv": .. or null}}}
       -- fixtures need only the fields required to exercise compatibility
       logic (task section 13); missing measured fields become None, not
       fabricated numbers.

    3. A Stage 16 multi-domain profile
       (schema_version "stage16_multidomain_profile_v1"): multiple
       independent named domains, each with its own domain_identity
       (target/tile/shape) and a "candidates" dict keyed by canonical
       CandidateKey id. Domains are never merged into one flat candidate
       list at the SOURCE level (each stays under its own domain_identity
       for provenance/readability), but since CandidateKey already
       uniquely encodes shape+tile+unroll, flattening them into one pool
       for ranking purposes is safe -- check_compatibility() (unchanged,
       Stage 14) is what actually prevents one domain's evidence from
       influencing another domain's query, not the file layout.

    Raises ScheduleSelectionError on malformed/unreadable input -- this is
    the "malformed provenance" rejection case from task section 5.
    """
    try:
        with open(profile_path) as f:
            raw = json.load(f)
    except Exception as e:
        raise ScheduleSelectionError(f"malformed or unreadable profile {profile_path}: {e}")

    if raw.get("schema_version") == "stage16_multidomain_profile_v1":
        methodology = raw.get("benchmark_methodology_version", "unknown")
        if methodology != cm.BENCHMARK_METHODOLOGY_VERSION:
            raise ScheduleSelectionError(
                f"profile {profile_path} declares benchmark_methodology_version={methodology!r}, "
                f"expected {cm.BENCHMARK_METHODOLOGY_VERSION!r} -- rejecting the entire profile as stale"
            )
        pool = []
        for domain_name, domain in raw.get("domains", {}).items():
            ident = domain.get("domain_identity", {})
            target_arch = ident.get("target_arch", "aarch64")
            target_cpu = ident.get("target_cpu", "unknown")
            target_features = ident.get("target_features", "none")
            dtype = ident.get("dtype", "f32")
            tile = ident.get("tile", {})
            shape_m, shape_n, shape_k = (int(x) for x in ident.get("shape", "0x0x0").split("x"))
            for canonical_id, c in domain.get("candidates", {}).items():
                key = cm.CandidateKey(
                    target_arch=target_arch, target_cpu=target_cpu, target_features=target_features, dtype=dtype,
                    shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
                    tile_m=tile.get("m"), tile_n=tile.get("n"), tile_k=tile.get("k"),
                    schedule_unroll_k=int(canonical_id.split(":uk")[1].split(":")[0]),
                )
                if key.canonical_id() != canonical_id:
                    raise ScheduleSelectionError(
                        f"profile {profile_path} domain {domain_name!r} entry key mismatch: "
                        f"reconstructed {key.canonical_id()!r} != stored {canonical_id!r}"
                    )
                cid = key.canonical_id()
                mk = lambda v: cm.ev(v, "raspberry_pi_measured", profile_path, "stage16_multidomain_v1", target_cpu, "stage16", cid)
                measured = cm.MeasuredHardwareEvidence(
                    median_latency_ms=mk(c.get("median_latency_ms")), p95_latency_ms=mk(None),
                    mean_latency_ms=mk(None), stddev_latency_ms=mk(None),
                    cv=mk(c.get("cv")), correctness_pass=mk(c.get("correctness_pass")),
                    hardware_identity=mk(target_cpu),
                )
                if measured.median_latency_ms.value is not None:
                    pool.append((key, measured))
        return pool

    pool = []
    if "pi_environment" in raw and "candidates" in raw:
        # Real Stage 13 harness output -- reuse the tested loader.
        tmp_records = {}
        try:
            cm.load_stage13_measured(profile_path, tmp_records)
        except Exception as e:
            raise ScheduleSelectionError(f"failed to parse Stage-13-shaped profile {profile_path}: {e}")
        for r in tmp_records.values():
            if r.measured is not None:
                pool.append((r.key, r.measured))
        return pool

    if raw.get("schema_version") != "stage15_schedule_profile_v1":
        raise ScheduleSelectionError(
            f"profile {profile_path} has unrecognized or missing schema_version "
            f"(got {raw.get('schema_version')!r}, expected 'stage15_schedule_profile_v1' or a real Stage 13 pi_validation_results.json)"
        )
    target_arch = raw.get("target_arch", "aarch64")
    target_cpu = raw.get("target_cpu", "unknown")
    target_features = raw.get("target_features", "none")
    dtype = raw.get("dtype", "f32")
    methodology = raw.get("benchmark_methodology_version", "unknown")
    # rank_candidates()'s internal compatibility check (Stage 14) always
    # compares against the CURRENT cm.BENCHMARK_METHODOLOGY_VERSION
    # constant -- it has no way to see what version a caller-supplied
    # profile actually declares, since the real Stage 13 harness output
    # never stores that field itself. Stage 15 fixtures DO declare it
    # explicitly, so validate it HERE, at load time: a stale/unsupported
    # value means every entry from this profile is unusable for
    # calibration, checked before any candidate-level compatibility logic
    # runs at all (task section 5's "stale or unsupported schema version"
    # case).
    if methodology != cm.BENCHMARK_METHODOLOGY_VERSION:
        raise ScheduleSelectionError(
            f"profile {profile_path} declares benchmark_methodology_version={methodology!r}, "
            f"expected {cm.BENCHMARK_METHODOLOGY_VERSION!r} -- rejecting the entire profile as stale"
        )
    for label, c in raw.get("candidates", {}).items():
        key = cm.CandidateKey(
            target_arch=target_arch, target_cpu=target_cpu, target_features=target_features, dtype=dtype,
            shape_m=c["shape_m"], shape_n=c["shape_n"], shape_k=c["shape_k"],
            tile_m=c["tile_m"], tile_n=c["tile_n"], tile_k=c["tile_k"],
            schedule_unroll_k=c["schedule_unroll_k"],
        )
        cid = key.canonical_id()
        mk = lambda v, note="": cm.ev(v, "raspberry_pi_measured" if target_cpu == "cortex-a76" else "manual_config",
                                       profile_path, "stage15_fixture_v1", target_cpu, "fixture", cid, note=note)
        measured = cm.MeasuredHardwareEvidence(
            median_latency_ms=mk(c.get("median_latency_ms")),
            p95_latency_ms=mk(c.get("p95_latency_ms")),
            mean_latency_ms=mk(c.get("mean_latency_ms")),
            stddev_latency_ms=mk(c.get("stddev_latency_ms")),
            cv=mk(c.get("cv")),
            correctness_pass=mk(c.get("correctness_pass")),
            hardware_identity=mk(target_cpu),
        )
        pool.append((key, measured))
    return pool


def blank_record_for(key):
    """A candidate with no real evidence at all -- every field explicitly
    None with a note, never a fabricated number. Used only so the
    candidate is still representable (and legality/count-visible) in a
    selection report even when it must fall back."""
    cid = key.canonical_id()
    empty = lambda: cm.ev(None, "manual_config", "none", "n/a", "n/a", "n/a", cid, note="no Stage 12/13 evidence exists for this candidate")
    return cm.CandidateEvidenceRecord(
        key=key, label=cid,
        static_ir=cm.StaticIRBEvidence(*[empty() for _ in dataclasses.fields(cm.StaticIRBEvidence)]),
        llvm_backend=cm.LLVMBackendEvidence(*[empty() for _ in dataclasses.fields(cm.LLVMBackendEvidence)]),
        measured=None,
    )


# ---------------------------------------------------------------------------
# Selection (task sections 2, 6, 11, 12)
# ---------------------------------------------------------------------------

def select_candidate(mode, candidates, evidence_by_key, profile_path=None, manual_unroll_k=None):
    """Returns a dict: requested_mode, effective_mode, selected_key,
    fallback_reason, confidence, cost_breakdown (list of dicts or None),
    evidence_accepted, evidence_rejected."""
    if mode == MODE_MANUAL:
        if manual_unroll_k is None:
            raise ScheduleSelectionError("--schedule-candidate-mode=manual requires --schedule-unroll-k")
        matching = [c for c in candidates if c.schedule_unroll_k == manual_unroll_k]
        if not matching:
            raise ScheduleSelectionError(f"--schedule-unroll-k={manual_unroll_k} is not a legal candidate for this shape/tile (see rejected_candidates)")
        return {
            "requested_mode": mode, "effective_mode": mode, "selected_key": matching[0],
            "fallback_reason": "none -- manual mode never overrides the user-supplied value",
            "confidence": 1.0, "cost_breakdown": None, "evidence_accepted": [], "evidence_rejected": [],
        }

    if mode not in (MODE_STATIC, MODE_CALIBRATED):
        raise ScheduleSelectionError(f"unknown --schedule-candidate-mode {mode!r}, must be one of {VALID_MODES}")

    if mode == MODE_CALIBRATED and not profile_path:
        raise ScheduleSelectionError("--schedule-candidate-mode=calibrated requires --schedule-profile")

    records = [evidence_by_key.get(c) or blank_record_for(c) for c in candidates]
    has_any_real_evidence = any(r.llvm_backend.object_bytes.value is not None for r in records)

    evidence_accepted, evidence_rejected = [], []
    effective_mode = mode
    fallback_reason = "none"

    if not has_any_real_evidence:
        # No static/backend evidence exists for this shape/tile at all --
        # cannot rank, must fall back deterministically to the
        # conservative baseline (task section 6, item 4).
        baseline = next((c for c in candidates if c.schedule_unroll_k == CONSERVATIVE_BASELINE_UNROLL_K), candidates[0] if candidates else None)
        if baseline is None:
            raise ScheduleSelectionError("no legal candidates and no evidence -- cannot select")
        return {
            "requested_mode": mode, "effective_mode": "fallback_conservative_baseline",
            "selected_key": baseline,
            "fallback_reason": f"no Stage 12/13 evidence exists for shape {baseline.shape_m}x{baseline.shape_n}x{baseline.shape_k} tile {baseline.tile_m}x{baseline.tile_n}x{baseline.tile_k} -- falling back to schedule-unroll-k={CONSERVATIVE_BASELINE_UNROLL_K}",
            "confidence": 0.0, "cost_breakdown": None, "evidence_accepted": [], "evidence_rejected": [{"reason": "no evidence for this configuration at all"}],
        }

    pool = []
    if mode == MODE_CALIBRATED:
        try:
            profile_pool = load_profile_pool(profile_path)
        except ScheduleSelectionError as e:
            evidence_rejected.append({"artifact": profile_path, "reason": str(e)})
            profile_pool = []

        # Real per-candidate compatibility check against every entry the
        # profile actually contains (not a self-comparison) -- exact
        # match, cross-shape-same-schedule, and shape-bucket are all
        # genuinely exercised here, mirroring cm.rank_candidates' own
        # internal search so the reported evidence_accepted/rejected
        # matches what the ranking call below will actually use.
        for query_key in candidates:
            best = None
            for other_key, other_measured in profile_pool:
                compat = cm.check_compatibility(query_key, other_key, cm.BENCHMARK_METHODOLOGY_VERSION)
                if compat["level"] == cm.INCOMPATIBLE:
                    continue
                if best is None or compat["confidence"] > best[0]["confidence"]:
                    best = (compat, other_key, other_measured)
            if best is not None:
                compat, other_key, other_measured = best
                evidence_accepted.append({
                    "candidate": query_key.canonical_id(), "matched_evidence": other_key.canonical_id(),
                    "compatibility": compat["level"], "confidence": compat["confidence"],
                })
                pool.append((other_key, other_measured))
            else:
                evidence_rejected.append({
                    "candidate": query_key.canonical_id(),
                    "reason": f"no compatible measured evidence found in {profile_path} for this candidate (checked {len(profile_pool)} profile entries)",
                })

        if not pool:
            effective_mode = "fallback_static"
            fallback_reason = f"calibrated mode requested but no compatible measured evidence found in {profile_path} for any generated candidate -- falling back to static scoring"

    ranking_mode = cm.RANKING_MODE_CALIBRATED_PI if (mode == MODE_CALIBRATED and pool) else cm.RANKING_MODE_STATIC_SOFT_PENALTY
    ranked = cm.rank_candidates(records, ranking_mode, measured_evidence_pool=pool)
    selectable = [b for b in ranked if not b.rejected]
    if not selectable:
        baseline = next((c for c in candidates if c.schedule_unroll_k == CONSERVATIVE_BASELINE_UNROLL_K), candidates[0])
        return {
            "requested_mode": mode, "effective_mode": "fallback_conservative_baseline",
            "selected_key": baseline,
            "fallback_reason": "every generated candidate was rejected (unsupported/incorrect) -- falling back to the conservative baseline",
            "confidence": 0.0, "cost_breakdown": [cm.breakdown_to_dict(b) for b in ranked],
            "evidence_accepted": evidence_accepted, "evidence_rejected": evidence_rejected,
        }

    winner = selectable[0]
    winner_key = next(c for c in candidates if c.canonical_id() == winner.candidate_id)
    return {
        "requested_mode": mode, "effective_mode": effective_mode if effective_mode != mode else mode,
        "selected_key": winner_key, "fallback_reason": fallback_reason if effective_mode != mode else "none",
        "confidence": winner.confidence, "cost_breakdown": [cm.breakdown_to_dict(b) for b in ranked],
        "evidence_accepted": evidence_accepted, "evidence_rejected": evidence_rejected,
    }


# ---------------------------------------------------------------------------
# Materialization: invoke the EXISTING, unmodified compile script with the
# selected candidate's values (task section 7/9 -- no second
# materialization mechanism)
# ---------------------------------------------------------------------------

def compile_selected_candidate(key: "cm.CandidateKey", fixture_path, output_dir, name):
    cmd = [
        "bash", COMPILE_SCRIPT, "--variant", "tiled-scheduled",
        "--tile-m", str(key.tile_m), "--tile-n", str(key.tile_n), "--tile-k", str(key.tile_k),
        "--schedule-unroll-k", str(key.schedule_unroll_k),
        fixture_path, output_dir, name,
    ]
    sh(cmd)
    obj_path = os.path.join(output_dir, f"{name}.o")
    return {
        "command": " ".join(cmd),
        "llvm_dialect_mlir": os.path.join(output_dir, f"{name}_llvm.mlir"),
        "llvm_ir": os.path.join(output_dir, f"{name}.ll"),
        "asm": os.path.join(output_dir, f"{name}.s"),
        "obj": obj_path,
        "obj_sha256": sha256_file(obj_path),
        "obj_bytes": os.path.getsize(obj_path),
    }


def verify_no_mismatch(selected_key: "cm.CandidateKey", compile_argv_tile, compile_argv_unroll_k, shape_m, shape_n, shape_k):
    """Hard guard (task section 7): recompute the candidate key from the
    LITERAL arguments passed to the compile script and assert it equals
    the selected candidate's key. Must never fail if the driver is
    correct; raising loudly here is strictly preferable to a report that
    claims one candidate while a different one was actually compiled."""
    compiled_key = cm.CandidateKey(
        shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
        tile_m=compile_argv_tile[0], tile_n=compile_argv_tile[1], tile_k=compile_argv_tile[2],
        schedule_unroll_k=compile_argv_unroll_k,
    )
    if compiled_key != selected_key:
        raise ArtifactIdentityMismatchError(
            f"HARD GUARD FAILURE: selected candidate key {selected_key.canonical_id()} "
            f"does not match the key reconstructed from the actual compile invocation "
            f"{compiled_key.canonical_id()} -- aborting rather than shipping a mismatched artifact"
        )
    return compiled_key


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_mlir")
    ap.add_argument("output_dir")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--schedule-candidate-mode", choices=VALID_MODES, default=MODE_MANUAL,
                     help="default: manual (existing behavior, unchanged). static/calibrated are opt-in.")
    ap.add_argument("--schedule-profile", help="required for calibrated mode: a Stage-13-shaped or Stage-15-fixture-shaped measured-evidence JSON")
    ap.add_argument("--schedule-unroll-k", type=int, help="required for manual mode")
    ap.add_argument("--tile-m", type=int, default=8)
    ap.add_argument("--tile-n", type=int, default=8)
    ap.add_argument("--tile-k", type=int, default=8)
    ap.add_argument("--emit-schedule-selection", help="path to write the full machine-readable selection report")
    ap.add_argument("--stage12-json", default=DEFAULT_STAGE12_JSON)
    ap.add_argument("--stage13-json", default=DEFAULT_STAGE13_JSON)
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.input_mlir))[0]
    shape_m, shape_n, shape_k = parse_shape_from_fixture(args.input_mlir)

    candidates, rejected_candidates = generate_supported_candidates(
        shape_m, shape_n, shape_k, args.tile_m, args.tile_n, args.tile_k)
    if not candidates:
        print(f"error: no legal candidates for shape {shape_m}x{shape_n}x{shape_k} tile {args.tile_m}x{args.tile_n}x{args.tile_k}", file=sys.stderr)
        for r in rejected_candidates:
            print(f"  rejected: {r}", file=sys.stderr)
        return 2

    evidence_by_key = {}
    if args.schedule_candidate_mode in (MODE_STATIC, MODE_CALIBRATED):
        try:
            evidence_by_key = load_available_evidence(args.stage12_json, args.stage13_json)
        except Exception as e:
            print(f"warning: could not load Stage 12/13 evidence: {e}", file=sys.stderr)

    try:
        selection = select_candidate(
            args.schedule_candidate_mode, candidates, evidence_by_key,
            profile_path=args.schedule_profile, manual_unroll_k=args.schedule_unroll_k,
        )
    except ScheduleSelectionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    selected_key = selection["selected_key"]
    os.makedirs(args.output_dir, exist_ok=True)

    manifest_before = {
        "selected_candidate_key": dataclasses.asdict(selected_key),
        "selected_candidate_canonical_id": selected_key.canonical_id(),
        "requested_mode": selection["requested_mode"], "effective_mode": selection["effective_mode"],
        "fallback_reason": selection["fallback_reason"], "confidence": selection["confidence"],
        "timestamp": time.time(),
    }
    with open(os.path.join(args.output_dir, f"{name}_pre_compile_manifest.json"), "w") as f:
        json.dump(manifest_before, f, indent=2)

    compiled = compile_selected_candidate(selected_key, args.input_mlir, args.output_dir, name)

    compiled_key = verify_no_mismatch(
        selected_key, (selected_key.tile_m, selected_key.tile_n, selected_key.tile_k),
        selected_key.schedule_unroll_k, shape_m, shape_n, shape_k,
    )

    report = {
        "schema_version": "stage15_schedule_selection_report_v1",
        "requested_mode": selection["requested_mode"],
        "effective_mode": selection["effective_mode"],
        "fallback_reason": selection["fallback_reason"],
        "confidence": selection["confidence"],
        "candidate_set": [dataclasses.asdict(c) for c in candidates],
        "rejected_candidates": rejected_candidates,
        "evidence_accepted": selection["evidence_accepted"],
        "evidence_rejected": selection["evidence_rejected"],
        "cost_breakdown": selection["cost_breakdown"],
        "selected_candidate_key": dataclasses.asdict(selected_key),
        "selected_candidate_canonical_id": selected_key.canonical_id(),
        "compiled_artifact_key_matches_selection": compiled_key == selected_key,
        "pass_pipeline_command": compiled["command"],
        "output_artifacts": {k: v for k, v in compiled.items() if k != "command"},
    }
    if args.emit_schedule_selection:
        with open(args.emit_schedule_selection, "w") as f:
            json.dump(report, f, indent=2)

    summary = (
        f"[schedule-selection] mode={selection['requested_mode']} effective={selection['effective_mode']} "
        f"selected={selected_key.canonical_id()} confidence={selection['confidence']:.2f} "
        f"fallback_reason={selection['fallback_reason']!r}"
    )
    print(summary, file=sys.stderr)
    print(f"[schedule-selection] object: {compiled['obj']} sha256={compiled['obj_sha256']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
