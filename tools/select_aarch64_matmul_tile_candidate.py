#!/usr/bin/env python3
"""select_aarch64_matmul_tile_candidate.py

Stage 11/12/13 for the AArch64 tile-candidate selection slice: a
transparent, explicit, non-ML scoring policy that picks one tile candidate
per shape from real measured evidence (Raspberry Pi latency, generated
object size, and assembly-derived spill evidence).

Policy (deliberately simple, weights explained inline -- no hidden terms,
no learned weights):

  Hard rejection (before any scoring):
    - not legal (fails shape%tile divisibility or the analytical
      register-demand hard limit)
    - not correct (failed the 1-call or 1000-call Raspberry Pi correctness
      check, or a max_abs_error >= 1e-3)
  A rejected candidate is EXCLUDED from selection entirely -- never kept as
  a "high-penalty" option (per the task brief's explicit instruction).

  Score (lower is better), computed per shape, only over legal+correct
  candidates for that shape:

    score = latency_ms / best_latency_ms                      [primary signal]
          + 0.20 * hot_loop_vector_spills                     [spills are expensive: each one is a
          + 0.20 * hot_loop_vector_reloads                      real store+load pair added to the hot
                                                                 loop's critical path every iteration --
                                                                 weighted equally to spills since a
                                                                 reload without its paired spill cannot
                                                                 occur]
          + 0.10 * (object_bytes / smallest_object_bytes)     [code-size growth vs. the smallest legal
                                                                 candidate for this shape -- a soft
                                                                 preference for compactness, weighted
                                                                 far below latency since object sizes
                                                                 here are all in the 1.7-2.7KB range,
                                                                 i.e. the size difference is real but
                                                                 minor next to a spill]
          + register_limit_penalty                             [0.05 per register referenced beyond 28
                                                                 (i.e. within 4 of the 32-register
                                                                 ceiling) -- a small, late-kicking-in
                                                                 penalty for candidates that leave no
                                                                 headroom, without contradicting the
                                                                 prior slice's finding that 32
                                                                 registers/0 spills is fine on its own]

  A tiny latency improvement should not automatically justify severe
  spills or code-size growth -- these weights are calibrated so that ONE
  hot-loop spill (+0.20) outweighs a latency advantage smaller than 20% of
  the best candidate's latency, but a latency advantage larger than that
  still wins even with a spill. This is a deliberate design choice, not a
  mathematical necessity; the weights are named constants below so they can
  be inspected and changed.

  Tie-break order (applied only when scores are equal after rounding to
  4 decimal places -- true ties are rare given continuous latency, but the
  order is still defined and used deterministically):
    1. Lower latency
    2. Zero spills (fewer hot_loop_vector_spills + hot_loop_vector_reloads)
    3. Smaller object_bytes
    4. Fewer vector_registers_referenced
    5. Smaller tile volume (tm*tn*tk), for robustness

Usage:
  python3 tools/select_aarch64_matmul_tile_candidate.py \
    --results candidate_results.json \
    --output selected_tiles.json \
    --scoring-policy scoring_policy.json
"""
import argparse
import json

WEIGHT_SPILL = 0.20
WEIGHT_RELOAD = 0.20
WEIGHT_OBJECT_SIZE = 0.10
REGISTER_SOFT_THRESHOLD = 28
WEIGHT_PER_REGISTER_OVER_THRESHOLD = 0.05


def shape_key(shape):
    return tuple(shape)


def tile_key(tile):
    return (tile["m"], tile["n"], tile["k"])


def is_eligible(c):
    if not c["legality"]["legal"]:
        return False, "illegal: " + "; ".join(c["legality"]["rejection_reasons"])
    corr = c.get("correctness", {})
    if not corr.get("passed"):
        return False, "failed Raspberry Pi correctness check"
    perf = c.get("performance", {})
    if perf.get("median_ms") is None:
        return False, "no benchmark evidence"
    return True, None


def score(c, best_latency_ms, smallest_object_bytes):
    perf = c["performance"]
    backend = c["backend"]
    latency_term = perf["median_ms"] / best_latency_ms
    spills = backend.get("hot_loop_vector_spills") or 0
    reloads = backend.get("hot_loop_vector_reloads") or 0
    spill_term = WEIGHT_SPILL * spills
    reload_term = WEIGHT_RELOAD * reloads
    size_term = WEIGHT_OBJECT_SIZE * (backend["object_bytes"] / smallest_object_bytes)
    regs = backend.get("vector_registers_referenced") or 0
    reg_penalty = WEIGHT_PER_REGISTER_OVER_THRESHOLD * max(0, regs - REGISTER_SOFT_THRESHOLD)
    total = latency_term + spill_term + reload_term + size_term + reg_penalty
    return total, {
        "latency_term": latency_term,
        "spill_term": spill_term,
        "reload_term": reload_term,
        "size_term": size_term,
        "register_penalty_term": reg_penalty,
    }


def tie_break_key(c):
    backend = c["backend"]
    perf = c["performance"]
    tile = c["tile"]
    spills_plus_reloads = (backend.get("hot_loop_vector_spills") or 0) + (backend.get("hot_loop_vector_reloads") or 0)
    return (
        perf["median_ms"],
        spills_plus_reloads,
        backend["object_bytes"],
        backend.get("vector_registers_referenced") or 0,
        tile["m"] * tile["n"] * tile["k"],
    )


def select_for_shape(candidates):
    eligible = []
    rejected = []
    for c in candidates:
        ok, reason = is_eligible(c)
        if ok:
            eligible.append(c)
        else:
            rejected.append({"tile": [c["tile"]["m"], c["tile"]["n"], c["tile"]["k"]], "reason": reason})

    if not eligible:
        return None, rejected

    best_latency = min(c["performance"]["median_ms"] for c in eligible)
    smallest_object = min(c["backend"]["object_bytes"] for c in eligible)

    scored = []
    for c in eligible:
        s, breakdown = score(c, best_latency, smallest_object)
        scored.append((round(s, 4), c, breakdown))

    scored.sort(key=lambda t: (t[0], tie_break_key(t[1])))
    winner_score, winner, winner_breakdown = scored[0]

    fastest = min(eligible, key=lambda c: c["performance"]["median_ms"])

    reasons = []
    if winner is fastest:
        reasons.append("lowest measured median latency")
    else:
        pct = 100.0 * (winner["performance"]["median_ms"] - fastest["performance"]["median_ms"]) / fastest["performance"]["median_ms"]
        reasons.append(f"not the single fastest candidate ({pct:.1f}% slower than fastest), but best overall score")
    wb = winner["backend"]
    if (wb.get("hot_loop_vector_spills") or 0) == 0 and (wb.get("hot_loop_vector_reloads") or 0) == 0:
        reasons.append("zero hot-loop spills/reloads")
    reasons.append(f"{wb['object_bytes']:,}-byte object")
    reasons.append(f"score={winner_score:.4f} (latency_term={winner_breakdown['latency_term']:.4f}, "
                    f"spill_term={winner_breakdown['spill_term']:.4f}, "
                    f"size_term={winner_breakdown['size_term']:.4f})")

    rejected_candidates = []
    for s, c, breakdown in scored[1:]:
        tile = c["tile"]
        wb2 = c["backend"]
        reason_bits = []
        spills = (wb2.get("hot_loop_vector_spills") or 0) + (wb2.get("hot_loop_vector_reloads") or 0)
        if spills > 0:
            reason_bits.append(f"{spills} hot-loop vector spill/reload events")
        if c["performance"]["median_ms"] > winner["performance"]["median_ms"]:
            reason_bits.append(f"{c['performance']['median_ms']:.6f}ms slower")
        if not reason_bits:
            reason_bits.append(f"higher overall score ({s:.4f} vs {winner_score:.4f})")
        rejected_candidates.append({
            "tile": [tile["m"], tile["n"], tile["k"]],
            "score": s,
            "reason": "; ".join(reason_bits),
        })

    result = {
        "shape": list(shape_key(winner["shape"])),
        "selected_tile": [winner["tile"]["m"], winner["tile"]["n"], winner["tile"]["k"]],
        "score": winner_score,
        "median_ms": winner["performance"]["median_ms"],
        "object_bytes": wb["object_bytes"],
        "hot_loop_vector_spills": wb.get("hot_loop_vector_spills") or 0,
        "hot_loop_vector_reloads": wb.get("hot_loop_vector_reloads") or 0,
        "vector_registers_referenced": wb.get("vector_registers_referenced"),
        "selection_reasons": reasons,
        "fastest_measured": {
            "tile": [fastest["tile"]["m"], fastest["tile"]["n"], fastest["tile"]["k"]],
            "median_ms": fastest["performance"]["median_ms"],
            "hot_loop_vector_spills": fastest["backend"].get("hot_loop_vector_spills") or 0,
        },
        "selected_matches_fastest": winner is fastest,
        "rejected_candidates": rejected_candidates,
    }
    return result, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--scoring-policy", default=None,
                     help="optional: write the scoring policy weights/description to this path")
    args = ap.parse_args()

    data = json.load(open(args.results))
    by_shape = {}
    for c in data["results"]:
        by_shape.setdefault(tuple(c["shape"]), []).append(c)

    selections = []
    for shape, candidates in sorted(by_shape.items()):
        result, illegal_or_incorrect = select_for_shape(candidates)
        if result is None:
            print(f"shape {shape}: NO ELIGIBLE CANDIDATE -- all rejected: {illegal_or_incorrect}")
            continue
        selections.append(result)
        matches = "MATCHES fastest" if result["selected_matches_fastest"] else "DIFFERS from fastest"
        print(f"shape {shape}: selected tile {result['selected_tile']} "
              f"(score={result['score']:.4f}, {result['median_ms']:.6f}ms, "
              f"{result['object_bytes']} bytes, {matches})")

    out = {"target": data["target"], "selections": selections}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(selections)} shape selections to {args.output}")

    if args.scoring_policy:
        policy = {
            "hard_rejection": ["not legal", "not correct (failed Pi repeated-call check or max_abs_error >= 1e-3)"],
            "score_formula": "latency_ms/best_latency_ms + 0.20*hot_loop_vector_spills + 0.20*hot_loop_vector_reloads + 0.10*(object_bytes/smallest_object_bytes) + 0.05*max(0, vector_registers_referenced-28)",
            "weights": {
                "spill": WEIGHT_SPILL,
                "reload": WEIGHT_RELOAD,
                "object_size": WEIGHT_OBJECT_SIZE,
                "register_soft_threshold": REGISTER_SOFT_THRESHOLD,
                "register_penalty_per_register_over_threshold": WEIGHT_PER_REGISTER_OVER_THRESHOLD,
            },
            "tie_break_order": [
                "lower latency", "zero spills+reloads", "smaller object_bytes",
                "fewer vector_registers_referenced", "smaller tile volume (tm*tn*tk)",
            ],
            "rationale": "Latency is the primary signal (coefficient 1.0). A single hot-loop spill "
                         "(+0.20) outweighs a latency advantage smaller than 20% of the best candidate's "
                         "latency; a larger latency advantage still wins even with one spill. Object-size "
                         "growth is a minor tiebreaker (0.10) since all candidates here are 1.7-2.7KB -- "
                         "the difference is real but small next to a spill. The register penalty only "
                         "activates within 4 registers of the 32-register ceiling, since the prior "
                         "single-tile slice already showed 32-registers-zero-spills is fine on its own; "
                         "it exists to break near-ties toward candidates with more headroom.",
        }
        with open(args.scoring_policy, "w") as f:
            json.dump(policy, f, indent=2)
        print(f"Wrote scoring policy to {args.scoring_policy}")


if __name__ == "__main__":
    main()
