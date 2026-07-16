# Ranking Experiment: Predicted vs. Measured (Task Section 10)

Three modes run against three matched candidate groups. Lower `total_cost`
is preferred (rank 1 = top). `RANKING_MODE_STATIC_HARD_REJECT` reproduces
the incorrect Stage 12 "any spill -> reject" policy for comparison only --
it is not exposed as a selectable production mode.

## Group: primary_32x32x32_unroll_family (primary uk1, uk2, full-K-unroll uk4)

| Mode | Rank 1 | Rank 2 | Rank 3 (rejected?) |
|---|---|---|---|
| static_hard_reject_spill | primary_unroll2 (2.341) | primary_unroll1 (4.310) | primary_full_unroll (**REJECTED**, spills>0) |
| static_soft_penalty (corrected default) | primary_unroll2 (2.341) | primary_unroll1 (4.310) | primary_full_unroll (41.589) |
| calibrated_raspberry_pi_5 | **primary_full_unroll (2.074)** | primary_unroll2 (2.352) | primary_unroll1 (2.593) |

**Measured order** (Stage 13 median-of-medians, fastest first):
primary_full_unroll (0.002074ms) > primary_unroll2 (0.002352ms) >
primary_unroll1 (0.002593ms)

Both static modes pick the WRONG top candidate (pairwise agreement with
measured order: **33%**). Calibrated mode matches exactly (100%).

## Group: alt_k_tile_8x8x4 (alt-K-tile uk1, uk2)

| Mode | Rank 1 | Rank 2 (rejected?) |
|---|---|---|
| static_hard_reject_spill | alt_k_tile_unroll1 (8.347) | alt_k_tile_unroll2 (**REJECTED**, spills>0) |
| static_soft_penalty | alt_k_tile_unroll1 (8.347) | alt_k_tile_unroll2 (11.437) |
| calibrated_raspberry_pi_5 | **alt_k_tile_unroll2 (2.611)** | alt_k_tile_unroll1 (2.963) |

**Measured order**: alt_k_tile_unroll2 (0.002611ms) > alt_k_tile_unroll1
(0.002963ms)

Both static modes pick the WRONG top candidate (pairwise agreement:
**0%** -- complete inversion). Calibrated mode matches exactly (100%).

## Group: cube64_8x8x8 (cube64 uk1, uk2)

| Mode | Rank 1 | Rank 2 |
|---|---|---|
| static_hard_reject_spill | cube64_unroll2 (4.575) | cube64_unroll1 (8.544) |
| static_soft_penalty | cube64_unroll2 (4.575) | cube64_unroll1 (8.544) |
| calibrated_raspberry_pi_5 | cube64_unroll2 (17.685) | cube64_unroll1 (19.760) |

**Measured order**: cube64_unroll2 (0.017685ms) > cube64_unroll1
(0.019760ms)

All three modes agree with measured order (100% pairwise agreement) --
this group has zero spills anywhere, so the static model's ranking
naturally aligns with reality.

## Interpretation

The static model (either variant) is a reliable predictor **only when no
candidate in the comparison set has real backend cost** (spills/reloads).
The moment a spilling candidate is measured to be genuinely faster (both
Group B diagnostics), a purely static ranking gets the winner wrong --
not because the static evidence is false, but because "more loop-control
reduction than the spill costs" is a real-hardware fact this static model
has no channel to observe. This is not a weight-tuning problem to be
overfit away with 5-14 data points; it is the expected, honest limit of
static evidence, and it is exactly why `RANKING_MODE_CALIBRATED_PI`
exists as a separate, explicitly-labeled mode rather than trying to force
one static formula to explain both zero-spill and spilling outcomes.

**Conclusion**: static-only ranking is trustworthy for filtering out
genuinely broken/unsupported/incorrect candidates and for comparing
candidates with identical backend-cost profiles (e.g. the cube64 pair).
It is not trustworthy for picking a winner among candidates whose backend
costs differ, unless compatible real hardware measurement is available to
calibrate against.
