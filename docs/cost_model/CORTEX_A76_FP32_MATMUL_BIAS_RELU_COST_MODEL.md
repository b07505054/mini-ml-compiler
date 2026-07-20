# Cortex-A76 FP32 MatMul-Bias-ReLU Cost Model

This document records the measured dataset and candidate-latency model for the
Raspberry Pi 5 Cortex-A76 FP32 MatMul-Bias-ReLU lowering scope. It is a
target-specific result, not a claim about other CPUs, dtypes, operators, or tile
families.

## Status and evidence boundary

- Dataset base commit: `907b29ee` (`Build production candidate latency dataset`).
- Dataset V2 was measured and trained from the working tree based on
  `a7190332cb38049430d2305aef5fb26ff5251971`.
- Target: Raspberry Pi 5, Cortex-A76, AArch64 NEON.
- Operator and dtype: FP32 MatMul + Bias + ReLU.
- Tile family: 8x8x8 where a tiled candidate applies.
- Benchmark governor: `ondemand`; the measured start frequency was 1.5 GHz.
- Every split is made by complete `shape_group_id`. All candidates for one
  shape remain in the same split.
- Held-out shapes did not influence features, hyperparameters, weighting, or
  the uncertainty policy.
- Production policy remains `analytical` by default. `gbdt` and `hybrid` are
  opt-in.

The source artifacts are:

- V2 shape matrix:
  `configs/cost_model_dataset/cortex_a76_fp32_matmul_bias_relu_v2_shapes.json`
- V2 measurements, split manifests, summary, and integrity hash:
  `artifacts/cost_model_dataset/cortex_a76_fp32_matmul_bias_relu_v2/`
- Frozen model, metrics, predictions, and generated evaluator:
  `configs/cost_model/cortex_a76_fp32_matmul_bias_relu_gbdt_v2/`

## Dataset V1 to V2

| Item | V1 | V2 |
| --- | ---: | ---: |
| Shape groups | 47 | 129 |
| Dataset rows | 162 | 378 |
| Valid latency labels | 161 | 377 |
| Train shapes | 29 | 79 |
| Validation shapes | 9 | 25 |
| Held-out shapes | 9 | 25 |

V2 adds 82 shapes and increases shape coverage by about 2.74x. Its shape
ranges are M=1..1024, N=1..1024, and K=1..2048. It contains six executable
candidate kinds. There is one compile failure and no runtime, numerical
correctness, or output-sentinel failures.

The only compile failure is:

```text
1x1x1 WholeShapeVectorNoPadding
```

Its residual `vector<1x1xf32>` cannot cross the LLVM boundary. The failed row
has no fabricated penalty latency and is excluded from valid latency labels.
The dataset summary therefore records 31 total rows but 30 valid rows for this
candidate.

### Shape coverage

The additional shapes are curated workload families rather than random samples
or examples chosen to favor a model. They cover:

- tile boundaries 31/32/33, 63/64/65, 95/96/97, 127/128/129, and
  255/256/257;
- dense K remainders around full reduction tiles;
- skinny, wide, tall, rectangular, and nearly-square matrices;
- regions where WholeShape competes with Tiled lowering;
- regions where Direct K-tail competes with Materialized tail handling; and
- extreme ratios such as `8x512x16`, `16x1024x32`, and `512x16x1024`.

These cases expose boundaries in tile utilization, padding traffic, tail
handling, and candidate ranking. They were not designed to make GBDT, Ridge,
or any other model win.

### Candidate coverage

| Candidate | Valid rows |
| --- | ---: |
| FusedScalar | 129 |
| WholeShapeVectorNoPadding | 30 |
| WholeShapeVectorMaterializedPadding | 22 |
| TiledVectorFullTiles | 32 |
| TiledVectorMaterializedTail | 97 |
| TiledVectorDirectCleanup | 67 |

Measured oracle winners are:

| Winner | Shapes |
| --- | ---: |
| TiledVectorDirectCleanup | 50 |
| TiledVectorFullTiles | 32 |
| TiledVectorMaterializedTail | 23 |
| WholeShapeVectorNoPadding | 23 |
| FusedScalar | 1 |
| WholeShapeVectorMaterializedPadding | 0 |

This distribution is more balanced than V1 and is no longer dominated by tiny
WholeShape cases. `WholeShapeVectorMaterializedPadding` remains represented by
22 valid measurements even though it wins none. It must not be deleted based
on this dataset alone. Follow-up work should determine whether coverage misses
its useful region, whether its lowering is intrinsically uncompetitive, or
whether a general legality/performance pruning rule can reject it earlier.

## Feature schema and leakage boundary

The production model has 98 features, all available when the compiler ranks a
lowering candidate. They include:

- M/N/K and logarithmic transformations;
- FLOPs, bytes, and arithmetic intensity;
- a stable one-hot candidate identity;
- independent tiling, vectorization, padding, and M/N/K tail strategies;
- tile counts, remainders, and tail invocation counts;
- temporary allocation, zero-fill, copy, and direct-tail work; and
- analytical code-size and register-pressure estimates.

The schema explicitly excludes the latency label, shape or split identity,
correctness result, binary and object hashes, measured object size, and
measured instruction/FMLA/branch counts. Those fields would either leak the
benchmark outcome or be unavailable to production selection before code
generation.

## Models

### Analytical

The Analytical model is a deterministic, manually structured and calibrated
cost formula. It remains the production default and the safety fallback for
unsupported or out-of-distribution learned-model inputs.

### Ridge

Ridge is linear regression with L2 regularization. It predicts
`log(median_ns)` and is an important baseline for determining whether tree
non-linearity adds useful ranking information.

### Single Decision Tree

The Single Tree is one regression-tree baseline. It measures how far a single
piecewise-constant partition can generalize without boosting.

### GBDT

GBDT is a sequence of small regression trees that correct earlier residuals.
It also predicts `log(median_ns)`. Its frozen configuration is:

```text
trees=25
depth=2
learning_rate=0.1
min_leaf=4
feature_fraction=0.9
weighting=candidate_balanced
seed=1729
uncertainty_margin=5%
```

### Hybrid

Hybrid is not independently trained. It combines GBDT ranking with
uncertainty and out-of-distribution checks, falling back to the Analytical
model when the learned prediction is unsupported or insufficiently confident.

## Metrics

**Exact match** means that the candidate ranked fastest by a model is the same
candidate as the measured benchmark oracle winner.

**Regret** measures the runtime cost of a selection mistake:

```text
regret = selected_candidate_latency / oracle_latency - 1
```

For example, regret `0.4189` means the selected candidate is 41.89% slower,
with latency approximately 1.4189x the oracle.

**MAE log** and **RMSE log** measure numerical error when predicting
`log(median_ns)`. Lower latency-regression error does not necessarily imply
better candidate ranking. A model can estimate absolute latencies well while
reversing the order of two close candidates.

## Validation results

Validation contains 25 shape groups.

| Model | Exact | Mean regret | Median regret | P95 regret | Worst regret | MAE log | RMSE log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Analytical | 88% | 2.34% | 0% | 0.53% | 56.79% | 0.646 | 0.699 |
| Ridge | 92% | 2.29% | 0% | 0% | 56.79% | 0.387 | 0.482 |
| Single Tree | 68% | 86.96% | 0% | 378.73% | 785.20% | 0.920 | 1.132 |
| GBDT | 88% | 2.34% | 0% | 0.53% | 56.79% | 0.681 | 0.804 |
| Hybrid | 88% | 2.34% | 0% | 0.53% | 56.79% | 0.681 | 0.804 |

Ridge selects 23/25 validation winners. Analytical, GBDT, and Hybrid each
select 22/25, so Ridge gets one additional shape correct. The Single Tree
generalizes poorly. This validation set alone is not sufficient evidence for
changing the production default.

## Held-out results

Held-out contains 25 shape groups and was evaluated after configuration freeze.

| Model | Exact | Mean regret | Median regret | P95 regret | Worst regret | MAE log | RMSE log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Analytical | 84% | 22.44% | 0% | 41.89% | 284.73% | 0.605 | 0.664 |
| Ridge | 92% | 30.39% | 0% | 0% | 512.95% | 0.427 | 0.537 |
| Single Tree | 68% | 125.31% | 0% | 428.21% | 600.90% | 0.891 | 1.136 |
| GBDT | 92% | 1.68% | 0% | 0% | 41.89% | 0.706 | 0.861 |
| Hybrid | 92% | 1.68% | 0% | 0% | 41.89% | 0.706 | 0.861 |

Ridge and GBDT both select 23/25 held-out winners. Ridge does not select more
winners. Their difference is the severity of the two mistakes. Ridge's mean
regret is 30.39% and its worst regret is 512.95%, meaning that the worst
selected candidate takes about `1 + 5.1295 = 6.1295x` the oracle latency.
GBDT's mean regret is 1.68% and its worst regret is 41.89%, or approximately
`1 + 0.4189 = 1.4189x` oracle latency. GBDT therefore has substantially lower
candidate-ranking risk despite Ridge's lower latency-regression MAE and RMSE.

### Important failure cases

- On `8x95x8` and `8x127x8`, GBDT correctly selects Direct while Analytical
  selects Materialized. Analytical regret is approximately 234% and 285%.
  This exposes a serious Analytical blind spot near the Direct/Materialized
  boundary.
- On `16x7x16` and `16x15x16`, Ridge selects Direct and produces very large
  held-out regret.
- On `32x7x32` and `32x15x32`, measurements prefer Materialized while most
  models select Direct. These should be retained as boundary-focused
  calibration and validation cases.

## Model conclusions

**Single Tree:** Its generalization is inadequate for production selection. It
remains useful only as a baseline.

**Ridge:** It has the best latency-regression MAE/RMSE and high exact-match
accuracy, but its rare candidate-ranking errors are catastrophic. It is not a
safe unprotected production default.

**Analytical:** It is explainable, deterministic, and safer under
out-of-distribution inputs, so it remains the default. The V2 evidence also
shows severe Direct/Materialized boundary mistakes.

**GBDT:** V2 provides the first clear held-out ranking evidence over Analytical:
92% versus 84% exact match, 1.68% versus 22.44% mean regret, and 41.89% versus
284.73% worst regret. It also clearly outperforms the Single Tree. Its
advantage is candidate ranking, not absolute latency-regression accuracy.

**Hybrid:** It conceptually supplies uncertainty and OOD fallback. Its offline
metrics equal pure GBDT in this evaluation, so this dataset does not yet show an
additional benefit from fallback. Future evaluation should report pure offline
GBDT, production OOD-aware GBDT, and production Hybrid separately.

## Evaluator and compiler integration

The V2 model is exported as a deterministic, dependency-free C++ evaluator.
It starts no Python process and needs no external ML runtime.

- Python/C++ equality: 377/377 valid dataset rows pass.
- Maximum absolute prediction difference: at most `1e-12`.
- Raw evaluator overhead: approximately 129 ns per candidate.
- Complete six-candidate evaluation: approximately 1.43 microseconds.
- Compiler modes: `analytical`, `gbdt`, and `hybrid`.
- OOD, schema, model, target, dtype, operator, feature, or finite-value
  failures use the Analytical safety path.

One held-out shape has N=1024 while the frozen training domain has N<=512.
The pure offline metrics above evaluate the learned model directly; production
selection classifies this shape as OOD and falls back to Analytical rather than
clamping it into the training domain.

## Production policy

```text
production default = analytical
gbdt = opt-in
hybrid = opt-in
```

The learned result is promising but does not yet justify changing the default:

- held-out contains only 25 shapes;
- GBDT worst regret remains 41.89%;
- one held-out shape is outside the production training domain;
- measurements used the `ondemand` governor;
- there is no independent second-day reproduction;
- there is no second machine or independent Cortex-A76 environment; and
- future lowering candidates are not represented in this model.

These limitations do not erase the positive result: V2 shows materially lower
held-out selection regret for GBDT than for Analytical or Ridge.

## Next steps

1. Freeze at least 50 additional unseen shapes for a second independent
   held-out set.
2. Emphasize K=7/8/9 and K=15/16/17 across different M/N tile counts and the
   Direct/Materialized near-tie region.
3. Re-measure sensitive Raspberry Pi cases with a performance governor.
4. Repeat measurements on a different day to quantify DVFS and order noise.
5. Report pure learned, production OOD-aware learned, and Hybrid policies
   separately.
6. Do not increase tree depth yet; coverage and measurement stability are the
   present limitations.
7. Add new candidate kinds only after their real kernels are executable,
   correct, and benchmarked. The model must not predict nonexistent
   implementations.
