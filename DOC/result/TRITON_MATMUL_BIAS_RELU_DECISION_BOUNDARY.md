# Triton MatMul-Bias-ReLU Decision Boundary

## Executive Summary

This tier separates compiler decision-making coverage from the canonical fusion
benefit suite. The canonical 33-workload suite remains intact and continues to
cover representative, stress, balanced, compute-heavy, and K-sweep fusion
behavior. The decision-boundary tier targets high-K, small-output regions where
the V1/V3 ranking can approach measurement noise.

Formal measurement on the GTX 1650 Max-Q completed 11 decision-boundary
workloads with 3 independent sessions, warmup 50, iterations 300, repeats 5,
and alternating candidate order. The result was:

- stable V1 wins: 0
- stable V3 wins: 3
- statistical ties: 3
- unstable: 5

The original `M=1,N=4096,K=65536` V1 observation was not reproduced as a
stable win. It remained a marginal/noisy case and is classified as unstable.
This means the current fixed-config V1/V3 pair can produce ties and stable V3
regions, but it still does not provide a clean balanced binary selection
dataset on this GPU.

## Environment

- host: `allen-ZenBook-UX534FTC-UX534FT`
- GPU: `NVIDIA GeForce GTX 1650 with Max-Q Design`
- compute capability: `7.5`
- driver: `595.71.05`
- CUDA reported by PyTorch: `13.0`
- PyTorch: `2.12.1+cu130`
- Triton: `3.7.1`
- fixed config: `BLOCK_M=16, BLOCK_N=16, BLOCK_K=32, num_warps=4, num_stages=3, precision_mode=ieee`

## Methodology

The formal run used `tools/run_triton_matmul_bias_relu_benchmark.py` in
`decision-boundary-sweep` mode:

- sessions: `3`
- warmup: `50`
- iterations: `300`
- repeats: `5`
- candidate order: `alternating`
- tie threshold: `abs((V1 - V3) / V1) <= 1%`
- stable CV limit: `5%`

Each workload records session-level medians, session-level winners, candidate
CV, GPU state before/after each session, correctness, and final classification.

## Formal Results

| Workload | M | N | K | Session winners | V1 median ms | V3 median ms | V1/V3 | Relative diff | Classification |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| boundary_m1_n4096_k65536 | 1 | 4096 | 65536 | V1,tie,tie | 18.503807 | 18.434824 | 1.0037 | 0.37% | unstable |
| boundary_m1_n2048_k65536 | 1 | 2048 | 65536 | V3,tie,tie | 9.369628 | 9.362283 | 1.0008 | 0.08% | unstable |
| boundary_m1_n32768_k4096 | 1 | 32768 | 4096 | tie,V1,tie | 9.770906 | 9.920946 | 0.9849 | -1.54% | unstable |
| boundary_m1_n11008_k8192 | 1 | 11008 | 8192 | tie,tie,tie | 7.296974 | 7.281593 | 1.0021 | 0.21% | statistical_tie |
| boundary_m1_n11008_k32768 | 1 | 11008 | 32768 | tie,tie,tie | 31.185654 | 31.186084 | 1.0000 | -0.00% | statistical_tie |
| boundary_m1_n1024_k65536 | 1 | 1024 | 65536 | tie,tie,tie | 4.777124 | 4.770836 | 1.0013 | 0.13% | statistical_tie |
| boundary_m64_n64_k65536 | 64 | 64 | 65536 | tie,V3,tie | 1.728846 | 1.707972 | 1.0122 | 1.21% | unstable |
| boundary_m1_n32768_k8192 | 1 | 32768 | 8192 | V1,tie,tie | 19.919964 | 19.929478 | 0.9995 | -0.05% | unstable |
| boundary_m64_n64_k8192 | 64 | 64 | 8192 | V3,V3,V3 | 0.221693 | 0.217613 | 1.0187 | 1.84% | stable_v3_win |
| boundary_m128_n128_k4096 | 128 | 128 | 4096 | V3,V3,V3 | 0.292522 | 0.287127 | 1.0188 | 1.84% | stable_v3_win |
| boundary_m256_n256_k2048 | 256 | 256 | 2048 | V3,V3,V3 | 0.621194 | 0.612955 | 1.0134 | 1.33% | stable_v3_win |

## Interpretation

The transition region is driven by very large K and small output shapes. In
these shapes, MatMul dominates and the launch/intermediate savings from V3
shrink toward measurement noise. The measured behavior is not a clean V1/V3
binary boundary: the V1-side probes are marginal and session-sensitive.

This is descriptive benchmark evidence only. No analytical or hybrid model is
fit in this PR.

## Canonical Versus Boundary

The canonical suite answers whether one-pass fusion is beneficial across the
original workload map. The decision-boundary tier answers whether the compiler
has a meaningful V1/V3 classification problem. These aggregates must remain
separate.

## Label Isolation

The manifest stores `expected_region` and the boundary artifact stores measured
classification labels, but `Workload.selection_input()` strips evaluation-only
fields. Future compiler selection and cost-model fitting must not consume:

- `expected_region`
- measured winner
- final classification
- oracle latency
- report labels

## Recommended PR C Split

Use a grouped split rather than random row splitting:

- primary: leave-one-K-band-out
- secondary stress check: leave-one-N-family-out

The held-out side must include stable V3, tie/near-boundary, and unstable
regions. Since no stable V1 class was reproduced, PR C should not claim a
balanced V1/V3 classifier. The next smallest implementation is a selector that
can predict stable V3 versus tie/unstable fallback, while continuing to search
for a reproducible V1 region or proposing a candidate-quality change separately.
