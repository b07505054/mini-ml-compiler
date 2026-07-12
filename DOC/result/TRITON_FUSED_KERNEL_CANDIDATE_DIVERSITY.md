# Triton Fused MatMul-Bias-ReLU Candidate Diversity

This report evaluates one-pass fused Triton configurations only. V1/V3 fusion attribution remains separate.

## Candidate Configurations

| Candidate | BLOCK_M | BLOCK_N | BLOCK_K | Warps | Stages | Hypothesis |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bm16_bn16_bk32_w4_s3 | 16 | 16 | 32 | 4 | 3 | small tile baseline for skinny or edge-heavy shapes |
| bm32_bn32_bk32_w4_s3 | 32 | 32 | 32 | 4 | 3 | balanced square tile for regular shapes |
| bm64_bn64_bk32_w4_s3 | 64 | 64 | 32 | 4 | 3 | larger tile for regular output reuse |
| bm16_bn64_bk32_w4_s3 | 16 | 64 | 32 | 4 | 3 | wide tile for small-M wide-N shapes |
| bm32_bn64_bk32_w4_s3 | 32 | 64 | 32 | 4 | 3 | moderate wide tile for LLM-like N-heavy shapes |

## Candidate Summary

| Candidate | Stable wins | Ties/near-best | Mean normalized latency | Worst regret | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| bm16_bn16_bk32_w4_s3 | 1 | 3 | 1.8528 | 2.3343 | retained |
| bm32_bn32_bk32_w4_s3 | 0 | 4 | 1.3466 | 0.6569 | retained |
| bm64_bn64_bk32_w4_s3 | 2 | 6 | 1.3704 | 1.2886 | retained |
| bm16_bn64_bk32_w4_s3 | 2 | 7 | 1.1931 | 0.7521 | retained |
| bm32_bn64_bk32_w4_s3 | 0 | 3 | 1.2098 | 0.5103 | retained |

## Winner Histogram

- `bm16_bn16_bk32_w4_s3`: `1`
- `bm32_bn32_bk32_w4_s3`: `2`
- `bm64_bn64_bk32_w4_s3`: `5`
- `bm16_bn64_bk32_w4_s3`: `4`
- `bm32_bn64_bk32_w4_s3`: `0`

## Per-Workload Results

| Workload | M | N | K | Oracle config | Second config | Margin | Stable? |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| rep_m1_k768_n3072 | 1 | 3072 | 768 | bm16_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.2730 | unstable |
| rep_m16_k768_n3072 | 16 | 3072 | 768 | bm16_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.2245 | unstable |
| rep_m128_k768_n3072 | 128 | 3072 | 768 | bm64_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.3034 | stable_candidate_win |
| stress_m512_n512_k32 | 512 | 512 | 32 | bm64_bn64_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | 0.0080 | statistical_tie |
| balanced_m64_n64_k64 | 64 | 64 | 64 | bm32_bn32_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | 0.0017 | statistical_tie |
| balanced_m128_n128_k128 | 128 | 128 | 128 | bm64_bn64_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | 0.0022 | statistical_tie |
| balanced_m512_n512_k512 | 512 | 512 | 512 | bm64_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.3002 | unstable |
| unfriendly_m128_n128_k1024 | 128 | 128 | 1024 | bm32_bn32_bk32_w4_s3 | bm16_bn64_bk32_w4_s3 | 0.0639 | unstable |
| unfriendly_m64_n64_k4096 | 64 | 64 | 4096 | bm16_bn16_bk32_w4_s3 | bm32_bn32_bk32_w4_s3 | 0.0486 | stable_candidate_win |
| boundary_m1_n4096_k65536 | 1 | 4096 | 65536 | bm16_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.2826 | stable_candidate_win |
| boundary_m1_n11008_k8192 | 1 | 11008 | 8192 | bm16_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.2869 | stable_candidate_win |
| boundary_m256_n256_k2048 | 256 | 256 | 2048 | bm64_bn64_bk32_w4_s3 | bm32_bn64_bk32_w4_s3 | 0.1674 | stable_candidate_win |

## Recommended Primary Candidate Set

`bm16_bn16_bk32_w4_s3`, `bm32_bn32_bk32_w4_s3`, `bm64_bn64_bk32_w4_s3`, `bm16_bn64_bk32_w4_s3`

## Retained Candidate Set

`bm16_bn16_bk32_w4_s3`, `bm32_bn32_bk32_w4_s3`, `bm64_bn64_bk32_w4_s3`, `bm16_bn64_bk32_w4_s3`, `bm32_bn64_bk32_w4_s3`

## Dominated Candidates

None

## Interpretation

Genuine selection diversity: `True`.
If one configuration still dominates, a specialized M=1 or split-K candidate should be proposed separately with evidence.
