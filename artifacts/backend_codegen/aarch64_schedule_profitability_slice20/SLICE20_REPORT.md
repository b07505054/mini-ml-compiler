# Slice 20 AArch64 schedule profitability

uk4 won every tested domain, but no universality claim is made

| Domain | uk1 p50 | uk2 p50 | uk4 p50 | Winner | Legacy | Revised | Revised regret |
|---|---:|---:|---:|---|---|---|---:|
| m16_n16_k32_tile8x8x8 | 0.001173240 | 0.001121295 | 0.001005920 | m16_n16_k32_tile8x8x8_uk4 | m16_n16_k32_tile8x8x8_uk1 | m16_n16_k32_tile8x8x8_uk4 | 0.000% |
| m32_n32_k32_tile8x8x8 | 0.004367639 | 0.004146623 | 0.003851541 | m32_n32_k32_tile8x8x8_uk4 | m32_n32_k32_tile8x8x8_uk1 | m32_n32_k32_tile8x8x8_uk4 | 0.000% |
| m64_n64_k64_tile8x8x8 | 0.027169286 | 0.025050286 | 0.023997429 | m64_n64_k64_tile8x8x8_uk4 | m64_n64_k64_tile8x8x8_uk1 | m64_n64_k64_tile8x8x8_uk4 | 0.000% |
| m32_n32_k128_tile8x8x8 | 0.010979000 | 0.009997467 | 0.009548133 | m32_n32_k128_tile8x8x8_uk4 | m32_n32_k128_tile8x8x8_uk1 | m32_n32_k128_tile8x8x8_uk4 | 0.000% |
| m32_n32_k64_tile8x8x8 | 0.006567267 | 0.006080233 | 0.005869733 | m32_n32_k64_tile8x8x8_uk4 | m32_n32_k64_tile8x8x8_uk1 | m32_n32_k64_tile8x8x8_uk4 | 0.000% |
| m64_n32_k32_tile8x8x8 | 0.008664800 | 0.008178367 | 0.007641333 | m64_n32_k32_tile8x8x8_uk4 | m64_n32_k32_tile8x8x8_uk1 | m64_n32_k32_tile8x8x8_uk4 | 0.000% |
| m32_n64_k32_tile8x8x8 | 0.008745033 | 0.008219733 | 0.007850600 | m32_n64_k32_tile8x8x8_uk4 | m32_n64_k32_tile8x8x8_uk1 | m32_n64_k32_tile8x8x8_uk4 | 0.000% |
| m8_n64_k64_tile8x8x8 | 0.003367328 | 0.003121426 | 0.002979049 | m8_n64_k64_tile8x8x8_uk4 | m8_n64_k64_tile8x8x8_uk1 | m8_n64_k64_tile8x8x8_uk4 | 0.000% |
| m8_n8_k32_tile8x8x8 | 0.000302965 | 0.000291480 | 0.000261665 | m8_n8_k32_tile8x8x8_uk4 | m8_n8_k32_tile8x8x8_uk1 | m8_n8_k32_tile8x8x8_uk4 | 0.000% |
| m16_n64_k32_tile8x8x8 | 0.004416213 | 0.004154508 | 0.003947787 | m16_n64_k32_tile8x8x8_uk4 | m16_n64_k32_tile8x8x8_uk1 | m16_n64_k32_tile8x8x8_uk4 | 0.000% |
