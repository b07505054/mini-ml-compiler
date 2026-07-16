# Known Failing-Test Baseline (AArch64 Backend Codegen Slice)

Recorded at commit time so future slices can prove they introduced zero new
regressions by diffing against this exact baseline.

## Numbers

```text
Baseline HEAD before this commit (sparse-compiler-removal commit):
cfa1252c354544968c64645c47af89ec26800b60

Committed baseline tests (tools/run_mlir_pass_tests.sh at that HEAD):
82 total
20 failing

Backend working-tree tests (same script, with this commit's changes applied):
85 total
20 failing

New backend tests introduced by this commit:
3 passed
0 failed

Failing-test-name sets (committed baseline vs. working tree): identical.

Conclusion:
No new regressions introduced.
```

## Note on an earlier reported baseline

An earlier working-tree snapshot in this project's history reported 89 tests
total / 23 failing (committed HEAD 86 total / 23 failing). Those numbers
predate a separate, already-committed change (`cfa1252c`, "Remove structured
2:4 sparse-compiler feature") that deleted 4 sparse-only test fixtures -- 3 of
which were among the 23 historical failures, and 1 of which was passing. That
accounts for the -4 total / -3 failing shift between the earlier snapshot and
the numbers recorded above. The regression-freeness claim in this file is
about *this* commit (the AArch64 backend slice) relative to its own immediate
parent HEAD, `cfa1252c`, and was verified directly against that parent, not
against the older snapshot.

## Historical failures grouped by root cause (20 total)

```text
14: linalg.map operand / mapper-arity verifier incompatibility
     (down from 17 before the sparse-compiler-removal commit, which deleted
     3 sparse-fusion test fixtures that failed for this same reason)
2:  hir.quantize clamp_min verifier requirement
2:  serving metadata FileCheck drift
2:  LLM frontend normalization FileCheck drift
```

All 20 are pre-existing and unrelated to the AArch64 backend-codegen slice.
None reference backend codegen, LLVM lowering, `mlir-translate`, `llc`, or
AArch64 in any way. Verified via a non-invasive `/tmp`-copy diff isolation
(the committed test-runner script, `git show <HEAD>:tools/run_mlir_pass_tests.sh`,
copied outside the repo and run standalone) rather than any modification to
the repository itself.

This task does not fix these 20 failures. They remain scoped as separate,
future work.
