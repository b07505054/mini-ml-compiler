#!/usr/bin/env python3
"""Assert the Phase 1 CPU schedule use-plan dispatcher never silently
substitutes an unrecognized candidate — it must fail hard instead."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: assert_cpu_fused_schedule_rejects_unknown_candidate.py <benchmark_exe> <output_dir>",
            file=sys.stderr,
        )
        return 1
    benchmark_exe, output_dir = sys.argv[1], Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    bad_plan = output_dir / "unknown_candidate.plan.json"
    bad_plan.write_text(
        json.dumps(
            {
                "backend": "cpu",
                "kernel": "fused_matmul_bias_relu",
                "schedule": {
                    "candidate_id": "bm99_bn99_bk99",
                    "block_m": 99,
                    "block_n": 99,
                    "block_k": 99,
                    "thread_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [benchmark_exe, "--mode", "use-plan", "--plan", str(bad_plan),
         "--output-dir", str(output_dir)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if completed.returncode == 0:
        print("FAIL: use-plan silently accepted an unknown candidate_id", file=sys.stderr)
        return 1
    if "no silent fallback" not in completed.stderr and "unknown candidate_id" not in completed.stderr:
        print(f"FAIL: rejection message did not name the cause: {completed.stderr}", file=sys.stderr)
        return 1

    # A mismatched block size for a KNOWN candidate_id must also be rejected
    # (drift between plan and registry must not be silently accepted).
    mismatched_plan = output_dir / "mismatched_block.plan.json"
    mismatched_plan.write_text(
        json.dumps(
            {
                "backend": "cpu",
                "kernel": "fused_matmul_bias_relu",
                "schedule": {
                    "candidate_id": "bm16_bn16_bk32",
                    "block_m": 16,
                    "block_n": 999,
                    "block_k": 32,
                    "thread_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    completed2 = subprocess.run(
        [benchmark_exe, "--mode", "use-plan", "--plan", str(mismatched_plan),
         "--output-dir", str(output_dir)],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if completed2.returncode == 0:
        print("FAIL: use-plan silently accepted a block-size/candidate_id mismatch", file=sys.stderr)
        return 1

    print("OK: unknown candidate_id and block-size mismatch are both rejected, not substituted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
