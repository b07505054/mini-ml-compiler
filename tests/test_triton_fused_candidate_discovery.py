#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNNER = TOOLS / "run_triton_fused_candidate_discovery.py"
sys.path.insert(0, str(TOOLS))

import run_triton_fused_candidate_discovery as discovery  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def one_pass_source() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("def _matmul_bias_relu_one_pass_kernel")
    end = text.index("return _matmul_bias_relu_one_pass_kernel", start)
    return text[start:end]


def fake_session(config_values: dict[str, float], winner: str | None = None, cv: float = 0.01) -> dict:
    candidates = {}
    for cid, value in config_values.items():
        candidates[cid] = {
            "status": "completed",
            "timing": {"statistics": {"median_ms": value, "coefficient_of_variation": cv}},
        }
    winner = winner or min(config_values.items(), key=lambda item: (item[1], item[0]))[0]
    return {"status": "completed", "oracle_config": winner, "candidates": candidates}


class Args:
    smoke_test_mode = False
    tie_threshold = 0.01
    near_best_threshold = 0.03
    stable_cv_limit = 0.05
    dominated_threshold = 0.05
    candidate_order = "alternating"
    seed = 0


def main() -> int:
    cfg = discovery.FusedCandidateConfig("bm32_bn32_bk32_w4_s3", 32, 32, 32, 4, 3)
    cfg.validate()
    typed = cfg.typed()
    require(typed["kernel_family_id"] == discovery.KERNEL_FAMILY_ID, "kernel family mismatch")
    require(typed["full_size_intermediates"] == 0, "fused configs must have zero intermediates")
    require(typed["expected_launches"] == 1, "fused configs must launch one kernel")
    require(typed["precision_mode"] == "ieee", "precision mode must remain ieee")

    try:
        discovery.FusedCandidateConfig("bad_warps", 16, 16, 32, 3, 3).validate()
        raise AssertionError("unsupported warp count should fail")
    except ValueError:
        pass

    src = one_pass_source()
    bias_pos = src.index("acc = acc + bias[None, :]")
    relu_pos = src.index("acc = tl.maximum(acc, 0.0)")
    store_pos = src.index("tl.store")
    require(bias_pos < store_pos, "bias must be applied before final store")
    require(relu_pos < store_pos, "ReLU must be applied before final store")
    require(src.count("tl.store") == 1, "one-pass candidate must have one final store")
    require("matmul_out" not in src and "bias_out" not in src, "one-pass candidates must not materialize full intermediates")

    configs = discovery.DEFAULT_CANDIDATES[:4]
    order0 = discovery.candidate_order(configs, Args, 0, 0)
    order1 = discovery.candidate_order(configs, Args, 0, 1)
    require([c.config_id for c in order0] != [c.config_id for c in order1], "alternating order should rotate")

    stable = discovery.classify_formal(
        [
            fake_session({"a": 1.00, "b": 1.05, "c": 1.10}, "a"),
            fake_session({"a": 1.01, "b": 1.06, "c": 1.11}, "a"),
            fake_session({"a": 1.02, "b": 1.07, "c": 1.12}, "a"),
        ],
        [
            discovery.FusedCandidateConfig("a", 16, 16, 32, 4, 3),
            discovery.FusedCandidateConfig("b", 32, 32, 32, 4, 3),
            discovery.FusedCandidateConfig("c", 64, 64, 32, 4, 3),
        ],
        Args,
    )
    require(stable["classification"] == "stable_candidate_win", "stable winner classification mismatch")
    require(stable["oracle_config"] == "a", "stable winner oracle mismatch")

    tie = discovery.classify_formal(
        [
            fake_session({"a": 1.00, "b": 1.005}, "a"),
            fake_session({"a": 1.00, "b": 1.004}, "a"),
            fake_session({"a": 1.00, "b": 1.003}, "a"),
        ],
        [
            discovery.FusedCandidateConfig("a", 16, 16, 32, 4, 3),
            discovery.FusedCandidateConfig("b", 32, 32, 32, 4, 3),
        ],
        Args,
    )
    require(tie["classification"] == "statistical_tie", "tie classification mismatch")

    unstable = discovery.classify_formal(
        [
            fake_session({"a": 1.00, "b": 1.05}, "a"),
            fake_session({"a": 1.07, "b": 1.00}, "b"),
            fake_session({"a": 1.00, "b": 1.05}, "a"),
        ],
        [
            discovery.FusedCandidateConfig("a", 16, 16, 32, 4, 3),
            discovery.FusedCandidateConfig("b", 32, 32, 32, 4, 3),
        ],
        Args,
    )
    require(unstable["classification"] == "unstable", "mixed session winners should be unstable")

    rows = [
        {"cross_session_median_ms_by_config": {"a": 1.0, "b": 1.2, "c": 1.01}},
        {"cross_session_median_ms_by_config": {"a": 1.3, "b": 1.0, "c": 1.02}},
    ]
    pruning = discovery.prune_candidates(
        rows,
        [
            discovery.FusedCandidateConfig("a", 16, 16, 32, 4, 3),
            discovery.FusedCandidateConfig("b", 32, 32, 32, 4, 3),
            discovery.FusedCandidateConfig("c", 64, 64, 32, 4, 3),
        ],
        Args,
    )
    require(pruning["winner_histogram"]["a"] == 1 and pruning["winner_histogram"]["b"] == 1, "winner histogram mismatch")
    require(pruning["candidate_summary"]["c"]["pruning_status"] == "retained", "near-best candidate should be retained")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        output = tmp_path / "sweep.json"
        report = tmp_path / "report.md"
        doc = tmp_path / "doc.md"
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--workload-id",
                "balanced_m64_n64_k64",
                "--candidate-config",
                "bm16_bn16_bk32_w4_s3",
                "--candidate-config",
                "bm32_bn32_bk32_w4_s3",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--repeats",
                "1",
                "--smoke-test-mode",
                "--output",
                str(output),
                "--report-output",
                str(report),
                "--doc-output",
                str(doc),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        require(payload["schema"] == discovery.SCHEMA, "schema mismatch")
        require(payload["candidate_purpose"] == "kernel_selection_fused_config_only", "candidate purpose mismatch")
        require(payload["fusion_attribution_candidate_set"] == ["V1", "V3"], "fusion candidate set must stay separate")
        require(all(c["fusion"] == "one_pass_epilogue" for c in payload["selection_candidate_set"]), "selection set must contain only fused candidates")
        require(payload.get("profile_status") in (None, "unavailable"), completed.stdout + completed.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
