#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
RUNNER = TOOLS / "run_triton_matmul_bias_relu_benchmark.py"
sys.path.insert(0, str(TOOLS))

import run_triton_matmul_bias_relu_benchmark as bench  # noqa: E402
from matmul_postop_workloads import canonical_workloads, decision_boundary_workloads, load_manifest, postop_shape_for  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_expect_failure(args: list[str]) -> str:
    completed = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    require(completed.returncode != 0, "command unexpectedly succeeded")
    return completed.stdout + completed.stderr


def fake_session(v1: float, v3: float, v1_cv: float = 0.01, v3_cv: float = 0.01) -> dict:
    return {
        "status": "completed",
        "variants": {
            "V1": {"timing": {"statistics": {"median_ms": v1, "coefficient_of_variation": v1_cv}}},
            "V3": {"timing": {"statistics": {"median_ms": v3, "coefficient_of_variation": v3_cv}}},
        },
    }


def one_pass_source() -> str:
    text = RUNNER.read_text(encoding="utf-8")
    start = text.index("def _matmul_bias_relu_one_pass_kernel")
    end = text.index("return _matmul_kernel", start)
    return text[start:end]


def main() -> int:
    manifest = load_manifest(ROOT / "benchmarks" / "matmul_postop_workloads.json")
    require(len(canonical_workloads(manifest)) == 33, "all 33 canonical manifest entries must be discoverable")
    require(len(decision_boundary_workloads(manifest)) >= 6, "decision-boundary entries must be discoverable")

    require(postop_shape_for("bias", 7, 11) == [11], "bias shape must be [N]")
    try:
        bench.make_generators(None, 0, "cuda")  # type: ignore[arg-type]
    except AttributeError:
        pass
    require(bench.V1_METADATA["runtime_operations"] == 3, "V1 must report three operations")
    require(bench.V1_METADATA["full_size_intermediates"] == 2, "V1 must report two intermediates")
    require(bench.V3_METADATA["runtime_operations"] == 1, "V3 must report one operation")
    require(bench.V3_METADATA["full_size_intermediates"] == 0, "V3 must report zero intermediates")

    class Args:
        block_m = 16
        block_n = 16
        block_k = 32
        num_warps = 4
        num_stages = 3
        precision_mode = "ieee"

    cfg = bench.fixed_config(Args)
    require(cfg["BLOCK_M"] == 16 and cfg["BLOCK_N"] == 16 and cfg["BLOCK_K"] == 32, "fixed config mismatch")
    require(cfg["precision_mode"] == "ieee", "precision mode mismatch")

    src = one_pass_source()
    bias_pos = src.index("acc = acc + bias[None, :]")
    relu_pos = src.index("acc = tl.maximum(acc, 0.0)")
    store_pos = src.index("tl.store")
    require(bias_pos < store_pos, "V3 must add bias before final store")
    require(relu_pos < store_pos, "V3 must apply ReLU before final store")
    require(src.count("tl.store") == 1, "V3 must contain exactly one output store")
    require("matmul_out" not in src and "bias_out" not in src, "V3 must not allocate full-size intermediates")

    required = bench.estimate_required_bytes(8, 16, 4)
    require(required["v1_matmul_intermediate"] == 8 * 16 * 4, "V1 matmul intermediate bytes mismatch")
    require(required["v1_bias_intermediate"] == 8 * 16 * 4, "V1 bias intermediate bytes mismatch")

    class BoundaryArgs:
        smoke_test_mode = False
        tie_threshold = 0.01
        stable_cv_limit = 0.05
        candidate_order = "alternating"
        seed = 0

    require(
        bench.classify_boundary_workload(
            [fake_session(1.00, 1.03), fake_session(1.01, 1.04), fake_session(1.02, 1.05)],
            BoundaryArgs,
        )["final_classification"] == "stable_v1_win",
        "stable V1 classification mismatch",
    )
    require(
        bench.classify_boundary_workload(
            [fake_session(1.04, 1.00), fake_session(1.05, 1.01), fake_session(1.06, 1.02)],
            BoundaryArgs,
        )["final_classification"] == "stable_v3_win",
        "stable V3 classification mismatch",
    )
    require(
        bench.classify_boundary_workload(
            [fake_session(1.00, 1.005), fake_session(1.00, 0.997), fake_session(1.00, 1.000)],
            BoundaryArgs,
        )["final_classification"] == "statistical_tie",
        "tie classification mismatch",
    )
    require(
        bench.classify_boundary_workload(
            [fake_session(1.00, 1.03), fake_session(1.03, 1.00), fake_session(1.00, 1.00)],
            BoundaryArgs,
        )["final_classification"] == "unstable",
        "mixed session winners should be unstable",
    )
    require(
        bench.classify_boundary_workload(
            [fake_session(1.00, 1.03, 0.10), fake_session(1.00, 1.03), fake_session(1.00, 1.03)],
            BoundaryArgs,
        )["final_classification"] == "unstable",
        "excessive CV should prevent stable classification",
    )
    require(bench.candidate_order(BoundaryArgs, 0, 0) == ["V1", "V3"], "alternating order first mismatch")
    require(bench.candidate_order(BoundaryArgs, 0, 1) == ["V3", "V1"], "alternating order second mismatch")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "profile.json"
        report = tmp_path / "report.md"
        zero_msg = run_expect_failure(
            [
                sys.executable,
                str(RUNNER),
                "--workload-id",
                "balanced_m64_n64_k64",
                "--warmup",
                "0",
                "--iterations",
                "300",
                "--repeats",
                "5",
                "--output",
                str(out),
                "--report-output",
                str(report),
            ]
        )
        require("warmup" in zero_msg or "positive" in zero_msg, "zero warmup failure should be explicit")

        low_budget_msg = run_expect_failure(
            [
                sys.executable,
                str(RUNNER),
                "--workload-id",
                "balanced_m64_n64_k64",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--repeats",
                "1",
                "--output",
                str(out),
                "--report-output",
                str(report),
            ]
        )
        require("formal runs require" in low_budget_msg, "formal low-budget rejection should be explicit")

        boundary_sessions_msg = run_expect_failure(
            [
                sys.executable,
                str(RUNNER),
                "--mode",
                "decision-boundary-sweep",
                "--all-eligible",
                "--sessions",
                "2",
                "--warmup",
                "50",
                "--iterations",
                "300",
                "--repeats",
                "5",
                "--output",
                str(out),
                "--report-output",
                str(report),
            ]
        )
        require("sessions >= 3" in boundary_sessions_msg, "formal boundary sweep should require three sessions")

        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--workload-id",
                "balanced_m64_n64_k64",
                "--warmup",
                "1",
                "--iterations",
                "1",
                "--repeats",
                "1",
                "--smoke-test-mode",
                "--output",
                str(out),
                "--report-output",
                str(report),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
        require(payload["schema"] == bench.SCHEMA, "profile schema mismatch")
        require(payload["backend"] == "triton_cuda", "backend mismatch")
        require(report.exists(), "report should be generated")
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
            require(payload["profile_status"] == "unavailable", "missing deps should produce unavailable profile")
            require("import failed" in payload["unavailable_reason"], "unavailable reason should name import failure")
        else:
            require(payload["profile_status"] in ("unavailable", "measured"), completed.stdout + completed.stderr)

    # Shape/device/dtype validation is enforced in the runner before benchmarking;
    # CPU-only CI reaches the dependency-unavailable path, while GPU validation is
    # exercised by the remote smoke/formal commands.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
