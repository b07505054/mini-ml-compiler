"""Local HTTP service exposing this repo's existing LLM-serving artifact
generation pipeline as POST /compile.

This wraps, unmodified, the same pure-Python stdlib pipeline already used by
tools/run_llm_serving_artifact_pipeline.sh:
  tools/analyze_llm_serving_mlir.py        (build_analysis)
  tools/emit_llm_artifacts_from_analysis.py (analysis_to_config)
  src/ml_graph_compiler_runtime/generate_llm_artifacts.py (generate_artifacts)
  tools/validate_llm_serving_artifacts.py  (validate_artifacts)

It does NOT touch the C++ Graph/Tensor/Node IR, op registry, pass manager,
or kernels -- that subsystem is a separate compiler entirely and is not
invoked here. It does NOT run any GPU/CUDA/Metal kernel and performs no
benchmarking.

Only one real graph is registered today: "tiny_gpt_serving" (backed by
mlir/tiny_gpt_serving.mlir + configs/tiny_gpt_llm_config.json). For that
graph, POST /compile returns result_type="compiled_artifact": the analysis
+ artifact-generation + validation pipeline runs fresh, in-process, per
request, writing to an isolated temp directory -- never to the committed
artifacts/apple_demo/, trace/, or integration_bundle/ paths. Any other
graph_name returns result_type="simulated_compile": a synthetic, clearly
labeled placeholder with no real analysis behind it.

Within the compiled_artifact branch, selected_passes is a fixed, hardcoded
pass_pipeline list already declared in generate_llm_artifacts.py (not a
dynamically computed optimizer decision), and fusion_decisions /
kernel_selection_summary are template-authored candidate plans from that
same file (not derived from real cost-model search or benchmarking). See
each response's truth_boundary field for the full disclosure.

Stdlib only; no new dependency.
"""

import argparse
import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src" / "ml_graph_compiler_runtime"))

import analyze_llm_serving_mlir  # noqa: E402
import emit_llm_artifacts_from_analysis  # noqa: E402
import generate_llm_artifacts  # noqa: E402
import validate_llm_serving_artifacts  # noqa: E402

DEFAULT_GRAPH_NAME = "tiny_gpt_serving"

GRAPH_REGISTRY = {
    "tiny_gpt_serving": {
        "mlir_path": ROOT / "mlir" / "tiny_gpt_serving.mlir",
        "config_path": ROOT / "configs" / "tiny_gpt_llm_config.json",
    },
}

MEMORY_PLAN_SUMMARY_KEYS = [
    "peak_prefill_memory_mb",
    "peak_decode_memory_mb",
    "kv_cache_memory_mb_at_2048_tokens",
    "kv_cache_memory_mb_at_target_concurrency",
    "memory_budget_mb",
    "fits_memory_budget",
]

NOT_CLAIMED_COMPILED = [
    "no GPU/CUDA/Metal kernel execution occurred",
    "no benchmarking was performed; latency/throughput figures in "
    "memory_plan_summary/kernel_selection_summary are pre-existing estimated "
    "template values from generate_llm_artifacts.py, not freshly measured",
    "selected_passes is a fixed, hardcoded pass_pipeline list declared in "
    "generate_llm_artifacts.py, not a dynamically computed optimizer decision",
    "fusion_decisions and kernel_selection_summary are template-authored "
    "candidate plans, not derived from real cost-model search or "
    "benchmarking at request time",
    "the C++ Graph/Tensor/Node IR, op registry, and pass manager were not invoked",
]

NOT_CLAIMED_SIMULATED = [
    "this is not derived from any real graph",
    "no GPU/CUDA/Metal kernel execution occurred",
    "no benchmarking was performed",
    "no compiler passes were run",
]


def _git_commit() -> str:
    return generate_llm_artifacts.git_commit(ROOT)


def _condensed_memory_plan(memory_plan: dict) -> dict:
    return {key: memory_plan[key] for key in MEMORY_PLAN_SUMMARY_KEYS if key in memory_plan}


def _fusion_decisions(candidate_plans: dict) -> list:
    return [
        {
            "plan_id": plan.get("plan_id"),
            "runtime_action": plan.get("runtime_action"),
            "notes": plan.get("notes"),
        }
        for plan in candidate_plans.get("plans", [])
    ]


def _kernel_selection_summary(candidate_plans: dict) -> dict:
    return {
        "selected_plan_id": candidate_plans.get("selected_plan_id"),
        "selection_reason": candidate_plans.get("selection_reason"),
        "candidates": [
            {
                "plan_id": plan.get("plan_id"),
                "backend": plan.get("backend"),
                "runtime_action": plan.get("runtime_action"),
                "estimated_latency_ms": plan.get("estimated_latency_ms"),
                "estimated_throughput_tokens_per_s": plan.get("estimated_throughput_tokens_per_s"),
            }
            for plan in candidate_plans.get("plans", [])
        ],
    }


def _graph_summary_from_analysis(analysis: dict) -> dict:
    return {
        "source": analysis.get("source"),
        "model": analysis.get("model"),
        "op_count": len(analysis.get("ops", [])),
        "phase_partition": analysis.get("phase_partition"),
        "kv_cache_required": analysis.get("kv_cache_analysis", {}).get("required"),
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_graph(graph_name: str, output_root: str | None = None) -> dict:
    registry_entry = GRAPH_REGISTRY.get(graph_name)
    if (
        registry_entry is None
        or not registry_entry["mlir_path"].exists()
        or not registry_entry["config_path"].exists()
    ):
        return _simulated_compile(graph_name)
    return _compiled_artifact(graph_name, registry_entry, output_root)


def _compiled_artifact(graph_name: str, registry_entry: dict, output_root: str | None) -> dict:
    mlir_path = registry_entry["mlir_path"]
    mlir_text = mlir_path.read_text(encoding="utf-8")
    analysis = analyze_llm_serving_mlir.build_analysis(mlir_path, mlir_text)

    base_config = _load_json(registry_entry["config_path"])
    config = emit_llm_artifacts_from_analysis.analysis_to_config(base_config, analysis)

    out_dir = Path(tempfile.mkdtemp(prefix="compile_service_", dir=output_root))
    written = generate_llm_artifacts.generate_artifacts(config, out_dir)
    artifact_paths = [str(out_dir / filename) for filename in written]

    memory_plan = _load_json(out_dir / "memory_plan.json")
    candidate_plans = _load_json(out_dir / "candidate_execution_plans.json")
    provenance = _load_json(out_dir / "artifact_provenance.json")

    checks = validate_llm_serving_artifacts.validate_artifacts(out_dir)
    passed = sum(1 for check in checks if check["passed"])
    failed = sum(1 for check in checks if not check["passed"])

    return {
        "result_type": "compiled_artifact",
        "graph_name": graph_name,
        "graph_summary": _graph_summary_from_analysis(analysis),
        "selected_passes": provenance.get("pass_pipeline", []),
        "fusion_decisions": _fusion_decisions(candidate_plans),
        "memory_plan_summary": _condensed_memory_plan(memory_plan),
        "kernel_selection_summary": _kernel_selection_summary(candidate_plans),
        "artifact_paths": artifact_paths,
        "validation": {
            "status": "passed" if failed == 0 else "failed",
            "passed": passed,
            "failed": failed,
            "total": len(checks),
        },
        "truth_boundary": {
            "compiler_logic_executed": True,
            "note": (
                "Ran this repo's existing LLM-serving artifact-generation pipeline "
                "(analyze_llm_serving_mlir + emit_llm_artifacts_from_analysis + "
                f"generate_llm_artifacts) in-process against the real {mlir_path.name} "
                f"graph and {registry_entry['config_path'].name} config, writing to an "
                "isolated temp directory. This is real Python execution of "
                "artifact-generation logic, not a C++ pass-manager run and not "
                "measured GPU/CUDA/Metal kernel execution."
            ),
            "not_claimed": NOT_CLAIMED_COMPILED,
        },
        "git_commit": _git_commit(),
    }


def _simulated_compile(graph_name: str) -> dict:
    return {
        "result_type": "simulated_compile",
        "graph_name": graph_name,
        "graph_summary": {
            "source": None,
            "model": {},
            "op_count": 0,
            "phase_partition": {"shared_ops": [], "prefill_ops": [], "decode_ops": []},
            "kv_cache_required": False,
        },
        "selected_passes": [],
        "fusion_decisions": [],
        "memory_plan_summary": {},
        "kernel_selection_summary": {
            "selected_plan_id": None,
            "selection_reason": None,
            "candidates": [],
        },
        "artifact_paths": [],
        "validation": {"status": "not_run", "passed": 0, "failed": 0, "total": 0},
        "truth_boundary": {
            "compiler_logic_executed": False,
            "note": (
                f"No registered real graph definition was found for "
                f"graph_name={graph_name!r}. This response is a synthetic "
                "placeholder; no MLIR analysis or artifact generation ran."
            ),
            "not_claimed": NOT_CLAIMED_SIMULATED,
        },
        "git_commit": _git_commit(),
    }


class CompileHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/compile":
            self._send_json({"error": "not_found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, status=400)
            return

        if not isinstance(payload, dict):
            self._send_json({"error": "payload must be a JSON object"}, status=400)
            return

        graph_name = payload.get("graph_name") or DEFAULT_GRAPH_NAME

        try:
            result = compile_graph(graph_name)
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=400)
            return

        self._send_json(result)

    def log_message(self, fmt, *args) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local compiler engine service (LLM serving artifact pipeline)."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8902)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), CompileHandler)
    print(f"compile service: http://{args.host}:{args.port}/compile")
    print(
        "compiled_artifact responses run this repo's existing LLM-serving "
        "artifact pipeline in-process; simulated_compile responses are "
        "synthetic placeholders for unknown graph_name values."
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
