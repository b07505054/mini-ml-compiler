import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import mlir_fusion_to_runtime_json as bridge


def candidate(candidate_id="cuda_rmsnorm_fp32_bs256_v1", p50=0.02, **updates):
    row = {"candidate_id": candidate_id, "operator": "rmsnorm", "semantics": "weighted_rmsnorm",
           "backend": "cuda", "kernel_family": "custom_cuda_rmsnorm", "kernel_entry_point": "fused_rmsnorm_forward",
           "dtype": "fp32", "tokens": 16, "hidden": 4096, "epsilon": 1e-6, "block_size": 256,
           "num_warps": None, "num_stages": None, "target": {"gpu_name": "GTX", "compute_capability": "7.5"},
           "p50_ms": p50, "correct": True, "selection_ready": True, "measurement_kind": "measured"}
    row.update(updates)
    return row


def test_fastest_exact_candidate_and_tie_break():
    profile = {"exact_candidates": [candidate(p50=0.03), candidate("cuda_rmsnorm_fp32_bs128_v1", 0.02, block_size=128)]}
    result = bridge.select_exact_rmsnorm_candidate(profile, {"tokens": 16, "hidden": 4096, "dtype": "f32"}, "GTX", "7.5")
    assert result["selected_candidate_id"] == "cuda_rmsnorm_fp32_bs128_v1"
    assert result["exact_match"] and not result["fallback_used"]


def test_wrong_shape_dtype_target_and_incorrect_fail_closed():
    rows = [candidate(tokens=1), candidate(dtype="fp16"), candidate(target={"gpu_name": "Other", "compute_capability": "7.5"}), candidate(correct=False, p50=0.001)]
    result = bridge.select_exact_rmsnorm_candidate({"exact_candidates": rows}, {"tokens": 16, "hidden": 4096, "dtype": "f32"}, "GTX", "7.5")
    assert result["selected_candidate_id"] == "torch_rmsnorm_fp32_v1"
    assert result["fallback_used"] and not result["exact_match"]
    assert len(result["rejected_candidates"]) == 4


def test_triton_backend_is_not_cpu(tmp_path):
    source = tmp_path / "triton.json"
    source.write_text(json.dumps({"environment": {"gpu_name": "GTX", "compute_capability": "7.5"}, "exact_candidates": [candidate("triton_rmsnorm_fp32_block4096_warps8_stages_default_v1", backend="triton", kernel_family="triton_rmsnorm", kernel_entry_point="rmsnorm_kernel", block_size=4096, num_warps=8, num_stages="default")]}))
    output = tmp_path / "cost.json"
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools/build_profile_cost_table.py"), "--profile", str(source), "--output", str(output)], check=True)
    row = json.loads(output.read_text())["exact_candidates"][0]
    assert row["backend"] == "triton"
    assert row["num_warps"] == 8
