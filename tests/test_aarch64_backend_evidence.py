import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import aarch64_backend_evidence as m


def fixture(tmp_path, uk, **updates):
    obj = tmp_path / f"uk{uk}.o"
    obj.write_bytes(f"object-{uk}".encode())
    e = {**m.identity(uk),
         "artifacts": {"object_ref": obj.name,
                       "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
                       "backend_evidence_ref": f"uk{uk}.json"},
         "static_backend_evidence": {
             "object_size_bytes": 100 + uk, "text_size_bytes": 80 + uk,
             "static_instruction_count": 10, "fmla_count": 4,
             "spill_store_count": uk == 4, "reload_load_count": uk == 4,
             "spill_slot_bytes": 16 if uk == 4 else 0,
             "physical_vector_registers_referenced": 8,
             "approximate_peak_live_vector_registers": 9},
         "estimated_backend_evidence": {"llvm_mca_estimated_cycles": None,
                                        "methodology": None},
         "measured_backend_evidence": None,
         "validation": {"codegen_succeeded": True, "llvm_ir_verified": True,
                        "correctness_passed": None, "measured_on_target": False},
         "provenance": {"compiler_revision": "x", "llvm_version": "x",
                        "working_tree_clean": False}}
    for path, value in updates.items():
        target = e
        bits = path.split(".")
        for bit in bits[:-1]:
            target = target[bit]
        target[bits[-1]] = value
    return e


def test_exact_candidate_ids_and_artifact_paths_are_unique(tmp_path):
    assert [m.candidate_id(x) for x in m.UKS] == [
        "tile8x8x8_uk1", "tile8x8x8_uk2", "tile8x8x8_uk4"]
    paths = [m.artifact_paths(tmp_path, x) for x in m.UKS]
    for key in ("llvm_dialect", "llvm_ir", "mir_post_isel",
                "mir_pre_scheduler", "mir_pre_ra", "mir_post_ra",
                "mir_post_prologue_epilogue", "assembly", "object", "evidence"):
        values = [str(p[key]) for p in paths]
        assert len(set(values)) == 3
        assert all(f"uk{uk}" in value for uk, value in zip(m.UKS, values))


def test_hard_gates_and_object_hash(tmp_path):
    good = fixture(tmp_path, 2)
    assert m.validate_evidence(good, root=tmp_path) == []
    cases = {
        "candidate_id": "wrong", "dtype": "f16", "shape.m": 31,
        "target.cpu": "other", "lowering.tile_m": 4,
        "lowering.schedule_unroll_k": 1, "lowering.vector_width_bits": 64,
        "lowering.pipeline_id": "wrong", "entry_point": "",
        "abi_version": "wrong", "validation.correctness_passed": False,
    }
    for field, value in cases.items():
        bad = copy.deepcopy(good)
        target = bad
        bits = field.split(".")
        for bit in bits[:-1]:
            target = target[bit]
        target[bits[-1]] = value
        assert m.validate_evidence(bad, root=tmp_path), field
    bad = copy.deepcopy(good)
    (tmp_path / "uk2.o").write_bytes(b"stale")
    assert "object_sha256_mismatch" in m.validate_evidence(bad, root=tmp_path)


def test_static_selection_is_deterministic_and_spills_are_not_hard_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    rows = [fixture(tmp_path, uk) for uk in m.UKS]
    result = m.select(list(reversed(rows)))
    assert result["selection_mode"] == "deterministic_static_lexicographic_estimate"
    assert result["selected_candidate_id"] == "tile8x8x8_uk1"
    assert all(x["accepted"] for x in result["selector_trace"])


def test_exact_measurement_selected_only_with_matching_target_and_correctness(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    rows = [fixture(tmp_path, uk) for uk in m.UKS]
    rows[1]["measured_backend_evidence"] = {
        "target_profile_id": m.PROFILE, "correctness_passed": True,
        "latency_p50_ms": 0.01}
    rows[2]["measured_backend_evidence"] = {
        "target_profile_id": "wrong", "correctness_passed": True,
        "latency_p50_ms": 0.001}
    result = m.select(rows)
    assert result["selection_mode"] == "exact_raspberry_pi_measurement"
    assert result["selected_candidate_id"] == "tile8x8x8_uk2"


def test_execution_plan_retains_exact_native_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    chosen = fixture(tmp_path, 1)
    decision = m.select([chosen])
    plan = m.execution_plan(chosen, decision, "selection.json")
    native = plan["function_plans"][0]["per_op_decisions"][0]["native_execution"]
    assert native["candidate_id"] == "tile8x8x8_uk1"
    assert native["object_sha256"] == chosen["artifacts"]["object_sha256"]
    assert native["entry_point"] == m.ENTRY
    assert native["runtime_no_redecision"] is True
