import copy, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import aarch64_backend_evidence as m

def pair(tmp,uk,p50):
 o=tmp/f"u{uk}.o";o.write_bytes(bytes([uk]))
 e={**m.identity(uk),"artifacts":{"object_ref":o.name,"object_sha256":hashlib.sha256(o.read_bytes()).hexdigest(),"backend_evidence_ref":f"u{uk}.json"},
 "static_backend_evidence":{"object_size_bytes":100+uk,"text_size_bytes":80+uk,"static_instruction_count":1,"fmla_count":1,"spill_store_count":0,"reload_load_count":0,"spill_slot_bytes":0,"physical_vector_registers_referenced":1,"approximate_peak_live_vector_registers":1},
 "estimated_backend_evidence":{"llvm_mca_estimated_cycles":None,"methodology":None},"measured_backend_evidence":None,
 "validation":{"codegen_succeeded":True,"llvm_ir_verified":True,"correctness_passed":None,"measured_on_target":False},
 "provenance":{"compiler_revision":"rev","llvm_version":"v","working_tree_clean":False}}
 sess={"metrics":{"p50_ms":p50,"p95_ms":p50*1.01,"mean_ms":p50},
       "identity_proof":{"runtime_redecision_count":0}}
 meas={"schema_version":1,"benchmark_protocol_version":m.PROTOCOL_VERSION,"candidate_id":e["candidate_id"],
 "target_fingerprint":{"architecture":"aarch64","cpu":m.CPU,"features":[]},
 "artifact_identity":{"object_sha256":e["artifacts"]["object_sha256"],"compiler_revision":"rev","runtime_revision":m.RUNTIME_REVISION,"entry_point":m.ENTRY,"abi_version":m.ABI},
 "workload_identity":{"operator":m.FAMILY,"kernel_family":e["kernel_family"],"dtype":"f32","shape":[32,32,32],"tile":[8,8,8],"schedule_unroll_k":uk,"vector_width_bits":128,"loop_order_id":m.LOOP,"lowering_pipeline_id":m.PIPELINE},
 "benchmark_protocol":{"version":m.PROTOCOL_VERSION},"correctness":{"passed":True,"repeated_call_correct":True,"guard_buffers_intact":True,"max_abs_error":0},
 "sessions":[copy.deepcopy(sess) for _ in range(6)]}
 meas["aggregate"]=m.aggregate_sessions(meas["sessions"])
 return e,meas

def test_exact_measurement_accepted_and_fastest_selected(tmp_path,monkeypatch):
 monkeypatch.setattr(m,"ROOT",tmp_path)
 pairs=[pair(tmp_path,u,p) for u,p in [(1,.010),(2,.008),(4,.012)]]
 r=m.calibrated_select([x[0] for x in pairs],{x[0]["candidate_id"]:x[1] for x in pairs})
 assert r["static_selected_candidate"]=="tile8x8x8_uk1"
 assert r["calibrated_selected_candidate"]=="tile8x8x8_uk2" and r["selection_changed"]

def test_equivalence_uses_text_size(tmp_path,monkeypatch):
 monkeypatch.setattr(m,"ROOT",tmp_path)
 pairs=[pair(tmp_path,u,p) for u,p in [(1,.010),(2,.0099),(4,.0101)]]
 r=m.calibrated_select([x[0] for x in pairs],{x[0]["candidate_id"]:x[1] for x in pairs})
 assert r["selected_candidate_id"]=="tile8x8x8_uk1"

def test_all_identity_protocol_and_correctness_mismatches_rejected(tmp_path,monkeypatch):
 monkeypatch.setattr(m,"ROOT",tmp_path);e,z=pair(tmp_path,1,.01)
 mutations=[("candidate_id","bad"),("benchmark_protocol_version","bad")]
 for key,val in mutations:
  q=copy.deepcopy(z);q[key]=val;assert m.validate_measurement(e,q)
 for section,key,val in [
  ("artifact_identity","object_sha256","0"*64),("artifact_identity","entry_point","bad"),("artifact_identity","abi_version","bad"),
  ("workload_identity","shape",[1,1,1]),("workload_identity","dtype","f16"),("workload_identity","tile",[4,8,8]),
  ("workload_identity","schedule_unroll_k",2),("workload_identity","vector_width_bits",64),
  ("workload_identity","loop_order_id","bad"),("workload_identity","lowering_pipeline_id","bad"),
  ("target_fingerprint","cpu","bad"),("correctness","passed",False),("correctness","guard_buffers_intact",False)]:
  q=copy.deepcopy(z);q[section][key]=val;assert m.validate_measurement(e,q)
 q=copy.deepcopy(z);q["sessions"][0]["identity_proof"]["runtime_redecision_count"]=1
 assert "measurement_runtime_redecision" in m.validate_measurement(e,q)

def test_missing_sessions_unavailable_and_calibrated_plan_refs(tmp_path,monkeypatch):
 monkeypatch.setattr(m,"ROOT",tmp_path);e,z=pair(tmp_path,1,.01);z["sessions"]=[]
 assert "measurement_sessions_unavailable" in m.validate_measurement(e,z)
 d={"selection_mode":"exact_target_calibrated","static_selected_candidate":"tile8x8x8_uk1","calibrated_selected_candidate":"tile8x8x8_uk1","measurement_evidence_ref":"uk1.json","measurement_policy_ref":"protocol.json","benchmark_protocol_version":m.PROTOCOL_VERSION}
 p=m.execution_plan(e,d,"trace.json");n=p["function_plans"][0]["per_op_decisions"][0]["native_execution"]
 assert n["measurement_evidence_ref"]=="uk1.json" and n["object_sha256"]==e["artifacts"]["object_sha256"]
