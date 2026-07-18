#!/usr/bin/env python3
"""Prepare and finalize the fixed Slice 19 exact-target calibration."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import aarch64_backend_evidence as m

BASE=ROOT/"artifacts/backend_codegen/aarch64_native_exact_slice18"
RUNTIME_REV=m.RUNTIME_REVISION
ORDERS=[[1,2,4],[2,4,1],[4,1,2],[1,4,2],[2,1,4],[4,2,1]]

def evidence():
 return [json.loads((BASE/f"tile8x8x8_uk{u}/backend_evidence.json").read_text()) for u in m.UKS]

def prepare(out):
 out.mkdir(parents=True,exist_ok=True)
 protocol={"schema_version":1,"benchmark_protocol_version":m.PROTOCOL_VERSION,
  "warmup_samples":30,"measured_samples_per_session":1000,"calls_per_sample":100,
  "session_count":6,"total_measured_calls_per_candidate":600000,"cpu_affinity":3,
  "timing_mode":"batched","session_orders":ORDERS,"equivalence_threshold_pct":3.0,
  "uncertainty_method":"deterministic_session_mean_bootstrap_10000_seed19"}
 (out/"measurement_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
 for e in evidence():
  (out/f"{e['candidate_id']}_measurement_plan.json").write_text(
   json.dumps(m.measurement_plan(e),indent=2)+"\n")
 (out/"static_selection_trace.json").write_text(json.dumps(m.select(evidence()),indent=2)+"\n")

def finalize(out,raw):
 es=evidence(); measurements={}
 for e in es:
  cid=e["candidate_id"]; sessions=[]
  for sid in range(1,7):
   p=raw/f"session{sid}_{cid}.json"; sessions.append(json.loads(p.read_text()))
  first=sessions[0]
  z={"schema_version":1,"benchmark_protocol_version":m.PROTOCOL_VERSION,
   "candidate_id":cid,
   "target_fingerprint":{"hostname":first["target"]["hostname"],"architecture":first["target"]["architecture"],
    "cpu":"cortex-a76","core_count":first["target"]["core_count"],"kernel":first["target"]["kernel"],
    "governor":first["target"]["governor"],"nominal_frequency_khz":2400000,
    "features":[]},
   "artifact_identity":{"object_sha256":e["artifacts"]["object_sha256"],
    "compiler_revision":e["provenance"]["compiler_revision"],"runtime_revision":RUNTIME_REV,
    "entry_point":e["entry_point"],"abi_version":e["abi_version"]},
   "workload_identity":{"operator":e["operator"],"kernel_family":e["kernel_family"],
    "dtype":e["dtype"],"shape":[32,32,32],
    "tile":[8,8,8],"schedule_unroll_k":e["lowering"]["schedule_unroll_k"],
    "vector_width_bits":128,"loop_order_id":m.LOOP,"lowering_pipeline_id":m.PIPELINE},
   "benchmark_protocol":json.loads((out/"measurement_protocol.json").read_text()),
   "correctness":{"passed":all(s["metrics"]["correct"] for s in sessions),
    "repeated_call_correct":all(s["metrics"]["repeated_call_correct"] for s in sessions),
    "guard_buffers_intact":all(s["metrics"]["guard_buffers_intact"] for s in sessions),
    "max_abs_error":max(s["metrics"]["max_abs_error"] for s in sessions)},
   "sessions":sessions}
  z["aggregate"]=m.aggregate_sessions(sessions);measurements[cid]=z
  (out/f"{cid}_measurement.json").write_text(json.dumps(z,indent=2)+"\n")
 decision=m.calibrated_select(es,measurements)
 (out/"calibrated_selection_trace.json").write_text(json.dumps(decision,indent=2)+"\n")
 (out/"candidate_comparison.json").write_text(json.dumps(decision["candidate_aggregates"],indent=2)+"\n")
 profile={"schema_version":1,"benchmark_protocol_version":m.PROTOCOL_VERSION,
          "measurements":{k:f"{k}_measurement.json" for k in measurements}}
 (out/"calibration_profile.json").write_text(json.dumps(profile,indent=2)+"\n")
 selected=next(e for e in es if e["candidate_id"]==decision["selected_candidate_id"])
 decision.update({"measurement_evidence_ref":f"artifacts/backend_codegen/aarch64_native_calibrated_slice19/{selected['candidate_id']}_measurement.json",
  "measurement_policy_ref":"artifacts/backend_codegen/aarch64_native_calibrated_slice19/measurement_protocol.json",
  "benchmark_protocol_version":m.PROTOCOL_VERSION,
  "target_fingerprint":measurements[selected["candidate_id"]]["target_fingerprint"]})
 plan=m.execution_plan(selected,decision,
  "artifacts/backend_codegen/aarch64_native_calibrated_slice19/calibrated_selection_trace.json")
 (out/"calibrated_execution_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
 print(json.dumps(decision,indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument("command",choices=["prepare","finalize"])
 ap.add_argument("--output-dir",required=True);ap.add_argument("--raw-dir")
 a=ap.parse_args();out=Path(a.output_dir)
 prepare(out) if a.command=="prepare" else finalize(out,Path(a.raw_dir))
if __name__=="__main__":main()
