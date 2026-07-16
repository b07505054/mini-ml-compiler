#!/usr/bin/env python3
"""Exact-profile measured selection for the single vLLM max_num_seqs dimension."""
import argparse,hashlib,json
from pathlib import Path

VALUES=(None,1,2,4,8)
IDS={None:"vllm_max_num_seqs_default",1:"vllm_max_num_seqs_1",2:"vllm_max_num_seqs_2",4:"vllm_max_num_seqs_4",8:"vllm_max_num_seqs_8"}
def candidates(fixed,target,workload,version):
 return [{"candidate_id":IDS[v],"max_num_seqs":v,"value_source":"default" if v is None else "explicit","model_identity":fixed["model"],"target_gpu_identity":target,"vllm_version":version,"dtype":fixed["dtype"],"max_model_len":fixed["max_model_len"],"max_num_batched_tokens":fixed["max_num_batched_tokens"],"enable_chunked_prefill":fixed["enable_chunked_prefill"],"enable_prefix_caching":fixed["enable_prefix_caching"],"tensor_parallel_size":1,"pipeline_parallel_size":1,"workload_identity":workload,"constraints":{"positive_when_explicit":True},"measurement_provenance":"exact_target_model_workload_measured_evidence","fallback_identity":"vllm_max_num_seqs_default","truth_boundary":"target_model_workload_specific_measured_vllm_policy"} for v in VALUES]
def select(rows,objective,defs):
 valid=[r for r in rows if r["classification"]=="VALID" and r["failure_count"]==0 and r["oom_count"]==0]
 if not valid:return None,"no_valid_candidate_declared_default_fallback"
 floor=defs[objective].get("minimum_output_tokens_per_second",0);limit=defs[objective].get("maximum_ttft_p95_ms",float("inf"));valid=[r for r in valid if r["output_token_throughput"]>=floor and r["ttft_ms"]["p95"]<=limit]
 if not valid:return None,"constraints_rejected_all_declared_default_fallback"
 if objective=="latency":key=lambda r:(r["ttft_ms"]["p95"],r["candidate_id"])
 elif objective=="throughput":key=lambda r:(-r["output_token_throughput"],r["candidate_id"])
 else:
  mins={k:min(r[k] if k=="output_token_throughput" else r[k]["p95"] for r in valid) for k in ("ttft_ms","tpot_ms","output_token_throughput")};maxmem=max(r["peak_gpu_memory_mib"] for r in valid) or 1;w=defs[objective]["weights"]
  for r in valid:r["objective_score"]=w["latency"]*r["ttft_ms"]["p95"]/mins["ttft_ms"]+w["tpot"]*r["tpot_ms"]["p95"]/mins["tpot_ms"]-w["throughput"]*r["output_token_throughput"]/mins["output_token_throughput"]+w["memory"]*r["peak_gpu_memory_mib"]/maxmem
  key=lambda r:(r["objective_score"],r["candidate_id"])
 return min(valid,key=key),"exact_profile_measured_objective_winner"
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--evidence",type=Path);ap.add_argument("--fixed",type=Path,required=True);ap.add_argument("--objectives",type=Path,required=True);ap.add_argument("--target",required=True);ap.add_argument("--model",required=True);ap.add_argument("--workload",required=True);ap.add_argument("--objective",choices=("latency","throughput","balanced"),required=True);ap.add_argument("--vllm-version",required=True);ap.add_argument("--measurement-candidate",choices=("default","1","2","4","8"));ap.add_argument("--out",type=Path,required=True);a=ap.parse_args();fixed=json.loads(a.fixed.read_text());defs=json.loads(a.objectives.read_text())
 if a.measurement_candidate is not None:
  value=None if a.measurement_candidate=="default" else int(a.measurement_candidate);reason="explicit_measurement_candidate"
 else:
  if a.evidence is None:ap.error("--evidence is required for policy selection")
  allrows=json.loads(a.evidence.read_text());rows=[r.copy() for r in allrows if r["target_identity"]==a.target and r["model_identity"]==a.model and r["workload_id"]==a.workload];winner,reason=select(rows,a.objective,defs);value=winner["max_num_seqs"] if winner else None
 plan={"schema_version":"vllm.max_num_seqs.policy.v1","execution_unit":"vllm_openai_server","backend":"vllm","model":a.model,"target_gpu_identity":a.target,"vllm_version":a.vllm_version,"candidate_id":IDS[value],"max_num_seqs":value,"value_source":"default" if value is None else "explicit","fixed_configuration":fixed,"workload_profile_identity":a.workload,"objective":a.objective,"constraints":defs[a.objective],"measurement_evidence_reference":str(a.evidence) if a.evidence else None,"selection_reason":reason,"fallback_identity":"vllm_max_num_seqs_default","truth_boundary":"target_model_workload_specific_measured_vllm_policy","runtime_no_policy_redecision":True,"candidate_matrix":candidates(fixed,a.target,a.workload,a.vllm_version)};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n");print(a.out)
if __name__=="__main__":main()
