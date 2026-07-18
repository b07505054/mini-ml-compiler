#!/usr/bin/env python3
"""Build/finalize the bounded Slice 20 cross-shape study."""
import argparse,csv,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import aarch64_schedule_profitability as p

# Held-out choices are fixed before measurement analysis.
INITIAL=[(16,16,32),(32,32,32),(64,64,64),(32,32,128),
         (32,32,64),(64,32,32),(32,64,32),(8,64,64)]
STRESS=[(8,8,32),(16,64,32)] # minimum legal K for uk4; low work/parallelism
DOMAINS=INITIAL+STRESS
HOLDOUT={p.sid((64,64,64)),p.sid((32,32,128))}
ORDERS=((1,2,4),(2,4,1),(4,1,2))

def prepare(out):
 out.mkdir(parents=True,exist_ok=True); artifacts=out/"candidate_artifacts"
 manifest={"schema_version":1,"domains":[]}
 protocol={"benchmark_protocol_version":p.PROTOCOL,"warmup_samples":30,
 "measured_samples_per_session":500,"independent_sessions":3,"cpu_affinity":3,
 "governor":"performance","timing_mode":"batched","session_orders":ORDERS,
 "calls_per_sample_by_domain":{},"equivalence_threshold_percent":3.0,
 "feature_thresholds":{"small_problem_flops_lte":65536,"high_K_gte":128,
 "low_K_lte":16,"M_limited_lte":8,"N_limited_lte":8}}
 for shape in DOMAINS:
  did=p.sid(shape); calls=max(1,min(200,2000000//(shape[0]*shape[1]*shape[2])))
  protocol["calls_per_sample_by_domain"][did]=calls
  row={"domain_id":did,"shape":shape,"tile":[8,8,8],"rationale":
   ("adaptive_uk4_stress_low_work_or_low_k" if shape in STRESS else "initial_domain"),
   "split":"held_out" if did in HOLDOUT else "development","candidates":[]}
  for uk in p.UKS:
   e=p.build(artifacts,shape,uk); row["candidates"].append(e["candidate_id"])
   plan_dir=out/"measurement_plans";plan_dir.mkdir(exist_ok=True)
   # Pi bundle places objects beside plans under objects/.
   plan=p.plan(e,"../objects",p.PROTOCOL)
   (plan_dir/f"{e['candidate_id']}.json").write_text(json.dumps(plan,indent=2)+"\n")
  manifest["domains"].append(row)
 (out/"domain_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 (out/"measurement_protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
 split={"development_domains":[x["domain_id"] for x in manifest["domains"] if x["split"]=="development"],
 "held_out_domains":[x["domain_id"] for x in manifest["domains"] if x["split"]=="held_out"],
 "fixed_before_finalizing_policy":True}
 (out/"development_holdout_split.json").write_text(json.dumps(split,indent=2)+"\n")

def finalize(out,raw):
 manifest=json.loads((out/"domain_manifest.json").read_text()); rows=[]; groups={};session_summaries=[]
 for d in manifest["domains"]:
  shape=tuple(d["shape"]); did=d["domain_id"];groups[did]=[]
  for uk in p.UKS:
   e=json.loads((out/"candidate_artifacts"/p.cid(shape,uk)/"backend_evidence.json").read_text())
   sessions=[json.loads((raw/f"{did}_session{s}_{p.cid(shape,uk)}.json").read_text()) for s in (1,2,3)]
   session_summaries.extend({"domain_id":did,"candidate_id":p.cid(shape,uk),
    "session_id":x["session_id"],"candidate_order_position":x["candidate_order_position"],
    "object_sha256":x["identity_proof"]["executed_object_sha256"],
    "plan_sha256":x["plan_sha256"],"environment_before":x["environment_before"],
    "target":x["target"],"metrics":x["metrics"],"identity_proof":x["identity_proof"]}
    for x in sessions)
   assert all(x["metrics"]["correct"] and x["metrics"]["repeated_call_correct"] and x["metrics"]["guard_buffers_intact"] for x in sessions)
   assert all(x["identity_proof"]["runtime_redecision_count"]==0 for x in sessions)
   row={"domain_id":did,"candidate_id":e["candidate_id"],"shape":list(shape),"tile":[8,8,8],
    "schedule_unroll_k":uk,"target":"cortex-a76","static_features":p.features(shape,uk,e),
    "backend_evidence":e["static_backend_evidence"],"measurement":p.aggregate(sessions),
    "correctness":{"passed":True,"max_abs_error":max(x["metrics"]["max_abs_error"] for x in sessions),
    "guard_buffers_intact":True,"runtime_redecision_count":0},
    "provenance":{"object_sha256":e["artifacts"]["object_sha256"],"sessions":[str(x) for x in sorted(raw.glob(f"{did}_session*_{e['candidate_id']}.json"))]}}
   rows.append(row);groups[did].append(row)
  p.classify(groups[did])
 (out/"candidate_profitability_dataset.json").write_text(json.dumps(rows,indent=2)+"\n")
 (out/"session_summaries.json").write_text(json.dumps(session_summaries,indent=2)+"\n")
 with (out/"candidate_profitability_dataset.csv").open("w",newline="") as f:
  w=csv.writer(f);w.writerow(["domain","candidate","uk","p50_ms","p95_ms","regret_pct","winner","text_bytes","reloads","spill_bytes"])
  for r in rows:w.writerow([r["domain_id"],r["candidate_id"],r["schedule_unroll_k"],r["measurement"]["median_session_p50_ms"],
   r["measurement"]["median_session_p95_ms"],r["classification"]["relative_regret_percent"],r["classification"]["domain_winner"],
   r["backend_evidence"]["text_size_bytes"],r["backend_evidence"]["reload_load_count"],r["backend_evidence"]["spill_slot_bytes"]])
 chooseuk=lambda u:lambda rs:next(r for r in rs if r["schedule_unroll_k"]==u)
 policies={"legacy_static":p.legacy,"always_uk1":chooseuk(1),"always_uk2":chooseuk(2),"always_uk4":chooseuk(4),"revised_static":p.revised}
 split=json.loads((out/"development_holdout_split.json").read_text())
 for name,fn in policies.items():
  result={"overall":p.evaluate(groups,fn),
   "development":p.evaluate({k:v for k,v in groups.items() if k in split["development_domains"]},fn),
   "held_out":p.evaluate({k:v for k,v in groups.items() if k in split["held_out_domains"]},fn)}
  (out/f"{name}_evaluation.json").write_text(json.dumps(result,indent=2)+"\n")
 traces=out/"selection_traces";traces.mkdir(exist_ok=True)
 compact=[]
 for did,rs in groups.items():
  win=min(rs,key=lambda r:r["measurement"]["median_session_p50_ms"])
  trace={"domain_id":did,"actual_measured_winner":win["candidate_id"],
   "legacy_static_candidate":p.legacy(rs)["candidate_id"],"revised_static_candidate":p.revised(rs)["candidate_id"],
   "exact_calibrated_candidate":win["candidate_id"],
   "precedence":["hard_gates","exact_compatible_measurement","revised_static_policy","legacy_static_fallback"],
   "candidates":[{"candidate_id":r["candidate_id"],"features":r["static_features"],
    "backend_evidence":r["backend_evidence"],"classification":r["classification"]} for r in rs]}
  (traces/f"{did}.json").write_text(json.dumps(trace,indent=2)+"\n");compact.append(trace)
 (out/"per_domain_results.json").write_text(json.dumps(compact,indent=2)+"\n")
 policy={"policy_id":"slice20_short_static_k_full_unroll_v1","kind":"deterministic_bounded_rule",
  "rule":"prefer highest legal schedule_unroll_k (4) for this measured fixed-tile domain",
  "exact_calibration_has_priority":True,"truth_boundary":"Cortex-A76 measured domains only; no cross-shape universality claim"}
 (out/"revised_static_policy.json").write_text(json.dumps(policy,indent=2)+"\n")
 uk4_losses=[x for x in compact if not x["actual_measured_winner"].endswith("_uk4")]
 counter={"initial_domains":[p.sid(x) for x in INITIAL],"adaptive_stress_domains":[p.sid(x) for x in STRESS],
  "hypothesis":"low work, low K, and limited M expose uk4 code/spill overhead",
  "uk4_counterexamples":[x["domain_id"] for x in uk4_losses],
  "result":"a counterexample to uk4 was found" if uk4_losses else "uk4 won every tested domain, but no universality claim is made"}
 (out/"counterexample_search.json").write_text(json.dumps(counter,indent=2)+"\n")
 report=["# Slice 20 AArch64 schedule profitability", "",counter["result"],"",
  "| Domain | uk1 p50 | uk2 p50 | uk4 p50 | Winner | Legacy | Revised | Revised regret |",
  "|---|---:|---:|---:|---|---|---|---:|"]
 for x in compact:
  rs=groups[x["domain_id"]]; q={r["schedule_unroll_k"]:r for r in rs}; rev=p.revised(rs)
  report.append(f"| {x['domain_id']} | {q[1]['measurement']['median_session_p50_ms']:.9f} | {q[2]['measurement']['median_session_p50_ms']:.9f} | {q[4]['measurement']['median_session_p50_ms']:.9f} | {x['actual_measured_winner']} | {x['legacy_static_candidate']} | {x['revised_static_candidate']} | {rev['classification']['relative_regret_percent']:.3f}% |")
 (out/"SLICE20_REPORT.md").write_text("\n".join(report)+"\n")

def main():
 a=argparse.ArgumentParser();a.add_argument("command",choices=("prepare","finalize"));a.add_argument("--output-dir",required=True);a.add_argument("--raw-dir")
 z=a.parse_args();out=Path(z.output_dir);prepare(out) if z.command=="prepare" else finalize(out,Path(z.raw_dir))
if __name__=="__main__":main()
