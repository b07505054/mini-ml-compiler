#!/usr/bin/env python3
"""Slice 20 cross-shape evidence, feature, and selector utilities.

This is deliberately limited to static f32, tile 8x8x8, Cortex-A76 and
uk={1,2,4}.  It extends the Slice 18 artifact chain and Slice 19 measurement
contract; it is not a general autotuner.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, random, re, statistics, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMPILE=ROOT/"mlir_passes/tools/compile_hir_matmul_bias_relu_aarch64.sh"
EXTRACT=ROOT/"tools/extract_aarch64_candidate_mir.py"
ANALYZE=ROOT/"tools/analyze_aarch64_candidate_mir.py"
TRIPLE,CPU,PROFILE="aarch64-linux-gnu","cortex-a76","raspberry-pi5-cortex-a76-cpu"
PIPELINE,LOOP,ABI="aarch64_tiled_scheduled_v1","tiled_mnk_row_major_v1","mlir_ciface_memref_f32_v1"
FAMILY="hir.fused_matmul_bias_relu"; UKS=(1,2,4)
PROTOCOL="slice20_aarch64_profitability_v1"

class Slice20Error(ValueError): pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def run(a):
 p=subprocess.run(a,cwd=ROOT,text=True,capture_output=True)
 if p.returncode: raise Slice20Error(f"{' '.join(map(str,a))}\n{p.stdout}\n{p.stderr}")
 return p.stdout
def sid(shape): return f"m{shape[0]}_n{shape[1]}_k{shape[2]}_tile8x8x8"
def cid(shape,uk): return f"{sid(shape)}_uk{uk}"
def entry(shape): return f"_mlir_ciface_matmul_bias_relu_tiled_{shape[0]}x{shape[1]}x{shape[2]}"
def legal(shape,uk):
 m,n,k=shape
 return all(x%8==0 for x in shape) and (k//8)%uk==0
def fixture(shape,path):
 m,n,k=shape; fn=f"matmul_bias_relu_tiled_{m}x{n}x{k}"
 path.write_text(f"""func.func @{fn}(%lhs: tensor<{m}x{k}xf32>, %rhs: tensor<{k}x{n}xf32>, %bias: tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32> attributes {{ llvm.emit_c_interface }} {{
  %0 = hir.fused_matmul_bias_relu %lhs, %rhs, %bias {{fusion.candidate = "matmul_bias_relu", kernel.selection = "native_cpu", lowering.source = "linalg.matmul_add_relu"}} : (tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>, tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  return %0 : tensor<{m}x{n}xf32>
}}\n""")
def paths(out,shape,uk):
 d=out/cid(shape,uk); stem=cid(shape,uk); prefix=f"{shape[0]}x{shape[1]}x{shape[2]}_tm8_tn8_tk8_uk{uk}_greedy_misched-default"
 return d,{ "llvm_dialect":d/f"{stem}_llvm.mlir","llvm_ir":d/f"{stem}.ll",
  "mir_post_isel":d/f"{prefix}_post_isel.mir","mir_pre_scheduler":d/f"{prefix}_pre_scheduler.mir",
  "mir_pre_ra":d/f"{prefix}_pre_ra.mir","mir_post_ra":d/f"{prefix}_post_ra.mir",
  "mir_post_prologue_epilogue":d/f"{prefix}_post_prologue_epilogue.mir",
  "assembly":d/f"{prefix}.s","object":d/f"{prefix}.o","register_metrics":d/"register_metrics.json"}
def asm_metrics(path):
 text=path.read_text()
 inst=sum(1 for x in text.splitlines() if x.strip() and not x.lstrip().startswith((".","#","//")) and not x.rstrip().endswith(":"))
 count=lambda pat:len(re.findall(pat,text,re.M))
 return {"static_instruction_count":inst,"fmla_count":count(r"^\s*fmla\s"),
  "branch_count":count(r"^\s+(?:b|b\.\w+|cbz|cbnz)\s"),
  "load_count":count(r"^\s+(?:ldr|ldp|ld1)\s"),"store_count":count(r"^\s+(?:str|stp|st1)\s"),
  "basic_block_count":count(r"^\.LBB\d+_\d+:")}
def build(out,shape,uk):
 if not legal(shape,uk): raise Slice20Error(f"illegal {shape} uk{uk}")
 d,p=paths(out,shape,uk);d.mkdir(parents=True,exist_ok=True); f=d/f"{sid(shape)}.mlir";fixture(shape,f)
 stem=cid(shape,uk)
 run(["bash",str(COMPILE),"--variant","tiled-scheduled","--tile-m","8","--tile-n","8","--tile-k","8",
      "--schedule-unroll-k",str(uk),str(f),str(d),stem])
 run(["python3",str(EXTRACT),"--llvm-ir",str(p["llvm_ir"]),"--cpu",CPU,
      "--shape","x".join(map(str,shape)),"--tile-m","8","--tile-n","8","--tile-k","8",
      "--schedule-unroll-k",str(uk),"--output-dir",str(d)])
 run(["python3",str(ANALYZE),"--post-isel",str(p["mir_post_isel"]),"--pre-ra",str(p["mir_pre_ra"]),
      "--post-ra",str(p["mir_post_ra"]),"--post-prologue-epilogue",str(p["mir_post_prologue_epilogue"]),
      "--output",str(p["register_metrics"])])
 r=json.loads(p["register_metrics"].read_text())["stages"]; static=asm_metrics(p["assembly"])
 size=subprocess.run(["llvm-size","-A",str(p["object"])],text=True,capture_output=True)
 z=re.search(r"^\.text\s+(\d+)",size.stdout,re.M); static.update({
  "object_size_bytes":p["object"].stat().st_size,"text_size_bytes":int(z.group(1)) if z else None,
  "spill_store_count":r["post_ra"]["spill_stores"],"reload_load_count":r["post_ra"]["spill_reloads"],
  "spill_slot_bytes":r["post_ra"]["spill_slot_bytes"],
  "physical_vector_registers_referenced":r["post_ra"]["physical_vector_registers_referenced"],
  "approximate_peak_live_vector_registers":r["pre_ra"]["approx_peak_live_vector_registers"],
  "stack_frame_size_bytes":None})
 rel=lambda x:os.path.relpath(x,ROOT)
 e={"schema_version":1,"candidate_id":cid(shape,uk),"operator":FAMILY,
  "kernel_family":"aarch64_generated_fused_matmul_bias_relu","dtype":"f32",
  "shape":{"m":shape[0],"n":shape[1],"k":shape[2]},
  "target":{"triple":TRIPLE,"cpu":CPU,"features":[],"target_profile_id":PROFILE},
  "lowering":{"pipeline_id":PIPELINE,"tile_m":8,"tile_n":8,"tile_k":8,
   "schedule_unroll_k":uk,"vector_width_bits":128,"loop_order_id":LOOP},
  "microkernel_id":"hir_fused_matmul_bias_relu_tiled_scheduled_v1","entry_point":entry(shape),"abi_version":ABI,
  "artifacts":{**{k+"_ref":rel(v) for k,v in p.items() if k!="register_metrics"},
               **{k+"_sha256":sha(v) for k,v in p.items() if v.is_file()}},
  "static_backend_evidence":static,"validation":{"codegen_succeeded":True,"llvm_ir_verified":True,
  "correctness_passed":None,"measured_on_target":False},
  "provenance":{"compiler_revision":run(["git","rev-parse","HEAD"]).strip(),"working_tree_clean":False}}
 e["artifacts"]["backend_evidence_ref"]=rel(d/"backend_evidence.json")
 (d/"backend_evidence.json").write_text(json.dumps(e,indent=2)+"\n");return e
def features(shape,uk,e):
 m,n,k=shape; flops=2*m*n*k; f=e["static_backend_evidence"]; reads=(m*k+k*n+m*n)*4; writes=m*n*4
 return {"M":m,"N":n,"K":k,"M_N_K":m*n*k,"FLOPs":flops,"m_tile_count":m//8,
 "n_tile_count":n//8,"k_tile_count":k//8,"k_schedule_trip_count":k//8,
 "unroll_factor":uk,"effective_unrolled_k_groups":(k//8)//uk,"tail_presence":False,
 "tile_volume":512,"output_elements":m*n,"estimated_bytes_read":reads,"estimated_bytes_written":writes,
 "arithmetic_intensity_flops_per_byte":flops/(reads+writes),
 "fmla_per_static_instruction":f["fmla_count"]/f["static_instruction_count"],
 "fmla_per_object_byte":f["fmla_count"]/f["object_size_bytes"],
 "reloads_per_fmla":f["reload_load_count"]/f["fmla_count"] if f["fmla_count"] else None,
 "spill_bytes_per_flop":f["spill_slot_bytes"]/flops,"approximate_live_vectors":f["approximate_peak_live_vector_registers"],
 "small_problem":flops<=65536,"high_K":k>=128,"low_K":k<=16,
 "rectangular":len({m,n,k})>1,"M_limited":m<=8,"N_limited":n<=8}
def plan(e,object_root,protocol=PROTOCOL):
 shape=e["shape"]; native={k:e[k] for k in ("candidate_id","operator","kernel_family","dtype","shape","target","lowering","microkernel_id","entry_point","abi_version")}
 native.update({"object_ref":str(Path(object_root)/Path(e["artifacts"]["object_ref"]).name),
  "object_sha256":e["artifacts"]["object_sha256"],"backend_evidence_ref":e["artifacts"]["backend_evidence_ref"],
  "selection_mode":"measurement_candidate","selection_trace_ref":"measurement_protocol.json",
  "runtime_no_redecision":True,"benchmark_protocol_version":protocol})
 return {"schema":"execution_plan","schema_version":"2.0.0","plan_id":"slice20-"+e["candidate_id"],
 "provenance":{"truth_boundary":"exact generated static-shape object"},"model_identity":{"model_id":sid(tuple(shape.values()))},
 "global_decisions":{"quantization":{},"memory":{},"serving":{}},"function_plans":[{
 "function_name":e["entry_point"].removeprefix("_mlir_ciface_"),"serving_phase":"other",
 "backend":{"selected_backend":"aarch64_native_object"},"per_op_decisions":[{"op_name":"fused_matmul_bias_relu",
 "op_type":FAMILY,"native_execution":native}]}]}
def aggregate(sessions,seed=20):
 vals=lambda k:[x["metrics"][k] for x in sessions]; means=vals("mean_ms");rng=random.Random(seed);boot=[]
 for _ in range(10000):boot.append(statistics.mean(rng.choices(means,k=len(means))))
 boot.sort()
 return {"median_session_p50_ms":statistics.median(vals("p50_ms")),"median_session_p95_ms":statistics.median(vals("p95_ms")),
 "mean_session_mean_ms":statistics.mean(means),"session_mean_stddev_ms":statistics.stdev(means),
 "session_mean_standard_error_ms":statistics.stdev(means)/math.sqrt(len(means)),
 "bootstrap_ci95_low_ms":boot[249],"bootstrap_ci95_high_ms":boot[9749],"session_count":len(sessions)}
def classify(rows,threshold=3.0):
 best=min(rows,key=lambda r:r["measurement"]["median_session_p50_ms"]);bp=best["measurement"]["median_session_p50_ms"]
 for r in rows:
  rel=(r["measurement"]["median_session_p50_ms"]/bp-1)*100
  overlap=not(r["measurement"]["bootstrap_ci95_low_ms"]>best["measurement"]["bootstrap_ci95_high_ms"] or best["measurement"]["bootstrap_ci95_low_ms"]>r["measurement"]["bootstrap_ci95_high_ms"])
  r["classification"]={"domain_winner":r is best,"within_equivalence_band":rel<=threshold or overlap,
   "relative_regret_percent":rel,"status":"clear winner" if r is best else ("statistical tie" if rel<=threshold or overlap else "clear loser")}
 return best
def evaluate(domains,choice):
 regrets=[];agree=within1=within3=0
 for did,rows in domains.items():
  win=min(rows,key=lambda r:r["measurement"]["median_session_p50_ms"]); picked=choice(rows)
  reg=(picked["measurement"]["median_session_p50_ms"]/win["measurement"]["median_session_p50_ms"]-1)*100
  regrets.append(reg);agree+=picked["candidate_id"]==win["candidate_id"];within1+=reg<=1;within3+=reg<=3
 s=sorted(regrets)
 return {"domain_count":len(regrets),"winner_agreement":agree,"winner_agreement_ratio":agree/len(regrets),
 "mean_regret_percent":statistics.mean(regrets),"median_regret_percent":statistics.median(regrets),
 "p95_regret_percent":s[math.ceil(.95*len(s))-1],"maximum_regret_percent":max(regrets),
 "domains_within_1_percent":within1,"domains_within_3_percent":within3}
def legacy(rows):
 return min(rows,key=lambda r:(r["backend_evidence"].get("reload_load_count") is None,r["backend_evidence"].get("reload_load_count") or 0,
  r["backend_evidence"].get("text_size_bytes") is None,r["backend_evidence"].get("text_size_bytes") or 0,
  r["backend_evidence"].get("object_size_bytes") or 0,r["candidate_id"]))
def revised(rows):
 # Development-derived, intentionally simple: fully unroll the short static
 # K schedule. Exact calibration still has priority outside this function.
 return max(rows,key=lambda r:(r["static_features"]["unroll_factor"],r["candidate_id"]))
