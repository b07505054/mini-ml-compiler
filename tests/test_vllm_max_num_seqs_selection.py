import importlib.util
from pathlib import Path
P=Path(__file__).parents[1]/"tools/select_vllm_max_num_seqs.py";s=importlib.util.spec_from_file_location("v",P);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
def test_candidates_are_distinct_and_default_is_omitted():
 c=v.candidates({"model":"m","dtype":"float16","max_model_len":1,"max_num_batched_tokens":1,"enable_chunked_prefill":False,"enable_prefix_caching":False},"g","w","1");assert len({x["candidate_id"] for x in c})==5 and c[0]["max_num_seqs"] is None and [x["max_num_seqs"] for x in c[1:]]==[1,2,4,8]
def test_measured_exact_rows_select_objective_winner_and_fallback():
 rows=[{"candidate_id":"a","classification":"VALID","failure_count":0,"oom_count":0,"ttft_ms":{"p95":2},"tpot_ms":{"p95":2},"output_token_throughput":4,"peak_gpu_memory_mib":2},{"candidate_id":"b","classification":"VALID","failure_count":0,"oom_count":0,"ttft_ms":{"p95":1},"tpot_ms":{"p95":3},"output_token_throughput":5,"peak_gpu_memory_mib":3}];defs={"latency":{},"throughput":{},"balanced":{"weights":{"latency":.35,"tpot":.25,"throughput":.3,"memory":.1}}};assert v.select(rows,"latency",defs)[0]["candidate_id"]=="b";assert v.select([],"latency",defs)[0] is None


def test_failed_and_constraint_violating_rows_are_not_selected():
 rows=[{"candidate_id":"fast_failed","classification":"REQUEST_FAILURE","failure_count":1,"oom_count":0,"ttft_ms":{"p95":1},"tpot_ms":{"p95":1},"output_token_throughput":100,"peak_gpu_memory_mib":1},{"candidate_id":"valid","classification":"VALID","failure_count":0,"oom_count":0,"ttft_ms":{"p95":5},"tpot_ms":{"p95":2},"output_token_throughput":4,"peak_gpu_memory_mib":2}]
 defs={"latency":{"minimum_output_tokens_per_second":2,"maximum_ttft_p95_ms":10}}
 assert v.select(rows,"latency",defs)[0]["candidate_id"]=="valid"


def test_objective_weights_are_explicit_and_balanced_selection_is_deterministic():
 rows=[{"candidate_id":"latency","classification":"VALID","failure_count":0,"oom_count":0,"ttft_ms":{"p95":1},"tpot_ms":{"p95":2},"output_token_throughput":2,"peak_gpu_memory_mib":2},{"candidate_id":"throughput","classification":"VALID","failure_count":0,"oom_count":0,"ttft_ms":{"p95":2},"tpot_ms":{"p95":2},"output_token_throughput":8,"peak_gpu_memory_mib":2}]
 defs={"balanced":{"weights":{"latency":.1,"tpot":.0,"throughput":.9,"memory":.0}}}
 assert set(defs["balanced"]["weights"])=={"latency","tpot","throughput","memory"}
 assert v.select(rows,"balanced",defs)[0]["candidate_id"]=="throughput"
