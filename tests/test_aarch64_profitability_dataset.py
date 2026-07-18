from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import aarch64_schedule_profitability as p

def fake(shape,uk):
 return {"static_backend_evidence":{"fmla_count":128*uk,"static_instruction_count":300,
 "object_size_bytes":2000,"reload_load_count":uk,"spill_slot_bytes":16,
 "approximate_peak_live_vector_registers":20}}
def test_identity_and_shape_specific_names():
 assert p.cid((16,16,32),4)=="m16_n16_k32_tile8x8x8_uk4"
 assert p.cid((16,16,32),4)!=p.cid((32,16,32),4)
 assert p.entry((16,16,32)).endswith("16x16x32")
def test_legality_and_features():
 assert p.legal((16,16,32),4)
 assert not p.legal((16,16,16),4)
 f=p.features((16,16,32),4,fake((16,16,32),4))
 assert f["FLOPs"]==16384 and f["k_schedule_trip_count"]==4
 assert f["effective_unrolled_k_groups"]==1
 assert f["reloads_per_fmla"]==4/512
 assert f["spill_bytes_per_flop"]==16/16384
def test_classification_and_policy_metrics():
 rows=[]
 for uk,v in ((1,1.2),(2,1.1),(4,1.0)):
  rows.append({"candidate_id":f"x_uk{uk}","schedule_unroll_k":uk,
      "measurement":{"median_session_p50_ms":v,"bootstrap_ci95_low_ms":v-.001,"bootstrap_ci95_high_ms":v+.001},
   "backend_evidence":{"reload_load_count":uk,"text_size_bytes":uk*10,"object_size_bytes":uk*20},
   "static_features":{"unroll_factor":uk}})
 assert p.classify(rows)["candidate_id"]=="x_uk4"
 assert p.legacy(rows)["candidate_id"]=="x_uk1"
 assert p.revised(rows)["candidate_id"]=="x_uk4"
 metrics=p.evaluate({"d":rows},p.revised)
 assert metrics["winner_agreement"]==1 and metrics["maximum_regret_percent"]==0
