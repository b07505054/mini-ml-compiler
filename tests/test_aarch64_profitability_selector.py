from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
def test_retained_split_and_exact_calibration_precedence():
 d=ROOT/"artifacts/backend_codegen/aarch64_schedule_profitability_slice20"
 split=json.loads((d/"development_holdout_split.json").read_text())
 assert len(split["development_domains"])>=6 and len(split["held_out_domains"])>=2
 trace=json.loads(next((d/"selection_traces").glob("*.json")).read_text())
 assert trace["precedence"][1]=="exact_compatible_measurement"
 assert trace["exact_calibrated_candidate"]==trace["actual_measured_winner"]
def test_dataset_identity_and_no_collisions():
 d=ROOT/"artifacts/backend_codegen/aarch64_schedule_profitability_slice20"
 rows=json.loads((d/"candidate_profitability_dataset.json").read_text())
 assert len(rows)>=24 and len({x["candidate_id"] for x in rows})==len(rows)
 assert all(x["correctness"]["runtime_redecision_count"]==0 for x in rows)
