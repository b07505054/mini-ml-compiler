import json
import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "artifacts/yoloseg_generic_frontend/yoloseg.execution_plan.json"
SCRIPT_PATH = REPO_ROOT / "scripts/run_yoloseg_execution_plan.sh"


class TestYoloSegExecutionPlan(unittest.TestCase):
    def test_script_requires_explicit_target_profile(self):
        env = os.environ.copy()
        env["YOLOSEG_TARGET_PROFILE"] = str(REPO_ROOT / "missing-target-profile.json")
        result = subprocess.run(
            [str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("target profile not found", result.stderr)

    def test_real_yoloseg_execution_plan_artifact_if_present(self):
        if not PLAN_PATH.exists():
            self.skipTest("real YOLO-Seg execution plan artifact is not present")

        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        self.assertEqual(plan["schema"], "execution_plan")
        self.assertEqual(plan["model_identity"]["model_family"], "yoloseg")
        self.assertIn("cv_extension", plan)

        cv = plan["cv_extension"]
        self.assertEqual(
            cv["truth_boundary"],
            "real_yoloseg_execution_plan_compiler_decisions_from_static_capability_and_analysis_no_runtime_execution_no_measured_performance_no_full_memory_slot_allocation",
        )

        outputs = {output["role"]: output for output in cv["outputs"]}
        self.assertEqual(outputs["detection"]["shape"], [1, 116, 8400])
        self.assertEqual(outputs["segmentation_prototype"]["shape"], [1, 32, 160, 160])
        self.assertEqual(outputs["detection"]["dtype"], "f32")
        self.assertEqual(outputs["segmentation_prototype"]["layout"], "nchw")

        regions = {region["region_id"]: region for region in cv["semantic_regions"]}
        self.assertIn("cv.region.detection_head", regions)
        self.assertIn("cv.region.segmentation_prototype", regions)
        self.assertIn("cv.region.mask_coefficient_branch", regions)
        self.assertIn("cv.region.feature_pyramid", regions)

        self.assertGreater(cv["memory_estimates"]["estimated_total_tensor_bytes"], 0)
        self.assertEqual(
            cv["memory_estimates"]["scope"],
            "static_tensor_byte_estimates_no_slot_allocation",
        )

        functions = plan["function_plans"]
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["function_name"], "main_graph")
        self.assertTrue(functions[0]["per_op_decisions"])
        self.assertIn("backend", functions[0])


if __name__ == "__main__":
    unittest.main()
