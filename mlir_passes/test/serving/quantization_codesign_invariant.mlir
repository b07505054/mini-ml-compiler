// Ranking-invariance input for QuantizationCoDesignPass.
//
// Run by tools/run_mlir_pass_tests.sh (run_quant_codesign_ranking_invariant):
// this module is compiled twice —
//   A: candidate-evaluation-pipeline,plan-selection-pipeline
//   B: quant-codesign-pipeline,candidate-evaluation-pipeline,plan-selection-pipeline
// and every `evaluation.*` and `selected_plan.*` signal extracted from both
// outputs must be byte-identical: quant_codesign.est.* evidence must never
// leak into CandidateEvaluation or PlanSelection inputs or outputs.
//
// The module deliberately enables the co-design policy, backend legality,
// and profile numbers so the pass does real work in run B.

module attributes {
  quant.codesign.policy = "systems_cost_only",
  target.backend_capabilities.cpu.supported_quant_modes = ["weight_only"],
  target.static_cost_profile.peak_flops_fp16 = 1.0e12 : f64,
  target.static_cost_profile.memory_bandwidth_bytes_per_sec = 1.0e11 : f64
} {
  func.func @ranking_invariant(%a: tensor<4x8xf16>) -> tensor<4x16xf16>
      attributes {representation.source_backend = "cpu"} {
    %0 = "compute.matmul"(%a) {
      weight.constant_satisfied = true,
      compiler.candidates = [{
        candidate_type = "direct_lower",
        required_boundary_ops = [],
        source_op = "matmul"
      }, {
        candidate_type = "backend_fallback",
        fallback_backend = "cpu",
        required_boundary_ops = [],
        source_op = "matmul"
      }]
    } : (tensor<4x8xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}
