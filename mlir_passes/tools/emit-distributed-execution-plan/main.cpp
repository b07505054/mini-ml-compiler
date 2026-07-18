// emit-distributed-execution-plan: D1 "Compiler-Planned TP=2 Multi-Process
// Simulation" — generate the fixed {TP1, TP2} distributed candidates for a
// synthetic sharded-matmul problem, apply D1 legality filtering, and export
// the selected candidate as a real ExecutionPlan schema_version 2.0.0
// artifact (mlir_passes/include/serving/ExecutionPlan.h).
//
// This is a standalone D1 entrypoint, not a hook into the 16-pass Qwen
// serving pipeline: it exercises the same DistributedPlanning + Export code
// a future integration would call, against a minimal, explicit, hand-built
// module identity and a synthetic matmul problem (A[M,K] x B[K,N], K
// partitioned across ranks) so the compiler-truth chain (candidate ->
// legality -> plan -> export) is real and testable without requiring the
// full ONNX/Qwen frontend. See docs/DISTRIBUTED_D1_COMPILER_PLANNED_TP2_MULTIPROCESS_REPORT.md
// in heterogeneous-inference-runtime for the runtime-consumption contract.

#include "serving/DistributedPlanning.h"
#include "serving/ExecutionPlanExporter.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

using namespace mlir::hir;

static llvm::cl::opt<std::string> Candidate(
    "candidate", llvm::cl::desc("Distributed candidate to select: tp1 | tp2"),
    llvm::cl::Required);

static llvm::cl::opt<int64_t> TensorDimK(
    "tensor-dim-k",
    llvm::cl::desc("K dimension of the synthetic sharded matmul (A[M,K] x B[K,N])"),
    llvm::cl::init(16));

static llvm::cl::opt<int64_t> TensorDimM(
    "tensor-dim-m", llvm::cl::desc("M dimension"), llvm::cl::init(4));

static llvm::cl::opt<int64_t> TensorDimN(
    "tensor-dim-n", llvm::cl::desc("N dimension"), llvm::cl::init(4));

static llvm::cl::opt<std::string> OutputPath(
    "output", llvm::cl::desc("Output path for the exported ExecutionPlan JSON"),
    llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "emit-distributed-execution-plan: D1 compiler-planned TP1/TP2 export\n");

  auto candidates = generateDistributedCandidates();
  const DistributedCandidate *selected = nullptr;
  for (const auto &c : candidates)
    if (c.candidate_id == Candidate)
      selected = &c;
  if (!selected) {
    llvm::errs() << "error: unknown --candidate '" << Candidate.getValue()
                 << "' (expected tp1 or tp2)\n";
    return 1;
  }

  auto legality = checkCandidateLegality(*selected, TensorDimK);
  if (!legality.legal) {
    llvm::errs() << "error: candidate '" << Candidate.getValue()
                 << "' failed legality for tensor_dim_k=" << TensorDimK.getValue()
                 << ":\n";
    for (const auto &reason : legality.rejection_reasons)
      llvm::errs() << "  - " << reason << "\n";
    return 1;
  }

  ExecutionPlan plan;
  plan.plan_id = "d1_distributed_" + Candidate.getValue() + "_synthetic_matmul";
  plan.provenance.compiler_tool = "emit-distributed-execution-plan";
  plan.provenance.model_spec_ref = "d1_synthetic_sharded_matmul";
  plan.provenance.capability_bundle.hardware_profile_ref =
      "d1_localhost_cpu_simulated";
  plan.provenance.capability_bundle.backend_profile_refs = {
      "d1_simulated_multiprocess"};
  plan.provenance.truth_boundary =
      "d1_simulated_localhost_multiprocess_ipc_not_real_gpu_not_nccl_not_"
      "measured_gpu_performance";

  plan.model_identity.model_id = "d1_synthetic_sharded_matmul";
  plan.model_identity.model_family = "synthetic_matmul";
  plan.model_identity.truth_boundary = "d1_synthetic_problem_not_a_real_model";

  FunctionPlan fp;
  fp.function_name = "d1_sharded_matmul";
  fp.serving_phase = ServingPhase::Prefill;
  fp.backend.selected_backend = "cpu_multiprocess_simulated";
  plan.function_plans.push_back(fp);

  // TP1 (world_size == 1) intentionally carries no `distributed` block —
  // it must round-trip as an ordinary legacy-shaped plan (backward
  // compatibility with pre-D1 ExecutionPlan artifacts).
  if (selected->world_size > 1) {
    auto distributed =
        buildDistributedPlan(*selected, TensorDimK, "partial_output");
    if (!distributed) {
      llvm::errs() << "error: legal candidate failed to build a plan "
                      "(internal inconsistency)\n";
      return 1;
    }
    auto structural = validateDistributedPlan(*distributed);
    if (!structural.legal) {
      llvm::errs() << "error: compiler-built plan failed structural "
                      "validation (must never happen):\n";
      for (const auto &reason : structural.rejection_reasons)
        llvm::errs() << "  - " << reason << "\n";
      return 1;
    }
    plan.distributed = distributed;
  }

  if (auto err = ExecutionPlanExporter::exportToFile(plan, OutputPath)) {
    llvm::errs() << "error: export failed: " << llvm::toString(std::move(err))
                 << "\n";
    return 1;
  }

  llvm::outs() << "wrote " << OutputPath.getValue() << " (candidate="
               << Candidate.getValue() << ", world_size=" << selected->world_size
               << ", tensor_dim_k=" << TensorDimK.getValue()
               << ", m=" << TensorDimM.getValue() << ", n=" << TensorDimN.getValue()
               << ")\n";
  return 0;
}
