#include "FusionPasses.h"
#include "HIR/IR/HIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_ATTENTIONCANDIDATEGENERATION
#define GEN_PASS_DEF_ATTENTIONLEGALITY
#define GEN_PASS_DEF_ATTENTIONSELECTIONLOWERING
#include "FusionPasses.h.inc"

static DictionaryAttr candidate(MLIRContext *ctx, StringRef phase,
                                int64_t workspaceBytes) {
  const bool prefill = phase == "prefill";
  NamedAttrList a;
  a.append("candidate_id", StringAttr::get(ctx, prefill
      ? "cpu_attention_prefill_fp32" : "cpu_attention_decode_fp32"));
  a.append("phase", StringAttr::get(ctx, phase));
  a.append("backend", StringAttr::get(ctx, "portable_cpu"));
  a.append("precision", StringAttr::get(ctx, "fp32"));
  a.append("kernel_id", StringAttr::get(ctx, prefill
      ? "cpu_attention_prefill_fp32" : "cpu_attention_decode_fp32"));
  a.append("entry_point", StringAttr::get(ctx, prefill
      ? "hir_cpu_attention_prefill_fp32" : "hir_cpu_attention_decode_fp32"));
  a.append("layout", StringAttr::get(ctx, "bhsd_contiguous"));
  a.append("query_length_constraint", StringAttr::get(ctx,
      prefill ? "query_length_gt_1_and_equals_context" : "query_length_eq_1"));
  a.append("head_constraint", StringAttr::get(ctx, "num_query_heads_eq_num_kv_heads"));
  a.append("head_dimension_constraint", StringAttr::get(ctx, "static_positive"));
  a.append("causal_required", BoolAttr::get(ctx, true));
  a.append("workspace_bytes",
           IntegerAttr::get(IntegerType::get(ctx, 64), workspaceBytes));
  a.append("fallback_identity", StringAttr::get(ctx, "unsupported_attention_explicit_failure"));
  a.append("evidence_classification", StringAttr::get(ctx, "not_measured"));
  a.append("provenance", StringAttr::get(ctx, "attention_candidate_generation_v1"));
  a.append("legality_status", StringAttr::get(ctx, "pending"));
  return a.getDictionary(ctx);
}

struct AttentionCandidateGenerationPass
    : impl::AttentionCandidateGenerationBase<AttentionCandidateGenerationPass> {
  void getDependentDialects(DialectRegistry &r) const override { r.insert<HIRDialect>(); }
  void runOnOperation() override {
    getOperation().walk([&](AttentionOp op) {
      int64_t workspaceBytes =
          op->getAttrOfType<IntegerAttr>("workspace_bytes").getInt();
      op->setAttr("attention.candidates", ArrayAttr::get(op.getContext(), {
          candidate(op.getContext(), "prefill", workspaceBytes),
          candidate(op.getContext(), "decode", workspaceBytes)}));
      op->setAttr("attention.candidate_generation_truth_boundary",
                  StringAttr::get(op.getContext(), "cartesian_phase_candidates_not_ranked_or_measured"));
    });
  }
};

struct AttentionLegalityPass
    : impl::AttentionLegalityBase<AttentionLegalityPass> {
  void getDependentDialects(DialectRegistry &r) const override { r.insert<HIRDialect>(); }
  void runOnOperation() override {
    getOperation().walk([&](AttentionOp op) {
      auto all = op->getAttrOfType<ArrayAttr>("attention.candidates");
      if (!all) { op.emitError("missing generated attention candidates"); signalPassFailure(); return; }
      SmallVector<Attribute> checked;
      StringRef actual = op->getAttrOfType<StringAttr>("phase").getValue();
      for (Attribute x : all) {
        auto d = cast<DictionaryAttr>(x); NamedAttrList a(d);
        StringRef proposed = cast<StringAttr>(d.get("phase")).getValue();
        bool legal = proposed == actual;
        a.set("legality_status", StringAttr::get(op.getContext(), legal ? "legal" : "rejected"));
        a.set("legality_reason", StringAttr::get(op.getContext(), legal
            ? "phase_and_semantic_contract_match" : "phase_mismatch"));
        checked.push_back(a.getDictionary(op.getContext()));
      }
      op->setAttr("attention.candidates", ArrayAttr::get(op.getContext(), checked));
    });
  }
};

static StringAttr moduleString(Operation *op, StringRef name) {
  Operation *module = op->getParentOp();
  while (module && !isa<ModuleOp>(module)) module = module->getParentOp();
  return module ? module->getAttrOfType<StringAttr>(name) : StringAttr{};
}

struct AttentionSelectionLoweringPass
    : impl::AttentionSelectionLoweringBase<AttentionSelectionLoweringPass> {
  void getDependentDialects(DialectRegistry &r) const override { r.insert<HIRDialect>(); }
  void runOnOperation() override {
    SmallVector<AttentionOp> ops;
    getOperation().walk([&](AttentionOp op) { ops.push_back(op); });
    for (AttentionOp op : ops) {
      auto ref = moduleString(op, "attention.cpu.artifact_ref");
      auto sha = moduleString(op, "attention.cpu.artifact_sha256");
      auto version = moduleString(op, "attention.cpu.artifact_version");
      if (!ref || !sha || !version) {
        op.emitError("selected CPU attention requires module artifact ref, sha256, and version");
        signalPassFailure(); return;
      }
      DictionaryAttr selected;
      for (Attribute x : op->getAttrOfType<ArrayAttr>("attention.candidates")) {
        auto d = cast<DictionaryAttr>(x);
        if (cast<StringAttr>(d.get("legality_status")).getValue() == "legal") selected = d;
      }
      if (!selected) { op.emitError("no legal phase-specific attention implementation"); signalPassFailure(); return; }
      OpBuilder b(op); NamedAttrList attrs(op->getAttrs());
      attrs.erase("attention.candidates");
      attrs.set("candidate_id", selected.get("candidate_id"));
      attrs.set("kernel_id", selected.get("kernel_id"));
      attrs.set("entry_point", selected.get("entry_point"));
      attrs.set("artifact_ref", ref); attrs.set("artifact_sha256", sha);
      attrs.set("artifact_version", version);
      attrs.set("fallback_identity", selected.get("fallback_identity"));
      attrs.set("runtime_execution_unit", b.getStringAttr("portable_cpu_attention"));
      attrs.set("required_isa", b.getStringAttr("scalar_fp32"));
      attrs.set("runtime_no_redecision", b.getBoolAttr(true));
      attrs.set("selection_policy", b.getStringAttr("deterministic_phase_policy"));
      attrs.set("truth_boundary", b.getStringAttr(
          "real_operator_level_fp32_cpu_attention_not_full_model_or_kv_lifetime"));
      auto lowered = b.create<CPUAttentionOp>(op.getLoc(),
          TypeRange{op.getOutput().getType()},
          ValueRange{op.getQuery(), op.getKey(), op.getValue()}, attrs.getAttrs());
      op.getOutput().replaceAllUsesWith(lowered.getOutput());
      auto fn = op->getParentOfType<func::FuncOp>();
      fn->setAttr("execution_provider.primary", b.getStringAttr("portable_cpu"));
      fn->setAttr("execution_provider.decision_source", b.getStringAttr("deterministic_phase_policy"));
      fn->setAttr("execution_provider.precision", b.getStringAttr("fp32"));
      fn->setAttr("execution_provider.kv_layout", b.getStringAttr("externally_supplied_contiguous"));
      fn->setAttr("execution_provider.requires_replay", b.getBoolAttr(false));
      fn->setAttr("execution_provider.truth_boundary", b.getStringAttr(
          "operator_level_attention_execution_provider_not_general_serving_backend"));
      op.erase();
    }
  }
};

} // namespace
std::unique_ptr<Pass> createAttentionCandidateGenerationPass() { return std::make_unique<AttentionCandidateGenerationPass>(); }
std::unique_ptr<Pass> createAttentionLegalityPass() { return std::make_unique<AttentionLegalityPass>(); }
std::unique_ptr<Pass> createAttentionSelectionLoweringPass() { return std::make_unique<AttentionSelectionLoweringPass>(); }
} // namespace mlir::hir
