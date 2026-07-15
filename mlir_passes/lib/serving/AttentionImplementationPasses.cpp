#include "FusionPasses.h"
#include "HIR/IR/HIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <limits>

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
      Operation *module=op->getParentOp();while(module&&!isa<ModuleOp>(module))module=module->getParentOp();
      auto capacity=module?module->getAttrOfType<IntegerAttr>("attention.cpu.kv_capacity_tokens"):IntegerAttr{};
      if (capacity) {
        NamedAttrList kv;
        kv.append("candidate_id",StringAttr::get(op.getContext(),"cpu_contiguous_kv_fp32_v1"));
        kv.append("backend",StringAttr::get(op.getContext(),"portable_cpu"));
        kv.append("dtype",StringAttr::get(op.getContext(),"fp32"));
        kv.append("layout",StringAttr::get(op.getContext(),"bhcd_contiguous"));
        kv.append("capacity_tokens",capacity);
        kv.append("compatible_prefill_kernel",StringAttr::get(op.getContext(),"cpu_attention_prefill_fp32"));
        kv.append("compatible_decode_kernel",StringAttr::get(op.getContext(),"cpu_attention_decode_fp32"));
        kv.append("abi_version",StringAttr::get(op.getContext(),"hir.contiguous_kv.v1"));
        kv.append("fallback_identity",StringAttr::get(op.getContext(),"unsupported_kv_explicit_failure"));
        kv.append("legality_status",StringAttr::get(op.getContext(),"pending"));
        op->setAttr("kv.candidates",ArrayAttr::get(op.getContext(),{kv.getDictionary(op.getContext())}));
      }
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
      if (auto candidates=op->getAttrOfType<ArrayAttr>("kv.candidates")) {
        SmallVector<Attribute> kvChecked;
        for(Attribute x:candidates){auto d=cast<DictionaryAttr>(x);NamedAttrList a(d);
          int64_t capacity=cast<IntegerAttr>(d.get("capacity_tokens")).getInt();
          int64_t context=op->getAttrOfType<IntegerAttr>("context_length").getInt();
          bool legal=capacity>0&&context<=capacity&&op->getAttrOfType<StringAttr>("dtype").getValue()=="fp32"&&op->getAttrOfType<IntegerAttr>("num_query_heads").getInt()==op->getAttrOfType<IntegerAttr>("num_kv_heads").getInt();
          a.set("legality_status",StringAttr::get(op.getContext(),legal?"legal":"rejected"));
          a.set("legality_reason",StringAttr::get(op.getContext(),legal?"static_contiguous_contract_match":"kv_capacity_or_semantic_mismatch"));kvChecked.push_back(a.getDictionary(op.getContext()));}
        op->setAttr("kv.candidates",ArrayAttr::get(op.getContext(),kvChecked));
      }
    });
  }
};

static StringAttr moduleString(Operation *op, StringRef name) {
  Operation *module = op->getParentOp();
  while (module && !isa<ModuleOp>(module)) module = module->getParentOp();
  return module ? module->getAttrOfType<StringAttr>(name) : StringAttr{};
}

static IntegerAttr moduleInteger(Operation *op, StringRef name) {
  Operation *module = op->getParentOp();
  while (module && !isa<ModuleOp>(module)) module = module->getParentOp();
  return module ? module->getAttrOfType<IntegerAttr>(name) : IntegerAttr{};
}

static bool checkedMul(int64_t a, int64_t b, int64_t &out) {
  if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b)
    return false;
  out = a * b; return true;
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
      auto capacityAttr = moduleInteger(op, "attention.cpu.kv_capacity_tokens");
      if (!ref || !sha || !version || !capacityAttr) {
        op.emitError("selected CPU attention requires module artifact ref, sha256, and version");
        signalPassFailure(); return;
      }
      DictionaryAttr selected;
      for (Attribute x : op->getAttrOfType<ArrayAttr>("attention.candidates")) {
        auto d = cast<DictionaryAttr>(x);
        if (cast<StringAttr>(d.get("legality_status")).getValue() == "legal") selected = d;
      }
      if (!selected) { op.emitError("no legal phase-specific attention implementation"); signalPassFailure(); return; }
      DictionaryAttr selectedKV;
      if(auto all=op->getAttrOfType<ArrayAttr>("kv.candidates"))for(Attribute x:all){auto d=cast<DictionaryAttr>(x);if(cast<StringAttr>(d.get("legality_status")).getValue()=="legal")selectedKV=d;}
      if (!selectedKV) { op.emitError("no legal contiguous KV implementation"); signalPassFailure(); return; }
      int64_t batch = op->getAttrOfType<IntegerAttr>("batch").getInt();
      int64_t heads = op->getAttrOfType<IntegerAttr>("num_kv_heads").getInt();
      int64_t dim = op->getAttrOfType<IntegerAttr>("head_dim").getInt();
      int64_t capacity = capacityAttr.getInt();
      int64_t bytesPerToken, elements, oneCacheBytes, totalBytes;
      if (capacity <= 0 || !checkedMul(batch, heads, elements) ||
          !checkedMul(elements, dim, elements) ||
          !checkedMul(elements, 8, bytesPerToken) ||
          !checkedMul(elements, capacity, elements) ||
          !checkedMul(elements, 4, oneCacheBytes) ||
          !checkedMul(oneCacheBytes, 2, totalBytes) ||
          op->getAttrOfType<IntegerAttr>("context_length").getInt() > capacity) {
        op.emitError("illegal or overflowing contiguous KV capacity contract");
        signalPassFailure(); return;
      }
      OpBuilder b(op); NamedAttrList attrs(op->getAttrs());
      attrs.erase("attention.candidates");
      attrs.erase("kv.candidates");
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
          "real_operator_level_fp32_cpu_attention_with_runtime_owned_contiguous_kv"));
      attrs.set("kv_candidate_id", b.getStringAttr("cpu_contiguous_kv_fp32_v1"));
      attrs.set("kv_cache_id", b.getStringAttr("operator_attention_kv_cache"));
      attrs.set("kv_dtype", b.getStringAttr("fp32"));
      attrs.set("kv_layout", b.getStringAttr("bhcd_contiguous"));
      attrs.set("capacity_tokens", b.getI64IntegerAttr(capacity));
      attrs.set("initial_valid_tokens", b.getI64IntegerAttr(
          op->getAttrOfType<StringAttr>("phase").getValue() == "prefill" ? 0 :
          op->getAttrOfType<IntegerAttr>("context_length").getInt() - 1));
      attrs.set("bytes_per_token", b.getI64IntegerAttr(bytesPerToken));
      attrs.set("k_cache_bytes", b.getI64IntegerAttr(oneCacheBytes));
      attrs.set("v_cache_bytes", b.getI64IntegerAttr(oneCacheBytes));
      attrs.set("total_cache_bytes", b.getI64IntegerAttr(totalBytes));
      attrs.set("kv_artifact_ref", ref); attrs.set("kv_artifact_sha256", sha);
      attrs.set("kv_artifact_version", b.getStringAttr("hir.contiguous_kv.v1"));
      attrs.set("kv_create_entry_point", b.getStringAttr("hir_contiguous_kv_initialize"));
      attrs.set("kv_prefill_write_entry_point", b.getStringAttr("hir_contiguous_kv_prefill_write"));
      attrs.set("kv_decode_append_entry_point", b.getStringAttr("hir_contiguous_kv_append"));
      attrs.set("kv_view_binding", b.getStringAttr("direct_contiguous_pointer_valid_prefix"));
      attrs.set("kv_reset_entry_point", b.getStringAttr("hir_contiguous_kv_reset"));
      attrs.set("compatible_prefill_kernel_id", b.getStringAttr("cpu_attention_prefill_fp32"));
      attrs.set("compatible_decode_kernel_id", b.getStringAttr("cpu_attention_decode_fp32"));
      attrs.set("runtime_no_layout_redecision", b.getBoolAttr(true));
      attrs.set("kv_operation_order", b.getStringAttr(
          op->getAttrOfType<StringAttr>("phase").getValue() == "prefill"
              ? "create_then_prefill_write_then_attention"
              : "append_then_view_then_decode_attention"));
      int64_t strideD = 1, strideC = dim, strideH = capacity * dim;
      int64_t strideB = heads * strideH;
      attrs.set("k_strides", b.getDenseI64ArrayAttr({strideB, strideH, strideC, strideD}));
      attrs.set("v_strides", b.getDenseI64ArrayAttr({strideB, strideH, strideC, strideD}));

      NamedAttrList kvAttrs;
      for (StringRef name : {"kv_candidate_id", "kv_cache_id", "kv_dtype", "kv_layout",
                             "kv_artifact_version"}) kvAttrs.set(name, attrs.get(name));
      for (StringRef name : {"batch", "num_kv_heads", "head_dim", "capacity_tokens",
                             "alignment_bytes"}) kvAttrs.set(name, attrs.get(name));
      auto handle = b.create<KVCacheCreateOp>(op.getLoc(), b.getI64Type(),
                                              ValueRange{}, kvAttrs.getAttrs());
      Value key = op.getKey(), value = op.getValue();
      StringRef phase = op->getAttrOfType<StringAttr>("phase").getValue();
      if (phase == "prefill") {
        b.create<KVCachePrefillWriteOp>(op.getLoc(), TypeRange{},
            ValueRange{handle.getHandle(), key, value}, kvAttrs.getAttrs());
      } else {
        b.create<KVCacheAppendOp>(op.getLoc(), TypeRange{},
            ValueRange{handle.getHandle(), key, value}, kvAttrs.getAttrs());
        auto qt = cast<RankedTensorType>(key.getType());
        auto viewType = RankedTensorType::get(
            {batch, heads, op->getAttrOfType<IntegerAttr>("context_length").getInt(), dim},
            qt.getElementType());
        auto view = b.create<KVCacheViewOp>(op.getLoc(), TypeRange{viewType, viewType},
                                            ValueRange{handle.getHandle()}, kvAttrs.getAttrs());
        key = view.getKey(); value = view.getValue();
      }
      auto lowered = b.create<CPUAttentionOp>(op.getLoc(),
          TypeRange{op.getOutput().getType()},
          ValueRange{op.getQuery(), key, value}, attrs.getAttrs());
      op.getOutput().replaceAllUsesWith(lowered.getOutput());
      auto fn = op->getParentOfType<func::FuncOp>();
      fn->setAttr("execution_provider.primary", b.getStringAttr("portable_cpu"));
      fn->setAttr("execution_provider.decision_source", b.getStringAttr("deterministic_phase_policy"));
      fn->setAttr("execution_provider.precision", b.getStringAttr("fp32"));
      fn->setAttr("execution_provider.kv_layout", b.getStringAttr("runtime_owned_bhcd_contiguous"));
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
