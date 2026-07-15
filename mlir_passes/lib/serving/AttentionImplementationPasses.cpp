#include "FusionPasses.h"
#include "HIR/IR/HIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include <limits>
#include <tuple>

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
        SmallVector<Attribute> values;
        for(auto spec:{std::tuple<StringRef,StringRef,StringRef,StringRef>{"cpu_contiguous_kv_fp32_v1","cpu_attention_decode_fp32","hir_cpu_attention_decode_contiguous_kv_fp32","dimension_major_strided_v_accumulation"},std::tuple<StringRef,StringRef,StringRef,StringRef>{"cpu_contiguous_kv_fp32_reordered_v1","cpu_attention_decode_contiguous_kv_reordered_fp32","hir_cpu_attention_decode_contiguous_kv_reordered_fp32","token_major_contiguous_v_accumulation"}}){
          NamedAttrList kv;kv.append("candidate_id",StringAttr::get(op.getContext(),std::get<0>(spec)));kv.append("backend",StringAttr::get(op.getContext(),"portable_cpu"));kv.append("dtype",StringAttr::get(op.getContext(),"fp32"));kv.append("layout",StringAttr::get(op.getContext(),"bhcd_contiguous"));kv.append("capacity_tokens",capacity);kv.append("compatible_prefill_kernel",StringAttr::get(op.getContext(),"cpu_attention_prefill_fp32"));kv.append("compatible_decode_kernel",StringAttr::get(op.getContext(),std::get<1>(spec)));kv.append("entry_point",StringAttr::get(op.getContext(),std::get<2>(spec)));kv.append("implementation_strategy",StringAttr::get(op.getContext(),std::get<3>(spec)));kv.append("workspace_bytes",IntegerAttr::get(IntegerType::get(op.getContext(),64),workspaceBytes));kv.append("head_dimension_constraint",StringAttr::get(op.getContext(),"static_positive"));kv.append("capacity_constraint",StringAttr::get(op.getContext(),"valid_tokens_lte_capacity"));kv.append("measurement_provenance",StringAttr::get(op.getContext(),"exact_target_workload_measured_evidence_required"));kv.append("abi_version",StringAttr::get(op.getContext(),"hir.contiguous_kv.v1"));kv.append("fallback_identity",StringAttr::get(op.getContext(),std::get<0>(spec)=="cpu_contiguous_kv_fp32_v1"?"unsupported_kv_explicit_failure":"cpu_contiguous_kv_fp32_v1"));kv.append("legality_status",StringAttr::get(op.getContext(),"pending"));values.push_back(kv.getDictionary(op.getContext()));
        }
        auto requested=module->getAttrOfType<StringAttr>("attention.cpu.requested_kv_layout");
        if(requested&&requested.getValue()=="paged"){
          auto pageTokens=module->getAttrOfType<IntegerAttr>("attention.cpu.page_tokens");auto pages=module->getAttrOfType<IntegerAttr>("attention.cpu.num_physical_pages");
          if(pageTokens&&pages){
            for(auto spec:{std::tuple<StringRef,StringRef,StringRef>{"cpu_paged_kv_fp32_token_major_v1","cpu_attention_decode_paged_kv_fp32","token_major_block_translation"},std::tuple<StringRef,StringRef,StringRef>{"cpu_paged_kv_fp32_page_major_v1","cpu_attention_decode_paged_kv_page_major_fp32","page_major_cached_page_base"}}){
              NamedAttrList p;p.append("candidate_id",StringAttr::get(op.getContext(),std::get<0>(spec)));p.append("backend",StringAttr::get(op.getContext(),"portable_cpu"));p.append("dtype",StringAttr::get(op.getContext(),"fp32"));p.append("layout",StringAttr::get(op.getContext(),"paged_phd_contiguous"));p.append("capacity_tokens",capacity);p.append("page_tokens",pageTokens);p.append("num_physical_pages",pages);p.append("compatible_prefill_kernel",StringAttr::get(op.getContext(),"cpu_attention_prefill_fp32"));p.append("compatible_decode_kernel",StringAttr::get(op.getContext(),std::get<1>(spec)));p.append("entry_point",StringAttr::get(op.getContext(),std::get<1>(spec)=="cpu_attention_decode_paged_kv_fp32"?"hir_cpu_attention_decode_paged_kv_fp32":"hir_cpu_attention_decode_paged_kv_page_major_fp32"));p.append("implementation_strategy",StringAttr::get(op.getContext(),std::get<2>(spec)));p.append("measurement_provenance",StringAttr::get(op.getContext(),"exact_target_workload_measured_evidence_required"));p.append("abi_version",StringAttr::get(op.getContext(),"hir.paged_kv.v1"));p.append("fallback_identity",StringAttr::get(op.getContext(),"cpu_contiguous_kv_fp32_v1"));p.append("legality_status",StringAttr::get(op.getContext(),"pending"));values.push_back(p.getDictionary(op.getContext()));
            }}}
        op->setAttr("kv.candidates",ArrayAttr::get(op.getContext(),values));
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
          bool paged=cast<StringAttr>(d.get("layout")).getValue()=="paged_phd_contiguous";bool legal=capacity>0&&context<=capacity&&op->getAttrOfType<StringAttr>("dtype").getValue()=="fp32"&&op->getAttrOfType<IntegerAttr>("batch").getInt()==1&&op->getAttrOfType<IntegerAttr>("num_query_heads").getInt()==op->getAttrOfType<IntegerAttr>("num_kv_heads").getInt();if(paged){int64_t pt=cast<IntegerAttr>(d.get("page_tokens")).getInt(),pages=cast<IntegerAttr>(d.get("num_physical_pages")).getInt();legal=legal&&pt>0&&pages>0&&capacity<=pt*pages;}
          a.set("legality_status",StringAttr::get(op.getContext(),legal?"legal":"rejected"));
          a.set("legality_reason",StringAttr::get(op.getContext(),legal?(paged?"static_paged_contract_match":"static_contiguous_contract_match"):"kv_capacity_or_semantic_mismatch"));kvChecked.push_back(a.getDictionary(op.getContext()));}
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
      DictionaryAttr selectedKV;auto requestedLayout=moduleString(op,"attention.cpu.requested_kv_layout");auto requestedPagedStrategy=moduleString(op,"attention.cpu.paged_decode_strategy");auto requestedContiguousStrategy=moduleString(op,"attention.cpu.contiguous_decode_strategy");
      if(auto all=op->getAttrOfType<ArrayAttr>("kv.candidates"))for(Attribute x:all){auto d=cast<DictionaryAttr>(x);if(cast<StringAttr>(d.get("legality_status")).getValue()!="legal")continue;bool isPaged=cast<StringAttr>(d.get("layout")).getValue()=="paged_phd_contiguous";if(requestedLayout&&requestedLayout.getValue()=="paged"){if(!isPaged)continue;StringRef desired=requestedPagedStrategy?requestedPagedStrategy.getValue():StringRef("token_major_block_translation");if(cast<StringAttr>(d.get("implementation_strategy")).getValue()!=desired)continue;}else{if(isPaged)continue;StringRef desired=requestedContiguousStrategy?requestedContiguousStrategy.getValue():StringRef("dimension_major_strided_v_accumulation");if(cast<StringAttr>(d.get("implementation_strategy")).getValue()!=desired)continue;}selectedKV=d;}
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
      bool paged = cast<StringAttr>(selectedKV.get("layout")).getValue() == "paged_phd_contiguous";
      if (paged) {
        int64_t pt=cast<IntegerAttr>(selectedKV.get("page_tokens")).getInt(), pages=cast<IntegerAttr>(selectedKV.get("num_physical_pages")).getInt();
        int64_t blocks=(capacity+pt-1)/pt, onePage, poolBytes;
        if(!checkedMul(heads,pt,elements)||!checkedMul(elements,dim,elements)||!checkedMul(elements,4,onePage)||!checkedMul(onePage,2*pages,poolBytes)){op.emitError("overflowing paged KV contract");signalPassFailure();return;}
        StringRef phase=op->getAttrOfType<StringAttr>("phase").getValue();
        attrs.set("pool_create_entry_point",b.getStringAttr("hir_paged_kv_initialize"));attrs.set("prefill_write_entry_point",b.getStringAttr("hir_paged_kv_prefill_write"));attrs.set("append_entry_point",b.getStringAttr("hir_paged_kv_append"));attrs.set("view_binding",b.getStringAttr("direct_int32_block_table_translation"));attrs.set("reset_entry_point",b.getStringAttr("hir_paged_kv_reset"));attrs.set("release_entry_point",b.getStringAttr("runtime_owned_pool_release"));
        attrs.set("truth_boundary",b.getStringAttr("real_single_request_cpu_paged_kv_not_general_paged_attention"));attrs.set("kv_layout_kind",b.getStringAttr("paged_phd_contiguous"));attrs.set("kv_candidate_id",selectedKV.get("candidate_id"));attrs.set("implementation_strategy",selectedKV.get("implementation_strategy"));attrs.set("measurement_provenance",selectedKV.get("measurement_provenance"));attrs.set("kv_dtype",b.getStringAttr("fp32"));attrs.set("page_tokens",b.getI64IntegerAttr(pt));attrs.set("num_physical_pages",b.getI64IntegerAttr(pages));attrs.set("maximum_logical_tokens",b.getI64IntegerAttr(capacity));attrs.set("maximum_logical_blocks",b.getI64IntegerAttr(blocks));attrs.set("block_table_length",b.getI64IntegerAttr(blocks));attrs.set("block_table_element_type",b.getStringAttr("int32"));attrs.set("invalid_page_sentinel",b.getI32IntegerAttr(-1));attrs.set("bytes_per_token",b.getI64IntegerAttr(bytesPerToken));attrs.set("bytes_per_k_page",b.getI64IntegerAttr(onePage));attrs.set("bytes_per_v_page",b.getI64IntegerAttr(onePage));attrs.set("bytes_per_combined_page",b.getI64IntegerAttr(onePage*2));attrs.set("total_pool_bytes",b.getI64IntegerAttr(poolBytes));attrs.set("k_page_strides",b.getDenseI64ArrayAttr({heads*pt*dim,pt*dim,dim,1}));attrs.set("v_page_strides",b.getDenseI64ArrayAttr({heads*pt*dim,pt*dim,dim,1}));attrs.set("pool_artifact_ref",ref);attrs.set("pool_artifact_sha256",sha);attrs.set("pool_artifact_version",b.getStringAttr("hir.paged_kv.v1"));attrs.set("paged_attention_kernel_id",selectedKV.get("compatible_decode_kernel"));attrs.set("paged_attention_entry_point",selectedKV.get("entry_point"));attrs.set("contiguous_fallback_identity",b.getStringAttr("cpu_contiguous_kv_fp32_v1"));attrs.set("runtime_no_layout_redecision",b.getBoolAttr(true));attrs.set("runtime_no_kernel_redecision",b.getBoolAttr(true));attrs.set("paged_operation_order",b.getStringAttr(phase=="prefill"?"create_pool_then_bind_blocks_then_prefill_write_then_attention":"ensure_page_then_append_then_view_then_paged_decode"));
        NamedAttrList pa;for(StringRef n:{"kv_candidate_id","kv_layout_kind","kv_dtype","block_table_element_type"})pa.set(n,attrs.get(n));for(StringRef n:{"page_tokens","num_physical_pages","maximum_logical_tokens","block_table_length","num_kv_heads","head_dim"})pa.set(n,attrs.get(n));
        auto pool=b.create<PagedKVPoolCreateOp>(op.getLoc(),b.getI64Type(),ValueRange{},pa.getAttrs());b.create<PagedKVBindBlockOp>(op.getLoc(),TypeRange{},ValueRange{pool.getPool()},pa.getAttrs());Value key=op.getKey(),value=op.getValue();
        if(phase=="prefill")b.create<PagedKVPrefillWriteOp>(op.getLoc(),TypeRange{},ValueRange{pool.getPool(),key,value},pa.getAttrs());else{b.create<PagedKVAppendOp>(op.getLoc(),TypeRange{},ValueRange{pool.getPool(),key,value},pa.getAttrs());auto qt=cast<RankedTensorType>(key.getType());auto viewType=RankedTensorType::get({batch,heads,op->getAttrOfType<IntegerAttr>("context_length").getInt(),dim},qt.getElementType());auto view=b.create<PagedKVViewOp>(op.getLoc(),TypeRange{viewType,viewType},ValueRange{pool.getPool()},pa.getAttrs());key=view.getKey();value=view.getValue();attrs.set("kernel_id",selectedKV.get("compatible_decode_kernel"));attrs.set("entry_point",selectedKV.get("entry_point"));}
        auto lowered=b.create<CPUAttentionOp>(op.getLoc(),TypeRange{op.getOutput().getType()},ValueRange{op.getQuery(),key,value},attrs.getAttrs());op.getOutput().replaceAllUsesWith(lowered.getOutput());auto fn=op->getParentOfType<func::FuncOp>();fn->setAttr("execution_provider.primary",b.getStringAttr("portable_cpu"));fn->setAttr("execution_provider.kv_layout",b.getStringAttr("runtime_owned_paged_phd_contiguous"));fn->setAttr("execution_provider.decision_source",b.getStringAttr("explicit_paged_request_legal"));op.erase();continue;
      }
      attrs.set("truth_boundary", b.getStringAttr(
          "real_operator_level_fp32_cpu_attention_with_runtime_owned_contiguous_kv"));
      attrs.set("kv_candidate_id", selectedKV.get("candidate_id"));
      attrs.set("implementation_strategy", selectedKV.get("implementation_strategy"));
      attrs.set("attention_implementation_strategy", b.getStringAttr(
          op->getAttrOfType<StringAttr>("phase").getValue() == "decode"
              ? cast<StringAttr>(selectedKV.get("implementation_strategy")).getValue()
              : StringRef("prefill_contiguous")));
      attrs.set("measurement_provenance", selectedKV.get("measurement_provenance"));
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
      attrs.set("compatible_decode_kernel_id", selectedKV.get("compatible_decode_kernel"));
      attrs.set("attention_entry_point", selectedKV.get("entry_point"));
      if (op->getAttrOfType<StringAttr>("phase").getValue() == "decode") {
        attrs.set("candidate_id", selectedKV.get("candidate_id"));
        attrs.set("kernel_id", selectedKV.get("compatible_decode_kernel"));
        attrs.set("entry_point", selectedKV.get("entry_point"));
      }
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
