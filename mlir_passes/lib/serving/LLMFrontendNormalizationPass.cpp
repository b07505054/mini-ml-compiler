#include "FusionPasses.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <optional>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_LLMFRONTENDNORMALIZATION
#include "FusionPasses.h.inc"

// Raw pseudo-LLM op names that define the recognizable attention pattern.
static constexpr llvm::StringLiteral kQProj   = "llm.q_proj";
static constexpr llvm::StringLiteral kKProj   = "llm.k_proj";
static constexpr llvm::StringLiteral kVProj   = "llm.v_proj";
static constexpr llvm::StringLiteral kScores  = "llm.attention_scores";
static constexpr llvm::StringLiteral kSoftmax = "llm.softmax";
static constexpr llvm::StringLiteral kAttnOut = "llm.attention_output";
static constexpr llvm::StringLiteral kKVWrite = "llm.kv_cache_write";
static constexpr llvm::StringLiteral kKVRead  = "llm.kv_cache_read";

// Canonical output op names — identical to qwen-to-serving-mlir output.
static constexpr llvm::StringLiteral kPrefillOp = "llm.attention_prefill";
static constexpr llvm::StringLiteral kDecodeOp  = "llm.attention_decode";

// One occurrence of the raw attention pattern, found within a single
// "occurrence group" (see runOnOperation). A function normalized from a
// real full-per-layer-expanded import may contain num_layers occurrences,
// one per serving.layer_index value; a legacy single-occurrence function
// (no layer_index attrs at all, e.g. mlir/raw_qwen_frontend_input.mlir) is
// treated as exactly one ungrouped occurrence, preserving prior behavior.
struct PatternOps {
  Operation *qProj   = nullptr;
  Operation *kProj    = nullptr;
  Operation *vProj    = nullptr;
  Operation *scores   = nullptr;
  Operation *softmax  = nullptr; // optional intermediate; erased if present
  Operation *attnOut  = nullptr;
  Operation *kvOp     = nullptr; // llm.kv_cache_write or llm.kv_cache_read
  bool isPrefill      = false;

  bool isComplete() const {
    return qProj && kProj && vProj && scores && attnOut && kvOp;
  }
};

static PatternOps findPatternOps(llvm::ArrayRef<Operation *> ops) {
  PatternOps p;
  for (Operation *op : ops) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == kQProj && !p.qProj)          p.qProj = op;
    else if (name == kKProj && !p.kProj)      p.kProj = op;
    else if (name == kVProj && !p.vProj)      p.vProj = op;
    else if (name == kScores && !p.scores)    p.scores = op;
    else if (name == kSoftmax && !p.softmax)  p.softmax = op;
    else if (name == kAttnOut && !p.attnOut)  p.attnOut = op;
    else if (name == kKVWrite && !p.kvOp)     { p.kvOp = op; p.isPrefill = true; }
    else if (name == kKVRead && !p.kvOp)      { p.kvOp = op; p.isPrefill = false; }
  }
  return p;
}

// Replace one local occurrence of the raw attention pattern with a single
// canonical llm.attention_prefill/decode op, wired to the REAL q/k/v
// producer values found in this occurrence (not a dummy placeholder), and
// leaves every other op in the function untouched. This makes the rewrite
// safe to apply once per occurrence in a fully-expanded, multi-layer
// function, rather than only once for a whole single-attention function.
//
// qProj/kProj/vProj are NOT erased: the canonical op's operands are their
// results, so they remain live producers feeding the canonical op instead
// of feeding scores/kvOp. Only the now-redundant intermediate computation
// (scores, optional softmax, attnOut, kvOp) is erased, leaf-first (consumers
// before producers) so no op is erased while it still has uses: attnOut's
// result is redirected to the new canonical op before attnOut itself is
// erased, then softmax -> scores -> kvOp is safe regardless of whether kvOp
// is a KV producer consuming k/v (prefill) or a KV producer feeding
// scores/attnOut (decode).
static void rewriteOccurrence(func::FuncOp funcOp, const PatternOps &p,
                              std::optional<int64_t> layerIndex,
                              MLIRContext *ctx) {
  OpBuilder builder(p.attnOut);
  Location loc = p.attnOut->getLoc();
  Type resultType = p.attnOut->getResult(0).getType();

  llvm::StringRef canonicalName = p.isPrefill ? kPrefillOp : kDecodeOp;
  llvm::StringRef kvRole        = p.isPrefill ? "producer" : "consumer";
  llvm::StringRef servingPhase  = p.isPrefill ? "prefill"  : "decode";
  int64_t promptTokens          = p.isPrefill ? 512 : 0;
  int64_t outputTokens          = p.isPrefill ? 0   : 1;

  OperationState state(loc, canonicalName);
  state.addOperands({p.qProj->getResult(0), p.kProj->getResult(0),
                      p.vProj->getResult(0)});
  state.addTypes({resultType});

  Type i64 = IntegerType::get(ctx, 64);
  state.addAttribute("kv_cache.role",    StringAttr::get(ctx, kvRole));
  state.addAttribute("serving.phase",    StringAttr::get(ctx, servingPhase));
  state.addAttribute("serving.prompt_tokens",
                     IntegerAttr::get(i64, promptTokens));
  state.addAttribute("serving.output_tokens",
                     IntegerAttr::get(i64, outputTokens));
  state.addAttribute("frontend.source",
                     StringAttr::get(ctx, "llm_graph_pattern"));
  state.addAttribute("frontend.truth_boundary",
                     StringAttr::get(ctx,
                       "attention_pattern_simplified_not_full_transformer_lowering"));
  state.addAttribute("frontend.normalized_from",
                     StringAttr::get(ctx, "simplified_llm_attention_graph"));
  if (layerIndex) {
    state.addAttribute("serving.layer_index", IntegerAttr::get(i64, *layerIndex));
    state.addAttribute("serving.layer_role", StringAttr::get(ctx, "decoder_layer"));
  }

  // Propagate GQA head counts from module attrs when present.
  if (Operation *parent = funcOp->getParentOp()) {
    if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_attention_heads"))
      state.addAttribute("llm.num_q_heads", IntegerAttr::get(i64, a.getInt()));
    if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_key_value_heads"))
      state.addAttribute("llm.num_kv_heads", IntegerAttr::get(i64, a.getInt()));
  }

  // builder.create(OperationState&) is the non-template overload for
  // unregistered ops; does not require a registered op name.
  Operation *canonicalOp = builder.create(state);
  p.attnOut->getResult(0).replaceAllUsesWith(canonicalOp->getResult(0));

  // qProj/kProj/vProj are intentionally NOT erased — canonicalOp's operands
  // are their results, so erasing them would leave canonicalOp with
  // dangling operands (MLIR's IsIsolatedFromAbove verifier would reject the
  // function with "operation's operand is unlinked").
  p.attnOut->erase();
  if (p.softmax) p.softmax->erase();
  p.scores->erase();
  p.kvOp->erase();
}

struct LLMFrontendNormalizationPass
    : impl::LLMFrontendNormalizationBase<LLMFrontendNormalizationPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    if (funcOp.getBody().empty())
      return;

    // Bucket the entry block's top-level ops by serving.layer_index. Ops
    // with no layer_index attr (legacy single-occurrence functions) share
    // one ungrouped bucket, so exactly-once functions keep working exactly
    // as before. Ops are only ever bucketed, never reordered.
    constexpr int64_t kUngrouped = -1;
    llvm::MapVector<int64_t, llvm::SmallVector<Operation *, 16>> groups;
    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      int64_t key = kUngrouped;
      if (auto a = op.getAttrOfType<IntegerAttr>("serving.layer_index"))
        key = a.getInt();
      groups[key].push_back(&op);
    }

    bool rewroteAny = false;
    for (auto &entry : groups) {
      PatternOps p = findPatternOps(entry.second);
      if (!p.isComplete())
        continue;
      std::optional<int64_t> layerIndex;
      if (entry.first != kUngrouped)
        layerIndex = entry.first;
      rewriteOccurrence(funcOp, p, layerIndex, ctx);
      rewroteAny = true;
    }

    if (!rewroteAny)
      return;

    // Annotate the func.func itself so downstream passes can identify path B.
    funcOp->setAttr("frontend.source",
                    StringAttr::get(ctx, "llm_graph_pattern"));
    funcOp->setAttr("frontend.truth_boundary",
                    StringAttr::get(ctx,
                      "attention_pattern_simplified_not_full_transformer_lowering"));
  }
};

} // namespace

std::unique_ptr<Pass> createLLMFrontendNormalizationPass() {
  return std::make_unique<LLMFrontendNormalizationPass>();
}

} // namespace mlir::hir
