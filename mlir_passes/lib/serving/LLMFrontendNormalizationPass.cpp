#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_LLMFRONTENDNORMALIZATION
#include "FusionPasses.h.inc"

// Raw pseudo-LLM op names that define the recognizable attention pattern.
static constexpr llvm::StringLiteral kQProj   = "llm.q_proj";
static constexpr llvm::StringLiteral kKProj   = "llm.k_proj";
static constexpr llvm::StringLiteral kVProj   = "llm.v_proj";
static constexpr llvm::StringLiteral kScores  = "llm.attention_scores";
static constexpr llvm::StringLiteral kAttnOut = "llm.attention_output";
static constexpr llvm::StringLiteral kKVWrite = "llm.kv_cache_write";
static constexpr llvm::StringLiteral kKVRead  = "llm.kv_cache_read";

// Canonical output op names — identical to qwen-to-serving-mlir output.
static constexpr llvm::StringLiteral kPrefillOp = "llm.attention_prefill";
static constexpr llvm::StringLiteral kDecodeOp  = "llm.attention_decode";

struct PatternResult {
  bool isAttentionPattern = false;
  bool hasKVWrite         = false;
  bool hasKVRead          = false;
};

static PatternResult detectPattern(func::FuncOp funcOp) {
  PatternResult r;
  bool hasQ = false, hasK = false, hasV = false;
  bool hasScores = false, hasAttnOut = false;

  funcOp.walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == kQProj)   hasQ       = true;
    if (name == kKProj)   hasK       = true;
    if (name == kVProj)   hasV       = true;
    if (name == kScores)  hasScores  = true;
    if (name == kAttnOut) hasAttnOut = true;
    if (name == kKVWrite) r.hasKVWrite = true;
    if (name == kKVRead)  r.hasKVRead  = true;
  });

  r.isAttentionPattern = hasQ && hasK && hasV && hasScores && hasAttnOut;
  return r;
}

// Erase the entire block bottom-up, then rebuild with:
//   <canonical attention op>
//   return <result>
// Preserves the function signature and return type.
static void rewriteToCanonical(func::FuncOp funcOp, bool isPrefill,
                               MLIRContext *ctx) {
  Block &block = funcOp.getBody().front();

  // Capture return type before any erasure.
  mlir::Type resultType;
  if (!funcOp.getResultTypes().empty())
    resultType = funcOp.getResultTypes()[0];

  // Erase all ops bottom-up (users before definitions — safe for use-def chains).
  while (!block.empty())
    block.back().erase();

  // First block arg is the hidden-state tensor; used as structural placeholder
  // for Q, K, and V operands. Labelled via frontend.truth_boundary.
  Value dummyArg = block.getArgument(0);

  OpBuilder builder(&block, block.begin());
  Location loc = funcOp.getLoc();

  llvm::StringRef canonicalName = isPrefill ? kPrefillOp : kDecodeOp;
  llvm::StringRef kvRole        = isPrefill ? "producer" : "consumer";
  llvm::StringRef servingPhase  = isPrefill ? "prefill"  : "decode";
  int64_t promptTokens          = isPrefill ? 512 : 0;
  int64_t outputTokens          = isPrefill ? 0   : 1;

  OperationState state(loc, canonicalName);
  state.addOperands({dummyArg, dummyArg, dummyArg});
  if (resultType)
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

  // Propagate GQA head counts from module attrs when present.
  if (Operation *parent = funcOp->getParentOp()) {
    if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_attention_heads"))
      state.addAttribute("llm.num_q_heads", IntegerAttr::get(i64, a.getInt()));
    if (auto a = parent->getAttrOfType<IntegerAttr>("llm.num_key_value_heads"))
      state.addAttribute("llm.num_kv_heads", IntegerAttr::get(i64, a.getInt()));
  }

  // builder.create(OperationState&) is the non-template overload for
  // unregistered ops; does not require a registered op name.
  Operation *attnOp = builder.create(state);

  // Emit return — preserves original function return type.
  // Use the new OpTy::create API (builder.create<T> is deprecated in LLVM 22).
  if (resultType && attnOp->getNumResults() > 0)
    func::ReturnOp::create(builder, loc, attnOp->getResult(0));
  else
    func::ReturnOp::create(builder, loc, ValueRange{});

  // Annotate the func.func itself so downstream passes can identify path B.
  funcOp->setAttr("frontend.source",
                  StringAttr::get(ctx, "llm_graph_pattern"));
  funcOp->setAttr("frontend.truth_boundary",
                  StringAttr::get(ctx,
                    "attention_pattern_simplified_not_full_transformer_lowering"));
}

struct LLMFrontendNormalizationPass
    : impl::LLMFrontendNormalizationBase<LLMFrontendNormalizationPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();

    PatternResult r = detectPattern(funcOp);
    if (!r.isAttentionPattern)
      return;
    if (!r.hasKVWrite && !r.hasKVRead)
      return;

    // kv_cache_write = KV producer = prefill phase.
    // kv_cache_read  = KV consumer = decode phase.
    bool isPrefill = r.hasKVWrite;
    rewriteToCanonical(funcOp, isPrefill, ctx);
  }
};

} // namespace

std::unique_ptr<Pass> createLLMFrontendNormalizationPass() {
  return std::make_unique<LLMFrontendNormalizationPass>();
}

} // namespace mlir::hir
