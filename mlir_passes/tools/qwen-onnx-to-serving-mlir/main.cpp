// qwen-onnx-to-serving-mlir: ONNX-graph-facts frontend (Phase 1 scaffold).
//
// Reads a graph-facts JSON describing an ONNX-shaped Qwen graph and emits
// RAW (pre-normalization), fully-expanded, flat, unrolled serving-aware MLIR
// text — one real op sequence per decoder layer (0..num_layers-1), not a
// single hand-templated block.
//
// Truth boundary: THIS TOOL DOES NOT PARSE A REAL ONNX PROTOBUF FILE.
// It reads a hand-authored graph-facts JSON
// (configs/models/qwen_0_5b_onnx_graph_facts.json) that stands in for facts
// a real ONNX importer would extract (op-type sequence per role, layer
// count, dims). See that file's "truth_boundary"/"notes" fields and
// tools/export_qwen_onnx.py for the current state of real ONNX export/
// introspection work. Do not describe this tool as a full ONNX frontend.
//
// What IS real here:
//   - Full per-layer expansion: all num_layers decoder layers are emitted
//     as distinct, really-chained ops (real SSA dataflow layer N -> N+1),
//     not one hand-templated block reused for every layer count.
//   - serving.layer_index / serving.layer_role stamped per op, so
//     downstream passes and the LLMFrontendNormalizationPass can identify
//     which real layer (or non-layer role) an op belongs to.
//   - The raw (pre-canonicalization) attention pattern
//     (q_proj/k_proj/v_proj/attention_scores/softmax/attention_output +
//     kv_cache_write|read) is emitted per layer, matching the existing
//     single-occurrence convention in mlir/raw_qwen_frontend_input.mlir, so
//     LLMFrontendNormalizationPass has real per-layer work to do.
//
// What is NOT real here (do not claim otherwise):
//   - No weights are imported.
//   - No residual adds are modeled (consistent with the existing
//     raw_qwen_frontend_input.mlir simplification).
//   - Tensor shapes are structural placeholders (dynamic where unconstrained),
//     not derived from real per-op shape inference.
//   - RoPE is not modeled as a distinct op.
//
// Usage:
//   qwen-onnx-to-serving-mlir \
//     --graph-facts configs/models/qwen_0_5b_onnx_graph_facts.json \
//     --out mlir/qwen_0_5b_onnx_raw.mlir
//
// Output must be run through the LLMFrontendNormalizationPass before
// compile-for-target, e.g.:
//   mlir-opt --allow-unregistered-dialect \
//     --load-pass-plugin=<plugin> \
//     --pass-pipeline='builtin.module(llm-frontend-normalization)' \
//     mlir/qwen_0_5b_onnx_raw.mlir -o mlir/qwen_0_5b_onnx_normalized.mlir

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> GraphFactsPath(
    "graph-facts",
    llvm::cl::desc("Path to Qwen ONNX graph-facts JSON"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> OutPath(
    "out",
    llvm::cl::desc("Output MLIR file path"),
    llvm::cl::Required);

// ---------------------------------------------------------------------------
// GraphFacts
// ---------------------------------------------------------------------------

struct GraphFacts {
  std::string model_name;
  std::string source;
  std::string truth_boundary;
  std::string dtype;
  int64_t num_layers              = 0;
  int64_t hidden_size             = 0;
  int64_t intermediate_size       = 0;
  int64_t num_attention_heads     = 0;
  int64_t num_key_value_heads     = 0;
  int64_t max_position_embeddings = 0;
  int64_t vocab_size              = 0;

  std::vector<std::string> embedding_ops;
  std::vector<std::string> decoder_layer_ops;
  std::vector<std::string> final_norm_ops;
  std::vector<std::string> lm_head_ops;
};

static llvm::Expected<std::vector<std::string>> requireStringArray(
    const llvm::json::Object &graph, llvm::StringRef key) {
  const llvm::json::Array *arr = graph.getArray(key);
  if (!arr)
    return llvm::make_error<llvm::StringError>(
        "graph facts missing required array field 'graph." + key.str() + "'",
        llvm::inconvertibleErrorCode());
  std::vector<std::string> out;
  for (const llvm::json::Value &v : *arr) {
    if (auto s = v.getAsString())
      out.push_back(s->str());
    else
      return llvm::make_error<llvm::StringError>(
          "graph.'" + key.str() + "' must be an array of strings",
          llvm::inconvertibleErrorCode());
  }
  return out;
}

static llvm::Expected<GraphFacts> parseGraphFacts(llvm::StringRef path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::make_error<llvm::StringError>(
        "cannot read graph facts '" + path.str() + "': " +
            buf.getError().message(),
        llvm::inconvertibleErrorCode());

  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::make_error<llvm::StringError>(
        "JSON parse error in graph facts '" + path.str() + "': " +
            llvm::toString(json.takeError()),
        llvm::inconvertibleErrorCode());

  const llvm::json::Object *obj = json->getAsObject();
  if (!obj)
    return llvm::make_error<llvm::StringError>(
        "graph facts must be a JSON object", llvm::inconvertibleErrorCode());

  GraphFacts facts;

  auto requireString = [&](llvm::StringRef key) -> llvm::Expected<std::string> {
    if (auto v = obj->getString(key))
      return v->str();
    return llvm::make_error<llvm::StringError>(
        "graph facts missing required field '" + key.str() + "'",
        llvm::inconvertibleErrorCode());
  };
  auto requireInt = [&](llvm::StringRef key) -> llvm::Expected<int64_t> {
    if (auto v = obj->getInteger(key))
      return *v;
    return llvm::make_error<llvm::StringError>(
        "graph facts missing required integer field '" + key.str() + "'",
        llvm::inconvertibleErrorCode());
  };

  if (auto r = requireString("model_name"))     facts.model_name     = *r; else return r.takeError();
  if (auto r = requireString("source"))         facts.source         = *r; else return r.takeError();
  if (auto r = requireString("truth_boundary")) facts.truth_boundary = *r; else return r.takeError();
  if (auto r = requireString("dtype"))          facts.dtype          = *r; else return r.takeError();
  if (auto r = requireInt("num_layers"))              facts.num_layers              = *r; else return r.takeError();
  if (auto r = requireInt("hidden_size"))             facts.hidden_size             = *r; else return r.takeError();
  if (auto r = requireInt("intermediate_size"))       facts.intermediate_size       = *r; else return r.takeError();
  if (auto r = requireInt("num_attention_heads"))     facts.num_attention_heads     = *r; else return r.takeError();
  if (auto r = requireInt("num_key_value_heads"))     facts.num_key_value_heads     = *r; else return r.takeError();
  if (auto r = requireInt("max_position_embeddings")) facts.max_position_embeddings = *r; else return r.takeError();
  if (auto r = requireInt("vocab_size"))              facts.vocab_size              = *r; else return r.takeError();

  if (facts.num_attention_heads <= 0)
    return llvm::make_error<llvm::StringError>(
        "num_attention_heads must be > 0", llvm::inconvertibleErrorCode());
  if (facts.hidden_size % facts.num_attention_heads != 0)
    return llvm::make_error<llvm::StringError>(
        "hidden_size must be divisible by num_attention_heads",
        llvm::inconvertibleErrorCode());
  if (facts.num_layers <= 0)
    return llvm::make_error<llvm::StringError>(
        "num_layers must be > 0", llvm::inconvertibleErrorCode());

  const llvm::json::Object *graph = obj->getObject("graph");
  if (!graph)
    return llvm::make_error<llvm::StringError>(
        "graph facts missing required object field 'graph'",
        llvm::inconvertibleErrorCode());

  if (auto r = requireStringArray(*graph, "embedding"))
    facts.embedding_ops = *r;
  else return r.takeError();
  if (auto r = requireStringArray(*graph, "decoder_layer"))
    facts.decoder_layer_ops = *r;
  else return r.takeError();
  if (auto r = requireStringArray(*graph, "final_norm"))
    facts.final_norm_ops = *r;
  else return r.takeError();
  if (auto r = requireStringArray(*graph, "lm_head"))
    facts.lm_head_ops = *r;
  else return r.takeError();

  static const llvm::StringRef kSupportedOps[] = {
      "embed", "rmsnorm", "q_proj", "k_proj", "v_proj", "attention_scores",
      "softmax", "attention_output", "kv_cache_boundary", "o_proj", "mlp",
      "lm_head_proj",
  };
  auto isSupported = [&](llvm::StringRef name) {
    for (llvm::StringRef s : kSupportedOps)
      if (s == name) return true;
    return false;
  };
  for (const auto &group : {facts.embedding_ops, facts.decoder_layer_ops,
                            facts.final_norm_ops, facts.lm_head_ops}) {
    for (const std::string &op : group) {
      if (!isSupported(op))
        return llvm::make_error<llvm::StringError>(
            "unsupported graph op '" + op + "': this scaffold importer only "
            "supports a fixed known vocabulary, not arbitrary ONNX op types",
            llvm::inconvertibleErrorCode());
    }
  }

  return facts;
}

// ---------------------------------------------------------------------------
// MLIR text emission
// ---------------------------------------------------------------------------

namespace {

// Threads a "current hidden state" SSA value plus attention side-channel
// values (q/k/v/scores/probs/attn/ck/cv) through one function body, emitting
// one real op per graph-facts entry. Not a general graph interpreter — the
// per-op-name wiring convention below is fixed to the known vocabulary
// validated in parseGraphFacts.
class LayerEmitter {
public:
  LayerEmitter(llvm::raw_ostream &os, const GraphFacts &facts, bool isPrefill)
      : os(os), facts(facts), isPrefill(isPrefill),
        hs(facts.hidden_size),
        kvDim(facts.num_key_value_heads * (facts.hidden_size / facts.num_attention_heads)),
        seqDim(isPrefill ? "?" : "1") {}

  // %h is the running hidden-state value name; updated in place.
  void emitEmbedding(std::string &h) {
    std::string tokensTy = "tensor<" + seqDim + "xi32>";
    std::string hTy = hiddenTy();
    std::string next = fresh("emb");
    os << "    %" << next << " = \"llm.embed\"(%tokens)"
       << " {serving.layer_role = \"embedding\"}"
       << " : (" << tokensTy << ") -> " << hTy << "\n";
    h = next;
  }

  void emitDecoderLayer(int64_t layerIndex, std::string &h) {
    std::string q, k, v, scores, probs, attn, ck, cv;
    for (const std::string &opName : facts.decoder_layer_ops)
      emitOne(opName, layerIndex, h, q, k, v, scores, probs, attn, ck, cv);
  }

  void emitFinalNorm(std::string &h) {
    for (const std::string &opName : facts.final_norm_ops) {
      // Only "rmsnorm" is expected/supported in the final_norm role.
      std::string next = fresh("finalnorm");
      os << "    %" << next << " = \"llm." << opName << "\"(%" << h << ")"
         << " {serving.layer_role = \"final_norm\"}"
         << " : (" << hiddenTy() << ") -> " << hiddenTy() << "\n";
      h = next;
    }
  }

  void emitLMHead(std::string &h, std::string &logits) {
    for (const std::string &opName : facts.lm_head_ops) {
      std::string next = fresh("logits");
      os << "    %" << next << " = \"llm." << opName << "\"(%" << h << ")"
         // lm_head_proj is a linear weight-bearing op; stamp the explicit
         // semantic marker planning passes key off of (see isLinearOpName).
         << " {serving.layer_role = \"lm_head\", serving.quantizable = true}"
         << " : (" << hiddenTy() << ") -> " << logitsTy() << "\n";
      h = next;
      logits = next;
    }
  }

  std::string hiddenTy() const {
    return "tensor<" + seqDim + "x" + std::to_string(hs) + "xf16>";
  }
  std::string kvTy() const {
    return "tensor<" + seqDim + "x" + std::to_string(kvDim) + "xf16>";
  }
  std::string scoresTy() const {
    return "tensor<" + seqDim + "x?xf16>";
  }
  std::string logitsTy() const {
    return "tensor<" + seqDim + "x" + std::to_string(facts.vocab_size) + "xf16>";
  }

private:
  std::string fresh(llvm::StringRef base) {
    return (base + "_" + std::to_string(counter++)).str();
  }

  // quantizable stamps serving.quantizable = true on linear weight-bearing
  // ops (q/k/v/o_proj, mlp) so downstream planning passes (Weight
  // Classification, Quantization Strategy) can key off an explicit semantic
  // attribute instead of pattern-matching the op name.
  std::string layerAttrs(int64_t layerIndex, bool quantizable) const {
    std::string s = "{serving.layer_index = " + std::to_string(layerIndex) +
                     " : i64, serving.layer_role = \"decoder_layer\"";
    if (quantizable)
      s += ", serving.quantizable = true";
    s += "}";
    return s;
  }

  void emitOne(const std::string &opName, int64_t layerIndex, std::string &h,
               std::string &q, std::string &k, std::string &v,
               std::string &scores, std::string &probs, std::string &attn,
               std::string &ck, std::string &cv) {
    const std::string attrs = layerAttrs(layerIndex, /*quantizable=*/false);
    // q/k/v/o_proj and mlp are linear weight-bearing ops; stamp the explicit
    // semantic marker planning passes key off of instead of the op name.
    const std::string linearAttrs = layerAttrs(layerIndex, /*quantizable=*/true);
    if (opName == "rmsnorm") {
      std::string next = fresh("norm");
      os << "    %" << next << " = \"llm.rmsnorm\"(%" << h << ") " << attrs
         << " : (" << hiddenTy() << ") -> " << hiddenTy() << "\n";
      h = next;
    } else if (opName == "q_proj") {
      q = fresh("q");
      os << "    %" << q << " = \"llm.q_proj\"(%" << h << ") " << linearAttrs
         << " : (" << hiddenTy() << ") -> " << hiddenTy() << "\n";
    } else if (opName == "k_proj") {
      k = fresh("k");
      os << "    %" << k << " = \"llm.k_proj\"(%" << h << ") " << linearAttrs
         << " : (" << hiddenTy() << ") -> " << kvTy() << "\n";
    } else if (opName == "v_proj") {
      v = fresh("v");
      os << "    %" << v << " = \"llm.v_proj\"(%" << h << ") " << linearAttrs
         << " : (" << hiddenTy() << ") -> " << kvTy() << "\n";
    } else if (opName == "attention_scores") {
      scores = fresh("scores");
      // Decode attends against the cache-sourced key when present; prefill
      // attends against the freshly-computed key. Symbolic wiring only, no
      // real numerics flow through this planning-only compiler.
      std::string keyOperand = (!isPrefill && !ck.empty()) ? ck : k;
      os << "    %" << scores << " = \"llm.attention_scores\"(%" << q << ", %"
         << keyOperand << ") " << attrs << " : (" << hiddenTy() << ", "
         << kvTy() << ") -> " << scoresTy() << "\n";
    } else if (opName == "softmax") {
      probs = fresh("probs");
      os << "    %" << probs << " = \"llm.softmax\"(%" << scores << ") "
         << attrs << " : (" << scoresTy() << ") -> " << scoresTy() << "\n";
    } else if (opName == "attention_output") {
      attn = fresh("attn");
      std::string valueOperand = (!isPrefill && !cv.empty()) ? cv : v;
      os << "    %" << attn << " = \"llm.attention_output\"(%" << probs
         << ", %" << valueOperand << ") " << attrs << " : (" << scoresTy()
         << ", " << kvTy() << ") -> " << hiddenTy() << "\n";
    } else if (opName == "kv_cache_boundary") {
      if (isPrefill) {
        os << "    \"llm.kv_cache_write\"(%" << k << ", %" << v << ") "
           << attrs << " : (" << kvTy() << ", " << kvTy() << ") -> ()\n";
      } else {
        ck = fresh("ck");
        cv = fresh("cv");
        os << "    %" << ck << ", %" << cv << " = \"llm.kv_cache_read\"(%"
           << q << ") " << attrs << " : (" << hiddenTy() << ") -> (" << kvTy()
           << ", " << kvTy() << ")\n";
      }
    } else if (opName == "o_proj") {
      std::string next = fresh("oproj");
      os << "    %" << next << " = \"llm.o_proj\"(%" << attn << ") " << linearAttrs
         << " : (" << hiddenTy() << ") -> " << hiddenTy() << "\n";
      h = next;
    } else if (opName == "mlp") {
      std::string next = fresh("mlp");
      os << "    %" << next << " = \"llm.mlp\"(%" << h << ") " << linearAttrs
         << " : (" << hiddenTy() << ") -> " << hiddenTy() << "\n";
      h = next;
    }
    // Unknown op names are rejected earlier in parseGraphFacts; nothing
    // reaches this function without being one of the known cases above.
  }

  llvm::raw_ostream &os;
  const GraphFacts &facts;
  bool isPrefill;
  int64_t hs;
  int64_t kvDim;
  std::string seqDim;
  int64_t counter = 0;
};

} // namespace

static void emitPhaseFunction(llvm::raw_ostream &os, const GraphFacts &facts,
                               bool isPrefill) {
  LayerEmitter emitter(os, facts, isPrefill);
  llvm::StringRef fnName = isPrefill ? "qwen_prefill" : "qwen_decode";

  os << "  func.func @" << fnName << "(%tokens: tensor<"
     << (isPrefill ? "?" : "1") << "xi32>) -> " << emitter.logitsTy()
     << " {\n";

  std::string h;
  emitter.emitEmbedding(h);
  for (int64_t layer = 0; layer < facts.num_layers; ++layer)
    emitter.emitDecoderLayer(layer, h);
  emitter.emitFinalNorm(h);
  std::string logits;
  emitter.emitLMHead(h, logits);

  os << "    return %" << logits << " : " << emitter.logitsTy() << "\n";
  os << "  }\n";
}

static void emitMLIR(const GraphFacts &facts, llvm::raw_ostream &os) {
  os << "// Generated by qwen-onnx-to-serving-mlir from ONNX-shaped graph facts.\n"
     << "// Truth boundary: " << facts.truth_boundary << "\n"
     << "// Source: " << facts.source << "\n"
     << "// Do not edit by hand; regenerate with:\n"
     << "//   qwen-onnx-to-serving-mlir --graph-facts "
        "configs/models/qwen_0_5b_onnx_graph_facts.json --out <path>\n"
     << "//\n"
     << "// This is RAW (pre-normalization) serving MLIR. Run\n"
     << "// llm-frontend-normalization (LLMFrontendNormalizationPass) before\n"
     << "// compile-for-target -- see this tool's main.cpp header comment.\n"
     << "//\n"
     << "// NOT implemented in this frontend:\n"
     << "//   - Real ONNX protobuf parsing (this reads a hand-authored fixture)\n"
     << "//   - HuggingFace weight import\n"
     << "//   - Residual adds, RoPE as a distinct op\n"
     << "// This tool emits a full per-layer-expanded, flat, unrolled graph\n"
     << "// from declared ONNX-shaped graph facts -- not a single hand-\n"
     << "// templated block reused for every layer, unlike qwen-to-serving-mlir.\n"
     << "\n";

  os << "module attributes {\n"
     << "  llm.model = \"" << facts.model_name << "\",\n"
     << "  llm.num_layers = " << facts.num_layers << " : i64,\n"
     << "  llm.hidden_size = " << facts.hidden_size << " : i64,\n"
     << "  llm.num_attention_heads = " << facts.num_attention_heads << " : i64,\n"
     << "  llm.num_key_value_heads = " << facts.num_key_value_heads << " : i64,\n"
     << "  llm.intermediate_size = " << facts.intermediate_size << " : i64,\n"
     << "  llm.vocab_size = " << facts.vocab_size << " : i64,\n"
     << "  llm.max_position_embeddings = " << facts.max_position_embeddings << " : i64,\n"
     << "  llm.dtype = \"" << facts.dtype << "\",\n"
     << "  llm.frontend_source = \"qwen_onnx_graph_import\",\n"
     << "  llm.truth_boundary = \"" << facts.truth_boundary << "\"\n"
     << "} {\n"
     << "\n";

  emitPhaseFunction(os, facts, /*isPrefill=*/true);
  os << "\n";
  emitPhaseFunction(os, facts, /*isPrefill=*/false);
  os << "}\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "qwen-onnx-to-serving-mlir: emit full per-layer-expanded serving MLIR "
      "from ONNX-shaped graph facts (Phase 1 scaffold; not a real ONNX "
      "protobuf frontend)\n");

  auto factsOrErr = parseGraphFacts(GraphFactsPath);
  if (!factsOrErr) {
    llvm::errs() << "error: " << llvm::toString(factsOrErr.takeError()) << "\n";
    return 1;
  }
  GraphFacts facts = std::move(*factsOrErr);

  {
    llvm::SmallString<256> parent(OutPath.getValue());
    llvm::sys::path::remove_filename(parent);
    if (!parent.empty()) {
      if (std::error_code ec = llvm::sys::fs::create_directories(
              parent, /*IgnoreExisting=*/true)) {
        llvm::errs() << "error: cannot create output directory '" << parent
                     << "': " << ec.message() << "\n";
        return 1;
      }
    }
  }

  std::error_code ec;
  llvm::raw_fd_ostream os(OutPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot open output file '" << OutPath
                 << "': " << ec.message() << "\n";
    return 1;
  }

  emitMLIR(facts, os);

  llvm::outs() << "qwen-onnx-to-serving-mlir: wrote " << OutPath << "\n"
               << "  model:        " << facts.model_name << "\n"
               << "  num_layers:   " << facts.num_layers << " (full expansion)\n"
               << "  hidden_size:  " << facts.hidden_size << "\n"
               << "  truth_boundary: " << facts.truth_boundary << "\n";

  return 0;
}
