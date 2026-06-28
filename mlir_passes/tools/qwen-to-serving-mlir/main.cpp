// qwen-to-serving-mlir: Qwen model-spec frontend.
//
// Reads a QwenModelSpec JSON and emits serving-aware MLIR text.
// Does NOT import weights, lower operators, or perform graph recognition.
// Truth boundary: declared model config -> serving-aware IR only.
//
// The emitted MLIR contains:
//   - module attrs (llm.model, llm.num_layers, llm.hidden_size,
//     llm.num_attention_heads, llm.num_key_value_heads, ...)
//   - @qwen_prefill with llm.attention_prefill (kv_cache.role = "producer")
//   - @qwen_decode  with llm.attention_decode  (kv_cache.role = "consumer")
//
// The output feeds directly into compile-for-target to produce a
// ServingExecutionPlan JSON artifact.
//
// Usage:
//   qwen-to-serving-mlir --model-spec configs/models/qwen_0_5b_spec.json \
//                         --out mlir/qwen_0_5b_serving.mlir

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> ModelSpecPath(
    "model-spec",
    llvm::cl::desc("Path to QwenModelSpec JSON"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> OutPath(
    "out",
    llvm::cl::desc("Output MLIR file path"),
    llvm::cl::Required);

// ---------------------------------------------------------------------------
// QwenModelSpec
// ---------------------------------------------------------------------------

struct QwenModelSpec {
  std::string model_name;
  std::string truth_boundary;
  std::string dtype;
  int64_t num_layers              = 0;
  int64_t hidden_size             = 0;
  int64_t intermediate_size       = 0;
  int64_t num_attention_heads     = 0;
  int64_t num_key_value_heads     = 0;
  int64_t max_position_embeddings = 0;
  int64_t vocab_size              = 0;
};

// ---------------------------------------------------------------------------
// JSON parsing
// ---------------------------------------------------------------------------

static llvm::Expected<QwenModelSpec> parseSpec(llvm::StringRef path) {
  auto buf = llvm::MemoryBuffer::getFile(path);
  if (!buf)
    return llvm::make_error<llvm::StringError>(
        "cannot read model spec '" + path.str() + "': " +
            buf.getError().message(),
        llvm::inconvertibleErrorCode());

  auto json = llvm::json::parse((*buf)->getBuffer());
  if (!json)
    return llvm::make_error<llvm::StringError>(
        "JSON parse error in model spec '" + path.str() + "': " +
            llvm::toString(json.takeError()),
        llvm::inconvertibleErrorCode());

  const llvm::json::Object *obj = json->getAsObject();
  if (!obj)
    return llvm::make_error<llvm::StringError>(
        "model spec must be a JSON object",
        llvm::inconvertibleErrorCode());

  QwenModelSpec spec;

  auto requireString = [&](llvm::StringRef key)
      -> llvm::Expected<std::string> {
    if (auto v = obj->getString(key))
      return v->str();
    return llvm::make_error<llvm::StringError>(
        "model spec missing required field '" + key.str() + "'",
        llvm::inconvertibleErrorCode());
  };
  auto requireInt = [&](llvm::StringRef key)
      -> llvm::Expected<int64_t> {
    if (auto v = obj->getInteger(key))
      return *v;
    return llvm::make_error<llvm::StringError>(
        "model spec missing required integer field '" + key.str() + "'",
        llvm::inconvertibleErrorCode());
  };

  if (auto r = requireString("model_name"))    spec.model_name    = *r; else return r.takeError();
  if (auto r = requireString("truth_boundary"))spec.truth_boundary = *r; else return r.takeError();
  if (auto r = requireString("dtype"))         spec.dtype         = *r; else return r.takeError();
  if (auto r = requireInt("num_layers"))              spec.num_layers              = *r; else return r.takeError();
  if (auto r = requireInt("hidden_size"))             spec.hidden_size             = *r; else return r.takeError();
  if (auto r = requireInt("intermediate_size"))       spec.intermediate_size       = *r; else return r.takeError();
  if (auto r = requireInt("num_attention_heads"))     spec.num_attention_heads     = *r; else return r.takeError();
  if (auto r = requireInt("num_key_value_heads"))     spec.num_key_value_heads     = *r; else return r.takeError();
  if (auto r = requireInt("max_position_embeddings")) spec.max_position_embeddings = *r; else return r.takeError();
  if (auto r = requireInt("vocab_size"))              spec.vocab_size              = *r; else return r.takeError();

  if (spec.num_attention_heads <= 0)
    return llvm::make_error<llvm::StringError>(
        "num_attention_heads must be > 0",
        llvm::inconvertibleErrorCode());
  if (spec.hidden_size % spec.num_attention_heads != 0)
    return llvm::make_error<llvm::StringError>(
        "hidden_size must be divisible by num_attention_heads",
        llvm::inconvertibleErrorCode());

  return spec;
}

// ---------------------------------------------------------------------------
// MLIR text emission
// ---------------------------------------------------------------------------

static void emitMLIR(const QwenModelSpec &spec, llvm::raw_ostream &os) {
  const int64_t head_dim    = spec.hidden_size / spec.num_attention_heads;
  const int64_t kv_proj_dim = spec.num_key_value_heads * head_dim;
  const int64_t hs          = spec.hidden_size;

  os << "// Generated by qwen-to-serving-mlir from declared model config.\n"
     << "// Truth boundary: " << spec.truth_boundary << "\n"
     << "// Do not edit by hand; regenerate with:\n"
     << "//   qwen-to-serving-mlir --model-spec configs/models/qwen_0_5b_spec.json\n"
     << "//                         --out mlir/qwen_0_5b_serving.mlir\n"
     << "//\n"
     << "// NOT implemented in this frontend:\n"
     << "//   - HuggingFace weight import\n"
     << "//   - Full operator graph (QKV matmul, softmax, RoPE, MLP)\n"
     << "//   - PyTorch / StableHLO / ONNX graph recognition\n"
     << "// This tool emits serving-aware IR from declared architecture constants.\n"
     << "\n";

  os << "module attributes {\n"
     << "  llm.model = \"" << spec.model_name << "\",\n"
     << "  llm.num_layers = " << spec.num_layers << " : i64,\n"
     << "  llm.hidden_size = " << hs << " : i64,\n"
     << "  llm.num_attention_heads = " << spec.num_attention_heads << " : i64,\n"
     << "  llm.num_key_value_heads = " << spec.num_key_value_heads << " : i64,\n"
     << "  llm.intermediate_size = " << spec.intermediate_size << " : i64,\n"
     << "  llm.vocab_size = " << spec.vocab_size << " : i64,\n"
     << "  llm.max_position_embeddings = " << spec.max_position_embeddings << " : i64,\n"
     << "  llm.dtype = \"" << spec.dtype << "\",\n"
     << "  llm.frontend_source = \"qwen_model_spec\",\n"
     << "  llm.truth_boundary = \"" << spec.truth_boundary << "\"\n"
     << "} {\n"
     << "\n";

  // Prefill function
  os << "  func.func @qwen_prefill(%tokens: tensor<?xi32>) -> tensor<?x" << hs << "xf16> {\n"
     << "    %0 = \"llm.embed\"(%tokens) : (tensor<?xi32>) -> tensor<?x" << hs << "xf16>\n"
     << "    %1 = \"llm.rmsnorm\"(%0) : (tensor<?x" << hs << "xf16>) -> tensor<?x" << hs << "xf16>\n"
     << "    %q, %k, %v = \"llm.qkv_projection\"(%1)"
     << " : (tensor<?x" << hs << "xf16>)"
     << " -> (tensor<?x" << hs << "xf16>, tensor<?x" << kv_proj_dim << "xf16>, tensor<?x" << kv_proj_dim << "xf16>)\n"
     << "    %2 = \"llm.attention_prefill\"(%q, %k, %v) {\n"
     << "        kv_cache.role = \"producer\",\n"
     << "        serving.phase = \"prefill\",\n"
     << "        serving.prompt_tokens = 512 : i64,\n"
     << "        serving.output_tokens = 0 : i64,\n"
     << "        llm.num_q_heads = " << spec.num_attention_heads << " : i64,\n"
     << "        llm.num_kv_heads = " << spec.num_key_value_heads << " : i64\n"
     << "    } : (tensor<?x" << hs << "xf16>, tensor<?x" << kv_proj_dim << "xf16>, tensor<?x" << kv_proj_dim << "xf16>)"
     << " -> tensor<?x" << hs << "xf16>\n"
     << "    %3 = \"llm.mlp\"(%2) : (tensor<?x" << hs << "xf16>) -> tensor<?x" << hs << "xf16>\n"
     << "    return %3 : tensor<?x" << hs << "xf16>\n"
     << "  }\n"
     << "\n";

  // Decode function
  os << "  func.func @qwen_decode(%token: tensor<1xi32>) -> tensor<1x" << hs << "xf16> {\n"
     << "    %0 = \"llm.embed\"(%token) : (tensor<1xi32>) -> tensor<1x" << hs << "xf16>\n"
     << "    %1 = \"llm.rmsnorm\"(%0) : (tensor<1x" << hs << "xf16>) -> tensor<1x" << hs << "xf16>\n"
     << "    %q, %k, %v = \"llm.qkv_projection\"(%1)"
     << " : (tensor<1x" << hs << "xf16>)"
     << " -> (tensor<1x" << hs << "xf16>, tensor<1x" << kv_proj_dim << "xf16>, tensor<1x" << kv_proj_dim << "xf16>)\n"
     << "    %2 = \"llm.attention_decode\"(%q, %k, %v) {\n"
     << "        kv_cache.role = \"consumer\",\n"
     << "        serving.phase = \"decode\",\n"
     << "        serving.prompt_tokens = 0 : i64,\n"
     << "        serving.output_tokens = 1 : i64,\n"
     << "        llm.num_q_heads = " << spec.num_attention_heads << " : i64,\n"
     << "        llm.num_kv_heads = " << spec.num_key_value_heads << " : i64\n"
     << "    } : (tensor<1x" << hs << "xf16>, tensor<1x" << kv_proj_dim << "xf16>, tensor<1x" << kv_proj_dim << "xf16>)"
     << " -> tensor<1x" << hs << "xf16>\n"
     << "    %3 = \"llm.mlp\"(%2) : (tensor<1x" << hs << "xf16>) -> tensor<1x" << hs << "xf16>\n"
     << "    return %3 : tensor<1x" << hs << "xf16>\n"
     << "  }\n"
     << "}\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "qwen-to-serving-mlir: emit serving-aware MLIR from Qwen model spec\n");

  auto specOrErr = parseSpec(ModelSpecPath);
  if (!specOrErr) {
    llvm::errs() << "error: " << llvm::toString(specOrErr.takeError()) << "\n";
    return 1;
  }
  QwenModelSpec spec = std::move(*specOrErr);

  // Create parent directory if needed.
  {
    llvm::SmallString<256> parent(OutPath.getValue());
    llvm::sys::path::remove_filename(parent);
    if (!parent.empty()) {
      if (std::error_code ec = llvm::sys::fs::create_directories(
              parent, /*IgnoreExisting=*/true)) {
        llvm::errs() << "error: cannot create output directory '"
                     << parent << "': " << ec.message() << "\n";
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

  emitMLIR(spec, os);

  const int64_t head_dim    = spec.hidden_size / spec.num_attention_heads;
  const int64_t kv_proj_dim = spec.num_key_value_heads * head_dim;

  llvm::outs() << "qwen-to-serving-mlir: wrote " << OutPath << "\n"
               << "  model:               " << spec.model_name << "\n"
               << "  layers:              " << spec.num_layers << "\n"
               << "  hidden_size:         " << spec.hidden_size << "\n"
               << "  num_attention_heads: " << spec.num_attention_heads << "\n"
               << "  num_kv_heads:        " << spec.num_key_value_heads
               << "  (GQA: " << spec.num_attention_heads << "q/"
               << spec.num_key_value_heads << "kv, "
               << spec.num_attention_heads / spec.num_key_value_heads
               << ":1 ratio)\n"
               << "  kv_proj_dim:         " << kv_proj_dim
               << "  (head_dim=" << head_dim << " * " << spec.num_key_value_heads << " kv heads)\n"
               << "  truth_boundary:      " << spec.truth_boundary << "\n";

  return 0;
}
