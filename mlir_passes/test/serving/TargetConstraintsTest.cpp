// CTest unit test for TargetConstraints fromModule / attachToModule.
// Pure C++. No GoogleTest. No Python. No JSON. No plugin loading.

#include "serving/TargetConstraints.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <cassert>
#include <cstdio>

// Helper: parse a module string and assert success.
static mlir::OwningOpRef<mlir::ModuleOp>
parseModule(const char *src, mlir::MLIRContext &ctx) {
  auto m = mlir::parseSourceString<mlir::ModuleOp>(src, &ctx);
  assert(m && "MLIR parse failed");
  return m;
}

// ---------------------------------------------------------------------------
// Case 1: module with no target.* attrs → all fields at defaults,
// all presence flags false.
// ---------------------------------------------------------------------------
static void testUnconstrained() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  auto m = parseModule(R"mlir(
    module attributes { llm.model = "tiny-gpt" } {}
  )mlir", ctx);

  mlir::hir::TargetConstraints tc =
      mlir::hir::TargetConstraints::fromModule(m.get());

  assert(tc.profile_id.empty()                      && "profile_id should be empty");
  assert(!tc.has_memory_budget                      && "has_memory_budget should be false");
  assert(tc.memory_budget_mb == 0.0                 && "memory_budget_mb should be 0");
  assert(!tc.has_static_shape_support               && "has_static_shape_support should be false");
  assert(tc.static_shape_support == true            && "static_shape_support default is true");
  assert(!tc.has_frame_latency_budget               && "has_frame_latency_budget should be false");
  assert(tc.preferred_backend.empty()               && "preferred_backend should be empty");
  assert(tc.allowed_backends.empty()                && "allowed_backends should be empty");
  assert(tc.supported_precisions.empty()            && "supported_precisions should be empty");
  assert(tc.paged_kv_compatible_backends.empty()    && "paged_kv_compatible_backends should be empty when attr absent");

  std::puts("  [PASS] testUnconstrained");
}

// ---------------------------------------------------------------------------
// Case 2: module with profile_id + memory_budget + static_shape_support=false.
// Matches the target_constraints.mlir FileCheck fixture.
// ---------------------------------------------------------------------------
static void testConstrainedModule() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  auto m = parseModule(R"mlir(
    module attributes {
      target.memory_budget_mb = 5.000000e+00 : f64,
      target.profile_id = "apple-a17pro-mobile",
      target.static_shape_support = false
    } {}
  )mlir", ctx);

  mlir::hir::TargetConstraints tc =
      mlir::hir::TargetConstraints::fromModule(m.get());

  assert(tc.profile_id == "apple-a17pro-mobile" && "profile_id mismatch");
  assert(tc.has_memory_budget                    && "has_memory_budget should be true");
  assert(tc.memory_budget_mb == 5.0              && "memory_budget_mb mismatch");
  assert(tc.has_static_shape_support             && "has_static_shape_support should be true");
  assert(tc.static_shape_support == false        && "static_shape_support should be false");
  assert(!tc.has_frame_latency_budget            && "has_frame_latency_budget should be false");
  assert(tc.allowed_backends.empty()             && "allowed_backends should be empty");

  std::puts("  [PASS] testConstrainedModule");
}

// ---------------------------------------------------------------------------
// Case 3: module with all fields including list attrs.
// ---------------------------------------------------------------------------
static void testFullProfile() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  auto m = parseModule(R"mlir(
    module attributes {
      target.profile_id = "cuda-a100",
      target.memory_budget_mb = 4.000000e+04 : f64,
      target.static_shape_support = true,
      target.frame_latency_budget_ms = 3.300000e+01 : f64,
      target.preferred_backend = "cuda",
      target.allowed_backends = ["cuda", "cpu"],
      target.supported_precisions = ["fp32", "fp16", "bf16"],
      target.paged_kv_compatible_backends = ["cuda", "vllm"]
    } {}
  )mlir", ctx);

  mlir::hir::TargetConstraints tc =
      mlir::hir::TargetConstraints::fromModule(m.get());

  assert(tc.profile_id == "cuda-a100"           && "profile_id mismatch");
  assert(tc.has_memory_budget                    && "has_memory_budget should be true");
  assert(tc.memory_budget_mb == 40000.0          && "memory_budget_mb mismatch");
  assert(tc.has_static_shape_support             && "has_static_shape_support should be true");
  assert(tc.static_shape_support == true         && "static_shape_support mismatch");
  assert(tc.has_frame_latency_budget             && "has_frame_latency_budget should be true");
  assert(tc.frame_latency_budget_ms == 33.0      && "frame_latency_budget_ms mismatch");
  assert(tc.preferred_backend == "cuda"          && "preferred_backend mismatch");

  assert(tc.allowed_backends.size() == 2         && "allowed_backends size mismatch");
  assert(tc.allowed_backends[0] == "cuda"        && "allowed_backends[0] mismatch");
  assert(tc.allowed_backends[1] == "cpu"         && "allowed_backends[1] mismatch");

  assert(tc.supported_precisions.size() == 3     && "supported_precisions size mismatch");
  assert(tc.supported_precisions[0] == "fp32"    && "supported_precisions[0] mismatch");
  assert(tc.supported_precisions[1] == "fp16"    && "supported_precisions[1] mismatch");
  assert(tc.supported_precisions[2] == "bf16"    && "supported_precisions[2] mismatch");

  assert(tc.paged_kv_compatible_backends.size() == 2   && "paged_kv_compatible_backends size mismatch");
  assert(tc.paged_kv_compatible_backends[0] == "cuda"  && "paged_kv_compatible_backends[0] mismatch");
  assert(tc.paged_kv_compatible_backends[1] == "vllm"  && "paged_kv_compatible_backends[1] mismatch");

  std::puts("  [PASS] testFullProfile");
}

// ---------------------------------------------------------------------------
// Case 4: roundtrip — attachToModule then fromModule should recover the same values.
// ---------------------------------------------------------------------------
static void testRoundtrip() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);
  auto m = parseModule("module {}", ctx);

  mlir::hir::TargetConstraints original;
  original.profile_id              = "apple-m4-pro";
  original.memory_budget_mb        = 8192.0;
  original.has_memory_budget       = true;
  original.static_shape_support    = false;
  original.has_static_shape_support = true;
  original.frame_latency_budget_ms = 16.67;
  original.has_frame_latency_budget = true;
  original.preferred_backend              = "metal";
  original.allowed_backends              = {"metal", "cpu"};
  original.supported_precisions          = {"fp16", "int8"};
  original.paged_kv_compatible_backends  = {"metal"};

  original.attachToModule(m.get(), &ctx);

  mlir::hir::TargetConstraints recovered =
      mlir::hir::TargetConstraints::fromModule(m.get());

  assert(recovered.profile_id == original.profile_id           && "roundtrip: profile_id");
  assert(recovered.has_memory_budget                            && "roundtrip: has_memory_budget");
  assert(recovered.memory_budget_mb == original.memory_budget_mb && "roundtrip: memory_budget_mb");
  assert(recovered.has_static_shape_support                     && "roundtrip: has_static_shape_support");
  assert(recovered.static_shape_support == original.static_shape_support
         && "roundtrip: static_shape_support");
  assert(recovered.has_frame_latency_budget                     && "roundtrip: has_frame_latency_budget");
  assert(recovered.frame_latency_budget_ms == original.frame_latency_budget_ms
         && "roundtrip: frame_latency_budget_ms");
  assert(recovered.preferred_backend == original.preferred_backend
         && "roundtrip: preferred_backend");
  assert(recovered.allowed_backends == original.allowed_backends
         && "roundtrip: allowed_backends");
  assert(recovered.supported_precisions == original.supported_precisions
         && "roundtrip: supported_precisions");
  assert(recovered.paged_kv_compatible_backends == original.paged_kv_compatible_backends
         && "roundtrip: paged_kv_compatible_backends");

  std::puts("  [PASS] testRoundtrip");
}

// ---------------------------------------------------------------------------
// Case 5: target.paged_kv_compatible_backends — explicit empty vs populated.
// Validates that paged KV compatibility is profile-driven, not hardcoded.
// ---------------------------------------------------------------------------
static void testPagedKVCompatibleBackends() {
  mlir::MLIRContext ctx;
  ctx.allowUnregisteredDialects(true);

  // 5a: Empty list written explicitly by attachToModule, recovered as empty.
  {
    auto m = parseModule("module {}", ctx);
    mlir::hir::TargetConstraints tc;
    tc.paged_kv_compatible_backends = {}; // Apple A17 Pro: no paged-KV backend yet.
    tc.attachToModule(m.get(), &ctx);

    mlir::hir::TargetConstraints recovered =
        mlir::hir::TargetConstraints::fromModule(m.get());
    assert(recovered.paged_kv_compatible_backends.empty()
           && "5a: empty paged_kv_compatible_backends should roundtrip as empty");
  }

  // 5b: Populated list (hypothetical CUDA target).
  {
    auto m = parseModule(R"mlir(
      module attributes {
        target.paged_kv_compatible_backends = ["cuda", "vllm"]
      } {}
    )mlir", ctx);
    mlir::hir::TargetConstraints tc =
        mlir::hir::TargetConstraints::fromModule(m.get());
    assert(tc.paged_kv_compatible_backends.size() == 2
           && "5b: should read 2 paged-KV backends");
    assert(tc.paged_kv_compatible_backends[0] == "cuda"
           && "5b: paged_kv_compatible_backends[0] should be cuda");
    assert(tc.paged_kv_compatible_backends[1] == "vllm"
           && "5b: paged_kv_compatible_backends[1] should be vllm");
  }

  // 5c: Apple-style target with metal as paged-KV-capable backend.
  {
    auto m = parseModule(R"mlir(
      module attributes {
        target.paged_kv_compatible_backends = ["metal"]
      } {}
    )mlir", ctx);
    mlir::hir::TargetConstraints tc =
        mlir::hir::TargetConstraints::fromModule(m.get());
    assert(tc.paged_kv_compatible_backends.size() == 1
           && "5c: should read 1 paged-KV backend");
    assert(tc.paged_kv_compatible_backends[0] == "metal"
           && "5c: paged_kv_compatible_backends[0] should be metal");
  }

  std::puts("  [PASS] testPagedKVCompatibleBackends");
}

int main() {
  std::puts("TargetConstraintsTest:");
  testUnconstrained();
  testConstrainedModule();
  testFullProfile();
  testRoundtrip();
  testPagedKVCompatibleBackends();
  std::puts("TargetConstraintsTest: PASS");
  return 0;
}
