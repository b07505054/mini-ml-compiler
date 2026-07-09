#pragma once

// OpShapeFacts.h — MLIR-side derivation of ShapeFacts and declared profile
// numbers, shared by ServingCostModelPass (shape_cost_model_v2) and
// TilePlanningPass.
//
// The pure formulas live in serving/ShapeCostModel.h (MLIR-free); this
// header owns everything that reads MLIR types and attrs. Moved out of
// ServingCostModelPass.cpp when TilePlanningPass became a second consumer.

#include "serving/ShapeCostModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include <initializer_list>
#include <string>

namespace mlir::hir {

// MLIR element type -> planner dtype string ("" when unrecognized).
inline std::string dtypeNameOf(Type elem) {
  if (elem.isF32())       return "fp32";
  if (elem.isF16())       return "fp16";
  if (elem.isBF16())      return "bf16";
  if (elem.isInteger(8))  return "int8";
  if (elem.isInteger(4))  return "int4";
  return "";
}

inline bool nameContainsAny(StringRef name,
                            std::initializer_list<StringRef> fragments) {
  for (StringRef f : fragments)
    if (name.contains(f)) return true;
  return false;
}

// Classify the op into a cost/tile kind. Mirrors the quantizable-op signal
// used by QuantizationStrategyPlanningPass: the explicit serving.quantizable
// marker wins; name fragments are the generic fallback.
inline std::string classifyOpKind(Operation& op) {
  StringRef full = op.getName().getStringRef();
  StringRef name = full;
  if (auto dot = full.find('.'); dot != StringRef::npos)
    name = full.substr(dot + 1);

  if (auto a = op.getAttrOfType<BoolAttr>("serving.quantizable");
      a && a.getValue())
    return "matmul_like";
  if (nameContainsAny(name, {"norm"}))
    return "normalization";
  if (nameContainsAny(name, {"matmul", "gemm", "linear", "proj", "mlp",
                             "lm_head", "conv", "qkv"}))
    return "matmul_like";
  if (nameContainsAny(name, {"add", "mul", "relu", "gelu", "silu",
                             "sigmoid", "softmax"}))
    return "elementwise";
  return "unknown";
}

// Static element count of a ranked tensor; -1 when dynamic/unranked.
inline int64_t staticElems(Type t) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt || !rt.hasStaticShape()) return -1;
  int64_t n = 1;
  for (int64_t d : rt.getShape()) n *= d;
  return n;
}

// Derive dtype-independent shape facts for one op. See ShapeCostModel.h for
// the per-kind formulas and their documented approximations.
inline ShapeFacts computeShapeFacts(Operation& op) {
  ShapeFacts facts;
  facts.op_kind = classifyOpKind(op);
  if (facts.op_kind == "unknown") {
    facts.status = "unsupported_op_kind";
    return facts;
  }
  if (op.getNumResults() == 0 ||
      !isa<RankedTensorType>(op.getResult(0).getType())) {
    facts.status = "no_tensor_result";
    return facts;
  }
  auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
  int64_t outElems = staticElems(resultType);
  if (outElems < 0) {
    facts.status = "dynamic_dims_unresolved";
    return facts;
  }
  facts.output_elems = outElems;

  // First ranked tensor operand = the activation input.
  RankedTensorType inputType;
  int64_t inputElemsSum = 0;
  bool anyDynamicOperand = false;
  for (Value operand : op.getOperands()) {
    auto rt = dyn_cast<RankedTensorType>(operand.getType());
    if (!rt) continue;
    int64_t n = staticElems(rt);
    if (n < 0) { anyDynamicOperand = true; break; }
    if (!inputType) inputType = rt;
    inputElemsSum += n;
  }
  if (anyDynamicOperand) {
    facts.status = "dynamic_dims_unresolved";
    return facts;
  }

  if (facts.op_kind == "matmul_like") {
    // 2*M*K*N: K from the input's last dim, N from the result's last dim,
    // M from the result's leading dims. Weight elems = K*N — projection-op
    // weights are typically not IR operands, so the weight shape is
    // inferred, not read.
    if (!inputType || inputType.getRank() < 1 || resultType.getRank() < 1) {
      facts.status = "no_tensor_result";
      return facts;
    }
    int64_t k = inputType.getShape().back();
    int64_t n = resultType.getShape().back();
    int64_t m = n > 0 ? facts.output_elems / n : 0;
    facts.m = m;
    facts.n = n;
    facts.k = k;
    facts.input_elems  = staticElems(inputType);
    facts.weight_elems = k * n;
    facts.flops        = 2 * m * k * n;
  } else if (facts.op_kind == "normalization") {
    facts.input_elems = inputType ? staticElems(inputType) : facts.output_elems;
    facts.flops       = 4 * facts.output_elems;
  } else { // elementwise
    facts.input_elems = inputElemsSum > 0 ? inputElemsSum : facts.output_elems;
    facts.flops       = facts.output_elems;
  }
  facts.status = "static_shapes";
  return facts;
}

// Read declared target.static_cost_profile.* module attrs (absent -> 0 /
// false). All values are declared theoretical numbers — never measured.
inline StaticCostProfileNums readProfileNums(Operation* module) {
  StaticCostProfileNums nums;
  if (!module) return nums;
  auto rF = [&](StringRef key) -> double {
    if (auto a = module->getAttrOfType<FloatAttr>(key))
      return a.getValueAsDouble();
    return 0.0;
  };
  auto rI = [&](StringRef key) -> int64_t {
    if (auto a = module->getAttrOfType<IntegerAttr>(key))
      return a.getInt();
    return 0;
  };
  auto rB = [&](StringRef key) -> bool {
    if (auto a = module->getAttrOfType<BoolAttr>(key))
      return a.getValue();
    return false;
  };
  nums.peak_flops_fp32 = rF("target.static_cost_profile.peak_flops_fp32");
  nums.peak_flops_fp16 = rF("target.static_cost_profile.peak_flops_fp16");
  nums.peak_flops_int8 = rF("target.static_cost_profile.peak_flops_int8");
  nums.mem_bandwidth_bytes_per_sec =
      rF("target.static_cost_profile.memory_bandwidth_bytes_per_sec");
  nums.local_memory_bytes =
      rI("target.static_cost_profile.local_memory_bytes");
  nums.cache_line_bytes = rI("target.static_cost_profile.cache_line_bytes");
  nums.supports_async_copy =
      rB("target.static_cost_profile.supports_async_copy");
  nums.supports_dma = rB("target.static_cost_profile.supports_dma");
  return nums;
}

// Resolve an op's activation/output dtype from existing metadata:
// quant.activation_dtype -> result element type -> effective dtype.
inline std::string resolveOpActivationDtype(Operation& op,
                                            const std::string& effectiveDtype) {
  if (auto a = op.getAttrOfType<StringAttr>("quant.activation_dtype"))
    if (dtypeBits(a.getValue().str()) > 0) return a.getValue().str();
  if (op.getNumResults() > 0)
    if (auto rt = dyn_cast<RankedTensorType>(op.getResult(0).getType())) {
      std::string fromType = dtypeNameOf(rt.getElementType());
      if (dtypeBits(fromType) > 0) return fromType;
    }
  return effectiveDtype;
}

} // namespace mlir::hir
