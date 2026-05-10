#pragma once

#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"

// Auto-generated operator onboarding stub for Concat.
//
// TODO:
// - Define runtime kernel semantics.
// - Add OpType::Concat to include/ir/node.h.
// - Register the operator in src/runtime/op_registry.cpp.
// - Add shape inference logic in src/analysis/shape_inference.cpp.
// - Add graph verification rules in src/analysis/graph_verifier.cpp.
// - Add unit/demo test under apps/.

void concat_kernel_stub(
    const Graph& graph,
    const Node& node
);

void concat_shape_inference_stub(
    Graph& graph,
    const Node& node
);

bool concat_verify_stub(
    const Graph& graph,
    const Node& node
);
