#pragma once

#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"

// Auto-generated operator onboarding stub for Slice.
//
// TODO:
// - Define runtime kernel semantics.
// - Add OpType::Slice to include/ir/node.h.
// - Register the operator in src/runtime/op_registry.cpp.
// - Add shape inference logic in src/analysis/shape_inference.cpp.
// - Add graph verification rules in src/analysis/graph_verifier.cpp.
// - Add unit/demo test under apps/.

void slice_kernel_stub(
    const Graph& graph,
    const Node& node
);

void slice_shape_inference_stub(
    Graph& graph,
    const Node& node
);

bool slice_verify_stub(
    const Graph& graph,
    const Node& node
);
