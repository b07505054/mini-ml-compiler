#include "generated_ops/slice_stub.h"

#include <iostream>

void slice_kernel_stub(
    const Graph& graph,
    const Node& node
) {
    std::cout << "[SliceStub] Runtime kernel is not implemented yet.\n";
    std::cout << "[SliceStub] Node: " << node.name << "\n";

    // TODO:
    // - Read input tensors from graph.get_tensor(node.inputs[i]).
    // - Allocate and write output tensors through graph.get_tensor(node.outputs[i]).
    // - Implement operator semantics.
}

void slice_shape_inference_stub(
    Graph& graph,
    const Node& node
) {
    std::cout << "[SliceStub] Shape inference is not implemented yet.\n";

    // TODO:
    // - Infer output shapes from input tensor shapes.
    // - Example:
    //   auto& input = graph.get_tensor(node.inputs[0]);
    //   auto& output = graph.get_tensor(node.outputs[0]);
    //   output.shape = input.shape;
}

bool slice_verify_stub(
    const Graph& graph,
    const Node& node
) {
    std::cout << "[SliceStub] Verifier is not implemented yet.\n";

    // TODO:
    // - Check input count.
    // - Check output count.
    // - Check supported tensor ranks and shape constraints.
    // - Return false if unsupported.

    return true;
}
