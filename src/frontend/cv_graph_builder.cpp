#include "frontend/cv_graph_builder.h"

#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"

CVGraphBuilder::CVGraphBuilder(CVModelSpec spec) : spec_(std::move(spec)) {}

Graph CVGraphBuilder::build() {
    Graph graph;

    // Derived spatial dimensions (no padding, stride 1)
    const int conv_h    = spec_.input_h - spec_.kernel_size + 1;
    const int conv_w    = spec_.input_w - spec_.kernel_size + 1;
    const int pool_h    = conv_h / 2;
    const int pool_w    = conv_w / 2;
    const int flat_size = spec_.num_conv_filters * pool_h * pool_w;

    // ---- Tensors -------------------------------------------------
    // IDs are assigned by add_tensor in insertion order; the ordering
    // below exactly matches apps/run_cv_graph_demo.cpp so that both
    // produce structurally identical graphs when using the default spec.

    int input = graph.add_tensor(
        Tensor("input",
               {1, spec_.input_channels, spec_.input_h, spec_.input_w}));

    int conv_weight = graph.add_tensor(
        Tensor("conv_weight",
               {spec_.num_conv_filters,
                spec_.input_channels,
                spec_.kernel_size,
                spec_.kernel_size}));

    int bn_param = graph.add_tensor(
        Tensor("bn_param", {spec_.num_conv_filters}));

    int conv_out = graph.add_tensor(
        Tensor("conv_out", {1, spec_.num_conv_filters, conv_h, conv_w}));

    int bn_out = graph.add_tensor(
        Tensor("bn_out", {1, spec_.num_conv_filters, conv_h, conv_w}));

    int relu_out = graph.add_tensor(
        Tensor("relu_out", {1, spec_.num_conv_filters, conv_h, conv_w}));

    int pool_out = graph.add_tensor(
        Tensor("pool_out", {1, spec_.num_conv_filters, pool_h, pool_w}));

    int flat_out = graph.add_tensor(
        Tensor("flat_out", {1, flat_size}));

    int linear_weight = graph.add_tensor(
        Tensor("linear_weight", {flat_size, spec_.num_classes}));

    int logits = graph.add_tensor(
        Tensor("logits", {1, spec_.num_classes}));

    // Mark weights and the input tensor as persistent (not intermediate)
    graph.get_tensor(input).persistent        = true;
    graph.get_tensor(conv_weight).persistent  = true;
    graph.get_tensor(bn_param).persistent     = true;
    graph.get_tensor(linear_weight).persistent = true;

    // ---- Nodes ---------------------------------------------------

    graph.add_node(Node("conv1",   OpType::Conv2D,    {input, conv_weight}, {conv_out}));
    graph.add_node(Node("bn1",     OpType::BatchNorm, {conv_out, bn_param}, {bn_out}));
    graph.add_node(Node("relu1",   OpType::ReLU,      {bn_out},             {relu_out}));
    graph.add_node(Node("pool1",   OpType::MaxPool,   {relu_out},           {pool_out}));
    graph.add_node(Node("flatten", OpType::Flatten,   {pool_out},           {flat_out}));
    graph.add_node(Node("linear",  OpType::Linear,    {flat_out, linear_weight}, {logits}));

    return graph;
}
