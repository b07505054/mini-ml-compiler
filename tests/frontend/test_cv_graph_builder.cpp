#include "frontend/cv_graph_builder.h"
#include "ir/graph.h"
#include "ir/node.h"
#include "ir/tensor.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// ---- helpers -------------------------------------------------------

static void check(bool cond, const char* label) {
    if (!cond) {
        std::cerr << "FAIL: " << label << "\n";
        std::exit(1);
    }
    std::cout << "  pass: " << label << "\n";
}

// ---- tests ---------------------------------------------------------

static void test_default_spec_produces_ten_tensors() {
    CVGraphBuilder b;
    Graph g = b.build();
    check(g.tensors.size() == 10,
          "default spec: tensor count == 10");
}

static void test_default_spec_produces_six_nodes() {
    CVGraphBuilder b;
    Graph g = b.build();
    check(g.nodes.size() == 6,
          "default spec: node count == 6");
}

static void test_node_op_sequence() {
    CVGraphBuilder b;
    Graph g = b.build();
    check(g.nodes[0].op == OpType::Conv2D,    "node[0] is Conv2D");
    check(g.nodes[1].op == OpType::BatchNorm, "node[1] is BatchNorm");
    check(g.nodes[2].op == OpType::ReLU,      "node[2] is ReLU");
    check(g.nodes[3].op == OpType::MaxPool,   "node[3] is MaxPool");
    check(g.nodes[4].op == OpType::Flatten,   "node[4] is Flatten");
    check(g.nodes[5].op == OpType::Linear,    "node[5] is Linear");
}

static void test_node_names() {
    CVGraphBuilder b;
    Graph g = b.build();
    check(g.nodes[0].name == "conv1",   "node[0].name == conv1");
    check(g.nodes[1].name == "bn1",     "node[1].name == bn1");
    check(g.nodes[2].name == "relu1",   "node[2].name == relu1");
    check(g.nodes[3].name == "pool1",   "node[3].name == pool1");
    check(g.nodes[4].name == "flatten", "node[4].name == flatten");
    check(g.nodes[5].name == "linear",  "node[5].name == linear");
}

static void test_persistent_tensors() {
    CVGraphBuilder b;
    Graph g = b.build();
    // input(0), conv_weight(1), bn_param(2), linear_weight(8) are persistent
    check(g.tensors[0].persistent == true,  "tensor[0] (input) is persistent");
    check(g.tensors[1].persistent == true,  "tensor[1] (conv_weight) is persistent");
    check(g.tensors[2].persistent == true,  "tensor[2] (bn_param) is persistent");
    check(g.tensors[8].persistent == true,  "tensor[8] (linear_weight) is persistent");
    check(g.tensors[3].persistent == false, "tensor[3] (conv_out) is not persistent");
    check(g.tensors[9].persistent == false, "tensor[9] (logits) is not persistent");
}

static void test_input_tensor_shape() {
    CVGraphBuilder b;
    Graph g = b.build();
    const auto& s = g.tensors[0].shape;
    check(s.size() == 4 && s[0]==1 && s[1]==3 && s[2]==224 && s[3]==224,
          "tensor[0] shape == {1,3,224,224}");
}

static void test_conv_weight_shape() {
    CVGraphBuilder b;
    Graph g = b.build();
    const auto& s = g.tensors[1].shape;
    check(s.size() == 4 && s[0]==16 && s[1]==3 && s[2]==3 && s[3]==3,
          "tensor[1] (conv_weight) shape == {16,3,3,3}");
}

static void test_flat_out_shape() {
    CVGraphBuilder b;
    Graph g = b.build();
    const auto& s = g.tensors[7].shape;
    // 16 filters * 111 * 111 = 197136
    check(s.size() == 2 && s[0]==1 && s[1]==197136,
          "tensor[7] (flat_out) shape == {1,197136}");
}

static void test_logits_shape() {
    CVGraphBuilder b;
    Graph g = b.build();
    const auto& s = g.tensors[9].shape;
    check(s.size() == 2 && s[0]==1 && s[1]==10,
          "tensor[9] (logits) shape == {1,10}");
}

static void test_conv1_connectivity() {
    CVGraphBuilder b;
    Graph g = b.build();
    // conv1: inputs=[0 (input), 1 (conv_weight)], outputs=[3 (conv_out)]
    const auto& n = g.nodes[0];
    check(n.inputs.size()==2 && n.inputs[0]==0 && n.inputs[1]==1,
          "conv1 inputs == [0, 1]");
    check(n.outputs.size()==1 && n.outputs[0]==3,
          "conv1 outputs == [3]");
}

static void test_linear_connectivity() {
    CVGraphBuilder b;
    Graph g = b.build();
    // linear: inputs=[7 (flat_out), 8 (linear_weight)], outputs=[9 (logits)]
    const auto& n = g.nodes[5];
    check(n.inputs.size()==2 && n.inputs[0]==7 && n.inputs[1]==8,
          "linear inputs == [7, 8]");
    check(n.outputs.size()==1 && n.outputs[0]==9,
          "linear outputs == [9]");
}

static void test_custom_spec_changes_shapes() {
    CVModelSpec spec;
    spec.input_channels   = 1;
    spec.input_h          = 32;
    spec.input_w          = 32;
    spec.num_conv_filters = 8;
    spec.kernel_size      = 3;
    spec.num_classes      = 5;

    CVGraphBuilder b(spec);
    Graph g = b.build();

    check(g.tensors.size() == 10, "custom spec: still 10 tensors");
    check(g.nodes.size()   == 6,  "custom spec: still 6 nodes");

    // input shape
    const auto& inp = g.tensors[0].shape;
    check(inp[1]==1 && inp[2]==32 && inp[3]==32,
          "custom spec: input shape {1,1,32,32}");

    // conv_out shape: (32-3+1)=30
    const auto& co = g.tensors[3].shape;
    check(co[1]==8 && co[2]==30 && co[3]==30,
          "custom spec: conv_out shape {1,8,30,30}");

    // flat_size = 8 * 15 * 15 = 1800
    const auto& flat = g.tensors[7].shape;
    check(flat[1] == 1800, "custom spec: flat_out size == 1800");

    // logits shape
    const auto& lg = g.tensors[9].shape;
    check(lg[1] == 5, "custom spec: logits shape {1,5}");
}

static void test_two_builds_are_independent() {
    CVGraphBuilder b;
    Graph g1 = b.build();
    Graph g2 = b.build();
    // Modifying g1 must not affect g2
    g1.nodes[0].name = "mutated";
    check(g2.nodes[0].name == "conv1",
          "two builds produce independent graphs");
}

// ---- main ----------------------------------------------------------

int main() {
    std::cout << "=== CVGraphBuilder tests ===\n";

    test_default_spec_produces_ten_tensors();
    test_default_spec_produces_six_nodes();
    test_node_op_sequence();
    test_node_names();
    test_persistent_tensors();
    test_input_tensor_shape();
    test_conv_weight_shape();
    test_flat_out_shape();
    test_logits_shape();
    test_conv1_connectivity();
    test_linear_connectivity();
    test_custom_spec_changes_shapes();
    test_two_builds_are_independent();

    std::cout << "\nAll CVGraphBuilder tests passed.\n";
    return 0;
}
