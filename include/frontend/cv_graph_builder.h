#pragma once

#include "frontend/graph_builder.h"

#include <string>

// Descriptor for a compact CV backbone graph.
// Default values reproduce the graph hand-built in apps/run_cv_graph_demo.cpp.
struct CVModelSpec {
    int input_channels  = 3;
    int input_h         = 224;
    int input_w         = 224;
    int num_conv_filters = 16;
    int kernel_size     = 3;
    int num_classes     = 10;
    std::string model_name = "cv_demo";
};

// Builds a Graph IR for a single-block CV backbone:
//   Conv2D → BatchNorm → ReLU → MaxPool → Flatten → Linear
//
// With default CVModelSpec the output graph is byte-for-byte equivalent
// to the graph constructed manually in apps/run_cv_graph_demo.cpp.
class CVGraphBuilder : public GraphBuilder {
public:
    explicit CVGraphBuilder(CVModelSpec spec = {});
    Graph build() override;

private:
    CVModelSpec spec_;
};
