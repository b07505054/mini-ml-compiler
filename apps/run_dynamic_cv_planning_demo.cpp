#include "ir/graph.h"
#include "pass/pass_manager.h"
#include "pass/canonicalization_pass.h"
#include "pass/compiler_pipeline_passes.h"
#include "pass/cost_report_pass.h"
#include "runtime/graph_lowerer.h"
#include "runtime/execution_plan_builder.h"
#include "runtime/static_scheduler.h"

#include <iostream>
#include <memory>
#include <vector>

Graph build_cv_graph(int batch) {
    Graph graph;

    int input = graph.add_tensor(
        Tensor("input", {batch, 3, 224, 224})
    );

    int conv_weight = graph.add_tensor(
        Tensor("conv_weight", {16, 3, 3, 3})
    );

    int bn_param = graph.add_tensor(
        Tensor("bn_param", {16})
    );

    int conv_out = graph.add_tensor(
        Tensor("conv_out", {batch, 16, 222, 222})
    );

    int bn_out = graph.add_tensor(
        Tensor("bn_out", {batch, 16, 222, 222})
    );

    int relu_out = graph.add_tensor(
        Tensor("relu_out", {batch, 16, 222, 222})
    );

    int pool_out = graph.add_tensor(
        Tensor("pool_out", {batch, 16, 111, 111})
    );

    int flat_out = graph.add_tensor(
        Tensor("flat_out", {batch, 197136})
    );

    int linear_weight = graph.add_tensor(
        Tensor("linear_weight", {197136, 10})
    );

    int logits = graph.add_tensor(
        Tensor("logits", {batch, 10})
    );

    graph.get_tensor(input).persistent = true;
    graph.get_tensor(conv_weight).persistent = true;
    graph.get_tensor(bn_param).persistent = true;
    graph.get_tensor(linear_weight).persistent = true;

    graph.add_node(
        Node(
            "conv1",
            OpType::Conv2D,
            {input, conv_weight},
            {conv_out}
        )
    );

    graph.add_node(
        Node(
            "bn1",
            OpType::BatchNorm,
            {conv_out, bn_param},
            {bn_out}
        )
    );

    graph.add_node(
        Node(
            "relu1",
            OpType::ReLU,
            {bn_out},
            {relu_out}
        )
    );

    graph.add_node(
        Node(
            "pool1",
            OpType::MaxPool,
            {relu_out},
            {pool_out}
        )
    );

    graph.add_node(
        Node(
            "flatten",
            OpType::Flatten,
            {pool_out},
            {flat_out}
        )
    );

    graph.add_node(
        Node(
            "linear",
            OpType::Linear,
            {flat_out, linear_weight},
            {logits}
        )
    );

    return graph;
}

void run_pipeline_for_batch(int batch) {
    std::cout
        << "\n==============================\n"
        << "Dynamic CV Planning Demo\n"
        << "batch = "
        << batch
        << "\n"
        << "==============================\n";

    Graph graph =
        build_cv_graph(batch);

    PassManager pm;

    pm.add_pass(
        std::make_unique<ShapeInferencePass>()
    );

    pm.add_pass(
        std::make_unique<CanonicalizationPass>()
    );

    pm.add_pass(
        std::make_unique<DTypePropagationPass>()
    );

    pm.add_pass(
        std::make_unique<FusionCandidatePass>()
    );

    pm.add_pass(
        std::make_unique<MemoryPlanningPass>()
    );

    pm.add_pass(
        std::make_unique<BackendPlacementPass>()
    );

    pm.add_pass(
        std::make_unique<SchedulingPass>()
    );

    pm.run(graph);

    GraphLowerer lowerer;

    LoweredGraph lowered =
        lowerer.lower(graph);

    lowered.dump();

    ExecutionPlanBuilder builder;

    ExecutionPlanV2 plan =
        builder.build(lowered);

    plan.dump();

    ExecutionSchedule sched =
        build_static_schedule(graph);

    sched.dump();

    CostReportPass cost_pass;

    CostReport report =
        cost_pass.run(graph);

    report.dump();

    std::string prefix =
        "../trace/dynamic_cv_batch_"
        + std::to_string(batch);

    lowered.export_json(
        prefix + "_lowered_graph.json"
    );

    plan.export_json(
        prefix + "_execution_plan.json"
    );

    sched.export_json(
        prefix + "_static_schedule.json"
    );

    report.export_json(
        prefix + "_cost_report.json"
    );
}

int main() {
    std::vector<int> batches =
        {1, 4, 8};

    for (int batch : batches) {
        run_pipeline_for_batch(batch);
    }

    std::cout
        << "\nDynamic CV planning demo complete.\n";

    return 0;
}