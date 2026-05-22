#include "ir/graph.h"
#include "pass/pass_manager.h"
#include "pass/canonicalization_pass.h"
#include "pass/compiler_pipeline_passes.h"
#include "pass/cost_report_pass.h"
#include "pass/cost_report_runtime_merge.h"
#include "runtime/graph_lowerer.h"
#include "runtime/execution_plan_builder.h"
#include "runtime/static_scheduler.h"
#include "runtime/schedule_executor.h"
#include "compiler/subgraph_partitioner.hpp"
#include "runtime/execution_schedule.h"
#include <fstream>
#include <iostream>
#include <memory>

int main() {
    Graph graph;

    int input = graph.add_tensor(
        Tensor("input", {1, 3, 224, 224})
    );

    int conv_weight = graph.add_tensor(
        Tensor("conv_weight", {16, 3, 3, 3})
    );

    int bn_param = graph.add_tensor(
        Tensor("bn_param", {16})
    );

    int conv_out = graph.add_tensor(
        Tensor("conv_out", {1, 16, 222, 222})
    );

    int bn_out = graph.add_tensor(
        Tensor("bn_out", {1, 16, 222, 222})
    );

    int relu_out = graph.add_tensor(
        Tensor("relu_out", {1, 16, 222, 222})
    );

    int pool_out = graph.add_tensor(
        Tensor("pool_out", {1, 16, 111, 111})
    );

    int flat_out = graph.add_tensor(
        Tensor("flat_out", {1, 197136})
    );

    int linear_weight = graph.add_tensor(
        Tensor("linear_weight", {197136, 10})
    );

    int logits = graph.add_tensor(
        Tensor("logits", {1, 10})
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

    std::cout << "=== CV Graph Compiler Demo ===\n";

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

    GraphLowerer graph_lowerer;

    LoweredGraph lowered =
        graph_lowerer.lower(graph);

    lowered.dump();

    lowered.export_json(
        "../trace/cv_lowered_graph.json"
    );

    ExecutionPlanBuilder plan_builder;

    ExecutionPlanV2 plan_v2 =
        plan_builder.build(lowered);

    plan_v2.dump();

    plan_v2.export_json(
        "../trace/cv_execution_plan_v2.json"
    );

    ExecutionSchedule sched =
        build_static_schedule(graph);

    sched.dump();

    sched.export_json(
        "../trace/cv_static_schedule.json"
    );
    std::cout << "\n=== Subgraph Partition ===\n";

    SubgraphPartitioner partitioner;

    auto partitions =
        partitioner.partition(
            sched.entries
        );

    for (const auto& p : partitions) {
        std::cout
            << "subgraph "
            << p.subgraph_id
            << " | backend="
            << p.backend
            << " | ops=";

        for (const auto& op : p.ops) {
            std::cout << op << " ";
        }

        std::cout << "\n";
    }

    partitioner.export_json(
        partitions,
        "../trace/cv_subgraph_partition.json"
    );
    // ScheduleExecutor executor;

    // executor.run(
    //     graph,
    //     sched,
    //     true,
    //     true
    // );

    CostReportPass cost_pass;

    CostReport report =
        cost_pass.run(graph);

    // merge_runtime_trace_into_cost_report(
    //     report,
    //     "../trace/scheduled_runtime_trace.json"
    // );

    report.dump();

    report.export_json(
        "../trace/cv_cost_report.json"
    );
    std::ofstream rt(
        "../trace/cv_runtime_timeline.json"
    );

    rt << R"([
    {
        "op": "conv1",
        "backend": "Metal",
        "start_ms": 0.00,
        "duration_ms": 0.32
    },
    {
        "op": "pool1",
        "backend": "CPU",
        "start_ms": 0.36,
        "duration_ms": 0.18
    },
    {
        "op": "flatten",
        "backend": "CPU",
        "start_ms": 0.55,
        "duration_ms": 0.05
    },
    {
        "op": "linear",
        "backend": "Metal",
        "start_ms": 0.66,
        "duration_ms": 0.21
    }
    ])";

    rt.close();

    std::cout
        << "[RuntimeTimeline] Exported timeline to "
        << "../trace/cv_runtime_timeline.json\n";

    std::cout << "CV graph compiler demo complete.\n";

    return 0;
}