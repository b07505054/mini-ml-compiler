#include "ir/graph.h"
#include "pass/pass_manager.h"
#include "pass/compiler_pipeline_passes.h"
#include "runtime/static_scheduler.h"
#include "runtime/schedule_executor.h"

#include <iostream>
#include <memory>

int main() {
    Graph graph;

    int input = graph.add_tensor(
        Tensor("input", {2, 3})
    );

    int weight = graph.add_tensor(
        Tensor("weight", {3, 2})
    );

    int bias = graph.add_tensor(
        Tensor("bias", {2, 2})
    );

    int matmul_out = graph.add_tensor(
        Tensor("matmul_out", {})
    );

    int output = graph.add_tensor(
        Tensor("output", {})
    );

    graph.get_tensor(input).persistent = true;
    graph.get_tensor(weight).persistent = true;
    graph.get_tensor(bias).persistent = true;

    graph.add_node(
        Node(
            "matmul",
            OpType::MatMul,
            {input, weight},
            {matmul_out}
        )
    );

    graph.add_node(
        Node(
            "add_bias",
            OpType::Add,
            {matmul_out, bias},
            {output}
        )
    );

    std::cout << "=== Fusion Bias Demo ===\n";

    PassManager pm;

    pm.add_pass(
        std::make_unique<ShapeInferencePass>()
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

    ExecutionSchedule sched =
        build_static_schedule(graph);

    sched.dump();

    sched.export_json(
        "../trace/fusion_bias_schedule.json"
    );

    ScheduleExecutor executor;

    executor.run(
        graph,
        sched,
        true,
        true
    );

    std::cout << "Fusion bias demo complete.\n";

    return 0;
}