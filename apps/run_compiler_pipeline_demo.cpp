#include "ir/graph.h"
#include "pass/pass_manager.h"
#include "pass/compiler_pipeline_passes.h"
#include "runtime/lowering.h"
#include "runtime/executor.h"
#include "runtime/static_scheduler.h"
#include <iostream>
#include <memory>
#include "runtime/schedule_executor.h"

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

    int add_out = graph.add_tensor(
        Tensor("add_out", {})
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
            "add",
            OpType::Add,
            {matmul_out, bias},
            {add_out}
        )
    );

    graph.add_node(
        Node(
            "relu",
            OpType::ReLU,
            {add_out},
            {output}
        )
    );

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

    std::cout << "\n=== Lowering After Compiler Pipeline ===\n";

    ExecutionPlan plan =
        lower_to_execution_plan(graph);
    ExecutionSchedule sched =
        build_static_schedule(graph);

    sched.dump();
    sched.export_json(
        "../trace/static_schedule.json"
    );
    ScheduleExecutor schedule_executor;

    schedule_executor.run(
        graph,
        sched,
        true,
        true
    );
    std::cout << "Compiler pipeline demo complete.\n";

    return 0;
}