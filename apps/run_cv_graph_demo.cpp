#include "frontend/cv_graph_builder.h"
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
#include "compiler/cost_based_planner.hpp"
#include "runtime/runtime_replanner.hpp"

int main() {
    CVGraphBuilder builder;
    Graph graph = builder.build();

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
    CostBasedPlanner planner;

    PlannerCandidate best =
    planner.choose_best_plan(
        graph,
        report
    );
    std::vector<RuntimeObservation> observations;

    observations.push_back({
        "Metal",
        2.84f,
        true
    });

    RuntimeReplanner replanner;

    PlannerCandidate replanned =
        replanner.replan(
            best,
            observations
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