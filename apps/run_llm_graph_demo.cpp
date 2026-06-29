#include "frontend/llm_graph_builder.h"
#include "ir/graph.h"
#include "pass/pass_manager.h"
#include "pass/compiler_pipeline_passes.h"
#include "runtime/graph_lowerer.h"
#include "runtime/execution_plan_builder.h"

#include <iostream>
#include <memory>

static LLMModelSpec default_spec() {
    LLMModelSpec spec;
    spec.model_name          = "qwen_0_5b";
    spec.num_layers          = 24;
    spec.hidden_size         = 1024;
    spec.num_attention_heads = 16;
    spec.num_key_value_heads = 16;
    spec.intermediate_size   = 2816;
    spec.vocab_size          = 151936;
    spec.max_seq_len         = 512;
    spec.batch_size          = 1;
    return spec;
}

static void run_pipeline(Graph& graph) {
    PassManager pm;
    pm.add_pass(std::make_unique<ShapeInferencePass>());
    pm.add_pass(std::make_unique<FusionCandidatePass>());
    pm.add_pass(std::make_unique<MemoryPlanningPass>());
    pm.add_pass(std::make_unique<BackendPlacementPass>());
    pm.add_pass(std::make_unique<SchedulingPass>());
    pm.run(graph);

    GraphLowerer lowerer;
    LoweredGraph lowered = lowerer.lower(graph);
    lowered.dump();

    ExecutionPlanBuilder plan_builder;
    ExecutionPlanV2 plan = plan_builder.build(lowered);
    plan.dump();
}

int main() {
    LLMModelSpec spec = default_spec();

    std::cout << "=== LLM Graph Compiler Demo ===\n";
    std::cout << "Model: " << spec.model_name << "\n";
    std::cout << "Truth boundary: " << spec.truth_boundary << "\n";

    // ---- Prefill graph ---------------------------------------------------
    std::cout << "\n=== Prefill Graph (KVCacheWrite) ===\n";
    {
        LLMGraphBuilder builder(spec);
        Graph graph = builder.build(LLMGraphMode::Prefill);

        std::cout << "Graph constructed: "
                  << graph.tensors.size() << " tensors, "
                  << graph.nodes.size() << " nodes\n\n";
        graph.dump();

        std::cout << "\n=== Pass Pipeline (Prefill) ===\n";
        run_pipeline(graph);
    }

    // ---- Decode graph ----------------------------------------------------
    std::cout << "\n=== Decode Graph (KVCacheRead) ===\n";
    {
        LLMGraphBuilder builder(spec);
        Graph graph = builder.build(LLMGraphMode::Decode);

        std::cout << "Graph constructed: "
                  << graph.tensors.size() << " tensors, "
                  << graph.nodes.size() << " nodes\n\n";
        graph.dump();

        std::cout << "\n=== Pass Pipeline (Decode) ===\n";
        run_pipeline(graph);
    }

    std::cout << "\nLLM graph compiler demo complete.\n";
    return 0;
}
