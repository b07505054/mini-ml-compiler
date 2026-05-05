#include "runtime/lowering.h"

#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>

ExecutionPlan lower_to_execution_plan(const Graph& graph) {
    ExecutionPlan plan;

    int num_nodes = static_cast<int>(graph.nodes.size());

    std::unordered_map<int, int> tensor_producer;
    std::vector<std::vector<int>> adjacency(num_nodes);
    std::vector<int> indegree(num_nodes, 0);

    // Map tensor id -> producer node id.
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
        for (int output_tensor : graph.nodes[node_id].outputs) {
            tensor_producer[output_tensor] = node_id;
        }
    }

    // Build dependency edges.
    // If node B uses tensor produced by node A, add edge A -> B.
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
        for (int input_tensor : graph.nodes[node_id].inputs) {
            auto it = tensor_producer.find(input_tensor);
            if (it != tensor_producer.end()) {
                int producer = it->second;

                if (producer != node_id) {
                    adjacency[producer].push_back(node_id);
                    indegree[node_id]++;
                }
            }
        }
    }

    std::queue<int> ready;

    for (int i = 0; i < num_nodes; ++i) {
        if (indegree[i] == 0) {
            ready.push(i);
        }
    }

    int visited = 0;

    while (!ready.empty()) {
        int node_id = ready.front();
        ready.pop();

        plan.ordered_nodes.push_back(graph.nodes[node_id]);
        visited++;

        for (int next : adjacency[node_id]) {
            indegree[next]--;

            if (indegree[next] == 0) {
                ready.push(next);
            }
        }
    }

    if (visited != num_nodes) {
        std::cerr << "[Lowering] Error: graph contains a cycle or invalid dependency\n";
        return plan;
    }

    std::cout << "[Lowering] Graph IR lowered to ExecutionPlan using topological scheduling\n";
    return plan;
}