#include "ir/graph.h"
#include "kernels/cpu_kernels.h"
#include "runtime/op_registry.h"
#include "utils/timer.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string find_plan_path() {
    for (const std::string& path : {
             "trace/mlir_execution_plan.json",
             "../trace/mlir_execution_plan.json",
         }) {
        std::ifstream in(path);
        if (in.good()) {
            return path;
        }
    }

    return "trace/mlir_execution_plan.json";
}

bool close_enough(const Tensor& a, const Tensor& b) {
    if (a.data.size() != b.data.size()) {
        return false;
    }

    for (size_t i = 0; i < a.data.size(); ++i) {
        if (std::fabs(a.data[i] - b.data[i]) > 1e-4f) {
            return false;
        }
    }

    return true;
}

void fill_inputs(Tensor& a, Tensor& b, Tensor& bias) {
    for (int i = 0; i < a.numel(); ++i) {
        a.data[i] = static_cast<float>((i % 7) + 1) * 0.01f;
    }
    for (int i = 0; i < b.numel(); ++i) {
        b.data[i] = static_cast<float>((i % 11) + 1) * 0.02f;
    }
    for (int i = 0; i < bias.numel(); ++i) {
        bias.data[i] = (i % 5 == 0) ? -0.25f : 0.1f;
    }
}

double run_unfused(
    const Tensor& a,
    const Tensor& b,
    const Tensor& bias,
    Tensor& out,
    int runs
) {
    Tensor matmul_out("matmul_out", out.shape);
    Tensor add_out("add_out", out.shape);
    Timer timer;
    timer.start();

    for (int i = 0; i < runs; ++i) {
        matmul(a, b, matmul_out);
        add(matmul_out, bias, add_out);
        relu(add_out, out);
    }

    return timer.stop_ms() / runs;
}

double run_fused_registry(Graph& graph, const Node& node, Tensor& out, int runs) {
    auto registry = create_default_registry();
    Timer timer;
    timer.start();

    for (int i = 0; i < runs; ++i) {
        registry.dispatch(OpType::FusedMatMulAddReLU, graph, node);
    }

    return timer.stop_ms() / runs;
}

} // namespace

int main() {
    const std::string plan_path = find_plan_path();
    const std::string plan = read_file(plan_path);

    if (!contains(plan, "\"runtime_kernel\": \"fused_matmul_add_relu\"") ||
        !contains(plan, "\"runtime_op_type\": \"FusedMatMulAddReLU\"")) {
        std::cerr
            << "MLIR execution plan does not reference the custom fused kernel path. "
            << "Run tools/run_mlir_fusion_pipeline.sh first.\n";
        return 1;
    }

    const int m = 128;
    const int k = 128;
    const int n = 128;
    const int runs = 30;

    Graph graph;
    int a_id = graph.add_tensor(Tensor("A", {m, k}));
    int b_id = graph.add_tensor(Tensor("B", {k, n}));
    int bias_id = graph.add_tensor(Tensor("bias", {m, n}));
    int fused_out_id = graph.add_tensor(Tensor("fused_out", {m, n}));

    fill_inputs(
        graph.get_tensor(a_id),
        graph.get_tensor(b_id),
        graph.get_tensor(bias_id)
    );

    Tensor unfused_out("unfused_out", {m, n});
    double unfused_ms = run_unfused(
        graph.get_tensor(a_id),
        graph.get_tensor(b_id),
        graph.get_tensor(bias_id),
        unfused_out,
        runs
    );

    Node fused_node(
        "mlir_lowered_fused_matmul_bias_relu",
        OpType::FusedMatMulAddReLU,
        {a_id, b_id, bias_id},
        {fused_out_id}
    );
    double fused_ms = run_fused_registry(
        graph,
        fused_node,
        graph.get_tensor(fused_out_id),
        runs
    );

    bool correct = close_enough(unfused_out, graph.get_tensor(fused_out_id));
    double speedup = fused_ms > 0.0 ? unfused_ms / fused_ms : 0.0;

    std::cout << "=== MLIR Fused Kernel Benchmark ===\n";
    std::cout << "Plan: " << plan_path << "\n";
    std::cout << "Lowered runtime op: FusedMatMulAddReLU\n";
    std::cout << "Runtime kernel: fused_matmul_add_relu\n";
    std::cout << "Shape: " << m << "x" << k << " * " << k << "x" << n << "\n";
    std::cout << "Runs: " << runs << "\n";
    std::cout << "Unfused avg latency: " << unfused_ms << " ms\n";
    std::cout << "Fused registry avg latency: " << fused_ms << " ms\n";
    std::cout << "Speedup: " << speedup << "x\n";
    std::cout << "Correctness: " << (correct ? "PASSED" : "FAILED") << "\n";

    return correct ? 0 : 1;
}
