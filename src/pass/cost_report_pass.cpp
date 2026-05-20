#include "pass/cost_report_pass.h"

#include "ir/op_type_utils.h"

#include <numeric>

namespace {

size_t num_elements_from_shape(
    const std::vector<int>& shape
) {
    if (shape.empty()) {
        return 0;
    }

    size_t total = 1;

    for (int dim : shape) {
        if (dim <= 0) {
            return 0;
        }

        total *= static_cast<size_t>(dim);
    }

    return total;
}

size_t tensor_bytes(
    const Graph& graph,
    int tensor_id
) {
    const auto& tensor =
        graph.tensors.at(
            static_cast<size_t>(tensor_id)
        );

    constexpr size_t dtype_size =
        4; // float32

    return num_elements_from_shape(
        tensor.shape
    ) * dtype_size;
}

size_t sum_tensor_bytes(
    const Graph& graph,
    const std::vector<int>& tensor_ids
) {
    size_t total = 0;

    for (int tid : tensor_ids) {
        total += tensor_bytes(
            graph,
            tid
        );
    }

    return total;
}

size_t estimate_matmul_flops(
    const Graph& graph,
    const Node& node
) {
    if (node.inputs.size() < 2) {
        return 0;
    }

    const auto& A =
        graph.tensors.at(
            static_cast<size_t>(
                node.inputs[0]
            )
        );

    const auto& B =
        graph.tensors.at(
            static_cast<size_t>(
                node.inputs[1]
            )
        );

    if (
        A.shape.size() != 2 ||
        B.shape.size() != 2
    ) {
        return 0;
    }

    size_t M =
        static_cast<size_t>(A.shape[0]);

    size_t K =
        static_cast<size_t>(A.shape[1]);

    size_t N =
        static_cast<size_t>(B.shape[1]);

    return 2 * M * N * K;
}

std::string backend_for_op(
    OpType op
) {
    if (
        op == OpType::MatMul ||
        op == OpType::FusedMatMulBias ||
        op == OpType::FusedMatMulAddReLU ||
        op == OpType::FusedAttention ||
        op == OpType::TiledAttention
    ) {
        return "Metal";
    }

    return "CPU";
}

double kernel_launch_cost(
    const std::string& backend
) {
    if (backend == "Metal") {
        return 0.08;
    }

    return 0.005;
}

} // namespace

CostReport CostReportPass::run(
    const Graph& graph
) const {
    CostReport report;

    std::string previous_backend;

    for (const auto& node : graph.nodes) {
        CostReportEntry e;

        e.op_name =
            node.name;

        e.op_type =
            op_type_to_string(
                node.op
            );

        e.backend =
            backend_for_op(
                node.op
            );

        e.estimated_read_bytes =
            sum_tensor_bytes(
                graph,
                node.inputs
            );

        e.estimated_write_bytes =
            sum_tensor_bytes(
                graph,
                node.outputs
            );

        e.estimated_flops = 0;

        if (
            node.op == OpType::MatMul ||
            node.op == OpType::FusedMatMulBias ||
            node.op == OpType::FusedMatMulAddReLU
        ) {
            e.estimated_flops =
                estimate_matmul_flops(
                    graph,
                    node
                );
        }

        size_t total_bytes =
            e.estimated_read_bytes
            + e.estimated_write_bytes;

        e.arithmetic_intensity =
            total_bytes == 0
                ? 0.0
                : static_cast<double>(
                    e.estimated_flops
                  ) / static_cast<double>(
                    total_bytes
                  );

        e.estimated_kernel_launch_cost =
            kernel_launch_cost(
                e.backend
            );

        if (
            !previous_backend.empty() &&
            previous_backend != e.backend
        ) {
            e.estimated_backend_switch_cost =
                0.02;
        } else {
            e.estimated_backend_switch_cost =
                0.0;
        }
        e.actual_backend = "N/A";

        e.actual_latency_ms = -1.0;

        previous_backend =
            e.backend;

        if (
            node.op ==
                OpType::FusedMatMulBias
        ) {
            e.fusion_note =
                "MatMul+Add fused";
        }

        if (
            node.op ==
                OpType::FusedMatMulAddReLU
        ) {
            e.fusion_note =
                "MatMul+Add+ReLU fused";
        }

        report.entries.push_back(e);
    }

    return report;
}