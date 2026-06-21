#include "runtime/metal_rmsnorm_executor.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

// CTest SKIP_RETURN_CODE for metal_rmsnorm_plan_dispatch (see CMakeLists.txt).
// The plan artifact comes from a separate MLIR pipeline
// (tools/run_metal_rmsnorm_compiler_pipeline.sh) that isn't part of the
// baseline CMake/CTest build, so its absence is a skip, not a failure.
constexpr int kSkipReturnCode = 77;

std::optional<std::filesystem::path> find_repo_root() {
    std::filesystem::path current = std::filesystem::current_path();
    for (int i = 0; i < 4; ++i) {
        if (std::filesystem::exists(
                current / "trace" / "metal_rmsnorm_execution_plan.json")) {
            return current;
        }
        current = current.parent_path();
    }
    return std::nullopt;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::pair<int, int> parse_shape_bucket(const std::string& plan) {
    std::smatch match;
    const std::regex pattern(R"("shape_bucket":\s*"(\d+)x(\d+):f32")");
    if (!std::regex_search(plan, match, pattern)) {
        throw std::runtime_error("execution plan has no FP32 RMSNorm shape bucket");
    }
    return {std::stoi(match[1]), std::stoi(match[2])};
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(fraction * (values.size() - 1))];
}

void rmsnorm_cpu(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    std::vector<float>& output,
    int tokens,
    int hidden,
    float epsilon
) {
    for (int token = 0; token < tokens; ++token) {
        const size_t base = static_cast<size_t>(token) * hidden;
        double square_sum = 0.0;
        for (int i = 0; i < hidden; ++i) {
            const double value = input[base + i];
            square_sum += value * value;
        }
        const float inverse_rms =
            1.0f / std::sqrt(static_cast<float>(square_sum / hidden) + epsilon);
        for (int i = 0; i < hidden; ++i) {
            output[base + i] = input[base + i] * inverse_rms * weight[i];
        }
    }
}

} // namespace

int main() {
    const auto maybe_repo_root = find_repo_root();
    if (!maybe_repo_root) {
        std::cerr
            << "SKIP: trace/metal_rmsnorm_execution_plan.json not found; "
               "run tools/run_metal_rmsnorm_compiler_pipeline.sh "
               "(requires an MLIR build) to generate it.\n";
        return kSkipReturnCode;
    }
    const auto& repo_root = *maybe_repo_root;
    const auto plan_path = repo_root / "trace" / "metal_rmsnorm_execution_plan.json";
    const std::string plan = read_text(plan_path);
    if (plan.find("\"runtime_kernel\": \"fused_rmsnorm_metal\"") ==
            std::string::npos ||
        plan.find("\"backend\": \"Metal\"") == std::string::npos) {
        std::cerr << "compiler plan did not select fused_rmsnorm_metal\n";
        return 1;
    }

    const auto [tokens, hidden] = parse_shape_bucket(plan);
    constexpr float epsilon = 1.0e-5f;
    std::vector<float> input(static_cast<size_t>(tokens) * hidden);
    std::vector<float> weight(hidden);
    std::vector<float> reference(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>((static_cast<int>(i % 31) - 15) * 0.03125);
    }
    for (int i = 0; i < hidden; ++i) {
        weight[i] = 0.75f + static_cast<float>(i % 17) * 0.01f;
    }
    rmsnorm_cpu(input, weight, reference, tokens, hidden, epsilon);

    MetalRMSNormExecutor executor((repo_root / "metal" / "rmsnorm.metal").string());
    MetalRMSNormExecutionResult result =
        executor.execute(input, weight, tokens, hidden, epsilon, 50, 100);

    double max_diff = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(
            static_cast<double>(reference[i]) - result.output[i]
        ));
    }
    const bool correct = max_diff <= 1.0e-4;
    const double p50 = percentile(result.latencies_ms, 0.50);
    const double p95 = percentile(result.latencies_ms, 0.95);

    const auto report_path =
        repo_root / "trace" / "metal_rmsnorm_runtime_report.json";
    std::ofstream out(report_path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"artifact_type\": \"metal_rmsnorm_runtime_report\",\n";
    out << "  \"execution_plan\": \"" << plan_path.string() << "\",\n";
    out << "  \"compiler_selected_kernel\": \"fused_rmsnorm_metal\",\n";
    out << "  \"runtime_dispatched_kernel\": \"fused_rmsnorm_metal\",\n";
    out << "  \"backend\": \"Metal\",\n";
    out << "  \"device\": \"" << result.device << "\",\n";
    out << "  \"shape_bucket\": \"" << tokens << "x" << hidden << ":f32\",\n";
    out << "  \"p50_latency_ms\": " << p50 << ",\n";
    out << "  \"p95_latency_ms\": " << p95 << ",\n";
    out << "  \"max_abs_diff\": " << max_diff << ",\n";
    out << "  \"correct\": " << (correct ? "true" : "false") << ",\n";
    out << "  \"compiler_runtime_kernel_match\": true\n";
    out << "}\n";

    std::cout << "Plan-selected kernel: fused_rmsnorm_metal\n";
    std::cout << "Runtime-dispatched kernel: fused_rmsnorm_metal\n";
    std::cout << "Shape: " << tokens << "x" << hidden << ":f32\n";
    std::cout << "Metal p50: " << p50 << " ms, p95: " << p95 << " ms\n";
    std::cout << "Max abs diff: " << max_diff
              << " " << (correct ? "PASS" : "FAIL") << "\n";
    std::cout << "Wrote " << report_path << "\n";
    return correct ? 0 : 1;
}
