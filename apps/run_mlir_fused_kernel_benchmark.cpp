#include "ir/graph.h"
#include "kernels/cpu_kernels.h"
#include "runtime/op_registry.h"
#include "utils/timer.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kDefaultTileSize = 32;
constexpr double kDefaultAtol = 1e-4;
constexpr double kDefaultRtol = 1e-4;

enum class Pattern {
    Bias,
    ElementwiseAdd,
    All,
};

enum class Variant {
    NaiveUnfused,
    TiledUnfused,
    NaiveOnePassFused,
    TiledOnePassFused,
    All,
};

enum class BenchmarkMode {
    ForceVariant,
    UsePlan,
    SweepCandidates,
};

struct KernelConfig {
    int tile_m = kDefaultTileSize;
    int tile_n = kDefaultTileSize;
    int tile_k = kDefaultTileSize;
};

struct PlannedOperation {
    std::string op_id;
    std::string op_type;
    std::string backend;
    std::string kernel_id;
    KernelConfig kernel_config;
    std::vector<std::string> input_ids;
    std::vector<std::string> output_ids;
};

struct RuntimeExecutionPlan {
    int schema_version = 0;
    std::string graph_id;
    std::vector<PlannedOperation> operations;
};

struct RuntimeTrace {
    std::string execution_plan;
    std::string planned_kernel;
    std::string actual_dispatched_kernel;
    std::string backend;
    int dispatch_count = 0;
    KernelConfig kernel_config;
    bool plan_matched_runtime = false;
};

struct BenchmarkConfig {
    int warmup = 50;
    int iterations = 300;
    int repeats = 5;
    int m = 128;
    int k = 128;
    int n = 128;
    int tile_size = kDefaultTileSize;
    double atol = kDefaultAtol;
    double rtol = kDefaultRtol;
    Pattern pattern = Pattern::All;
    Variant variant = Variant::All;
    BenchmarkMode mode = BenchmarkMode::ForceVariant;
    std::string execution_plan_path = "trace/mlir_execution_plan.json";
    bool force_invalid_bias_shape = false;
    bool force_invalid_elementwise_shape = false;
    bool smoke_test_mode = false;
    std::string output_path = "trace/matmul_postop_relu_kernel_profile.json";
    std::string report_path = "trace/matmul_postop_relu_benchmark_report.md";
};

struct Stats {
    int sample_count = 0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double coefficient_of_variation = 0.0;
};

struct CorrectnessResult {
    bool passed = false;
    double atol = kDefaultAtol;
    double rtol = kDefaultRtol;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool contains_nan = false;
    bool contains_inf = false;
};

struct ImplementationProperties {
    std::string matmul_strategy;
    std::string post_op_strategy;
    int intermediate_tensor_count = 0;
    int full_output_post_op_passes = 0;
    int final_output_store_passes = 1;
    int tile_size = 0;
};

struct VariantResult {
    Variant variant = Variant::NaiveUnfused;
    std::string function_name;
    std::string kernel_id;
    std::vector<double> samples_ms;
    Stats stats;
    CorrectnessResult correctness;
    ImplementationProperties implementation;
    RuntimeTrace runtime_trace;
    bool has_runtime_trace = false;
    int rank = 0;
    bool oracle_best = false;
};

struct PatternResult {
    Pattern pattern = Pattern::Bias;
    std::vector<VariantResult> variants;
};

struct Inputs {
    Tensor a;
    Tensor b;
    Tensor postop;

    Inputs(int m, int k, int n, const std::vector<int>& postop_shape)
        : a("A", {m, k}),
          b("B", {k, n}),
          postop("postop", postop_shape) {}
};

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::string run_command_capture(const std::string& command) {
    std::array<char, 256> buffer{};
    std::string result;
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return "";
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (char c : text) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

size_t find_matching(const std::string& text, size_t open_pos, char open_ch, char close_ch) {
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == open_ch) {
            ++depth;
        } else if (c == close_ch) {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    throw std::runtime_error("malformed JSON: unmatched delimiter");
}

std::string extract_object_field(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("plan missing required object field: " + key);
    }
    const size_t open_pos = text.find('{', key_pos + needle.size());
    if (open_pos == std::string::npos) {
        throw std::runtime_error("plan field is not an object: " + key);
    }
    const size_t close_pos = find_matching(text, open_pos, '{', '}');
    return text.substr(open_pos, close_pos - open_pos + 1);
}

std::string extract_array_field_raw(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        throw std::runtime_error("plan missing required array field: " + key);
    }
    const size_t open_pos = text.find('[', key_pos + needle.size());
    if (open_pos == std::string::npos) {
        throw std::runtime_error("plan field is not an array: " + key);
    }
    const size_t close_pos = find_matching(text, open_pos, '[', ']');
    return text.substr(open_pos, close_pos - open_pos + 1);
}

std::string extract_string_field(const std::string& text, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("plan missing required string field: " + key);
    }
    return match[1].str();
}

int extract_int_field(const std::string& text, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        throw std::runtime_error("plan missing required integer field: " + key);
    }
    return std::stoi(match[1].str());
}

std::vector<std::string> extract_string_array_field(const std::string& text, const std::string& key) {
    const std::string raw = extract_array_field_raw(text, key);
    std::regex item("\"([^\"]*)\"");
    std::vector<std::string> values;
    for (std::sregex_iterator it(raw.begin(), raw.end(), item), end; it != end; ++it) {
        values.push_back((*it)[1].str());
    }
    if (values.empty()) {
        throw std::runtime_error("plan array field is empty: " + key);
    }
    return values;
}

std::vector<std::string> extract_object_array(const std::string& text, const std::string& key) {
    const std::string raw = extract_array_field_raw(text, key);
    std::vector<std::string> objects;
    size_t pos = 0;
    while (true) {
        const size_t open_pos = raw.find('{', pos);
        if (open_pos == std::string::npos) {
            break;
        }
        const size_t close_pos = find_matching(raw, open_pos, '{', '}');
        objects.push_back(raw.substr(open_pos, close_pos - open_pos + 1));
        pos = close_pos + 1;
    }
    if (objects.empty()) {
        throw std::runtime_error("plan has no operation objects");
    }
    return objects;
}

std::string pattern_key(Pattern pattern) {
    switch (pattern) {
        case Pattern::Bias: return "bias";
        case Pattern::ElementwiseAdd: return "elementwise_add";
        case Pattern::All: return "all";
    }
    return "unknown";
}

std::string pattern_label(Pattern pattern) {
    switch (pattern) {
        case Pattern::Bias: return "bias";
        case Pattern::ElementwiseAdd: return "elementwise-add";
        case Pattern::All: return "all";
    }
    return "unknown";
}

std::string variant_key(Variant variant) {
    switch (variant) {
        case Variant::NaiveUnfused: return "naive_unfused";
        case Variant::TiledUnfused: return "tiled_unfused";
        case Variant::NaiveOnePassFused: return "naive_one_pass_fused";
        case Variant::TiledOnePassFused: return "tiled_one_pass_fused";
        case Variant::All: return "all";
    }
    return "unknown";
}

std::string variant_label(Variant variant) {
    switch (variant) {
        case Variant::NaiveUnfused: return "naive-unfused";
        case Variant::TiledUnfused: return "tiled-unfused";
        case Variant::NaiveOnePassFused: return "naive-one-pass-fused";
        case Variant::TiledOnePassFused: return "tiled-one-pass-fused";
        case Variant::All: return "all";
    }
    return "unknown";
}

std::string kernel_id_for(Pattern pattern, Variant variant) {
    const std::string semantic =
        pattern == Pattern::Bias ? "matmul_bias_relu" : "matmul_add_relu";
    switch (variant) {
        case Variant::NaiveUnfused:
            return "cpu_naive_" + semantic + "_unfused_f32";
        case Variant::TiledUnfused:
            return "cpu_tiled_" + semantic + "_unfused_f32";
        case Variant::NaiveOnePassFused:
            return "cpu_naive_" + semantic + "_one_pass_f32";
        case Variant::TiledOnePassFused:
            return "cpu_tiled_" + semantic + "_one_pass_f32";
        case Variant::All:
            break;
    }
    throw std::invalid_argument("kernel ID is not defined for variant all");
}

Pattern pattern_from_kernel_id(const std::string& kernel_id) {
    if (kernel_id.find("_matmul_bias_relu_") != std::string::npos) {
        return Pattern::Bias;
    }
    if (kernel_id.find("_matmul_add_relu_") != std::string::npos) {
        return Pattern::ElementwiseAdd;
    }
    throw std::invalid_argument("unknown kernel semantic pattern for kernel_id: " + kernel_id);
}

Variant variant_from_kernel_id(const std::string& kernel_id) {
    if (kernel_id.find("cpu_naive_") == 0 &&
        kernel_id.find("_unfused_f32") != std::string::npos) {
        return Variant::NaiveUnfused;
    }
    if (kernel_id.find("cpu_tiled_") == 0 &&
        kernel_id.find("_unfused_f32") != std::string::npos) {
        return Variant::TiledUnfused;
    }
    if (kernel_id.find("cpu_naive_") == 0 &&
        kernel_id.find("_one_pass_f32") != std::string::npos) {
        return Variant::NaiveOnePassFused;
    }
    if (kernel_id.find("cpu_tiled_") == 0 &&
        kernel_id.find("_one_pass_f32") != std::string::npos) {
        return Variant::TiledOnePassFused;
    }
    throw std::invalid_argument("unknown kernel_id: " + kernel_id);
}

bool is_known_kernel_id(const std::string& kernel_id) {
    try {
        (void)pattern_from_kernel_id(kernel_id);
        (void)variant_from_kernel_id(kernel_id);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string mode_label(BenchmarkMode mode) {
    switch (mode) {
        case BenchmarkMode::ForceVariant: return "force-variant";
        case BenchmarkMode::UsePlan: return "use-plan";
        case BenchmarkMode::SweepCandidates: return "sweep-candidates";
    }
    return "unknown";
}

std::vector<Pattern> selected_patterns(Pattern pattern) {
    if (pattern == Pattern::All) {
        return {Pattern::Bias, Pattern::ElementwiseAdd};
    }
    return {pattern};
}

std::vector<Variant> selected_variants(Variant variant) {
    if (variant == Variant::All) {
        return {
            Variant::NaiveUnfused,
            Variant::TiledUnfused,
            Variant::NaiveOnePassFused,
            Variant::TiledOnePassFused,
        };
    }
    return {variant};
}

bool contains_string(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

PlannedOperation parse_planned_operation(const std::string& object_text) {
    PlannedOperation op;
    op.op_id = extract_string_field(object_text, "op_id");
    op.op_type = extract_string_field(object_text, "op_type");
    op.backend = extract_string_field(object_text, "backend");
    op.kernel_id = extract_string_field(object_text, "selected_kernel");
    op.input_ids = extract_string_array_field(object_text, "inputs");
    op.output_ids = extract_string_array_field(object_text, "outputs");
    const std::string kernel_config = extract_object_field(object_text, "kernel_config");
    op.kernel_config.tile_m = extract_int_field(kernel_config, "tile_m");
    op.kernel_config.tile_n = extract_int_field(kernel_config, "tile_n");
    op.kernel_config.tile_k = extract_int_field(kernel_config, "tile_k");
    return op;
}

RuntimeExecutionPlan load_execution_plan(const std::string& path) {
    const std::string text = read_text_file(path);
    if (text.find('{') == std::string::npos) {
        throw std::runtime_error("malformed JSON: missing object");
    }
    RuntimeExecutionPlan plan;
    plan.schema_version = extract_int_field(text, "schema_version");
    plan.graph_id = extract_string_field(text, "graph_id");
    for (const auto& op_text : extract_object_array(text, "operations")) {
        plan.operations.push_back(parse_planned_operation(op_text));
    }
    return plan;
}

void validate_planned_operation(const PlannedOperation& op, Pattern requested_pattern) {
    if (op.backend != "cpu") {
        throw std::runtime_error("unsupported backend in execution plan: " + op.backend);
    }
    const Pattern kernel_pattern = pattern_from_kernel_id(op.kernel_id);
    if (kernel_pattern != requested_pattern) {
        throw std::runtime_error(
            "plan kernel semantic pattern does not match requested pattern: " +
            op.kernel_id
        );
    }
    if (kernel_pattern == Pattern::Bias && op.op_type != "FusedMatMulBiasRelu") {
        throw std::runtime_error("unknown or mismatched op_type for bias kernel: " + op.op_type);
    }
    if (kernel_pattern == Pattern::ElementwiseAdd && op.op_type != "FusedMatMulAddRelu") {
        throw std::runtime_error("unknown or mismatched op_type for elementwise-add kernel: " + op.op_type);
    }
    if (!is_known_kernel_id(op.kernel_id)) {
        throw std::runtime_error("unknown kernel_id in execution plan: " + op.kernel_id);
    }
    if (op.kernel_config.tile_m <= 0 || op.kernel_config.tile_n <= 0 || op.kernel_config.tile_k <= 0) {
        throw std::runtime_error("invalid kernel_config: tile dimensions must be positive");
    }
    if (op.kernel_config.tile_m != op.kernel_config.tile_n ||
        op.kernel_config.tile_m != op.kernel_config.tile_k) {
        throw std::runtime_error("invalid kernel_config: benchmark dispatcher requires equal tile_m/tile_n/tile_k");
    }
    if (!contains_string(op.input_ids, "A") || !contains_string(op.input_ids, "B")) {
        throw std::runtime_error("execution plan missing required tensor IDs A/B");
    }
    if (kernel_pattern == Pattern::Bias && !contains_string(op.input_ids, "bias")) {
        throw std::runtime_error("bias execution plan missing required tensor ID: bias");
    }
    if (kernel_pattern == Pattern::ElementwiseAdd && !contains_string(op.input_ids, "addend")) {
        throw std::runtime_error("elementwise-add execution plan missing required tensor ID: addend");
    }
    if (!contains_string(op.output_ids, "output")) {
        throw std::runtime_error("execution plan missing required output tensor ID: output");
    }
}

PlannedOperation select_and_validate_plan_operation(
    const RuntimeExecutionPlan& plan,
    Pattern requested_pattern
) {
    if (plan.schema_version != 2) {
        throw std::runtime_error("unsupported execution plan schema_version: " + std::to_string(plan.schema_version));
    }
    if (plan.operations.empty()) {
        throw std::runtime_error("execution plan contains no operations");
    }
    if (plan.operations.size() != 1) {
        throw std::runtime_error("benchmark expects exactly one planned operation");
    }
    validate_planned_operation(plan.operations.front(), requested_pattern);
    return plan.operations.front();
}

Pattern parse_pattern(const std::string& value) {
    if (value == "bias") {
        return Pattern::Bias;
    }
    if (value == "elementwise-add" || value == "elementwise_add") {
        return Pattern::ElementwiseAdd;
    }
    if (value == "all") {
        return Pattern::All;
    }
    throw std::invalid_argument("invalid pattern: " + value);
}

Variant parse_variant(const std::string& value) {
    if (value == "naive-unfused" || value == "naive_unfused") {
        return Variant::NaiveUnfused;
    }
    if (value == "tiled-unfused" || value == "tiled_unfused") {
        return Variant::TiledUnfused;
    }
    if (value == "naive-one-pass-fused" || value == "naive_one_pass_fused") {
        return Variant::NaiveOnePassFused;
    }
    if (value == "tiled-one-pass-fused" || value == "tiled_one_pass_fused") {
        return Variant::TiledOnePassFused;
    }
    if (value == "all") {
        return Variant::All;
    }
    throw std::invalid_argument("invalid variant: " + value);
}

BenchmarkMode parse_mode(const std::string& value) {
    if (value == "force-variant") {
        return BenchmarkMode::ForceVariant;
    }
    if (value == "use-plan") {
        return BenchmarkMode::UsePlan;
    }
    if (value == "sweep-candidates") {
        return BenchmarkMode::SweepCandidates;
    }
    throw std::invalid_argument("invalid mode: " + value);
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [--mode force-variant|use-plan|sweep-candidates]\n"
        << "       [--execution-plan PATH]\n"
        << "       [--pattern bias|elementwise-add|all]\n"
        << "       [--variant naive-unfused|tiled-unfused|naive-one-pass-fused|tiled-one-pass-fused|all]\n"
        << "       [--m M] [--n N] [--k K]\n"
        << "       [--warmup N] [--iterations N] [--repeats N]\n"
        << "       [--output PATH] [--report-output PATH]\n";
}

int parse_positive_int(const std::string& name, const std::string& value) {
    try {
        size_t parsed = 0;
        int result = std::stoi(value, &parsed);
        if (parsed != value.size() || result <= 0) {
            throw std::invalid_argument("not a positive integer");
        }
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be a positive integer");
    }
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }

        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(name + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--pattern") {
            config.pattern = parse_pattern(require_value(arg));
        } else if (arg == "--mode") {
            config.mode = parse_mode(require_value(arg));
        } else if (arg == "--execution-plan") {
            config.execution_plan_path = require_value(arg);
        } else if (arg == "--variant") {
            config.variant = parse_variant(require_value(arg));
        } else if (arg == "--m") {
            config.m = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--n") {
            config.n = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--k") {
            config.k = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--warmup") {
            config.warmup = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--iterations") {
            config.iterations = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--repeats") {
            config.repeats = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--tile-size") {
            config.tile_size = parse_positive_int(arg, require_value(arg));
        } else if (arg == "--output") {
            config.output_path = require_value(arg);
        } else if (arg == "--report-output") {
            config.report_path = require_value(arg);
        } else if (arg == "--force-invalid-bias-shape") {
            config.force_invalid_bias_shape = true;
        } else if (arg == "--force-invalid-elementwise-shape") {
            config.force_invalid_elementwise_shape = true;
        } else if (arg == "--smoke-test-mode") {
            config.smoke_test_mode = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (config.warmup <= 0 && !config.smoke_test_mode) {
        throw std::invalid_argument("warmup must be nonzero unless --smoke-test-mode is set");
    }
    if (config.mode == BenchmarkMode::UsePlan && config.pattern == Pattern::All) {
        throw std::invalid_argument("--mode use-plan requires a concrete --pattern");
    }
    return config;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = (static_cast<double>(values.size() - 1) * p) / 100.0;
    const auto lower = static_cast<size_t>(std::floor(rank));
    const auto upper = static_cast<size_t>(std::ceil(rank));
    if (lower == upper) {
        return values[lower];
    }
    const double weight = rank - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

Stats summarize(const std::vector<double>& values) {
    Stats stats;
    stats.sample_count = static_cast<int>(values.size());
    if (values.empty()) {
        return stats;
    }
    stats.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    stats.median_ms = percentile(values, 50.0);
    stats.p50_ms = stats.median_ms;
    stats.p95_ms = percentile(values, 95.0);
    stats.min_ms = *std::min_element(values.begin(), values.end());
    stats.max_ms = *std::max_element(values.begin(), values.end());

    if (values.size() > 1) {
        double sum_squared_diff = 0.0;
        for (double value : values) {
            const double diff = value - stats.mean_ms;
            sum_squared_diff += diff * diff;
        }
        stats.stddev_ms = std::sqrt(
            sum_squared_diff / static_cast<double>(values.size() - 1)
        );
    }
    if (stats.mean_ms > 0.0) {
        stats.coefficient_of_variation = stats.stddev_ms / stats.mean_ms;
    }
    return stats;
}

void validate_shapes(Pattern pattern, const Tensor& a, const Tensor& b, const Tensor& postop) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument("A and B must be rank-2 tensors");
    }
    const int m = a.shape[0];
    const int k = a.shape[1];
    const int n = b.shape[1];
    if (b.shape[0] != k) {
        throw std::invalid_argument("MatMul K dimension mismatch");
    }
    if (pattern == Pattern::Bias) {
        const bool rank1_bias = postop.shape.size() == 1 && postop.shape[0] == n;
        if (!rank1_bias) {
            throw std::invalid_argument("bias pattern requires bias shape [N]");
        }
    } else if (pattern == Pattern::ElementwiseAdd) {
        const bool elementwise = postop.shape.size() == 2 &&
            postop.shape[0] == m &&
            postop.shape[1] == n;
        if (!elementwise) {
            throw std::invalid_argument("elementwise-add pattern requires addend shape [M,N]");
        }
    }
}

float postop_value(Pattern pattern, const Tensor& postop, int i, int j, int n) {
    if (pattern == Pattern::Bias) {
        return postop.data[j];
    }
    return postop.data[i * n + j];
}

void add_postop_separate(Pattern pattern, const Tensor& matmul_out, const Tensor& postop, Tensor& add_out) {
    const int m = matmul_out.shape[0];
    const int n = matmul_out.shape[1];
    if (pattern == Pattern::Bias) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                add_out.data[i * n + j] = matmul_out.data[i * n + j] + postop.data[j];
            }
        }
    } else {
        add(matmul_out, postop, add_out);
    }
}

void matmul_naive_f32(const Tensor& a, const Tensor& b, Tensor& out) {
    matmul(a, b, out);
}

void matmul_tiled_f32_same_strategy_as_fused(
    const Tensor& a,
    const Tensor& b,
    Tensor& out,
    int tile_size,
    std::vector<float>& tile_scratch
) {
    const int m = a.shape[0];
    const int k_dim = a.shape[1];
    const int n = b.shape[1];

    for (int ii = 0; ii < m; ii += tile_size) {
        for (int jj = 0; jj < n; jj += tile_size) {
            const int i_end = std::min(ii + tile_size, m);
            const int j_end = std::min(jj + tile_size, n);
            const int tile_rows = i_end - ii;
            const int tile_cols = j_end - jj;
            const size_t scratch_size = static_cast<size_t>(tile_rows * tile_cols);
            if (tile_scratch.size() < scratch_size) {
                throw std::runtime_error("tile scratch buffer is too small");
            }
            std::fill(tile_scratch.begin(), tile_scratch.begin() + scratch_size, 0.0f);

            for (int kk = 0; kk < k_dim; kk += tile_size) {
                const int k_end = std::min(kk + tile_size, k_dim);
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        const float a_value = a.data[i * k_dim + k];
                        for (int j = jj; j < j_end; ++j) {
                            tile_scratch[(i - ii) * tile_cols + (j - jj)] +=
                                a_value * b.data[k * n + j];
                        }
                    }
                }
            }

            for (int i = ii; i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    out.data[i * n + j] = tile_scratch[(i - ii) * tile_cols + (j - jj)];
                }
            }
        }
    }
}

void run_naive_unfused(
    Pattern pattern,
    const Tensor& a,
    const Tensor& b,
    const Tensor& postop,
    Tensor& matmul_out,
    Tensor& add_out,
    Tensor& out
) {
    matmul_naive_f32(a, b, matmul_out);
    add_postop_separate(pattern, matmul_out, postop, add_out);
    relu(add_out, out);
}

void run_tiled_unfused(
    Pattern pattern,
    const Tensor& a,
    const Tensor& b,
    const Tensor& postop,
    Tensor& matmul_out,
    Tensor& add_out,
    Tensor& out,
    int tile_size,
    std::vector<float>& tile_scratch
) {
    matmul_tiled_f32_same_strategy_as_fused(a, b, matmul_out, tile_size, tile_scratch);
    add_postop_separate(pattern, matmul_out, postop, add_out);
    relu(add_out, out);
}

void matmul_postop_relu_naive_one_pass_f32(
    Pattern pattern,
    const Tensor& a,
    const Tensor& b,
    const Tensor& postop,
    Tensor& out
) {
    const int m = a.shape[0];
    const int k_dim = a.shape[1];
    const int n = b.shape[1];

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < k_dim; ++k) {
                sum += a.data[i * k_dim + k] * b.data[k * n + j];
            }
            const float with_postop = sum + postop_value(pattern, postop, i, j, n);
            out.data[i * n + j] = std::max(0.0f, with_postop);
        }
    }
}

void run_naive_one_pass_fused(
    Pattern pattern,
    const Tensor& a,
    const Tensor& b,
    const Tensor& postop,
    Tensor& out
) {
    matmul_postop_relu_naive_one_pass_f32(pattern, a, b, postop, out);
}

void matmul_postop_relu_tiled_one_pass_f32(
    Pattern pattern,
    const Tensor& a,
    const Tensor& b,
    const Tensor& postop,
    Tensor& out,
    int tile_size,
    std::vector<float>& tile_scratch
) {
    const int m = a.shape[0];
    const int k_dim = a.shape[1];
    const int n = b.shape[1];

    for (int ii = 0; ii < m; ii += tile_size) {
        for (int jj = 0; jj < n; jj += tile_size) {
            const int i_end = std::min(ii + tile_size, m);
            const int j_end = std::min(jj + tile_size, n);
            const int tile_rows = i_end - ii;
            const int tile_cols = j_end - jj;
            const size_t scratch_size = static_cast<size_t>(tile_rows * tile_cols);
            if (tile_scratch.size() < scratch_size) {
                throw std::runtime_error("tile scratch buffer is too small");
            }
            std::fill(tile_scratch.begin(), tile_scratch.begin() + scratch_size, 0.0f);

            for (int kk = 0; kk < k_dim; kk += tile_size) {
                const int k_end = std::min(kk + tile_size, k_dim);
                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {
                        const float a_value = a.data[i * k_dim + k];
                        for (int j = jj; j < j_end; ++j) {
                            tile_scratch[(i - ii) * tile_cols + (j - jj)] +=
                                a_value * b.data[k * n + j];
                        }
                    }
                }
            }

            for (int i = ii; i < i_end; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    const float with_postop =
                        tile_scratch[(i - ii) * tile_cols + (j - jj)] +
                        postop_value(pattern, postop, i, j, n);
                    out.data[i * n + j] = std::max(0.0f, with_postop);
                }
            }
        }
    }
}

void run_tiled_one_pass_fused(
    Pattern pattern,
    const Tensor& a,
    const Tensor& b,
    const Tensor& postop,
    Tensor& out,
    int tile_size,
    std::vector<float>& tile_scratch
) {
    matmul_postop_relu_tiled_one_pass_f32(pattern, a, b, postop, out, tile_size, tile_scratch);
}

ImplementationProperties implementation_properties(Variant variant, int tile_size) {
    switch (variant) {
        case Variant::NaiveUnfused:
            return {"naive", "separate", 2, 2, 1, 0};
        case Variant::TiledUnfused:
            return {"tiled", "separate", 2, 2, 1, tile_size};
        case Variant::NaiveOnePassFused:
            return {"naive", "one_pass_fused", 0, 0, 1, 0};
        case Variant::TiledOnePassFused:
            return {"tiled", "one_pass_fused", 0, 0, 1, tile_size};
        case Variant::All:
            break;
    }
    throw std::invalid_argument("implementation metadata is not defined for variant all");
}

std::string function_name_for_variant(Variant variant) {
    switch (variant) {
        case Variant::NaiveUnfused: return "run_naive_unfused";
        case Variant::TiledUnfused: return "run_tiled_unfused";
        case Variant::NaiveOnePassFused: return "run_naive_one_pass_fused";
        case Variant::TiledOnePassFused: return "run_tiled_one_pass_fused";
        case Variant::All: break;
    }
    return "unknown";
}

void execute_variant_once(
    Variant variant,
    Pattern pattern,
    const Inputs& inputs,
    Tensor& matmul_out,
    Tensor& add_out,
    Tensor& out,
    int tile_size,
    std::vector<float>& tile_scratch
) {
    switch (variant) {
        case Variant::NaiveUnfused:
            run_naive_unfused(pattern, inputs.a, inputs.b, inputs.postop, matmul_out, add_out, out);
            return;
        case Variant::TiledUnfused:
            run_tiled_unfused(pattern, inputs.a, inputs.b, inputs.postop, matmul_out, add_out, out, tile_size, tile_scratch);
            return;
        case Variant::NaiveOnePassFused:
            run_naive_one_pass_fused(pattern, inputs.a, inputs.b, inputs.postop, out);
            return;
        case Variant::TiledOnePassFused:
            run_tiled_one_pass_fused(pattern, inputs.a, inputs.b, inputs.postop, out, tile_size, tile_scratch);
            return;
        case Variant::All:
            break;
    }
    throw std::invalid_argument("cannot execute variant all directly");
}

class RuntimeKernelDispatcher {
public:
    RuntimeKernelDispatcher(Pattern pattern, Variant variant) {
        register_kernel(kernel_id_for(pattern, variant), pattern, variant);
    }

    RuntimeKernelDispatcher() {
        for (Pattern pattern : {Pattern::Bias, Pattern::ElementwiseAdd}) {
            for (Variant variant : selected_variants(Variant::All)) {
                register_kernel(kernel_id_for(pattern, variant), pattern, variant);
            }
        }
    }

    RuntimeTrace dispatch_kernel(
        const std::string& execution_plan_path,
        const PlannedOperation& planned_op,
        const Inputs& inputs,
        Tensor& matmul_out,
        Tensor& add_out,
        Tensor& out,
        std::vector<float>& tile_scratch
    ) const {
        auto it = kernels_.find(planned_op.kernel_id);
        if (it == kernels_.end()) {
            throw std::runtime_error("no runtime kernel registered for kernel_id: " + planned_op.kernel_id);
        }
        RuntimeTrace trace;
        trace.execution_plan = execution_plan_path;
        trace.planned_kernel = planned_op.kernel_id;
        trace.actual_dispatched_kernel = planned_op.kernel_id;
        trace.backend = planned_op.backend;
        trace.kernel_config = planned_op.kernel_config;
        trace.dispatch_count = dispatch_count_for(variant_from_kernel_id(planned_op.kernel_id));
        trace.plan_matched_runtime = trace.planned_kernel == trace.actual_dispatched_kernel;

        it->second(planned_op, inputs, matmul_out, add_out, out, tile_scratch);
        return trace;
    }

private:
    using DispatchFn = std::function<void(
        const PlannedOperation&,
        const Inputs&,
        Tensor&,
        Tensor&,
        Tensor&,
        std::vector<float>&
    )>;

    static int dispatch_count_for(Variant variant) {
        return (variant == Variant::NaiveUnfused || variant == Variant::TiledUnfused) ? 3 : 1;
    }

    void register_kernel(const std::string& kernel_id, Pattern pattern, Variant variant) {
        kernels_[kernel_id] = [pattern, variant](
            const PlannedOperation& planned_op,
            const Inputs& inputs,
            Tensor& matmul_out,
            Tensor& add_out,
            Tensor& out,
            std::vector<float>& tile_scratch
        ) {
            execute_variant_once(
                variant,
                pattern,
                inputs,
                matmul_out,
                add_out,
                out,
                planned_op.kernel_config.tile_m,
                tile_scratch
            );
        };
    }

    std::unordered_map<std::string, DispatchFn> kernels_;
};

void fill_inputs(Inputs& inputs, Pattern pattern) {
    for (int i = 0; i < inputs.a.numel(); ++i) {
        inputs.a.data[i] = static_cast<float>((i % 13) - 6) * 0.017f;
    }
    for (int i = 0; i < inputs.b.numel(); ++i) {
        inputs.b.data[i] = static_cast<float>((i % 17) - 8) * 0.011f;
    }
    for (int i = 0; i < inputs.postop.numel(); ++i) {
        if (pattern == Pattern::Bias) {
            inputs.postop.data[i] = static_cast<float>((i % 7) - 3) * 0.031f;
        } else {
            inputs.postop.data[i] = static_cast<float>((i % 11) - 5) * 0.019f;
        }
    }
}

Inputs make_inputs(const BenchmarkConfig& config, Pattern pattern) {
    std::vector<int> postop_shape;
    if (pattern == Pattern::Bias) {
        postop_shape = config.force_invalid_bias_shape
            ? std::vector<int>{config.m, config.n}
            : std::vector<int>{config.n};
    } else {
        postop_shape = config.force_invalid_elementwise_shape
            ? std::vector<int>{config.n}
            : std::vector<int>{config.m, config.n};
    }
    Inputs inputs(config.m, config.k, config.n, postop_shape);
    fill_inputs(inputs, pattern);
    validate_shapes(pattern, inputs.a, inputs.b, inputs.postop);
    return inputs;
}

CorrectnessResult compare_outputs(
    const Tensor& expected,
    const Tensor& actual,
    double atol,
    double rtol
) {
    CorrectnessResult result;
    result.atol = atol;
    result.rtol = rtol;
    result.passed = expected.data.size() == actual.data.size();
    if (!result.passed) {
        return result;
    }

    for (size_t i = 0; i < expected.data.size(); ++i) {
        const float exp = expected.data[i];
        const float got = actual.data[i];
        result.contains_nan = result.contains_nan || std::isnan(exp) || std::isnan(got);
        result.contains_inf = result.contains_inf || std::isinf(exp) || std::isinf(got);
        const double abs_error = std::fabs(static_cast<double>(got) - static_cast<double>(exp));
        const double rel_denom = std::max(std::fabs(static_cast<double>(exp)), 1e-12);
        const double rel_error = abs_error / rel_denom;
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);
        if (abs_error > atol + rtol * std::fabs(static_cast<double>(exp))) {
            result.passed = false;
        }
    }

    if (result.contains_nan || result.contains_inf) {
        result.passed = false;
    }
    return result;
}

VariantResult benchmark_variant(
    Variant variant,
    Pattern pattern,
    const BenchmarkConfig& config,
    const Inputs& inputs,
    const Tensor& reference
) {
    Tensor matmul_out("matmul_out", {config.m, config.n});
    Tensor add_out("add_out", {config.m, config.n});
    Tensor out("out", {config.m, config.n});
    std::vector<float> tile_scratch(
        static_cast<size_t>(config.tile_size * config.tile_size),
        0.0f
    );

    execute_variant_once(
        variant,
        pattern,
        inputs,
        matmul_out,
        add_out,
        out,
        config.tile_size,
        tile_scratch
    );

    VariantResult result;
    result.variant = variant;
    result.function_name = function_name_for_variant(variant);
    result.kernel_id = kernel_id_for(pattern, variant);
    result.correctness = compare_outputs(reference, out, config.atol, config.rtol);
    result.implementation = implementation_properties(variant, config.tile_size);
    result.samples_ms.reserve(config.repeats);

    for (int repeat = 0; repeat < config.repeats; ++repeat) {
        for (int i = 0; i < config.warmup; ++i) {
            execute_variant_once(
                variant,
                pattern,
                inputs,
                matmul_out,
                add_out,
                out,
                config.tile_size,
                tile_scratch
            );
        }

        Timer timer;
        timer.start();
        for (int i = 0; i < config.iterations; ++i) {
            execute_variant_once(
                variant,
                pattern,
                inputs,
                matmul_out,
                add_out,
                out,
                config.tile_size,
                tile_scratch
            );
        }
        result.samples_ms.push_back(timer.stop_ms() / static_cast<double>(config.iterations));
    }
    result.stats = summarize(result.samples_ms);
    return result;
}

VariantResult benchmark_plan_operation(
    Pattern pattern,
    const BenchmarkConfig& config,
    const PlannedOperation& planned_op,
    const Inputs& inputs,
    const Tensor& reference
) {
    Tensor matmul_out("matmul_out", {config.m, config.n});
    Tensor add_out("add_out", {config.m, config.n});
    Tensor out("out", {config.m, config.n});
    std::vector<float> tile_scratch(
        static_cast<size_t>(planned_op.kernel_config.tile_m * planned_op.kernel_config.tile_n),
        0.0f
    );

    RuntimeKernelDispatcher dispatcher;
    RuntimeTrace trace = dispatcher.dispatch_kernel(
        config.execution_plan_path,
        planned_op,
        inputs,
        matmul_out,
        add_out,
        out,
        tile_scratch
    );
    if (!trace.plan_matched_runtime) {
        throw std::runtime_error("planned kernel does not match actual dispatched kernel");
    }

    Variant variant = variant_from_kernel_id(planned_op.kernel_id);
    VariantResult result;
    result.variant = variant;
    result.function_name = function_name_for_variant(variant);
    result.kernel_id = planned_op.kernel_id;
    result.correctness = compare_outputs(reference, out, config.atol, config.rtol);
    result.implementation = implementation_properties(variant, planned_op.kernel_config.tile_m);
    result.runtime_trace = trace;
    result.has_runtime_trace = true;
    result.samples_ms.reserve(config.repeats);

    for (int repeat = 0; repeat < config.repeats; ++repeat) {
        for (int i = 0; i < config.warmup; ++i) {
            trace = dispatcher.dispatch_kernel(
                config.execution_plan_path,
                planned_op,
                inputs,
                matmul_out,
                add_out,
                out,
                tile_scratch
            );
        }

        Timer timer;
        timer.start();
        for (int i = 0; i < config.iterations; ++i) {
            trace = dispatcher.dispatch_kernel(
                config.execution_plan_path,
                planned_op,
                inputs,
                matmul_out,
                add_out,
                out,
                tile_scratch
            );
        }
        result.samples_ms.push_back(timer.stop_ms() / static_cast<double>(config.iterations));
    }

    result.runtime_trace = trace;
    result.stats = summarize(result.samples_ms);
    return result;
}

PatternResult benchmark_pattern(Pattern pattern, const BenchmarkConfig& config) {
    Inputs inputs = make_inputs(config, pattern);
    Tensor matmul_out("reference_matmul_out", {config.m, config.n});
    Tensor add_out("reference_add_out", {config.m, config.n});
    Tensor reference("reference", {config.m, config.n});
    run_naive_unfused(pattern, inputs.a, inputs.b, inputs.postop, matmul_out, add_out, reference);

    PatternResult result;
    result.pattern = pattern;

    if (config.mode == BenchmarkMode::UsePlan) {
        RuntimeExecutionPlan plan = load_execution_plan(config.execution_plan_path);
        PlannedOperation planned_op = select_and_validate_plan_operation(plan, pattern);
        result.variants.push_back(
            benchmark_plan_operation(pattern, config, planned_op, inputs, reference)
        );
        return result;
    }

    const Variant variants_to_run =
        config.mode == BenchmarkMode::SweepCandidates ? Variant::All : config.variant;
    for (Variant variant : selected_variants(variants_to_run)) {
        result.variants.push_back(benchmark_variant(variant, pattern, config, inputs, reference));
    }

    std::vector<VariantResult*> ranked;
    for (auto& variant : result.variants) {
        ranked.push_back(&variant);
    }
    std::sort(ranked.begin(), ranked.end(), [](const VariantResult* lhs, const VariantResult* rhs) {
        return lhs->stats.mean_ms < rhs->stats.mean_ms;
    });
    for (size_t i = 0; i < ranked.size(); ++i) {
        ranked[i]->rank = static_cast<int>(i + 1);
        ranked[i]->oracle_best = i == 0;
    }
    return result;
}

const VariantResult* find_variant(const PatternResult& result, Variant variant) {
    for (const auto& candidate : result.variants) {
        if (candidate.variant == variant) {
            return &candidate;
        }
    }
    return nullptr;
}

double speedup(const VariantResult* baseline, const VariantResult* candidate) {
    if (!baseline || !candidate || candidate->stats.mean_ms <= 0.0) {
        return 0.0;
    }
    return baseline->stats.mean_ms / candidate->stats.mean_ms;
}

double latency_reduction_percent(double speedup_value) {
    if (speedup_value <= 0.0) {
        return 0.0;
    }
    return (1.0 - (1.0 / speedup_value)) * 100.0;
}

void write_stats_json(std::ostream& out, const Stats& stats, int indent) {
    const std::string pad(indent, ' ');
    out << pad << "\"sample_count\": " << stats.sample_count << ",\n";
    out << pad << "\"mean_ms\": " << stats.mean_ms << ",\n";
    out << pad << "\"median_ms\": " << stats.median_ms << ",\n";
    out << pad << "\"p50_ms\": " << stats.p50_ms << ",\n";
    out << pad << "\"p95_ms\": " << stats.p95_ms << ",\n";
    out << pad << "\"min_ms\": " << stats.min_ms << ",\n";
    out << pad << "\"max_ms\": " << stats.max_ms << ",\n";
    out << pad << "\"stddev_ms\": " << stats.stddev_ms << ",\n";
    out << pad << "\"coefficient_of_variation\": " << stats.coefficient_of_variation << "\n";
}

void write_correctness_json(std::ostream& out, const CorrectnessResult& correctness, int indent) {
    const std::string pad(indent, ' ');
    out << pad << "\"passed\": " << (correctness.passed ? "true" : "false") << ",\n";
    out << pad << "\"atol\": " << correctness.atol << ",\n";
    out << pad << "\"rtol\": " << correctness.rtol << ",\n";
    out << pad << "\"max_abs_error\": " << correctness.max_abs_error << ",\n";
    out << pad << "\"max_rel_error\": " << correctness.max_rel_error << ",\n";
    out << pad << "\"contains_nan\": " << (correctness.contains_nan ? "true" : "false") << ",\n";
    out << pad << "\"contains_inf\": " << (correctness.contains_inf ? "true" : "false") << "\n";
}

void write_implementation_json(std::ostream& out, const ImplementationProperties& impl, int indent) {
    const std::string pad(indent, ' ');
    out << pad << "\"matmul_strategy\": \"" << impl.matmul_strategy << "\",\n";
    out << pad << "\"post_op_strategy\": \"" << impl.post_op_strategy << "\",\n";
    out << pad << "\"intermediate_tensor_count\": " << impl.intermediate_tensor_count << ",\n";
    out << pad << "\"full_output_post_op_passes\": " << impl.full_output_post_op_passes << ",\n";
    out << pad << "\"final_output_store_passes\": " << impl.final_output_store_passes << ",\n";
    out << pad << "\"tile_size\": " << impl.tile_size << "\n";
}

void write_runtime_trace_json(std::ostream& out, const RuntimeTrace& trace, int indent) {
    const std::string pad(indent, ' ');
    out << pad << "\"execution_plan\": \"" << json_escape(trace.execution_plan) << "\",\n";
    out << pad << "\"planned_kernel\": \"" << json_escape(trace.planned_kernel) << "\",\n";
    out << pad << "\"actual_dispatched_kernel\": \"" << json_escape(trace.actual_dispatched_kernel) << "\",\n";
    out << pad << "\"backend\": \"" << json_escape(trace.backend) << "\",\n";
    out << pad << "\"dispatch_count\": " << trace.dispatch_count << ",\n";
    out << pad << "\"kernel_config\": {\n";
    out << pad << "  \"tile_m\": " << trace.kernel_config.tile_m << ",\n";
    out << pad << "  \"tile_n\": " << trace.kernel_config.tile_n << ",\n";
    out << pad << "  \"tile_k\": " << trace.kernel_config.tile_k << "\n";
    out << pad << "},\n";
    out << pad << "\"plan_matched_runtime\": "
        << (trace.plan_matched_runtime ? "true" : "false") << "\n";
}

void write_samples_json(std::ostream& out, const std::vector<double>& samples) {
    out << "[";
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << samples[i];
    }
    out << "]";
}

void write_comparisons_json(std::ostream& out, const PatternResult& result, int indent) {
    const auto* v0 = find_variant(result, Variant::NaiveUnfused);
    const auto* v1 = find_variant(result, Variant::TiledUnfused);
    const auto* v2 = find_variant(result, Variant::NaiveOnePassFused);
    const auto* v3 = find_variant(result, Variant::TiledOnePassFused);
    const std::string pad(indent, ' ');
    const std::array<std::pair<std::string, double>, 4> comparisons{{
        {"tiling_speedup", speedup(v0, v1)},
        {"fusion_speedup_naive", speedup(v0, v2)},
        {"fusion_speedup_fair", speedup(v1, v3)},
        {"full_stack_speedup", speedup(v0, v3)},
    }};

    for (size_t i = 0; i < comparisons.size(); ++i) {
        out << pad << "\"" << comparisons[i].first << "\": {\n";
        out << pad << "  \"speedup\": " << comparisons[i].second << ",\n";
        out << pad << "  \"latency_reduction_percent\": "
            << latency_reduction_percent(comparisons[i].second) << "\n";
        out << pad << "}";
        if (i + 1 < comparisons.size()) {
            out << ",";
        }
        out << "\n";
    }
}

std::string compiler_id() {
#if defined(__clang__)
    return "clang " __clang_version__;
#elif defined(__GNUC__)
    return "gcc " __VERSION__;
#else
    return "unknown";
#endif
}

void write_json(
    const std::string& path,
    const BenchmarkConfig& config,
    const std::vector<PatternResult>& results
) {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"schema\": \"kernel_benchmark_profile\",\n";
    out << "  \"schema_version\": 2,\n";
    out << "  \"benchmark\": \"matmul_postop_relu\",\n";
    out << "  \"source\": \"apps/run_mlir_fused_kernel_benchmark.cpp\",\n";
    out << "  \"mode\": \"" << mode_label(config.mode) << "\",\n";
    out << "  \"execution_plan\": \"" << json_escape(config.execution_plan_path) << "\",\n";
    out << "  \"build\": {\n";
    out << "    \"type\": \"Release\",\n";
    out << "    \"compiler\": \"" << json_escape(compiler_id()) << "\",\n";
#ifdef NDEBUG
    out << "    \"ndebug\": true,\n";
#else
    out << "    \"ndebug\": false,\n";
#endif
    out << "    \"flags\": [\"Release\", \"-O3\", \"-DNDEBUG\"]\n";
    out << "  },\n";
    out << "  \"machine\": {\n";
    out << "    \"hostname\": \"" << json_escape(run_command_capture("hostname")) << "\",\n";
    out << "    \"uname\": \"" << json_escape(run_command_capture("uname -a")) << "\",\n";
    out << "    \"cpu_summary\": \"" << json_escape(run_command_capture("lscpu 2>/dev/null | sed -n '1,16p'")) << "\",\n";
    out << "    \"nproc\": \"" << json_escape(run_command_capture("nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null")) << "\",\n";
    out << "    \"cpu_affinity\": \"" << json_escape(run_command_capture("taskset -pc $$ 2>/dev/null || true")) << "\"\n";
    out << "  },\n";
    out << "  \"configuration\": {\n";
    out << "    \"warmup\": " << config.warmup << ",\n";
    out << "    \"iterations\": " << config.iterations << ",\n";
    out << "    \"repeats\": " << config.repeats << ",\n";
    out << "    \"dtype\": \"f32\",\n";
    out << "    \"m\": " << config.m << ",\n";
    out << "    \"n\": " << config.n << ",\n";
    out << "    \"k\": " << config.k << ",\n";
    out << "    \"tile_size\": " << config.tile_size << "\n";
    out << "  },\n";
    out << "  \"patterns\": {\n";
    for (size_t p = 0; p < results.size(); ++p) {
        const auto& pattern_result = results[p];
        out << "    \"" << pattern_key(pattern_result.pattern) << "\": {\n";
        out << "      \"postop_semantics\": \""
            << (pattern_result.pattern == Pattern::Bias ? "bias_shape_N" : "elementwise_add_shape_MxN")
            << "\",\n";
        out << "      \"variants\": {\n";
        for (size_t v = 0; v < pattern_result.variants.size(); ++v) {
            const auto& variant = pattern_result.variants[v];
            out << "        \"" << variant_key(variant.variant) << "\": {\n";
            out << "          \"function\": \"" << variant.function_name << "\",\n";
            out << "          \"kernel_id\": \"" << variant.kernel_id << "\",\n";
            out << "          \"rank\": " << variant.rank << ",\n";
            out << "          \"oracle_best\": " << (variant.oracle_best ? "true" : "false") << ",\n";
            out << "          \"samples_ms\": ";
            write_samples_json(out, variant.samples_ms);
            out << ",\n";
            out << "          \"statistics\": {\n";
            write_stats_json(out, variant.stats, 12);
            out << "          },\n";
            out << "          \"correctness\": {\n";
            write_correctness_json(out, variant.correctness, 12);
            out << "          },\n";
            out << "          \"implementation_properties\": {\n";
            write_implementation_json(out, variant.implementation, 12);
            out << "          }\n";
            if (variant.has_runtime_trace) {
                out << ",\n";
                out << "          \"runtime_trace\": {\n";
                write_runtime_trace_json(out, variant.runtime_trace, 12);
                out << "          }\n";
            }
            out << "        }";
            if (v + 1 < pattern_result.variants.size()) {
                out << ",";
            }
            out << "\n";
        }
        out << "      },\n";
        out << "      \"comparisons\": {\n";
        write_comparisons_json(out, pattern_result, 8);
        out << "      }\n";
        out << "    }";
        if (p + 1 < results.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  }\n";
    out << "}\n";
}

void write_report(
    const std::string& path,
    const BenchmarkConfig& config,
    const std::vector<PatternResult>& results
) {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(6);
    out << "# MatMul Post-Op ReLU Benchmark Report\n\n";
    out << "## Configuration\n\n";
    out << "- Mode: `" << mode_label(config.mode) << "`\n";
    out << "- Execution plan: `" << config.execution_plan_path << "`\n";
    out << "- Build type: `Release`\n";
    out << "- Warmup iterations per repeat: `" << config.warmup << "`\n";
    out << "- Measured iterations per repeat: `" << config.iterations << "`\n";
    out << "- Repeats: `" << config.repeats << "`\n";
    out << "- Shape: `M=" << config.m << ", K=" << config.k << ", N=" << config.n << "`\n";
    out << "- Tile size: `" << config.tile_size << "`\n";
    out << "- Host: `" << run_command_capture("hostname") << "`\n";
    out << "- Machine: `" << run_command_capture("uname -a") << "`\n\n";

    out << "## Variant Results\n\n";
    out << "| Pattern | Variant | Kernel | Mean ms | p50 ms | p95 ms | Stddev | CV | Correct | Rank |\n";
    out << "| ------- | ------- | ------ | ------: | -----: | -----: | -----: | -: | ------: | ---: |\n";
    for (const auto& pattern_result : results) {
        for (const auto& variant : pattern_result.variants) {
            out << "| " << pattern_key(pattern_result.pattern)
                << " | " << variant_key(variant.variant)
                << " | " << variant.kernel_id
                << " | " << variant.stats.mean_ms
                << " | " << variant.stats.p50_ms
                << " | " << variant.stats.p95_ms
                << " | " << variant.stats.stddev_ms
                << " | " << variant.stats.coefficient_of_variation
                << " | " << (variant.correctness.passed ? "true" : "false")
                << " | " << variant.rank
                << " |\n";
        }
    }

    out << "\n## Runtime Trace\n\n";
    for (const auto& pattern_result : results) {
        for (const auto& variant : pattern_result.variants) {
            if (!variant.has_runtime_trace) {
                continue;
            }
            out << "- Pattern `" << pattern_key(pattern_result.pattern) << "` planned `"
                << variant.runtime_trace.planned_kernel << "`, dispatched `"
                << variant.runtime_trace.actual_dispatched_kernel << "`, dispatch_count=`"
                << variant.runtime_trace.dispatch_count << "`, plan_matched_runtime=`"
                << (variant.runtime_trace.plan_matched_runtime ? "true" : "false")
                << "`.\n";
        }
    }

    out << "\n## Comparisons\n\n";
    out << "Fair fusion comparison: `tiled_unfused` vs `tiled_one_pass_fused`.\n\n";
    out << "Full-stack comparison: `naive_unfused` vs `tiled_one_pass_fused`.\n\n";
    out << "| Pattern | Comparison | Speedup | Latency reduction |\n";
    out << "| ------- | ---------- | ------: | ----------------: |\n";
    for (const auto& pattern_result : results) {
        const auto* v0 = find_variant(pattern_result, Variant::NaiveUnfused);
        const auto* v1 = find_variant(pattern_result, Variant::TiledUnfused);
        const auto* v2 = find_variant(pattern_result, Variant::NaiveOnePassFused);
        const auto* v3 = find_variant(pattern_result, Variant::TiledOnePassFused);
        const std::array<std::pair<std::string, double>, 4> comparisons{{
            {"tiling_speedup", speedup(v0, v1)},
            {"fusion_speedup_naive", speedup(v0, v2)},
            {"fusion_speedup_fair", speedup(v1, v3)},
            {"full_stack_speedup", speedup(v0, v3)},
        }};
        for (const auto& comparison : comparisons) {
            out << "| " << pattern_key(pattern_result.pattern)
                << " | " << comparison.first
                << " | " << comparison.second
                << " | " << latency_reduction_percent(comparison.second) << "% |\n";
        }
    }

    out << "\n## One-Pass Evidence\n\n";
    out << "- `run_naive_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.\n";
    out << "- `run_tiled_one_pass_fused`: `intermediate_tensor_count=0`, `full_output_post_op_passes=0`, `final_output_store_passes=1`.\n";
    out << "- These are static implementation properties, not measured hardware memory traffic.\n";
}

bool all_correct(const std::vector<PatternResult>& results) {
    for (const auto& pattern : results) {
        for (const auto& variant : pattern.variants) {
            if (!variant.correctness.passed) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    BenchmarkConfig config;
    try {
        config = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        std::vector<PatternResult> results;
        for (Pattern pattern : selected_patterns(config.pattern)) {
            results.push_back(benchmark_pattern(pattern, config));
        }

        write_json(config.output_path, config, results);
        write_report(config.report_path, config, results);

        std::cout << "=== MatMul Post-Op ReLU Benchmark ===\n";
        std::cout << "Mode: " << mode_label(config.mode) << "\n";
        std::cout << "Execution plan: " << config.execution_plan_path << "\n";
        std::cout << "Pattern: " << pattern_label(config.pattern) << "\n";
        std::cout << "Variant: " << variant_label(config.variant) << "\n";
        std::cout << "Shape: " << config.m << "x" << config.k << " * "
                  << config.k << "x" << config.n << "\n";
        std::cout << "Tile size: " << config.tile_size << "\n";
        std::cout << "Warmup iterations: " << config.warmup << "\n";
        std::cout << "Measured iterations: " << config.iterations << "\n";
        std::cout << "Repeats: " << config.repeats << "\n";
        std::cout << "JSON: " << config.output_path << "\n";
        std::cout << "Report: " << config.report_path << "\n";
        std::cout << "Correctness: " << (all_correct(results) ? "PASSED" : "FAILED") << "\n";

        return all_correct(results) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
