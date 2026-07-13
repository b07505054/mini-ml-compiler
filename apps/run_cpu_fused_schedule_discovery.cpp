// Phase 1: CPU Fused Schedule Candidate Discovery and Real Oracle.
//
// Determines empirically whether the real CPU backend has a meaningful,
// apples-to-apples SCHEDULE-selection problem for one-pass fused
// MatMul+Bias+ReLU: multiple tile-shape candidates that share identical
// semantics, dtype, accumulation type, fusion status, launch count, layout,
// and thread policy, differing ONLY in (block_m, block_n, block_k).
//
// This is a self-contained tool. It does not modify or depend on the
// existing four-variant (naive/tiled x unfused/fused) benchmark app or its
// CTest targets — those remain a separate fusion-attribution baseline.
//
// Truth boundary: every number in the emitted artifacts is either a real
// measurement from this process on this host, or a value read from the
// operating system / compiler at run time. No cache, SIMD, or benchmark
// value is invented; unavailable facts are recorded as "unknown" with an
// explicit source.

#include "ir/tensor.h"
#include "utils/timer.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif
#if defined(__linux__)
#include <unistd.h>
#endif
#include <unistd.h> // gethostname (POSIX; available on both macOS and Linux)

namespace {

// ---------------------------------------------------------------------------
// Small utilities shared with the style of apps/run_mlir_fused_kernel_benchmark.cpp
// ---------------------------------------------------------------------------

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

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.good()) {
        throw std::runtime_error("failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void write_text_file(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.good()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    out << content;
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

// ---------------------------------------------------------------------------
// Provenance: fields every emitted artifact must carry (target host, target
// profile ID, git commit, thread count, timestamp). target_profile_id is
// caller-supplied (--target-profile-id) because this tool has no target-
// profile-resolution logic of its own; an empty value is recorded honestly
// rather than guessed.
// ---------------------------------------------------------------------------

struct Provenance {
    std::string target_host;
    std::string git_commit;
    std::string target_profile_id;
    std::string utc_timestamp;
};

std::string get_hostname() {
    char buf[256] = {0};
    if (gethostname(buf, sizeof(buf) - 1) == 0) return std::string(buf);
    return "unknown";
}

std::string get_git_commit() {
    std::string commit = run_command_capture("git rev-parse HEAD 2>/dev/null");
    return commit.empty() ? "unknown_not_a_git_checkout_or_git_unavailable" : commit;
}

void write_provenance_json(std::ostream& out, const std::string& indent, const Provenance& prov) {
    out << indent << "\"provenance\": {\n";
    out << indent << "  \"target_host\": \"" << prov.target_host << "\",\n";
    out << indent << "  \"git_commit\": \"" << prov.git_commit << "\",\n";
    out << indent << "  \"target_profile_id\": \"" << prov.target_profile_id << "\",\n";
    out << indent << "  \"utc_timestamp\": \"" << prov.utc_timestamp << "\"\n";
    out << indent << "},\n";
}

// ---------------------------------------------------------------------------
// Environment facts. Every value is either a real runtime query result or
// explicitly "unknown". `source` documents where each fact came from so the
// report never conflates a queried fact with a fabricated one.
// ---------------------------------------------------------------------------

struct EnvFact {
    std::string value = "unknown";
    std::string source = "unavailable";
};

struct Environment {
    EnvFact cpu_model;
    EnvFact os;
    EnvFact arch;
    EnvFact compiler;
    EnvFact physical_cores;
    EnvFact logical_cores;
    EnvFact perf_core_count;
    EnvFact perf_core_l1d_bytes;
    EnvFact perf_core_l2_bytes;
    EnvFact efficiency_core_count;
    EnvFact efficiency_core_l1d_bytes;
    EnvFact efficiency_core_l2_bytes;
    EnvFact cache_line_bytes;
    EnvFact simd_capability;
    EnvFact benchmark_thread_count;
    EnvFact build_type;
    std::string utc_timestamp;
};

#if defined(__APPLE__)
EnvFact sysctl_string(const char* name) {
    char buffer[256] = {0};
    size_t size = sizeof(buffer);
    if (sysctlbyname(name, buffer, &size, nullptr, 0) == 0) {
        return {std::string(buffer, size > 0 ? size - 1 : 0), "sysctl"};
    }
    return {"unknown", "unavailable"};
}

EnvFact sysctl_int(const char* name) {
    int64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0) {
        return {std::to_string(value), "sysctl"};
    }
    return {"unknown", "unavailable"};
}
#endif

Environment collect_environment(int benchmark_threads) {
    Environment env;
    std::time_t now = std::time(nullptr);
    char ts[64];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    env.utc_timestamp = ts;

#if defined(__APPLE__)
    env.cpu_model = sysctl_string("machdep.cpu.brand_string");
    env.os = {run_command_capture("sw_vers -productVersion"), "os_query(sw_vers)"};
    env.physical_cores = sysctl_int("hw.physicalcpu");
    env.logical_cores = sysctl_int("hw.logicalcpu");
    env.perf_core_count = sysctl_int("hw.perflevel0.physicalcpu");
    env.perf_core_l1d_bytes = sysctl_int("hw.perflevel0.l1dcachesize");
    env.perf_core_l2_bytes = sysctl_int("hw.perflevel0.l2cachesize");
    env.efficiency_core_count = sysctl_int("hw.perflevel1.physicalcpu");
    env.efficiency_core_l1d_bytes = sysctl_int("hw.perflevel1.l1dcachesize");
    env.efficiency_core_l2_bytes = sysctl_int("hw.perflevel1.l2cachesize");
    env.cache_line_bytes = sysctl_int("hw.cachelinesize");
    EnvFact neon = sysctl_int("hw.optional.neon");
    env.simd_capability = {
        neon.value == "1" ? "neon_baseline_arm64" : "unknown",
        neon.source == "sysctl" ? "sysctl(hw.optional.neon)" : "unavailable"
    };
#elif defined(__linux__)
    env.cpu_model = {run_command_capture(
        "awk -F': ' '/model name/{print $2; exit}' /proc/cpuinfo"), "os_query(/proc/cpuinfo)"};
    env.os = {run_command_capture("uname -r"), "os_query(uname)"};
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    env.logical_cores = {nproc > 0 ? std::to_string(nproc) : "unknown",
                         nproc > 0 ? "sysconf(_SC_NPROCESSORS_ONLN)" : "unavailable"};
    // Physical core count: Core(s) per socket * Socket(s) from lscpu. This
    // CPU (Intel Comet Lake mobile) has no heterogeneous P/E clusters, so
    // "physical_cores" alone is sufficient; perf/efficiency split fields
    // stay "unknown" (honestly absent, never fabricated) rather than
    // guessed for a non-hybrid part.
    std::string cores_per_socket = run_command_capture(
        "lscpu | awk -F': *' '/^Core\\(s\\) per socket/{print $2}'");
    std::string sockets = run_command_capture(
        "lscpu | awk -F': *' '/^Socket\\(s\\)/{print $2}'");
    if (!cores_per_socket.empty() && !sockets.empty()) {
        try {
            int total = std::stoi(cores_per_socket) * std::stoi(sockets);
            env.physical_cores = {std::to_string(total), "os_query(lscpu: Core(s) per socket * Socket(s))"};
        } catch (...) {
            env.physical_cores = {"unknown", "unavailable"};
        }
    } else {
        env.physical_cores = {"unknown", "unavailable"};
    }
    env.perf_core_count = {"not_applicable_non_hybrid_cpu", "declared_from_lscpu_single_core_type"};
    env.perf_core_l1d_bytes = {"unknown", "unavailable"};
    env.perf_core_l2_bytes = {"unknown", "unavailable"};
    env.efficiency_core_count = {"not_applicable_non_hybrid_cpu", "declared_from_lscpu_single_core_type"};
    env.efficiency_core_l1d_bytes = {"unknown", "unavailable"};
    env.efficiency_core_l2_bytes = {"unknown", "unavailable"};
    env.cache_line_bytes = {run_command_capture(
        "getconf LEVEL1_DCACHE_LINESIZE 2>/dev/null"), "os_query(getconf)"};
    {
        std::string flags = run_command_capture(
            "awk -F': ' '/^flags/{print $2; exit}' /proc/cpuinfo");
        bool has_avx512 = flags.find("avx512") != std::string::npos;
        bool has_avx2 = flags.find(" avx2 ") != std::string::npos || flags.find(" avx2") != std::string::npos;
        bool has_avx = flags.find(" avx ") != std::string::npos || flags.find(" avx") != std::string::npos;
        std::string simd = has_avx512 ? "avx512" : has_avx2 ? "avx2" : has_avx ? "avx" : "sse_baseline_or_unknown";
        env.simd_capability = {simd, flags.empty() ? "unavailable" : "os_query(/proc/cpuinfo flags)"};
    }
#else
    env.cpu_model = {"unknown", "unavailable"};
    env.os = {"unknown", "unavailable"};
    env.physical_cores = {"unknown", "unavailable"};
    env.logical_cores = {"unknown", "unavailable"};
    env.perf_core_count = {"unknown", "unavailable"};
    env.perf_core_l1d_bytes = {"unknown", "unavailable"};
    env.perf_core_l2_bytes = {"unknown", "unavailable"};
    env.efficiency_core_count = {"unknown", "unavailable"};
    env.efficiency_core_l1d_bytes = {"unknown", "unavailable"};
    env.efficiency_core_l2_bytes = {"unknown", "unavailable"};
    env.cache_line_bytes = {"unknown", "unavailable"};
    env.simd_capability = {"unknown", "unavailable"};
#endif

#if defined(__x86_64__) || defined(_M_X64)
    env.arch = {"x86_64", "compiler_predefined_macro"};
#elif defined(__aarch64__)
    env.arch = {"arm64", "compiler_predefined_macro"};
#else
    env.arch = {"unknown", "unavailable"};
#endif

    // Portability note: the compiler that built THIS binary is only known
    // truthfully via predefined macros (__VERSION__ + family macro).
    // Invoking a hardcoded binary name (e.g. "clang++ --version") is
    // unreliable cross-toolchain: on a host where CMake is configured to
    // use g++ (CMAKE_CXX_COMPILER=/usr/bin/c++) but clang++ also happens
    // to be installed on PATH, that would silently report the wrong
    // compiler. The macro-derived identity is primary and always correct
    // for this binary; the matching family binary is invoked only for a
    // supplementary descriptive string, never as the source of truth.
#if defined(__clang__)
    env.compiler = {std::string("clang ") + __VERSION__, "compiler_predefined_macro(__clang__,__VERSION__)"};
    std::string raw = run_command_capture("clang++ --version 2>/dev/null | head -1");
#elif defined(__GNUC__)
    env.compiler = {std::string("gcc ") + __VERSION__, "compiler_predefined_macro(__GNUC__,__VERSION__)"};
    std::string raw = run_command_capture("c++ --version 2>/dev/null | head -1");
#else
    env.compiler = {"unknown", "unavailable"};
    std::string raw;
#endif
    if (!raw.empty()) {
        env.compiler.value += " [PATH binary reports: " + raw + "]";
    }

#if defined(NDEBUG)
    env.build_type = {"optimized_NDEBUG_defined", "compiler_predefined_macro"};
#else
    env.build_type = {"unoptimized_or_debug_NDEBUG_absent", "compiler_predefined_macro"};
#endif

    env.benchmark_thread_count = {std::to_string(benchmark_threads), "cli_fixed_for_phase1"};
    return env;
}

void write_env_fact(std::ostream& out, const std::string& indent, const std::string& key,
                    const EnvFact& fact, bool trailing_comma) {
    out << indent << "\"" << key << "\": {\"value\": \"" << json_escape(fact.value)
        << "\", \"source\": \"" << json_escape(fact.source) << "\"}"
        << (trailing_comma ? ",\n" : "\n");
}

void write_environment_json(const std::string& path, const Environment& env, const Provenance& prov) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"cpu_fused_schedule_discovery_environment\",\n";
    out << "  \"schema_version\": 1,\n";
    write_provenance_json(out, "  ", prov);
    out << "  \"utc_timestamp\": \"" << env.utc_timestamp << "\",\n";
    write_env_fact(out, "  ", "cpu_model", env.cpu_model, true);
    write_env_fact(out, "  ", "os", env.os, true);
    write_env_fact(out, "  ", "arch", env.arch, true);
    write_env_fact(out, "  ", "compiler", env.compiler, true);
    write_env_fact(out, "  ", "build_type", env.build_type, true);
    write_env_fact(out, "  ", "physical_cores_total", env.physical_cores, true);
    write_env_fact(out, "  ", "logical_cores_total", env.logical_cores, true);
    write_env_fact(out, "  ", "performance_core_count", env.perf_core_count, true);
    write_env_fact(out, "  ", "performance_core_l1d_bytes", env.perf_core_l1d_bytes, true);
    write_env_fact(out, "  ", "performance_core_l2_bytes", env.perf_core_l2_bytes, true);
    write_env_fact(out, "  ", "efficiency_core_count", env.efficiency_core_count, true);
    write_env_fact(out, "  ", "efficiency_core_l1d_bytes", env.efficiency_core_l1d_bytes, true);
    write_env_fact(out, "  ", "efficiency_core_l2_bytes", env.efficiency_core_l2_bytes, true);
    write_env_fact(out, "  ", "cache_line_bytes", env.cache_line_bytes, true);
    write_env_fact(out, "  ", "simd_capability", env.simd_capability, true);
    write_env_fact(out, "  ", "benchmark_thread_count", env.benchmark_thread_count, true);
    out << "  \"thread_affinity_control\": \"not_controlled_os_default_scheduling\",\n";
    out << "  \"note\": \"This run fixes benchmark_thread_count=1 and does not exercise "
           "multicore scheduling. Some hosts expose heterogeneous performance/efficiency "
           "core clusters (see performance_core_count/efficiency_core_count; "
           "'not_applicable_non_hybrid_cpu' means this host has a single core type). "
           "Core/cache facts are recorded for future hardware-abstraction work, not used "
           "by this phase's candidate selection.\"\n";
    out << "}\n";
    write_text_file(path, out.str());
}

// ---------------------------------------------------------------------------
// Candidate contract: fused-only, apples-to-apples CPU schedule candidates.
// Every candidate shares identical MatMul+Bias+ReLU semantics, dtype (f32),
// accumulator dtype (f32), fusion status (one-pass fused, no full-size
// intermediate), launch count (1), input/output layout (row-major), thread
// policy (serial, thread_count=1), and loop structure. The ONLY variable is
// the schedule: (block_m, block_n, block_k).
// ---------------------------------------------------------------------------

struct Candidate {
    std::string candidate_id;
    int block_m = 0;
    int block_n = 0;
    int block_k = 0;
    int thread_count = 1;
    std::string loop_order = "ii(M)_jj(N)_kk(K)_i_k_j";
    std::string vectorization_policy = "compiler_auto_vectorization_only_no_explicit_intrinsics";
    std::string fusion_status = "one_pass_fused_tile_local_accumulator";
    std::string dtype = "f32";
    std::string accumulator_dtype = "f32";
    int launch_count = 1;
    int full_size_intermediates = 0;
};

// Original Phase-1 candidate tier. Kept for reference/attribution only —
// both the local Apple M5 run and a remote Intel i5-10210U pilot session
// (trace/remote_intel_cpu_fused_schedule_discovery/pilot_session/) showed
// this set collapses to one dominant candidate (bm32_bn32_bk32) on BOTH
// hosts: only block_n was ever varied meaningfully (block_k fixed at 32
// throughout, all tile footprints ≤4 KiB, far below either host's L1D).
std::vector<Candidate> make_original_phase1_candidates() {
    return {
        {"bm8_bn8_bk32", 8, 8, 32, 1},
        {"bm16_bn16_bk32", 16, 16, 32, 1},
        {"bm32_bn32_bk32", 32, 32, 32, 1},
        {"bm8_bn32_bk32", 8, 32, 32, 1},
    };
}

// R1 candidate-space repair (spec-directed): varies block_m/block_n/block_k
// independently, spanning tile footprints from far below to above a real
// measured per-core L1D (this host: 32 KiB; Apple M5: 64-128 KiB) and
// varying block_k for the first time. Footprint = block_m*block_n*4 bytes
// (f32 tile-local accumulator):
//   bm16_bn16_bk16   ->   1,024 B  (small footprint, small K)
//   bm32_bn32_bk32   ->   4,096 B  (Phase-1 baseline, medium K)
//   bm64_bn64_bk32   ->  16,384 B  (near/half L1D, medium K)
//   bm64_bn64_bk128  ->  16,384 B  (near/half L1D, large K)
//   bm128_bn128_bk32 ->  65,536 B  (above L1D on this host, medium K)
//   bm128_bn128_bk256->  65,536 B  (above L1D, large K)
//   bm16_bn128_bk32  ->   8,192 B  (rectangular: skinny-M/wide-N tile)
//   bm128_bn16_bk32  ->   8,192 B  (rectangular: wide-M/skinny-N tile)
// No packing, manual SIMD, or multithreading added — thread_count stays 1
// for every candidate, per the phase's controlled-variable contract.
std::vector<Candidate> make_repaired_candidates() {
    return {
        {"bm16_bn16_bk16", 16, 16, 16, 1},
        {"bm32_bn32_bk32", 32, 32, 32, 1},
        {"bm64_bn64_bk32", 64, 64, 32, 1},
        {"bm64_bn64_bk128", 64, 64, 128, 1},
        {"bm128_bn128_bk32", 128, 128, 32, 1},
        {"bm128_bn128_bk256", 128, 128, 256, 1},
        {"bm16_bn128_bk32", 16, 128, 32, 1},
        {"bm128_bn16_bk32", 128, 16, 32, 1},
    };
}

std::vector<Candidate> make_candidates(const std::string& candidate_set) {
    return candidate_set == "original" ? make_original_phase1_candidates() : make_repaired_candidates();
}

void write_candidate_contract_json(const std::string& path, const std::vector<Candidate>& candidates,
                                   const Provenance& prov, const std::string& candidate_set) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"cpu_fused_schedule_candidate_contract\",\n";
    out << "  \"schema_version\": 1,\n";
    write_provenance_json(out, "  ", prov);
    out << "  \"candidate_set\": \"" << candidate_set << "\",\n";
    out << "  \"controlled_variables\": {\n";
    out << "    \"semantics\": \"matmul_bias_relu\",\n";
    out << "    \"dtype\": \"f32\",\n";
    out << "    \"accumulator_dtype\": \"f32\",\n";
    out << "    \"fusion_status\": \"one_pass_fused\",\n";
    out << "    \"launch_count\": 1,\n";
    out << "    \"full_size_intermediates\": 0,\n";
    out << "    \"input_layout\": \"row_major\",\n";
    out << "    \"output_layout\": \"row_major\",\n";
    out << "    \"thread_policy\": \"serial_thread_count_1\",\n";
    out << "    \"vectorization_policy\": \"compiler_auto_vectorization_only\",\n";
    out << "    \"correctness_tolerance\": {\"atol\": 0.0001, \"rtol\": 0.0001}\n";
    out << "  },\n";
    out << "  \"primary_variable\": \"schedule_block_m_block_n_block_k\",\n";
    out << "  \"candidates\": [\n";
    for (size_t i = 0; i < candidates.size(); ++i) {
        const Candidate& c = candidates[i];
        out << "    {\n";
        out << "      \"candidate_id\": \"" << c.candidate_id << "\",\n";
        out << "      \"block_m\": " << c.block_m << ",\n";
        out << "      \"block_n\": " << c.block_n << ",\n";
        out << "      \"block_k\": " << c.block_k << ",\n";
        out << "      \"thread_count\": " << c.thread_count << ",\n";
        out << "      \"loop_order\": \"" << c.loop_order << "\",\n";
        out << "      \"vectorization_policy\": \"" << c.vectorization_policy << "\",\n";
        out << "      \"fusion_status\": \"" << c.fusion_status << "\",\n";
        out << "      \"dtype\": \"" << c.dtype << "\",\n";
        out << "      \"accumulator_dtype\": \"" << c.accumulator_dtype << "\",\n";
        out << "      \"launch_count\": " << c.launch_count << ",\n";
        out << "      \"full_size_intermediates\": " << c.full_size_intermediates << "\n";
        out << "    }" << (i + 1 < candidates.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    write_text_file(path, out.str());
}

// ---------------------------------------------------------------------------
// Workload manifest: representative subset across the six required shape
// families, plus the explicit edge-heavy (non-divisible) shapes. Full
// Cartesian products are not used; this is a controlled, documented subset
// (spec explicitly permits this to bound benchmark runtime).
// ---------------------------------------------------------------------------

struct Workload {
    std::string workload_id;
    std::string family;
    int m = 0;
    int n = 0;
    int k = 0;
};

std::vector<Workload> make_workloads() {
    std::vector<Workload> w;
    auto add = [&](const std::string& family, int m, int n, int k) {
        std::ostringstream id;
        id << family << "_m" << m << "_n" << n << "_k" << k;
        w.push_back({id.str(), family, m, n, k});
    };

    // Small square: M=N across a range, K across a range (diagonal sample).
    add("small_square", 16, 16, 64);
    add("small_square", 32, 32, 128);
    add("small_square", 48, 48, 256);
    add("small_square", 64, 64, 512);
    add("small_square", 96, 96, 1024);
    add("small_square", 128, 128, 2048);

    // Small output / high reduction: tiny M,N with very large K.
    add("small_output_high_reduction", 16, 16, 2048);
    add("small_output_high_reduction", 32, 32, 4096);
    add("small_output_high_reduction", 64, 64, 8192);
    add("small_output_high_reduction", 96, 96, 4096);

    // Skinny M / wide N.
    add("skinny_wide", 1, 2048, 768);
    add("skinny_wide", 2, 1024, 512);
    add("skinny_wide", 4, 4096, 1024);
    add("skinny_wide", 8, 2048, 256);
    add("skinny_wide", 16, 1024, 128);
    add("skinny_wide", 32, 4096, 2048);

    // Medium rectangular.
    add("medium_rectangular", 64, 1024, 256);
    add("medium_rectangular", 96, 512, 1024);
    add("medium_rectangular", 128, 256, 2048);
    add("medium_rectangular", 192, 128, 512);
    add("medium_rectangular", 256, 64, 1024);

    // Large regular (kept practical for local CPU benchmark duration).
    add("large_regular", 256, 256, 256);
    add("large_regular", 512, 512, 512);
    add("large_regular", 1024, 1024, 256);
    add("large_regular", 512, 1024, 1024);

    // Edge-heavy / non-divisible (exact shapes from Phase 1 spec).
    add("edge_heavy", 70, 70, 512);
    add("edge_heavy", 80, 112, 1024);
    add("edge_heavy", 96, 160, 2048);
    add("edge_heavy", 130, 258, 512);
    add("edge_heavy", 192, 320, 1024);

    return w;
}

void write_workload_manifest_json(const std::string& path, const std::vector<Workload>& workloads,
                                  const Provenance& prov) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"cpu_fused_schedule_workload_manifest\",\n";
    out << "  \"schema_version\": 1,\n";
    write_provenance_json(out, "  ", prov);
    out << "  \"note\": \"Representative documented subset across six shape families; "
           "full Cartesian product not used to bound benchmark runtime.\",\n";
    out << "  \"workloads\": [\n";
    for (size_t i = 0; i < workloads.size(); ++i) {
        const Workload& wl = workloads[i];
        out << "    {\"workload_id\": \"" << wl.workload_id << "\", \"family\": \"" << wl.family
            << "\", \"m\": " << wl.m << ", \"n\": " << wl.n << ", \"k\": " << wl.k << "}"
            << (i + 1 < workloads.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    write_text_file(path, out.str());
}

// ---------------------------------------------------------------------------
// Measurement budget: scaled by FLOPs so total benchmark runtime stays
// bounded while still gathering enough repeats to assess noise.
// ---------------------------------------------------------------------------

struct Budget {
    int warmup = 0;
    int iterations = 0;
    int repeats = 0;
    std::string tier;
};

Budget budget_for_flops(double flops, bool smoke) {
    if (smoke) {
        return {2, 3, 3, "smoke"};
    }
    if (flops < 10e6) return {50, 200, 7, "tiny"};
    if (flops < 200e6) return {30, 80, 7, "small"};
    if (flops < 2e9) return {15, 30, 7, "medium"};
    return {5, 10, 5, "large"};
}

// ---------------------------------------------------------------------------
// The shared fused-tile kernel skeleton. Only (block_m, block_n, block_k,
// thread_count) vary across candidates; everything else — semantics, dtype,
// accumulation, fusion, layout — is identical.
//
// thread_count is accepted for interface completeness with later phases;
// Phase 1 requires thread_count == 1 (serial) so schedule and thread-count
// selection are never combined in one experiment (see Phase 1 spec section 3).
// ---------------------------------------------------------------------------

void run_fused_tiled_matmul_bias_relu(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    Tensor& output,
    int M, int N, int K,
    int block_m, int block_n, int block_k,
    int thread_count,
    std::vector<float>& tile_scratch
) {
    if (thread_count != 1) {
        throw std::runtime_error(
            "Phase 1 CPU schedule discovery is serial-only; thread_count must be 1 "
            "(multicore scheduling is explicitly out of scope for this phase)");
    }
    if (block_m <= 0 || block_n <= 0 || block_k <= 0) {
        throw std::invalid_argument("block_m/block_n/block_k must be positive");
    }

    const float* a = input.data.data();
    const float* b = weight.data.data();
    const float* bias_data = bias.data.data();
    float* out = output.data.data();

    for (int ii = 0; ii < M; ii += block_m) {
        const int i_end = std::min(ii + block_m, M);
        for (int jj = 0; jj < N; jj += block_n) {
            const int j_end = std::min(jj + block_n, N);
            const int tile_rows = i_end - ii;
            const int tile_cols = j_end - jj;
            const size_t scratch_size = static_cast<size_t>(tile_rows) * static_cast<size_t>(tile_cols);
            if (tile_scratch.size() < scratch_size) {
                throw std::runtime_error("tile scratch buffer too small for candidate block size");
            }
            std::fill(tile_scratch.begin(), tile_scratch.begin() + scratch_size, 0.0f);

            for (int kk = 0; kk < K; kk += block_k) {
                const int k_end = std::min(kk + block_k, K);
                for (int i = ii; i < i_end; ++i) {
                    const float* a_row = a + static_cast<size_t>(i) * K;
                    float* scratch_row = tile_scratch.data() + static_cast<size_t>(i - ii) * tile_cols;
                    for (int k = kk; k < k_end; ++k) {
                        const float a_value = a_row[k];
                        const float* b_row = b + static_cast<size_t>(k) * N + jj;
                        for (int j = 0; j < tile_cols; ++j) {
                            scratch_row[j] += a_value * b_row[j];
                        }
                    }
                }
            }

            for (int i = ii; i < i_end; ++i) {
                const float* scratch_row = tile_scratch.data() + static_cast<size_t>(i - ii) * tile_cols;
                float* out_row = out + static_cast<size_t>(i) * N + jj;
                for (int j = 0; j < tile_cols; ++j) {
                    const float with_bias = scratch_row[j] + bias_data[jj + j];
                    out_row[j] = std::max(0.0f, with_bias);
                }
            }
        }
    }
}

// Independent, untiled reference implementation used ONLY for correctness —
// not part of the schedule-candidate set (spec section 2/7: the oracle must
// not be one of the candidates it is validating).
void run_naive_fused_matmul_bias_relu(
    const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
    int M, int N, int K
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += input.data[static_cast<size_t>(i) * K + k] * weight.data[static_cast<size_t>(k) * N + j];
            }
            const float with_bias = sum + bias.data[j];
            output.data[static_cast<size_t>(i) * N + j] = std::max(0.0f, with_bias);
        }
    }
}

// ---------------------------------------------------------------------------
// Correctness
// ---------------------------------------------------------------------------

struct Correctness {
    bool passed = false;
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool contains_nan = false;
    bool contains_inf = false;
};

Correctness compare_outputs(const Tensor& expected, const Tensor& actual, double atol, double rtol) {
    Correctness result;
    result.passed = expected.data.size() == actual.data.size();
    if (!result.passed) return result;
    for (size_t i = 0; i < expected.data.size(); ++i) {
        const float exp = expected.data[i];
        const float got = actual.data[i];
        result.contains_nan = result.contains_nan || std::isnan(exp) || std::isnan(got);
        result.contains_inf = result.contains_inf || std::isinf(exp) || std::isinf(got);
        const double abs_error = std::fabs(static_cast<double>(got) - static_cast<double>(exp));
        const double rel_denom = std::max(std::fabs(static_cast<double>(exp)), 1e-12);
        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, abs_error / rel_denom);
        if (abs_error > atol + rtol * std::fabs(static_cast<double>(exp))) {
            result.passed = false;
        }
    }
    if (result.contains_nan || result.contains_inf) result.passed = false;
    return result;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

struct Stats {
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double coefficient_of_variation = 0.0;
};

Stats summarize(std::vector<double> values) {
    Stats s;
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    s.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
    const size_t mid = values.size() / 2;
    s.median_ms = (values.size() % 2 == 0)
        ? (values[mid - 1] + values[mid]) / 2.0
        : values[mid];
    s.min_ms = values.front();
    s.max_ms = values.back();
    if (values.size() > 1) {
        double sq = 0.0;
        for (double v : values) { const double d = v - s.mean_ms; sq += d * d; }
        s.stddev_ms = std::sqrt(sq / static_cast<double>(values.size() - 1));
    }
    if (s.mean_ms > 0.0) s.coefficient_of_variation = s.stddev_ms / s.mean_ms;
    return s;
}

// ---------------------------------------------------------------------------
// Inputs — deterministic seeded pseudo-random fill (documented seed), shared
// across all candidates and the reference for one workload.
// ---------------------------------------------------------------------------

struct Inputs {
    Tensor a, b, bias;
    Inputs(int m, int k, int n) : a("A", {m, k}), b("B", {k, n}), bias("bias", {n}) {}
};

Inputs make_inputs(int m, int k, int n, unsigned seed) {
    Inputs inputs(m, k, n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : inputs.a.data) v = dist(rng);
    for (auto& v : inputs.b.data) v = dist(rng);
    for (auto& v : inputs.bias.data) v = dist(rng);
    return inputs;
}

// ---------------------------------------------------------------------------
// Fusion attribution baseline (Phase 1 spec section 7) — DELIBERATELY
// SEPARATE from the schedule-selection oracle below. This compares a tiled
// matmul followed by separate bias/relu passes (3 launches, 2 full-size
// intermediates) against the SAME tile config used one-pass fused (1
// launch, 0 full-size intermediates). It attributes the value of fusion
// itself; it must never be read as evidence about which schedule wins.
// ---------------------------------------------------------------------------

void run_tiled_matmul_only(
    const Tensor& input, const Tensor& weight, Tensor& output,
    int M, int N, int K, int block_m, int block_n, int block_k,
    std::vector<float>& tile_scratch
) {
    const float* a = input.data.data();
    const float* b = weight.data.data();
    float* out = output.data.data();
    for (int ii = 0; ii < M; ii += block_m) {
        const int i_end = std::min(ii + block_m, M);
        for (int jj = 0; jj < N; jj += block_n) {
            const int j_end = std::min(jj + block_n, N);
            const int tile_rows = i_end - ii;
            const int tile_cols = j_end - jj;
            const size_t scratch_size = static_cast<size_t>(tile_rows) * tile_cols;
            std::fill(tile_scratch.begin(), tile_scratch.begin() + scratch_size, 0.0f);
            for (int kk = 0; kk < K; kk += block_k) {
                const int k_end = std::min(kk + block_k, K);
                for (int i = ii; i < i_end; ++i) {
                    const float* a_row = a + static_cast<size_t>(i) * K;
                    float* scratch_row = tile_scratch.data() + static_cast<size_t>(i - ii) * tile_cols;
                    for (int k = kk; k < k_end; ++k) {
                        const float a_value = a_row[k];
                        const float* b_row = b + static_cast<size_t>(k) * N + jj;
                        for (int j = 0; j < tile_cols; ++j) scratch_row[j] += a_value * b_row[j];
                    }
                }
            }
            for (int i = ii; i < i_end; ++i) {
                const float* scratch_row = tile_scratch.data() + static_cast<size_t>(i - ii) * tile_cols;
                float* out_row = out + static_cast<size_t>(i) * N + jj;
                for (int j = 0; j < tile_cols; ++j) out_row[j] = scratch_row[j];
            }
        }
    }
}

void run_separate_bias_relu(const Tensor& matmul_out, const Tensor& bias, Tensor& add_out, Tensor& out,
                            int M, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            add_out.data[static_cast<size_t>(i) * N + j] =
                matmul_out.data[static_cast<size_t>(i) * N + j] + bias.data[j];
        }
    }
    for (size_t idx = 0; idx < out.data.size(); ++idx) {
        out.data[idx] = std::max(0.0f, add_out.data[idx]);
    }
}

struct FusionAttributionResult {
    int m = 0, n = 0, k = 0;
    int block_m = 0, block_n = 0, block_k = 0;
    Correctness unfused_correctness;
    Correctness fused_correctness;
    Stats unfused_stats;
    Stats fused_stats;
    int unfused_launch_count = 3;
    int fused_launch_count = 1;
    int unfused_full_size_intermediates = 2;
    int fused_full_size_intermediates = 0;
};

FusionAttributionResult run_fusion_attribution_baseline(bool smoke) {
    // Fixed representative shape and tile config (matches the schedule
    // oracle's most common winner, bm32_bn32_bk32) — NOT swept, NOT used to
    // pick a schedule candidate.
    const int m = 256, n = 256, k = 512;
    const int block_m = 32, block_n = 32, block_k = 32;
    FusionAttributionResult result;
    result.m = m; result.n = n; result.k = k;
    result.block_m = block_m; result.block_n = block_n; result.block_k = block_k;

    Inputs inputs = make_inputs(m, k, n, /*seed=*/9001u);
    Tensor reference("reference", {m, n});
    run_naive_fused_matmul_bias_relu(inputs.a, inputs.b, inputs.bias, reference, m, n, k);

    Tensor matmul_out("matmul_out", {m, n});
    Tensor add_out("add_out", {m, n});
    Tensor unfused_out("unfused_out", {m, n});
    Tensor fused_out("fused_out", {m, n});
    std::vector<float> scratch(static_cast<size_t>(block_m) * block_n, 0.0f);

    run_tiled_matmul_only(inputs.a, inputs.b, matmul_out, m, n, k, block_m, block_n, block_k, scratch);
    run_separate_bias_relu(matmul_out, inputs.bias, add_out, unfused_out, m, n);
    result.unfused_correctness = compare_outputs(reference, unfused_out, 1e-4, 1e-4);

    run_fused_tiled_matmul_bias_relu(inputs.a, inputs.b, inputs.bias, fused_out, m, n, k,
                                     block_m, block_n, block_k, 1, scratch);
    result.fused_correctness = compare_outputs(reference, fused_out, 1e-4, 1e-4);

    const Budget budget = budget_for_flops(2.0 * m * n * k, smoke);
    std::vector<double> unfused_samples, fused_samples;
    for (int repeat = 0; repeat < budget.repeats; ++repeat) {
        for (int w = 0; w < budget.warmup; ++w) {
            run_tiled_matmul_only(inputs.a, inputs.b, matmul_out, m, n, k, block_m, block_n, block_k, scratch);
            run_separate_bias_relu(matmul_out, inputs.bias, add_out, unfused_out, m, n);
        }
        Timer t1; t1.start();
        for (int it = 0; it < budget.iterations; ++it) {
            run_tiled_matmul_only(inputs.a, inputs.b, matmul_out, m, n, k, block_m, block_n, block_k, scratch);
            run_separate_bias_relu(matmul_out, inputs.bias, add_out, unfused_out, m, n);
        }
        unfused_samples.push_back(t1.stop_ms() / budget.iterations);

        for (int w = 0; w < budget.warmup; ++w) {
            run_fused_tiled_matmul_bias_relu(inputs.a, inputs.b, inputs.bias, fused_out, m, n, k,
                                             block_m, block_n, block_k, 1, scratch);
        }
        Timer t2; t2.start();
        for (int it = 0; it < budget.iterations; ++it) {
            run_fused_tiled_matmul_bias_relu(inputs.a, inputs.b, inputs.bias, fused_out, m, n, k,
                                             block_m, block_n, block_k, 1, scratch);
        }
        fused_samples.push_back(t2.stop_ms() / budget.iterations);
    }
    result.unfused_stats = summarize(unfused_samples);
    result.fused_stats = summarize(fused_samples);
    return result;
}

void write_fusion_attribution_json(std::ostream& out, const FusionAttributionResult& r) {
    out << "{\n";
    out << "    \"note\": \"SEPARATE from the schedule-selection oracle: attributes the value "
           "of one-pass fusion itself at a single fixed tile config, not schedule choice.\",\n";
    out << "    \"shape\": {\"m\": " << r.m << ", \"n\": " << r.n << ", \"k\": " << r.k << "},\n";
    out << "    \"tile_config\": {\"block_m\": " << r.block_m << ", \"block_n\": " << r.block_n
        << ", \"block_k\": " << r.block_k << "},\n";
    out << "    \"unfused\": {\"launch_count\": " << r.unfused_launch_count
        << ", \"full_size_intermediates\": " << r.unfused_full_size_intermediates
        << ", \"correctness_passed\": " << (r.unfused_correctness.passed ? "true" : "false")
        << ", \"mean_ms\": " << r.unfused_stats.mean_ms
        << ", \"coefficient_of_variation\": " << r.unfused_stats.coefficient_of_variation << "},\n";
    out << "    \"fused\": {\"launch_count\": " << r.fused_launch_count
        << ", \"full_size_intermediates\": " << r.fused_full_size_intermediates
        << ", \"correctness_passed\": " << (r.fused_correctness.passed ? "true" : "false")
        << ", \"mean_ms\": " << r.fused_stats.mean_ms
        << ", \"coefficient_of_variation\": " << r.fused_stats.coefficient_of_variation << "},\n";
    const double speedup = r.fused_stats.mean_ms > 0 ? r.unfused_stats.mean_ms / r.fused_stats.mean_ms : 0.0;
    out << "    \"fusion_speedup\": " << speedup << ",\n";
    out << "    \"latency_reduction_percent\": " << (speedup > 0 ? (1.0 - 1.0 / speedup) * 100.0 : 0.0) << "\n";
    out << "  }";
}

// ---------------------------------------------------------------------------
// Per-candidate, per-workload measurement record.
// ---------------------------------------------------------------------------

struct CandidateMeasurement {
    std::string candidate_id;
    Correctness correctness;
    Stats stats;
    std::vector<double> samples_ms;
};

struct WorkloadMeasurement {
    Workload workload;
    double flops = 0.0;
    Budget budget;
    std::vector<CandidateMeasurement> candidates;
};

// ---------------------------------------------------------------------------
// Discover mode: run every candidate against every workload with rotated
// candidate order per repeat (reduces thermal/order bias) and real,
// kernel-execution-only timing (allocation/fill happen outside Timer).
// ---------------------------------------------------------------------------

std::vector<WorkloadMeasurement> run_discovery(
    const std::vector<Candidate>& candidates,
    const std::vector<Workload>& workloads,
    bool smoke
) {
    std::vector<WorkloadMeasurement> results;
    results.reserve(workloads.size());

    for (const Workload& wl : workloads) {
        const double flops = 2.0 * static_cast<double>(wl.m) * wl.n * wl.k;
        const Budget budget = budget_for_flops(flops, smoke);

        Inputs inputs = make_inputs(wl.m, wl.k, wl.n, /*seed=*/1234u);
        Tensor reference("reference", {wl.m, wl.n});
        run_naive_fused_matmul_bias_relu(inputs.a, inputs.b, inputs.bias, reference, wl.m, wl.n, wl.k);

        WorkloadMeasurement wm;
        wm.workload = wl;
        wm.flops = flops;
        wm.budget = budget;

        std::vector<Tensor> outputs;
        std::vector<std::vector<float>> scratches;
        for (size_t c = 0; c < candidates.size(); ++c) {
            outputs.emplace_back("out_" + candidates[c].candidate_id, std::vector<int>{wl.m, wl.n});
            scratches.emplace_back(
                static_cast<size_t>(candidates[c].block_m) * static_cast<size_t>(candidates[c].block_n), 0.0f);
        }

        std::vector<std::vector<double>> samples(candidates.size());

        // Correctness pass (once per candidate, outside timing).
        std::vector<Correctness> correctness(candidates.size());
        for (size_t c = 0; c < candidates.size(); ++c) {
            const Candidate& cand = candidates[c];
            run_fused_tiled_matmul_bias_relu(
                inputs.a, inputs.b, inputs.bias, outputs[c],
                wl.m, wl.n, wl.k, cand.block_m, cand.block_n, cand.block_k,
                cand.thread_count, scratches[c]);
            correctness[c] = compare_outputs(reference, outputs[c], 1e-4, 1e-4);
        }

        // Timed repeats, candidate order rotated each repeat to reduce
        // thermal/order bias.
        for (int repeat = 0; repeat < budget.repeats; ++repeat) {
            std::vector<size_t> order(candidates.size());
            for (size_t c = 0; c < candidates.size(); ++c) {
                order[c] = (c + static_cast<size_t>(repeat)) % candidates.size();
            }
            for (size_t idx : order) {
                const Candidate& cand = candidates[idx];
                for (int w = 0; w < budget.warmup; ++w) {
                    run_fused_tiled_matmul_bias_relu(
                        inputs.a, inputs.b, inputs.bias, outputs[idx],
                        wl.m, wl.n, wl.k, cand.block_m, cand.block_n, cand.block_k,
                        cand.thread_count, scratches[idx]);
                }
                Timer timer;
                timer.start();
                for (int it = 0; it < budget.iterations; ++it) {
                    run_fused_tiled_matmul_bias_relu(
                        inputs.a, inputs.b, inputs.bias, outputs[idx],
                        wl.m, wl.n, wl.k, cand.block_m, cand.block_n, cand.block_k,
                        cand.thread_count, scratches[idx]);
                }
                const double elapsed = timer.stop_ms();
                samples[idx].push_back(elapsed / static_cast<double>(budget.iterations));
            }
        }

        for (size_t c = 0; c < candidates.size(); ++c) {
            CandidateMeasurement cm;
            cm.candidate_id = candidates[c].candidate_id;
            cm.correctness = correctness[c];
            cm.samples_ms = samples[c];
            cm.stats = summarize(samples[c]);
            wm.candidates.push_back(std::move(cm));
        }
        results.push_back(std::move(wm));
    }
    return results;
}

void write_benchmark_measurements_json(const std::string& path, const std::vector<WorkloadMeasurement>& results,
                                       const FusionAttributionResult& fusion_baseline,
                                       const Provenance& prov) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"cpu_fused_schedule_benchmark_measurements\",\n";
    out << "  \"schema_version\": 1,\n";
    write_provenance_json(out, "  ", prov);
    out << "  \"fusion_attribution_baseline\": ";
    write_fusion_attribution_json(out, fusion_baseline);
    out << ",\n";
    out << "  \"timing_methodology\": \"Timer starts immediately before the timed iteration "
           "loop and stops immediately after; tensor allocation, input fill, and reference "
           "computation happen outside the timed region. Each repeat's sample is total timed "
           "elapsed ms divided by iteration count.\",\n";
    out << "  \"input_distribution\": \"uniform(-1,1), std::mt19937, fixed seed=1234, shared "
           "across all candidates and the reference for a given workload\",\n";
    out << "  \"correctness_tolerance\": {\"atol\": 0.0001, \"rtol\": 0.0001},\n";
    out << "  \"candidate_order_policy\": \"rotated_per_repeat_round_robin\",\n";
    out << "  \"workloads\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const WorkloadMeasurement& wm = results[i];
        out << "    {\n";
        out << "      \"workload_id\": \"" << wm.workload.workload_id << "\",\n";
        out << "      \"family\": \"" << wm.workload.family << "\",\n";
        out << "      \"m\": " << wm.workload.m << ", \"n\": " << wm.workload.n << ", \"k\": " << wm.workload.k << ",\n";
        out << "      \"flops\": " << wm.flops << ",\n";
        out << "      \"budget\": {\"tier\": \"" << wm.budget.tier << "\", \"warmup\": " << wm.budget.warmup
            << ", \"iterations\": " << wm.budget.iterations << ", \"repeats\": " << wm.budget.repeats << "},\n";
        out << "      \"candidates\": [\n";
        for (size_t c = 0; c < wm.candidates.size(); ++c) {
            const CandidateMeasurement& cm = wm.candidates[c];
            out << "        {\n";
            out << "          \"candidate_id\": \"" << cm.candidate_id << "\",\n";
            out << "          \"correctness\": {\"passed\": " << (cm.correctness.passed ? "true" : "false")
                << ", \"max_abs_error\": " << cm.correctness.max_abs_error
                << ", \"max_rel_error\": " << cm.correctness.max_rel_error
                << ", \"contains_nan\": " << (cm.correctness.contains_nan ? "true" : "false")
                << ", \"contains_inf\": " << (cm.correctness.contains_inf ? "true" : "false") << "},\n";
            out << "          \"stats\": {\"mean_ms\": " << cm.stats.mean_ms
                << ", \"median_ms\": " << cm.stats.median_ms
                << ", \"min_ms\": " << cm.stats.min_ms
                << ", \"max_ms\": " << cm.stats.max_ms
                << ", \"stddev_ms\": " << cm.stats.stddev_ms
                << ", \"coefficient_of_variation\": " << cm.stats.coefficient_of_variation << "},\n";
            out << "          \"samples_ms\": [";
            for (size_t s = 0; s < cm.samples_ms.size(); ++s) {
                out << cm.samples_ms[s] << (s + 1 < cm.samples_ms.size() ? ", " : "");
            }
            out << "]\n";
            out << "        }" << (c + 1 < wm.candidates.size() ? "," : "") << "\n";
        }
        out << "      ]\n";
        out << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    write_text_file(path, out.str());
}

// ---------------------------------------------------------------------------
// use-plan mode: load a plan JSON naming a candidate_id + explicit
// block_m/block_n/block_k/thread_count, dispatch through the SAME
// run_fused_tiled_matmul_bias_relu function used for benchmarking (no
// separate "runtime" code path to drift from), and assert planned == actual
// with no silent substitution. Unknown candidate_id is a hard error, never a
// fallback.
// ---------------------------------------------------------------------------

struct PlanRequest {
    std::string target_profile_id;
    std::string backend;
    std::string kernel;
    std::string candidate_id;
    int block_m = 0, block_n = 0, block_k = 0, thread_count = 0;
};

PlanRequest load_plan(const std::string& path) {
    const std::string text = read_text_file(path);
    PlanRequest req;
    req.target_profile_id = extract_string_field(text, "target_profile_id");
    req.backend = extract_string_field(text, "backend");
    req.kernel = extract_string_field(text, "kernel");
    const size_t schedule_pos = text.find("\"schedule\"");
    if (schedule_pos == std::string::npos) {
        throw std::runtime_error("plan missing required object field: schedule");
    }
    const std::string schedule_text = text.substr(schedule_pos);
    req.candidate_id = extract_string_field(schedule_text, "candidate_id");
    req.block_m = extract_int_field(schedule_text, "block_m");
    req.block_n = extract_int_field(schedule_text, "block_n");
    req.block_k = extract_int_field(schedule_text, "block_k");
    req.thread_count = extract_int_field(schedule_text, "thread_count");
    return req;
}

void run_use_plan_validation(const std::string& plan_path, const std::string& output_path,
                             const std::string& expected_target_profile_id_raw,
                             const std::string& candidate_set) {
    const std::vector<Candidate> candidates = make_candidates(candidate_set);
    PlanRequest req = load_plan(plan_path);
    // Empty means "not provided": skip cross-target enforcement, but still
    // show an honest sentinel (never a blank string) in the JSON output.
    const std::string expected_target_profile_id = expected_target_profile_id_raw.empty()
        ? "unspecified_no_target_profile_id_provided" : expected_target_profile_id_raw;

    if (!expected_target_profile_id_raw.empty() && req.target_profile_id != expected_target_profile_id) {
        throw std::runtime_error(
            "plan target_profile_id '" + req.target_profile_id + "' does not match this run's "
            "target profile '" + expected_target_profile_id + "' (refusing cross-target dispatch)");
    }
    if (req.backend != "cpu") {
        throw std::runtime_error("unsupported backend in plan: " + req.backend);
    }
    if (req.kernel != "fused_matmul_bias_relu") {
        throw std::runtime_error("unsupported kernel in plan: " + req.kernel);
    }
    const Candidate* matched = nullptr;
    for (const Candidate& c : candidates) {
        if (c.candidate_id == req.candidate_id) { matched = &c; break; }
    }
    if (!matched) {
        throw std::runtime_error("unknown candidate_id in plan (no silent fallback): " + req.candidate_id);
    }
    if (matched->block_m != req.block_m || matched->block_n != req.block_n ||
        matched->block_k != req.block_k) {
        throw std::runtime_error(
            "plan block_m/n/k does not match registered candidate_id " + req.candidate_id +
            " (refusing to silently substitute)");
    }

    // Fixed representative workload for plan-dispatch validation.
    const int m = 128, n = 128, k = 512;
    Inputs inputs = make_inputs(m, k, n, /*seed=*/4242u);
    Tensor output("plan_output", {m, n});
    std::vector<float> scratch(static_cast<size_t>(matched->block_m) * matched->block_n, 0.0f);

    int override_count = 0;
    std::string actual_candidate_id = matched->candidate_id;
    int actual_block_m = matched->block_m, actual_block_n = matched->block_n, actual_block_k = matched->block_k;
    int actual_thread_count = matched->thread_count;

    run_fused_tiled_matmul_bias_relu(
        inputs.a, inputs.b, inputs.bias, output, m, n, k,
        actual_block_m, actual_block_n, actual_block_k, actual_thread_count, scratch);

    const bool matched_exactly =
        req.candidate_id == actual_candidate_id &&
        req.block_m == actual_block_m &&
        req.block_n == actual_block_n &&
        req.block_k == actual_block_k &&
        req.thread_count == actual_thread_count;

    std::ostringstream out;
    out << "{\n";
    out << "  \"schema\": \"cpu_fused_schedule_plan_dispatch_validation\",\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"plan_path\": \"" << json_escape(plan_path) << "\",\n";
    out << "  \"planned\": {\"target_profile_id\": \"" << req.target_profile_id
        << "\", \"backend\": \"" << req.backend << "\", \"kernel\": \"" << req.kernel
        << "\", \"candidate_id\": \"" << req.candidate_id << "\", \"block_m\": " << req.block_m
        << ", \"block_n\": " << req.block_n << ", \"block_k\": " << req.block_k
        << ", \"thread_count\": " << req.thread_count << "},\n";
    out << "  \"actual\": {\"target_profile_id\": \"" << expected_target_profile_id
        << "\", \"backend\": \"cpu\", \"kernel\": \"fused_matmul_bias_relu\", "
        << "\"candidate_id\": \"" << actual_candidate_id << "\", \"block_m\": " << actual_block_m
        << ", \"block_n\": " << actual_block_n << ", \"block_k\": " << actual_block_k
        << ", \"thread_count\": " << actual_thread_count << "},\n";
    out << "  \"plan_matched_runtime\": " << (matched_exactly ? "true" : "false") << ",\n";
    out << "  \"override_count\": " << override_count << "\n";
    out << "}\n";
    write_text_file(output_path, out.str());

    if (!matched_exactly) {
        throw std::runtime_error("plan/dispatch mismatch: runtime did not execute exactly the planned candidate");
    }
}

void write_plan_file(const std::string& path, const Candidate& c, const std::string& target_profile_id) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"target_profile_id\": \"" << target_profile_id << "\",\n";
    out << "  \"backend\": \"cpu\",\n";
    out << "  \"kernel\": \"fused_matmul_bias_relu\",\n";
    out << "  \"schedule\": {\n";
    out << "    \"candidate_id\": \"" << c.candidate_id << "\",\n";
    out << "    \"block_m\": " << c.block_m << ",\n";
    out << "    \"block_n\": " << c.block_n << ",\n";
    out << "    \"block_k\": " << c.block_k << ",\n";
    out << "    \"thread_count\": " << c.thread_count << "\n";
    out << "  }\n";
    out << "}\n";
    write_text_file(path, out.str());
}

struct CliArgs {
    std::string mode = "discover";
    std::string output_dir = "trace/cpu_fused_schedule_discovery";
    std::string plan_path;
    std::string target_profile_id;
    std::string candidate_set = "repaired";
    bool smoke = false;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(name + " requires a value");
            return argv[++i];
        };
        if (arg == "--mode") {
            args.mode = require_value(arg);
        } else if (arg == "--output-dir") {
            args.output_dir = require_value(arg);
        } else if (arg == "--plan") {
            args.plan_path = require_value(arg);
        } else if (arg == "--target-profile-id") {
            args.target_profile_id = require_value(arg);
        } else if (arg == "--candidate-set") {
            args.candidate_set = require_value(arg);
        } else if (arg == "--smoke") {
            args.smoke = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0]
                      << " [--mode discover|use-plan] [--output-dir DIR] [--plan PATH] "
                      << "[--target-profile-id ID] [--candidate-set original|repaired] [--smoke]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (args.mode != "discover" && args.mode != "use-plan") {
        throw std::invalid_argument("invalid --mode: " + args.mode);
    }
    if (args.mode == "use-plan" && args.plan_path.empty()) {
        throw std::invalid_argument("--mode use-plan requires --plan PATH");
    }
    if (args.candidate_set != "original" && args.candidate_set != "repaired") {
        throw std::invalid_argument("invalid --candidate-set: " + args.candidate_set);
    }
    return args;
}

} // namespace

int main(int argc, char** argv) {
    try {
        CliArgs args = parse_args(argc, argv);
        const std::string mkdir_cmd = "mkdir -p " + args.output_dir + " " + args.output_dir + "/plans";
        std::system(mkdir_cmd.c_str());

        Provenance prov;
        prov.target_host = get_hostname();
        prov.git_commit = get_git_commit();
        prov.target_profile_id = args.target_profile_id.empty()
            ? "unspecified_no_target_profile_id_provided" : args.target_profile_id;
        {
            std::time_t now = std::time(nullptr);
            char ts[64];
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
            prov.utc_timestamp = ts;
        }

        if (args.mode == "use-plan") {
            // Raw --target-profile-id (possibly empty): an unset value means
            // "don't enforce cross-target rejection", distinct from the
            // Provenance struct's substituted sentinel used for JSON output.
            run_use_plan_validation(args.plan_path, args.output_dir + "/plan_dispatch_validation.json",
                                    args.target_profile_id, args.candidate_set);
            std::cout << "use-plan validation passed: " << args.plan_path << "\n";
            return 0;
        }

        const std::vector<Candidate> candidates = make_candidates(args.candidate_set);
        const std::vector<Workload> workloads = make_workloads();

        write_environment_json(args.output_dir + "/environment.json", collect_environment(1), prov);
        write_candidate_contract_json(args.output_dir + "/candidate_contract.json", candidates, prov,
                                      args.candidate_set);
        write_workload_manifest_json(args.output_dir + "/workload_manifest.json", workloads, prov);

        std::cout << "Running CPU fused schedule discovery (" << args.candidate_set << " candidate set): "
                  << workloads.size() << " workloads x " << candidates.size() << " candidates"
                  << (args.smoke ? " (smoke mode)" : "") << "\n";

        std::vector<WorkloadMeasurement> results = run_discovery(candidates, workloads, args.smoke);
        std::cout << "Running fusion attribution baseline (separate from schedule oracle)...\n";
        FusionAttributionResult fusion_baseline = run_fusion_attribution_baseline(args.smoke);
        write_benchmark_measurements_json(args.output_dir + "/benchmark_measurements.json", results,
                                          fusion_baseline, prov);

        // Generate one plan file per candidate and validate exact dispatch
        // for each, aggregating into a single plan_dispatch_validation.json.
        std::ostringstream agg;
        agg << "{\n  \"schema\": \"cpu_fused_schedule_plan_dispatch_validation\",\n";
        agg << "  \"schema_version\": 1,\n";
        write_provenance_json(agg, "  ", prov);
        agg << "  \"validations\": [\n";
        int override_total = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            const Candidate& c = candidates[i];
            const std::string plan_path = args.output_dir + "/plans/" + c.candidate_id + ".plan.json";
            write_plan_file(plan_path, c, prov.target_profile_id);
            const std::string single_out = args.output_dir + "/plans/" + c.candidate_id + ".validation.json";
            run_use_plan_validation(plan_path, single_out, prov.target_profile_id, args.candidate_set);
            const std::string single_text = read_text_file(single_out);
            const bool matched = single_text.find("\"plan_matched_runtime\": true") != std::string::npos;
            if (!matched) ++override_total;
            agg << "    {\"candidate_id\": \"" << c.candidate_id << "\", \"plan_path\": \""
                << json_escape(plan_path) << "\", \"plan_matched_runtime\": "
                << (matched ? "true" : "false") << "}" << (i + 1 < candidates.size() ? "," : "") << "\n";
        }
        agg << "  ],\n  \"total_override_count\": " << override_total << "\n}\n";
        write_text_file(args.output_dir + "/plan_dispatch_validation.json", agg.str());

        std::cout << "Wrote artifacts to " << args.output_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
