#pragma once

#include <string>
#include <vector>

struct ProfileEvent {
    std::string op_name;
    std::string backend;
    double latency_ms;
};

class Profiler {
public:
    void record(const std::string& op_name, double latency_ms);
    void record(const std::string& op_name, const std::string& backend, double latency_ms);

    void print_summary() const;
    void export_json(const std::string& path) const;
    void reset();

private:
    std::vector<ProfileEvent> events;
};