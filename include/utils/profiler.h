#pragma once

#include <string>
#include <vector>

struct ProfileEvent {
    std::string op_name;
    double latency_ms;
};

class Profiler {
public:
    void record(const std::string& op_name, double latency_ms);

    void print_summary() const;

    void reset();

private:
    std::vector<ProfileEvent> events;
};