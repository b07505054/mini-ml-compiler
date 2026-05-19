#include "runtime/runtime_tracer.h"

#include <chrono>
#include <thread>

double now_ms() {
    auto t =
        std::chrono::high_resolution_clock
            ::now()
            .time_since_epoch();

    return std::chrono::duration<
        double,
        std::milli
    >(t).count();
}

int main() {
    RuntimeTracer tracer;

    {
        double s = now_ms();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(14)
        );

        double e = now_ms();

        tracer.add_event({
            "matmul",
            "Metal",
            16,
            s,
            e,
            e - s
        });
    }

    {
        double s = now_ms();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(4)
        );

        double e = now_ms();

        tracer.add_event({
            "add",
            "CPU",
            20,
            s,
            e,
            e - s
        });
    }

    {
        double s = now_ms();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(2)
        );

        double e = now_ms();

        tracer.add_event({
            "relu",
            "CPU",
            16,
            s,
            e,
            e - s
        });
    }

    tracer.dump();

    tracer.export_json(
        "../trace/runtime_trace.json"
    );

    return 0;
}