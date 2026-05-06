#include "runtime/thread_pool.h"

ThreadPool::ThreadPool(int num_threads)
    : stop(false), active_tasks(0)
{
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back([this]() { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    stop = true;
    cv.notify_all();

    for (auto& t : workers) {
        t.join();
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(mutex);

            cv.wait(lock, [this]() {
                return stop || !tasks.empty();
            });

            if (stop && tasks.empty()) return;

            task = std::move(tasks.front());
            tasks.pop();
            active_tasks++;
        }

        task();

        active_tasks--;
        cv.notify_all();
    }
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        tasks.push(std::move(task));
    }
    cv.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() {
        return tasks.empty() && active_tasks == 0;
    });
}