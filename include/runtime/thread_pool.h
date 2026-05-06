#pragma once

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>

class ThreadPool {
public:
    ThreadPool(int num_threads);
    ~ThreadPool();

    void enqueue(std::function<void()> task);
    void wait();

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex mutex;
    std::condition_variable cv;

    std::atomic<bool> stop;
    std::atomic<int> active_tasks;

    void worker_loop();
};