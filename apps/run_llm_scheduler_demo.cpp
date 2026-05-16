#include "runtime/llm_scheduler.h"

int main() {
    LLMScheduler scheduler;

    LLMRequest req1;

    req1.request_id = 1;
    req1.prompt = "Hello";
    req1.prompt_tokens = {1, 2, 3, 4};
    req1.max_new_tokens = 3;

    LLMRequest req2;

    req2.request_id = 2;
    req2.prompt = "Explain transformers";
    req2.prompt_tokens = {5, 6, 7, 8, 9, 10};
    req2.max_new_tokens = 2;

    scheduler.submit(req1);
    scheduler.submit(req2);

    scheduler.run();

    return 0;
}