
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <esp_task_wdt.h>

#include "api/context.hpp"
#include "api/executor.hpp"
#include "api/server.hpp"

#include "hal/transport.hpp"

#if LOG_LEVEL >= LOG_LEVEL_DEBUG
#include "core/debug.hpp"
#endif

#include <cstdio>

#define EXECUTOR_TASK_BOUND          16
#define EXECUTOR_TASK_STACK_SIZE     40960
#define EXECUTOR_TASK_PRIORITY       3
#define EXECUTOR_TASK_NAME           "Executor"
#define EXECUTOR_TASK_TIMEOUT_MS     10000
#define EXECUTOR_TASK_PIN_TO_CORE    1
#define EXECUTOR_TASK_ROUND_DELAY_MS 1

#define SERVER_COMMAND_BUFFER_SIZE 2048
#define SERVER_TASK_ROUND_DELAY_MS 10

static api::Context *context_instance = nullptr;
static api::Executor *executor_instance = nullptr;

static void executor_yield()
{
    int ret = esp_task_wdt_reset();
    if (ret != ESP_OK) [[unlikely]]
    {
        LOG(ERROR, "Failed to reset task watchdog: %d", ret);
    }
    taskYIELD();
    vTaskDelay(pdMS_TO_TICKS(EXECUTOR_TASK_ROUND_DELAY_MS));
}

static void executor_task(void *)
{
    if (!executor_instance) [[unlikely]]
    {
        LOG(ERROR, "Executor instance is null");
        vTaskDelete(nullptr);
        return;
    }

    int ret = esp_task_wdt_add(nullptr);
    if (ret != ESP_OK) [[unlikely]]
    {
        LOG(ERROR, "Failed to add task to watchdog: %d", ret);
    }

    static esp_task_wdt_config_t wdt_config = {
        .timeout_ms = EXECUTOR_TASK_TIMEOUT_MS,
        .idle_core_mask = 1 << EXECUTOR_TASK_PIN_TO_CORE,
        .trigger_panic = false,
    };
    ret = esp_task_wdt_reconfigure(&wdt_config);
    if (ret != ESP_OK) [[unlikely]]
    {
        LOG(WARNING, "Failed to reconfigure task watchdog: %d", ret);
    }

    executor_yield();

    while (true)
    {
#if LOG_LEVEL >= LOG_LEVEL_DEBUG
        static size_t counter = 0;
        if (++counter % 1000 == 0)
            core::printHeapInfo();
#endif
        const auto status = executor_instance->execute(executor_yield);
        if (!status) [[unlikely]]
        {
            LOG(DEBUG, "Executor task fail: %d, %s", status.code(), status.message().c_str());
            if (context_instance) [[likely]]
            {
                context_instance->report(status);
            }
            continue;
        }
        executor_yield();
    }

    LOG(INFO, "Executor task exiting");
    vTaskDelete(nullptr);
}

static void executor_pick()
{
    if (!executor_instance) [[unlikely]]
    {
        return;
    }

    const auto status = executor_instance->pick();
    if (!status) [[unlikely]]
    {
        LOG(DEBUG, "Executor task fail: %d, %s", status.code(), status.message().c_str());
        if (context_instance) [[likely]]
        {
            context_instance->report(status);
        }
    }
}

extern "C" void app_main()
{
    LOG(INFO, "Starting...");

    LOG(INFO, "Initializing API Context");
    context_instance = api::Context::create();
    if (!context_instance)
    {
        LOG(ERROR, "Failed to create API context");
        return;
    }
    if (!context_instance->status())
    {
        LOG(ERROR, "API context initialization failed: %d, %s", context_instance->status().code(),
            context_instance->status().message().c_str());
        return;
    }

    LOG(INFO, "Initializing Executor");
    auto executor = api::Executor(EXECUTOR_TASK_BOUND);
    executor_instance = &executor;
    int ret = xTaskCreatePinnedToCore(executor_task, EXECUTOR_TASK_NAME, EXECUTOR_TASK_STACK_SIZE, nullptr,
        EXECUTOR_TASK_PRIORITY, nullptr, EXECUTOR_TASK_PIN_TO_CORE);
    if (ret != pdPASS)
    {
        LOG(ERROR, "Failed to create executor task: %d", ret);
    }

    LOG(INFO, "Initializing API Server");
    auto server = api::Server(*context_instance, executor, SERVER_COMMAND_BUFFER_SIZE);
    auto stop_token = [](const core::Status &status) noexcept -> bool {
        if (!status)
        {
            LOG(DEBUG, "Server stop token triggered: %d, %s", status.code(), status.message().c_str());
        }
        executor_pick();
        vTaskDelay(pdMS_TO_TICKS(SERVER_TASK_ROUND_DELAY_MS));
        return false;
    };

    LOG(INFO, "Waiting for commands...");
    while (1)
    {
        auto status = server.serve(stop_token);
        LOG(WARNING, "Server stopped: %d, %s", status.code(), status.message().c_str());

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
