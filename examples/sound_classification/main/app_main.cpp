#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include "engine/tflm.hpp"
#include "hal/engine.hpp"

#include <chrono>
#include <iostream>

extern "C" void app_main()
{
    std::cout << "Starting..." << std::endl;

    bridge::__REGISTER_ENGINES__();

    auto engine = hal::EngineRegistry::getEngine(1);
    if (!engine)
    {
        std::cout << "Engine with ID 1 not found" << std::endl;
        return;
    }
    std::cout << "Engine found: " << engine->info().name << std::endl;

    auto status = engine->init();
    if (!status)
    {
        std::cout << "Failed to initialize engine: " << status.message() << std::endl;
        return;
    }
    std::cout << "Engine initialized successfully" << std::endl;

    std::shared_ptr<core::Model> model;
    auto info = engine->modelInfo([](const core::Model::Info &info) { return info.id == 1; });
    status = engine->loadModel(info, model);
    if (!status)
    {
        std::cout << "Failed to load model: " << status.message() << std::endl;
        return;
    }
    std::cout << "Model loaded successfully: " << model->info()->name << std::endl;

    auto graph = model->graph(0);
    if (!graph)
    {
        std::cout << "Failed to get graph from model" << std::endl;
        return;
    }
    std::cout << "Graph loaded successfully: " << graph->name() << std::endl;

    auto s = std::chrono::steady_clock::now();
    status = graph->forward();
    auto e = std::chrono::steady_clock::now();

    if (!status)
    {
        std::cout << "Graph forward failed: " << status.message() << std::endl;
        return;
    }
    std::cout << "Graph forward executed successfully in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

    while (1)
    {
        auto s = std::chrono::steady_clock::now();
        status = graph->forward();
        auto e = std::chrono::steady_clock::now();

        if (!status)
        {
            std::cout << "Graph forward failed: " << status.message() << std::endl;
            return;
        }
        std::cout << "Graph forward executed successfully in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
