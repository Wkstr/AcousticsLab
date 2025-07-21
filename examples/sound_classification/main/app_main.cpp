#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include "engine/tflm.hpp"
#include "hal/engine.hpp"
#include "hal/sensor.hpp"

#include <chrono>
#include <iostream>

extern "C" void app_main()
{
    std::cout << "Starting..." << std::endl;

    bridge::__REGISTER_ENGINES__();
    bridge::__REGISTER_SENSORS__();

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
    for (size_t i = 0; i < 3; ++i)
    {
        auto s = std::chrono::steady_clock::now();
        status = graph->forward();
        auto e = std::chrono::steady_clock::now();

        if (!status)
        {
            std::cout << "Graph forward " << i << " failed: " << status.message() << std::endl;
            return;
        }
        std::cout << "Graph forward " << i << " executed successfully in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << " ms" << std::endl;

        vTaskDelay(pdMS_TO_TICKS(5));
    }

    auto mic = hal::SensorRegistry::getSensor(2);
    if (!mic)
    {
        std::cout << "Sensor with ID 2 not found" << std::endl;
        return;
    }
    std::cout << "Sensor found: " << mic->info().name << std::endl;

    status = mic->init();
    if (!status)
    {
        std::cout << "Failed to initialize sensor: " << status.message() << std::endl;
        return;
    }
    std::cout << "Sensor initialized successfully" << std::endl;

    s = std::chrono::steady_clock::now();
    while (1)
    {
        auto duration_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s).count();
        float seconds = duration_ms / 1000.0f;
        auto available = mic->dataAvailable();
        std::cout << "Time since start: " << seconds << " seconds, data available: " << available << std::endl;
        if (available)
        {
            auto rs = std::chrono::steady_clock::now();
            auto data_frame = core::DataFrame<std::unique_ptr<core::Tensor>>();
            status = mic->readDataFrame(data_frame, 44100 / 2);
            auto re = std::chrono::steady_clock::now();
            if (!status)
            {
                std::cout << "Failed to read data frame: " << status.message() << std::endl;
            }
            else
            {
                std::cout << "Data frame read successfully in "
                          << std::chrono::duration_cast<std::chrono::milliseconds>(re - rs).count() << " ms, "
                          << "data size: " << data_frame.data->size() << " bytes" << std::endl;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(500));
    }

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
