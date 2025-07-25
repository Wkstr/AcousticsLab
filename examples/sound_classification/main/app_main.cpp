#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/timers.h>

#include "engine/tflm.hpp"
#include "hal/engine.hpp"
#include "hal/sensor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#include "feature_generator.h"

static constexpr int kNumLabels = 3;
static const char *kLabels[kNumLabels] = { "Background Noise", "one", "two" };

extern "C" void app_main()
{
    std::cout << "=== Sound Classification Example ===" << std::endl;

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

    auto model_infos = engine->modelInfos();
    for (const auto &info: model_infos)
    {
        std::cout << "  Model ID: " << info->id << ", Name: " << info->name
                  << ", Type: " << static_cast<int>(info->type) << std::endl;
    }

    std::shared_ptr<core::Model> model;
    auto info = engine->modelInfo([](const core::Model::Info &info) { return info.id == 1; });
    if (!info)
    {
        std::cout << "No model found with ID 1" << std::endl;
        return;
    }
    std::cout << "Found model: " << info->name << " (ID: " << info->id << ")" << std::endl;

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
    for (size_t i = 0; i < 5; ++i)
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
    std::cout << "Initializing feature generator..." << std::endl;
    FeatureGenerator feature_generator;
    if (!feature_generator.init())
    {
        std::cout << "Failed to initialize feature generator" << std::endl;
        return;
    }
    std::cout << "Feature generator initialized successfully" << std::endl;

    std::cout << "Allocating feature output buffer..." << std::endl;
    float *feature_buffer = (float *)heap_caps_aligned_alloc(16, CONFIG_OUTPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!feature_buffer)
    {
        std::cout << "Failed to allocate feature buffer" << std::endl;
        return;
    }
    std::cout << "Feature buffer allocated successfully, size: " << CONFIG_OUTPUT_SIZE << " features" << std::endl;

    std::cout << "\n=== Starting Streaming Audio Classification ===" << std::endl;
    std::cout << "Processing audio in real-time with sliding window..." << std::endl;

    const size_t ANALYSIS_WINDOW_SAMPLES = 44032;
    const size_t INFERENCE_STRIDE_SAMPLES = 22016;
    const size_t AUDIO_CHUNK_SAMPLES = 4410;

    std::unique_ptr<int16_t[]> analysis_window(new int16_t[ANALYSIS_WINDOW_SAMPLES]);
    if (!analysis_window)
    {
        std::cout << "Failed to allocate analysis window buffer" << std::endl;
        return;
    }

    memset(analysis_window.get(), 0, ANALYSIS_WINDOW_SAMPLES * sizeof(int16_t));

    s = std::chrono::steady_clock::now();
    size_t samples_since_last_inference = 0;
    int classification_count = 0;
    while (1)
    {
        auto current_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - s).count();
        float seconds = duration_ms / 1000.0f;

        core::DataFrame<std::unique_ptr<core::Tensor>> audio_chunk;
        auto status = mic->readDataFrame(audio_chunk, AUDIO_CHUNK_SAMPLES);

        const int16_t *chunk_data = audio_chunk.data->data<int16_t>();
        if (!chunk_data)
        {
            std::cout << "Failed to get chunk data" << std::endl;
            continue;
        }

        memmove(analysis_window.get(), analysis_window.get() + AUDIO_CHUNK_SAMPLES,
            (ANALYSIS_WINDOW_SAMPLES - AUDIO_CHUNK_SAMPLES) * sizeof(int16_t));
        memcpy(analysis_window.get() + (ANALYSIS_WINDOW_SAMPLES - AUDIO_CHUNK_SAMPLES), chunk_data,
            AUDIO_CHUNK_SAMPLES * sizeof(int16_t));

        samples_since_last_inference += AUDIO_CHUNK_SAMPLES;

        if (samples_since_last_inference >= INFERENCE_STRIDE_SAMPLES)
        {
            std::cout << "\n\n--- Audio Classification #" << (++classification_count) << " ---" << std::endl;
            std::cout << "Time: " << seconds << "s, Processing " << ANALYSIS_WINDOW_SAMPLES << " samples" << std::endl;

            float samples_duration = static_cast<float>(ANALYSIS_WINDOW_SAMPLES) / 44100.0f;
            float window_end_time = seconds;
            float window_start_time = window_end_time - samples_duration;
            std::cout << "Audio window: [" << std::fixed << std::setprecision(1) << window_start_time << "s - "
                      << window_end_time << "s] (1.0s duration)" << std::endl;

            auto process_start = std::chrono::steady_clock::now();

            auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - process_start)
                                    .count();

            size_t samples_processed = ANALYSIS_WINDOW_SAMPLES;
            float audio_duration = static_cast<float>(samples_processed) / 44100.0f; // Assuming 44.1kHz sample rate

            std::cout << "Audio processing: " << process_time << "ms" << std::endl;
            std::cout << "  - Data size: " << samples_processed * sizeof(int16_t) << " bytes" << std::endl;
            std::cout << "  - Samples: " << samples_processed << " (expected: " << ANALYSIS_WINDOW_SAMPLES << ")"
                      << std::endl;
            std::cout << "  - Duration: " << std::fixed << std::setprecision(3) << audio_duration
                      << "s (expected: 1.0s)" << std::endl;

            auto feature_start = std::chrono::steady_clock::now();
            feature_generator.generate(analysis_window.get(), feature_buffer);
            auto feature_end = std::chrono::steady_clock::now();

            auto feature_time
                = std::chrono::duration_cast<std::chrono::milliseconds>(feature_end - feature_start).count();
            std::cout << "Feature extraction: " << feature_time << "ms" << std::endl;

            auto input_tensor = graph->input(0);
            if (!input_tensor)
            {
                std::cout << "Failed to get input tensor" << std::endl;
                continue;
            }

            size_t input_size = input_tensor->shape().dot();
            size_t elements_to_copy = std::min(input_size, static_cast<size_t>(CONFIG_OUTPUT_SIZE));

            std::cout << "Input tensor validation:" << std::endl;
            std::cout << "  - Tensor size: " << input_size << " elements" << std::endl;
            std::cout << "  - Feature buffer size: " << CONFIG_OUTPUT_SIZE << " elements" << std::endl;
            std::cout << "  - Elements to copy: " << elements_to_copy << " elements" << std::endl;

            if (input_size == 0)
            {
                std::cout << "ERROR: Input tensor has zero size!" << std::endl;
                continue;
            }
            if (elements_to_copy == 0)
            {
                std::cout << "ERROR: No elements to copy!" << std::endl;
                continue;
            }
            if (elements_to_copy < CONFIG_OUTPUT_SIZE)
            {
                std::cout << "WARNING: Input tensor smaller than feature buffer, will copy " << elements_to_copy
                          << " elements" << std::endl;
            }
            std::cout << "  - First 5 values: ";
            for (size_t i = 0; i < std::min(static_cast<size_t>(5), elements_to_copy); ++i)
            {
                std::cout << std::fixed << std::setprecision(3) << feature_buffer[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "Input tensor type: " << static_cast<int>(input_tensor->dtype()) << std::endl;

            if (input_tensor->dtype() == core::Tensor::Type::Float32)
            {
                float *float_data = input_tensor->data<float>();
                if (float_data)
                {
                    std::memcpy(float_data, feature_buffer, elements_to_copy * sizeof(float));
                    std::cout << "Features copied as float (direct copy)" << std::endl;
                }
                else
                {
                    std::cout << "ERROR: Failed to get float data pointer" << std::endl;
                    continue;
                }
            }
            else if (input_tensor->dtype() == core::Tensor::Type::Int8)
            {
                auto input_quant_params = graph->inputQuantParams(0);
                std::cout << "Input quantization params: scale=" << input_quant_params.scale()
                          << ", zero_point=" << input_quant_params.zeroPoint() << std::endl;

                int8_t *quantized_data = input_tensor->data<int8_t>();
                if (!quantized_data)
                {
                    std::cout << "ERROR: Failed to get int8 data pointer" << std::endl;
                    continue;
                }

                float scale = input_quant_params.scale();
                int32_t zero_point = input_quant_params.zeroPoint();

                for (size_t i = 0; i < elements_to_copy; ++i)
                {
                    int32_t quantized_value = static_cast<int32_t>(std::round(feature_buffer[i] / scale)) + zero_point;

                    quantized_value
                        = std::max(static_cast<int32_t>(-128), std::min(static_cast<int32_t>(127), quantized_value));
                    quantized_data[i] = static_cast<int8_t>(quantized_value);
                }
            }
            else
            {
                std::cout << "Unsupported input tensor data type" << std::endl;
                continue;
            }
            auto inference_start = std::chrono::steady_clock::now();
            status = graph->forward();
            auto inference_end = std::chrono::steady_clock::now();

            if (!status)
            {
                std::cout << "Model inference failed: " << status.message() << std::endl;
            }
            else
            {
                auto inference_time
                    = std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - inference_start).count();
                std::cout << "Model inference: " << inference_time << "ms" << std::endl;

                auto output_tensor = graph->output(0);
                if (!output_tensor)
                {
                    std::cout << "Failed to get output tensor" << std::endl;
                    continue;
                }

                std::cout << "\n--- Classification Results ---" << std::endl;

                size_t output_size = output_tensor->shape().dot();
                size_t num_classes = std::min(output_size, static_cast<size_t>(kNumLabels));

                std::cout << "Output tensor info: size=" << output_size << ", processing " << num_classes << " classes"
                          << std::endl;

                float max_score = 0;
                int max_index = 0;

                std::cout << "Output tensor type: " << static_cast<int>(output_tensor->dtype()) << std::endl;

                if (output_tensor->dtype() == core::Tensor::Type::Float32)
                {
                    const float *output_data = output_tensor->data<float>();
                    if (!output_data)
                    {
                        std::cout << "ERROR: Failed to get float output data pointer" << std::endl;
                        continue;
                    }
                    std::cout << "Reading float output directly" << std::endl;

                    for (size_t i = 0; i < num_classes; ++i)
                    {
                        float score = output_data[i];
                        std::cout << kLabels[i] << ": " << std::fixed << std::setprecision(6) << score << std::endl;

                        if (score > max_score)
                        {
                            max_score = score;
                            max_index = i;
                        }
                    }
                }
                else if (output_tensor->dtype() == core::Tensor::Type::Int8)
                {
                    const int8_t *output_data = output_tensor->data<int8_t>();
                    if (!output_data)
                    {
                        std::cout << "ERROR: Failed to get int8 output data pointer" << std::endl;
                        continue;
                    }

                    auto output_quant_params = graph->outputQuantParams(0);
                    std::cout << "Output quantization params: scale=" << output_quant_params.scale()
                              << ", zero_point=" << output_quant_params.zeroPoint() << std::endl;

                    for (size_t i = 0; i < num_classes; ++i)
                    {
                        float score = static_cast<float>(output_data[i] - output_quant_params.zeroPoint())
                                      * output_quant_params.scale();
                        std::cout << kLabels[i] << ": " << std::fixed << std::setprecision(6) << score
                                  << " (raw: " << static_cast<int>(output_data[i]) << ")" << std::endl;

                        if (score > max_score)
                        {
                            max_score = score;
                            max_index = i;
                        }
                    }
                }
                else
                {
                    std::cout << "Unsupported output tensor data type" << std::endl;
                    continue;
                }

                std::cout << "\nPredicted class: " << kLabels[max_index] << " (score: " << std::fixed
                          << std::setprecision(6) << max_score << ")" << std::endl;
                std::cout << "--- End Results ---\n" << std::endl;

                auto total_time = process_time + feature_time + inference_time;
                std::cout << "Total processing time: " << total_time << "ms" << std::endl;
                std::cout << "Classification completed successfully!" << std::endl;

                samples_since_last_inference = 0;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}