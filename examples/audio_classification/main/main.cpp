#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// AcousticsLab includes
#include "api/context.hpp"
#include "api/inference_pipeline.hpp"
#include "hal/engine.hpp"
#include "hal/processor.hpp"
#include "hal/sensor.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

static const char *TAG = "AudioClassification";

// Model labels matching faudioclassify
static const std::unordered_map<int, std::string> MODEL_LABELS
    = { { 0, "Background Noise" }, { 1, "one" }, { 2, "two" } };

// Declare embedded model data
extern const uint8_t fixed2conv_model_tflite_start[] asm("_binary_fixed2conv_model_tflite_start");
extern const uint8_t fixed2conv_model_tflite_end[] asm("_binary_fixed2conv_model_tflite_end");

/**
 * @brief Load TFLite model from embedded data
 *
 * @param model_data Output vector to store model data
 * @return true if successful, false otherwise
 */
bool loadModelFromEmbedded(std::vector<uint8_t> &model_data)
{
    size_t model_size = fixed2conv_model_tflite_end - fixed2conv_model_tflite_start;

    ESP_LOGI(TAG, "Loading embedded model, size: %zu bytes", model_size);

    if (model_size == 0)
    {
        ESP_LOGE(TAG, "Embedded model size is zero");
        return false;
    }

    model_data.resize(model_size);
    memcpy(model_data.data(), fixed2conv_model_tflite_start, model_size);

    ESP_LOGI(TAG, "Successfully loaded embedded model, size: %zu bytes", model_size);
    return true;
}

class AudioClassificationDemo
{
public:
    AudioClassificationDemo() : _microphone(nullptr), _pipeline(nullptr), _running(false) { }

    ~AudioClassificationDemo()
    {
        stop();
    }

    bool init()
    {
        ESP_LOGI(TAG, "Initializing Audio Classification Demo...");

        // Initialize AcousticsLab context
        auto context = api::Context::create();
        if (!context)
        {
            ESP_LOGE(TAG, "Failed to create AcousticsLab context");
            return false;
        }
        if (!context->status())
        {
            ESP_LOGE(TAG, "Failed to initialize AcousticsLab context: %s", context->status().message().c_str());
            return false;
        }

        // Get microphone sensor (ID 2 based on microphone.hpp)
        _microphone = hal::SensorRegistry::getSensor(2);
        if (!_microphone)
        {
            ESP_LOGE(TAG, "Microphone sensor not found");
            return false;
        }

        // Initialize microphone
        auto status = _microphone->init();
        if (!status)
        {
            ESP_LOGE(TAG, "Failed to initialize microphone: %s", status.message().c_str());
            return false;
        }

        // Create inference pipeline with configuration matching faudioclassify
        api::InferencePipelineConfig config;
        config.audio_samples = 44032;    // Match faudioclassify CONFIG_AUDIO_SAMPLES
        config.frame_length = 2048;      // Match faudioclassify CONFIG_FRAME_LEN
        config.hop_length = 1024;        // Match faudioclassify CONFIG_HOP_LEN
        config.fft_size = 2048;          // Match faudioclassify CONFIG_FFT_SIZE
        config.features_per_frame = 232; // Match faudioclassify CONFIG_FEATURES_PER_FRAME
        config.use_psram = true;
        config.normalize_features = true;
        config.tensor_arena_size = 2048 * 1024;
        config.model_name = "fixed2conv_model";
        config.model_version = "1.0.0";
        config.model_path = "embedded://fixed2conv_model.tflite";
        config.labels = MODEL_LABELS;

        ESP_LOGI(TAG, "Pipeline config: samples=%zu, frame_len=%zu, hop_len=%zu, fft_size=%zu, features_per_frame=%zu",
            config.audio_samples, config.frame_length, config.hop_length, config.fft_size, config.features_per_frame);

        _pipeline = std::make_unique<api::InferencePipeline>(config);
        if (!_pipeline)
        {
            ESP_LOGE(TAG, "Failed to create inference pipeline");
            return false;
        }

        // Initialize pipeline
        status = _pipeline->init();
        if (!status)
        {
            ESP_LOGE(TAG, "Failed to initialize inference pipeline: %s", status.message().c_str());
            return false;
        }

        // Load model from embedded data
        if (!loadModelFromEmbedded(_model_data))
        {
            ESP_LOGE(TAG, "Failed to load embedded model");
            return false;
        }

        status = _pipeline->loadModel(_model_data.data(), _model_data.size(), MODEL_LABELS);
        if (!status)
        {
            ESP_LOGE(TAG, "Failed to load model: %s", status.message().c_str());
            return false;
        }

        ESP_LOGI(TAG, "Audio Classification Demo initialized successfully");
        return true;
    }

    void start()
    {
        if (_running)
        {
            ESP_LOGW(TAG, "Demo already running");
            return;
        }

        _running = true;

        // Create classification task
        xTaskCreate(classificationTask, "classification_task", 8192, this, 5, &_task_handle);

        ESP_LOGI(TAG, "Audio Classification Demo started");
    }

    void stop()
    {
        if (!_running)
        {
            return;
        }

        _running = false;

        // Wait for task to finish
        if (_task_handle)
        {
            vTaskDelete(_task_handle);
            _task_handle = nullptr;
        }

        // Cleanup pipeline
        if (_pipeline)
        {
            _pipeline->deinit();
            _pipeline.reset();
        }

        // Cleanup microphone
        if (_microphone)
        {
            _microphone->deinit();
            _microphone = nullptr;
        }

        ESP_LOGI(TAG, "Audio Classification Demo stopped");
    }

    void printPerformanceStats()
    {
        if (!_pipeline)
        {
            return;
        }

        auto stats = _pipeline->getPerformanceStats();
        ESP_LOGI(TAG, "Performance Statistics:");
        ESP_LOGI(TAG, "  Total inferences: %zu", stats.total_inferences);
        ESP_LOGI(TAG, "  Avg feature extraction: %.2f ms", stats.avg_feature_extraction_ms);
        ESP_LOGI(TAG, "  Avg inference: %.2f ms", stats.avg_inference_ms);
        ESP_LOGI(TAG, "  Avg total: %.2f ms", stats.avg_total_ms);
    }

private:
    static void classificationTask(void *param)
    {
        auto *demo = static_cast<AudioClassificationDemo *>(param);
        demo->runClassification();
    }

    void runClassification()
    {
        ESP_LOGI(TAG, "Starting audio classification loop...");

        // Initialize sliding window buffer similar to faudioclassify
        const size_t ANALYSIS_WINDOW_SAMPLES = 44032;
        const size_t INFERENCE_STRIDE_SAMPLES = 22016; // Half window for 50% overlap
        const size_t AUDIO_CHUNK_SAMPLES = 4410;       // 100ms chunks from microphone

        // Allocate analysis window buffer
        std::unique_ptr<int16_t[]> analysis_window(new int16_t[ANALYSIS_WINDOW_SAMPLES]);
        std::unique_ptr<float[]> float_buffer(new float[ANALYSIS_WINDOW_SAMPLES]);

        if (!analysis_window || !float_buffer)
        {
            ESP_LOGE(TAG, "Failed to allocate analysis buffers");
            return;
        }

        // Initialize analysis window with zeros
        memset(analysis_window.get(), 0, ANALYSIS_WINDOW_SAMPLES * sizeof(int16_t));

        size_t samples_since_last_inference = 0;
        size_t frame_count = 0;
        api::InferenceResult result;

        ESP_LOGI(TAG, "Waiting for initial audio data to fill buffer...");

        while (_running)
        {
            // Read small chunks from microphone (similar to faudioclassify)
            core::DataFrame<core::Tensor> audio_chunk;
            auto status = _microphone->readDataFrame(audio_chunk, AUDIO_CHUNK_SAMPLES);
            if (!status)
            {
                ESP_LOGD(TAG, "Waiting for audio data: %s", status.message().c_str());
                vTaskDelay(pdMS_TO_TICKS(50)); // Shorter delay for chunk reading
                continue;
            }

            // Convert chunk to int16_t array
            const int16_t *chunk_data = audio_chunk.data.dataAs<int16_t>();
            if (!chunk_data)
            {
                ESP_LOGE(TAG, "Failed to get chunk data");
                continue;
            }

            // Update sliding window (shift left and append new data)
            memmove(analysis_window.get(), analysis_window.get() + AUDIO_CHUNK_SAMPLES,
                (ANALYSIS_WINDOW_SAMPLES - AUDIO_CHUNK_SAMPLES) * sizeof(int16_t));
            memcpy(analysis_window.get() + (ANALYSIS_WINDOW_SAMPLES - AUDIO_CHUNK_SAMPLES), chunk_data,
                AUDIO_CHUNK_SAMPLES * sizeof(int16_t));

            // Accumulate samples for inference trigger
            samples_since_last_inference += AUDIO_CHUNK_SAMPLES;

            // Check if we have enough new samples to trigger inference
            if (samples_since_last_inference >= INFERENCE_STRIDE_SAMPLES)
            {
                samples_since_last_inference -= INFERENCE_STRIDE_SAMPLES;

                ESP_LOGI(TAG, "Triggering inference...");
                int64_t start_time = esp_timer_get_time();

                // Convert analysis window to float (normalize to [-1.0, 1.0] range)
                for (size_t i = 0; i < ANALYSIS_WINDOW_SAMPLES; i++)
                {
                    float_buffer[i] = static_cast<float>(analysis_window[i]) / 32768.0f;
                }

                // Debug: Print first few samples to verify data
                ESP_LOGD(TAG, "Audio samples (first 10): %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f",
                    float_buffer[0], float_buffer[1], float_buffer[2], float_buffer[3], float_buffer[4],
                    float_buffer[5], float_buffer[6], float_buffer[7], float_buffer[8], float_buffer[9]);

                // Perform inference using float buffer
                ESP_LOGI(TAG, "Calling pipeline->infer with %zu samples", ANALYSIS_WINDOW_SAMPLES);
                status = _pipeline->infer(float_buffer.get(), ANALYSIS_WINDOW_SAMPLES, result);
                if (!status)
                {
                    ESP_LOGE(TAG, "Inference failed: %s", status.message().c_str());
                    continue;
                }

                int64_t end_time = esp_timer_get_time();
                frame_count++;

                // Print results in a format similar to faudioclassify
                printf("----------------------------------------\n");
                printf("Inference completed, class probabilities: (Frame: %zu, Time: %lld us)\n", frame_count,
                    end_time - start_time);

                for (size_t i = 0; i < result.scores.size(); ++i)
                {
                    std::string label = result.getLabel(static_cast<int>(i));
                    printf("%s: %.6f\n", label.c_str(), result.scores[i]);
                    ESP_LOGD(TAG, "  %s: %.6f", label.c_str(), result.scores[i]);
                }

                printf("----------------------------------------\n");

                // Also log in ESP format
                auto [top_class, top_score] = result.getTopClass();
                if (top_class >= 0)
                {
                    std::string top_label = result.getLabel(top_class);
                    ESP_LOGI(TAG, ">> Top class: %s (%.6f)", top_label.c_str(), top_score);
                }

                // Print performance stats every 5 inferences
                if (frame_count % 5 == 0)
                {
                    printPerformanceStats();
                }
            }

            // No additional delay needed - the natural audio chunk timing provides the right pace
        }

        ESP_LOGI(TAG, "Classification loop ended");
    }

    hal::Sensor *_microphone;
    std::unique_ptr<api::InferencePipeline> _pipeline;
    std::vector<uint8_t> _model_data;
    bool _running;
    TaskHandle_t _task_handle = nullptr;
};

// Global demo instance
static std::unique_ptr<AudioClassificationDemo> g_demo;

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== AcousticsLab Audio Classification Example ===");

    // Model is loaded from absolute path, no file system initialization needed

    // Create and initialize demo
    g_demo = std::make_unique<AudioClassificationDemo>();
    if (!g_demo->init())
    {
        ESP_LOGE(TAG, "Failed to initialize demo");
        return;
    }

    // Start classification
    g_demo->start();

    ESP_LOGI(TAG, "Demo is running. Press Ctrl+C to stop.");

    // Keep main task alive
    while (true)
    {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "Demo still running...");
    }
}
