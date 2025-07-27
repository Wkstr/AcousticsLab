#include "feature_generator.h"
#include <cmath>
#include <cstring>
#include <esp_heap_caps.h>
#include <esp_log.h>

static const char *TAG = "FeatureGenerator";
static const float PI = 3.14159265358979323846264338327950288419716939937510582097494459231f;

FeatureGenerator::FeatureGenerator()
    : initialized(false), fft_handle(nullptr), fft_input_buffer(nullptr), log_features_buffer(nullptr),
      audio_float_buffer(nullptr), blackman_window(nullptr)
{
}

FeatureGenerator::~FeatureGenerator()
{
    if (fft_handle)
    {
        dl_rfft_f32_deinit(fft_handle);
        fft_handle = nullptr;
    }
    if (fft_input_buffer)
    {
        heap_caps_free(fft_input_buffer);
        fft_input_buffer = nullptr;
    }
    if (log_features_buffer)
    {
        heap_caps_free(log_features_buffer);
        log_features_buffer = nullptr;
    }
    if (audio_float_buffer)
    {
        heap_caps_free(audio_float_buffer);
        audio_float_buffer = nullptr;
    }
    if (blackman_window)
    {
        heap_caps_free(blackman_window);
        blackman_window = nullptr;
    }
}

bool FeatureGenerator::init()
{
    // Allocate buffers
    fft_input_buffer = (float *)heap_caps_aligned_alloc(16, CONFIG_FFT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    log_features_buffer = (float *)heap_caps_aligned_alloc(16, CONFIG_OUTPUT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    audio_float_buffer = (float *)heap_caps_aligned_alloc(16, CONFIG_AUDIO_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);

    if (!fft_input_buffer || !log_features_buffer || !audio_float_buffer)
    {
        ESP_LOGE(TAG, "Buffer allocation failed");
        return false;
    }

    // Initialize ESP-DL real FFT
    fft_handle = dl_rfft_f32_init(CONFIG_FFT_SIZE, MALLOC_CAP_SPIRAM);
    if (!fft_handle)
    {
        ESP_LOGE(TAG, "ESP-DL RFFT initialization failed");
        return false;
    }

    // Initialize Blackman window
    init_blackman_window();
    if (!blackman_window)
    {
        ESP_LOGE(TAG, "Blackman window initialization failed");
        return false;
    }

    ESP_LOGI(TAG, "FeatureGenerator initialized successfully");
    initialized = true;
    return true;
}

void FeatureGenerator::init_blackman_window()
{
    blackman_window = (float *)heap_caps_aligned_alloc(16, CONFIG_FRAME_LEN * sizeof(float), MALLOC_CAP_SPIRAM);

    if (!blackman_window)
    {
        ESP_LOGE(TAG, "Failed to allocate memory for Blackman window");
        return;
    }

    for (size_t n = 0; n < CONFIG_FRAME_LEN; ++n)
    {
        float n_norm = (float)n / (float)CONFIG_FRAME_LEN;
        blackman_window[n] = 0.42f - 0.5f * cos(2.0f * PI * n_norm) + 0.08f * cos(4.0f * PI * n_norm);
    }

    ESP_LOGI(TAG, "Blackman window initialized with %d coefficients", CONFIG_FRAME_LEN);
}

void FeatureGenerator::convert_int16_to_float(const int16_t *input, float *output)
{
    for (size_t i = 0; i < CONFIG_AUDIO_SAMPLES; ++i)
    {
        output[i] = (float)input[i] / 32768.0f;
    }
}

void FeatureGenerator::prepare_frame(const float *audio_data, size_t frame_idx)
{
    size_t start_in_padded = frame_idx * CONFIG_HOP_LEN;
    size_t padding_size = CONFIG_HOP_LEN;
    for (size_t i = 0; i < CONFIG_HOP_LEN; ++i)
    {
        size_t pos_in_padded = start_in_padded + i;
        if (pos_in_padded < padding_size)
        {
            fft_input_buffer[i] = 0.0f;
        }
        else
        {
            fft_input_buffer[i] = audio_data[pos_in_padded - padding_size];
        }
    }
    memcpy(fft_input_buffer + CONFIG_HOP_LEN, audio_data + (frame_idx * CONFIG_HOP_LEN),
        CONFIG_HOP_LEN * sizeof(float));

    for (size_t i = 0; i < CONFIG_FRAME_LEN; ++i)
    {
        fft_input_buffer[i] *= blackman_window[i];
    }
}

void FeatureGenerator::normalize_globally(float *features, size_t total_features)
{
    double mean = 0.0f;
    for (size_t i = 0; i < total_features; ++i)
    {
        mean += features[i];
    }
    mean /= total_features;

    double variance = 0.0f;
    for (size_t i = 0; i < total_features; ++i)
    {
        float diff = features[i] - mean;
        variance += diff * diff;
    }
    variance /= total_features;
    float std_dev = sqrtf(variance);

    const float epsilon = 0.00009999999747378752f;
    float inv_std = 1.0f / (std_dev + epsilon);

    for (size_t i = 0; i < total_features; ++i)
    {
        features[i] = (features[i] - float(mean)) * inv_std;
    }
}

void FeatureGenerator::generate(const int16_t *raw_audio_data, float *output_features)
{
    if (!initialized)
    {
        ESP_LOGE(TAG, "FeatureGenerator not initialized");
        return;
    }

    ESP_LOGI(TAG, "Starting feature generation for %d frames", CONFIG_NUM_FRAMES);

    convert_int16_to_float(raw_audio_data, audio_float_buffer);

    for (size_t frame_idx = 0; frame_idx < CONFIG_NUM_FRAMES; ++frame_idx)
    {
        prepare_frame(audio_float_buffer, frame_idx);

        dl_rfft_f32_run(fft_handle, fft_input_buffer);

        float *current_log_features = log_features_buffer + (frame_idx * CONFIG_FEATURES_PER_FRAME);
        for (size_t j = 0; j < CONFIG_FEATURES_PER_FRAME; ++j)
        {
            float real = fft_input_buffer[j * 2 + 0];
            float imag = fft_input_buffer[j * 2 + 1];
            float mag = sqrtf(real * real + imag * imag);
            current_log_features[j] = logf(mag < 1e-12f ? 1e-12f : mag);
        }
    }

    normalize_globally(log_features_buffer, CONFIG_OUTPUT_SIZE);

    memcpy(output_features, log_features_buffer, CONFIG_OUTPUT_SIZE * sizeof(float));

    ESP_LOGI(TAG, "Feature generation completed, output size: %d", CONFIG_OUTPUT_SIZE);
}
