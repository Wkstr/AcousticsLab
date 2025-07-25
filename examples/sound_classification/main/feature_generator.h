#pragma once

#include "dl_rfft.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#define CONFIG_AUDIO_SAMPLES      44032
#define CONFIG_FRAME_LEN          2048
#define CONFIG_HOP_LEN            1024
#define CONFIG_NUM_FRAMES         43
#define CONFIG_FFT_SIZE           2048
#define CONFIG_FEATURES_PER_FRAME 232
#define CONFIG_OUTPUT_SIZE        (CONFIG_NUM_FRAMES * CONFIG_FEATURES_PER_FRAME)

class FeatureGenerator
{
public:
    FeatureGenerator();
    ~FeatureGenerator();

    bool init();
    void generate(const int16_t *raw_audio_data, float *output_features);

private:
    bool initialized;
    dl_fft_f32_t *fft_handle;
    float *fft_input_buffer;
    float *log_features_buffer;
    float *audio_float_buffer;

    void convert_int16_to_float(const int16_t *input, float *output);
    void prepare_frame(const float *audio_data, size_t frame_idx);
    void normalize_globally(float *features, size_t total_features);
};
