#pragma once
#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include "../processor.hpp"
#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>

// KissFFT includes
extern "C" {
#include "kiss_fft.h"
#include "kiss_fftr.h"
}

namespace hal {

/**
 * @brief Default configuration for FeatureExtractor processor
 */
inline core::ConfigObjectMap DEFAULT_FEATURE_EXTRACTOR_CONFIGS()
{
    core::ConfigObjectMap configs;

    // Configuration matching faudioclassify for audio classification
    configs.emplace("audio_samples",
        core::ConfigObject::createInteger("audio_samples", "Number of audio samples to process", 44032, 1024, 88064));
    configs.emplace("frame_length",
        core::ConfigObject::createInteger("frame_length", "Frame length for STFT", 2048, 512, 4096));
    configs.emplace("hop_length",
        core::ConfigObject::createInteger("hop_length", "Hop length for STFT", 1024, 256, 2048));
    configs.emplace("fft_size", core::ConfigObject::createInteger("fft_size", "FFT size", 2048, 512, 4096));
    configs.emplace("features_per_frame",
        core::ConfigObject::createInteger("features_per_frame", "Number of features per frame", 232, 64, 1024));
    configs.emplace("use_psram",
        core::ConfigObject::createBoolean("use_psram", "Use PSRAM for buffer allocation", true));
    configs.emplace("normalize_features",
        core::ConfigObject::createBoolean("normalize_features", "Apply global normalization", true));

    return configs;
}

/**
 * @brief Audio feature extraction processor
 *
 * This processor implements STFT-based feature extraction from audio signals.
 * It converts time-domain audio data into log-magnitude spectral features
 * suitable for machine learning models.
 */
class ProcessorFeatureExtractor final: public Processor
{
public:
    /**
     * @brief Constructor - registers processor with default configuration
     */
    ProcessorFeatureExtractor()
        : Processor(Info(1, "Audio Feature Extractor", Type::FeatureExtractor, DEFAULT_FEATURE_EXTRACTOR_CONFIGS())),
          _initialized(false), _use_psram(true), _normalize_features(true), _audio_samples(44032), _frame_length(2048),
          _hop_length(1024), _fft_size(2048), _features_per_frame(232), _num_frames(42), _fftr_cfg(nullptr),
          _fft_input_buffer(nullptr), _fft_output_buffer(nullptr), _log_features_buffer(nullptr),
          _hanning_window(nullptr)
    {
        _output_size = _num_frames * _features_per_frame;
    }

    /**
     * @brief Destructor - cleanup resources
     */
    ~ProcessorFeatureExtractor() override
    {
        deinit();
    }

    /**
     * @brief Initialize the feature extractor
     */
    core::Status init() override;

    /**
     * @brief Deinitialize and cleanup resources
     */
    core::Status deinit() override;

    /**
     * @brief Update processor configuration
     */
    core::Status updateConfig(const core::ConfigMap &configs) override;

    /**
     * @brief Process audio tensor to extract features
     *
     * @param input Input tensor containing audio samples (Float32, shape: [audio_samples])
     * @param output Output tensor for features (Float32, shape: [num_frames * features_per_frame])
     * @return core::Status indicating success or failure
     */
    core::Status process(const core::Tensor &input, core::Tensor &output) override;

    /**
     * @brief Process audio data frame to extract features
     *
     * @param input_frame Input data frame containing audio tensor
     * @param output_frame Output data frame containing feature tensor
     * @return core::Status indicating success or failure
     */
    core::Status processDataFrame(const core::DataFrame<core::Tensor> &input_frame,
        core::DataFrame<core::Tensor> &output_frame) override;

    /**
     * @brief Get expected input tensor specification
     *
     * @return Pair of shape and type for input tensor
     */
    std::pair<core::Tensor::Shape, core::Tensor::Type> getInputSpec() const override
    {
        return std::make_pair(core::Tensor::Shape({ _audio_samples }), core::Tensor::Type::Float32);
    }

    /**
     * @brief Get expected output tensor specification
     *
     * @return Pair of shape and type for output tensor
     */
    std::pair<core::Tensor::Shape, core::Tensor::Type> getOutputSpec() const override
    {
        return std::make_pair(core::Tensor::Shape({ _output_size }), core::Tensor::Type::Float32);
    }

private:
    /**
     * @brief Initialize Hanning window coefficients
     */
    core::Status initHanningWindow();

    /**
     * @brief Allocate processing buffers
     */
    core::Status allocateBuffers();

    /**
     * @brief Free processing buffers
     */
    void freeBuffers();

    /**
     * @brief Apply global normalization to features
     *
     * @param features Feature array to normalize
     * @param total_features Total number of features
     */
    void normalizeGlobally(float *features, size_t total_features);

    /**
     * @brief Extract features from audio data
     *
     * @param audio_data Input audio samples
     * @param output_features Output feature array
     */
    void extractFeatures(const float *audio_data, float *output_features);

    /**
     * @brief Update configuration from info object
     */
    core::Status updateConfigFromInfo();

    // Configuration parameters
    bool _initialized;
    bool _use_psram;
    bool _normalize_features;
    size_t _audio_samples;
    size_t _frame_length;
    size_t _hop_length;
    size_t _fft_size;
    size_t _features_per_frame;
    size_t _num_frames;
    size_t _output_size;

    // Processing resources
    kiss_fftr_cfg _fftr_cfg;
    float *_fft_input_buffer;
    kiss_fft_cpx *_fft_output_buffer;
    float *_log_features_buffer;
    float *_hanning_window;
};

} // namespace hal

#endif // FEATURE_EXTRACTOR_HPP
