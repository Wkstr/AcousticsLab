#pragma once
#ifndef INFERENCE_PIPELINE_HPP
#define INFERENCE_PIPELINE_HPP

#include "core/config_object.hpp"
#include "core/data_frame.hpp"
#include "core/logger.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "hal/engine.hpp"
#include "hal/processor.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace api {

/**
 * @brief Audio inference pipeline configuration
 */
struct InferencePipelineConfig
{
    // Feature extraction configuration - match main repository (16kHz)
    size_t audio_samples = 16000;
    size_t frame_length = 512;
    size_t hop_length = 256;
    size_t fft_size = 512;
    size_t features_per_frame = 257;
    bool use_psram = true;
    bool normalize_features = true;

    // Inference engine configuration
    size_t tensor_arena_size = 2048 * 1024;
    bool enable_profiling = false;

    // Model configuration
    std::string model_name = "audio_classifier";
    std::string model_version = "1.0.0";
    std::string model_path = "";
    std::unordered_map<int, std::string> labels;
};

/**
 * @brief Inference result containing classification scores
 */
struct InferenceResult
{
    std::vector<float> scores;
    std::unordered_map<int, std::string> labels;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    size_t frame_index;

    /**
     * @brief Get the class with highest score
     */
    std::pair<int, float> getTopClass() const
    {
        if (scores.empty())
            return { -1, 0.0f };

        int max_idx = 0;
        float max_score = scores[0];
        for (size_t i = 1; i < scores.size(); ++i)
        {
            if (scores[i] > max_score)
            {
                max_score = scores[i];
                max_idx = static_cast<int>(i);
            }
        }
        return { max_idx, max_score };
    }

    /**
     * @brief Get label for a class index
     */
    std::string getLabel(int class_idx) const
    {
        auto it = labels.find(class_idx);
        return (it != labels.end()) ? it->second : "Unknown";
    }
};

/**
 * @brief Unified audio inference pipeline
 *
 * This class provides a high-level API for audio classification that combines
 * feature extraction and model inference in a single, easy-to-use interface.
 * It manages the entire pipeline from raw audio data to classification results.
 */
class InferencePipeline
{
public:
    /**
     * @brief Constructor with configuration
     */
    explicit InferencePipeline(const InferencePipelineConfig &config = InferencePipelineConfig {});

    /**
     * @brief Destructor
     */
    ~InferencePipeline();

    /**
     * @brief Initialize the inference pipeline
     *
     * @return core::Status indicating success or failure
     */
    core::Status init();

    /**
     * @brief Deinitialize the pipeline
     *
     * @return core::Status indicating success or failure
     */
    core::Status deinit();

    /**
     * @brief Check if pipeline is initialized
     */
    bool initialized() const noexcept
    {
        return _initialized;
    }

    /**
     * @brief Update pipeline configuration
     *
     * @param config New configuration
     * @return core::Status indicating success or failure
     */
    core::Status updateConfig(const InferencePipelineConfig &config);

    /**
     * @brief Load a model for inference
     *
     * @param model_data Pointer to model data (e.g., TFLite model bytes)
     * @param model_size Size of model data in bytes
     * @param labels Label mapping for the model
     * @return core::Status indicating success or failure
     */
    core::Status loadModel(const void *model_data, size_t model_size,
        const std::unordered_map<int, std::string> &labels);

    /**
     * @brief Perform inference on audio data
     *
     * @param audio_data Pointer to audio samples (float32)
     * @param audio_samples Number of audio samples
     * @param result Output inference result
     * @return core::Status indicating success or failure
     */
    core::Status infer(const float *audio_data, size_t audio_samples, InferenceResult &result);

    /**
     * @brief Perform inference on audio data frame
     *
     * @param input_frame Input data frame containing audio tensor
     * @param result Output inference result
     * @return core::Status indicating success or failure
     */
    core::Status inferDataFrame(const core::DataFrame<core::Tensor> &input_frame, InferenceResult &result);

    /**
     * @brief Get current configuration
     */
    const InferencePipelineConfig &getConfig() const noexcept
    {
        return _config;
    }

    /**
     * @brief Get performance statistics
     */
    struct PerformanceStats
    {
        size_t total_inferences = 0;
        double avg_feature_extraction_ms = 0.0;
        double avg_inference_ms = 0.0;
        double avg_total_ms = 0.0;
    };

    PerformanceStats getPerformanceStats() const;

private:
    /**
     * @brief Initialize feature extractor
     */
    core::Status initFeatureExtractor();

    /**
     * @brief Initialize inference engine
     */
    core::Status initInferenceEngine();

    /**
     * @brief Create model configuration
     */
    core::ConfigMap createModelConfig() const;

    /**
     * @brief Create processor configuration
     */
    core::ConfigMap createProcessorConfig() const;

    // Configuration
    InferencePipelineConfig _config;

    // State
    bool _initialized;
    mutable std::mutex _lock;

    // Components
    hal::Processor *_feature_extractor;
    hal::Engine *_inference_engine;
    std::shared_ptr<core::Model> _model;

    // Working tensors
    std::unique_ptr<core::Tensor> _feature_tensor;
    std::unique_ptr<core::Tensor> _output_tensor;

    // Performance tracking
    mutable PerformanceStats _perf_stats;
    size_t _frame_counter;
};

} // namespace api

#endif // INFERENCE_PIPELINE_HPP
