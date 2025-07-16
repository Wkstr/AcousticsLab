#pragma once
#ifndef TFLITE_ENGINE_HPP
#define TFLITE_ENGINE_HPP

#include "../engine.hpp"
#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations for TensorFlow Lite Micro
namespace tflite {
class Model;
class MicroInterpreter;
template<unsigned int tOpCount>
class MicroMutableOpResolver;
} // namespace tflite

struct TfLiteTensor;

namespace hal {

/**
 * @brief Default configuration for TFLite Engine
 */
inline core::ConfigObjectMap DEFAULT_TFLITE_ENGINE_CONFIGS()
{
    core::ConfigObjectMap configs;

    configs.emplace("tensor_arena_size", core::ConfigObject::createInteger("tensor_arena_size",
                                             "Size of tensor arena in bytes", 2048 * 1024, 64 * 1024, 8 * 1024 * 1024));
    configs.emplace("use_psram",
        core::ConfigObject::createBoolean("use_psram", "Use PSRAM for tensor arena allocation", true));
    configs.emplace("enable_profiling",
        core::ConfigObject::createBoolean("enable_profiling", "Enable performance profiling", false));
    configs.emplace("max_op_count",
        core::ConfigObject::createInteger("max_op_count", "Maximum number of operations", 20, 5, 100));

    return configs;
}

/**
 * @brief TensorFlow Lite Micro inference engine
 *
 * This engine provides TensorFlow Lite Micro inference capabilities within
 * the AcousticsLab framework. It supports quantized models and provides
 * automatic quantization/dequantization for float inputs/outputs.
 */
class EngineTFLite final: public Engine
{
public:
    /**
     * @brief Constructor - registers engine with default configuration
     */
    EngineTFLite();

    /**
     * @brief Destructor - cleanup resources
     */
    ~EngineTFLite() override;

    /**
     * @brief Initialize the TFLite engine
     */
    core::Status init() override;

    /**
     * @brief Deinitialize and cleanup resources
     */
    core::Status deinit() override;

    /**
     * @brief Update engine configuration
     */
    core::Status updateConfig(const core::ConfigMap &configs) override;

    /**
     * @brief Set model data for loading
     *
     * @param model_data Pointer to model data
     * @param model_size Size of model data in bytes
     * @return core::Status indicating success or failure
     */
    core::Status setModelData(const void *model_data, size_t model_size);

    /**
     * @brief Load a TFLite model
     *
     * @param info Model information including path and metadata
     * @param model Output shared pointer to loaded model
     * @return core::Status indicating success or failure
     */
    const core::Status loadModel(const core::Model::Info &info, std::shared_ptr<core::Model> &model) noexcept override;

private:
    /**
     * @brief Update configuration from info object
     */
    core::Status updateConfigFromInfo();

    /**
     * @brief Allocate tensor arena
     */
    core::Status allocateTensorArena();

    /**
     * @brief Free tensor arena
     */
    void freeTensorArena();

    /**
     * @brief Setup operation resolver with common operations
     */
    core::Status setupOpResolver();

    /**
     * @brief Create TFLite model graph
     *
     * @param model_data Pointer to model data
     * @param model_size Size of model data
     * @param labels Label mapping for the model
     * @return Shared pointer to created graph
     */
    std::shared_ptr<core::Model::Graph> createModelGraph(const void *model_data, size_t model_size,
        const std::unordered_map<int, std::string> &labels);

    /**
     * @brief Forward pass callback for TFLite inference
     *
     * @param graph Model graph to execute
     * @return core::Status indicating success or failure
     */
    core::Status forwardCallback(core::Model::Graph &graph);

    // Configuration parameters
    bool _initialized;
    size_t _tensor_arena_size;
    bool _use_psram;
    bool _enable_profiling;
    size_t _max_op_count;

    // TensorFlow Lite Micro resources
    unsigned char *_tensor_arena;

    // Thread safety
    std::mutex _lock;

    // Internal implementation details (using void* to avoid exposing TFLite headers)
    struct TFLiteImpl;
    std::unique_ptr<TFLiteImpl> _impl;
};

} // namespace hal

#endif // TFLITE_ENGINE_HPP
