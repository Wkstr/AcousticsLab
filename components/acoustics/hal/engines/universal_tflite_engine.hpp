#ifndef UNIVERSAL_TFLITE_ENGINE_HPP
#define UNIVERSAL_TFLITE_ENGINE_HPP

#include "core/config.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"
#include "engine.hpp"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <memory>
#include <string>
#include <vector>

namespace hal {

/**
 * @brief Universal TensorFlow Lite Micro Engine for AcousticsLab
 *
 * This engine can automatically load and run any model converted with the
 * improved model converter script. It dynamically adapts to different:
 * - Input/output tensor sizes
 * - Number of classes
 * - Quantization parameters
 * - Class labels
 */
class UniversalTFLiteEngine final: public Engine
{
public:
    /**
     * @brief Model configuration structure
     */
    struct ModelConfig
    {
        // Model data
        const unsigned char *model_data;
        unsigned int model_size;

        // Input specifications
        int input_height;
        int input_width;
        int input_channels;
        int input_size;
        float input_scale;
        int input_zero_point;

        // Output specifications
        int output_size;
        int num_classes;
        float output_scale;
        int output_zero_point;

        // Class labels
        std::vector<std::string> class_labels;

        // Validation
        bool isValid() const
        {
            return model_data != nullptr && model_size > 0 && input_size > 0 && output_size > 0 && num_classes > 0
                   && class_labels.size() == static_cast<size_t>(num_classes);
        }
    };

    /**
     * @brief Constructor with automatic model detection
     *
     * This constructor will attempt to load model configuration from
     * the acousticslab_model.h header file if available.
     */
    UniversalTFLiteEngine();

    /**
     * @brief Constructor with explicit model configuration
     *
     * @param config Model configuration structure
     */
    explicit UniversalTFLiteEngine(const ModelConfig &config);

    /**
     * @brief Destructor - cleanup resources
     */
    ~UniversalTFLiteEngine() override;

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
     * @brief Run inference on input tensor
     *
     * @param input Input tensor (features)
     * @param output Output tensor (class probabilities)
     * @return Status of the operation
     */
    core::Status forward(const core::Tensor &input, core::Tensor &output) override;

    /**
     * @brief Get expected input tensor specification
     */
    std::pair<core::Tensor::Shape, core::Tensor::Type> getInputSpec() const override;

    /**
     * @brief Get expected output tensor specification
     */
    std::pair<core::Tensor::Shape, core::Tensor::Type> getOutputSpec() const override;

    /**
     * @brief Get class label for given class ID
     *
     * @param class_id Class index (0-based)
     * @return Class label string
     */
    std::string getClassLabel(int class_id) const;

    /**
     * @brief Get all class labels
     *
     * @return Vector of all class labels
     */
    const std::vector<std::string> &getClassLabels() const;

    /**
     * @brief Get number of classes
     *
     * @return Number of output classes
     */
    int getNumClasses() const;

    /**
     * @brief Load model configuration from header file
     *
     * This method attempts to load configuration from the generated
     * acousticslab_model.h header file.
     *
     * @return Status of the operation
     */
    core::Status loadModelConfig();

    /**
     * @brief Set model configuration manually
     *
     * @param config Model configuration structure
     * @return Status of the operation
     */
    core::Status setModelConfig(const ModelConfig &config);

    /**
     * @brief Get current model configuration
     *
     * @return Current model configuration
     */
    const ModelConfig &getModelConfig() const;

private:
    // TensorFlow Lite Micro components
    std::unique_ptr<tflite::MicroInterpreter> _interpreter;
    std::unique_ptr<tflite::MicroMutableOpResolver<10>> _resolver;
    std::unique_ptr<uint8_t[]> _tensor_arena;
    const tflite::Model *_model;

    // Model configuration
    ModelConfig _config;
    bool _initialized;
    bool _config_loaded;

    // Memory management
    static constexpr size_t kTensorArenaSize = 100 * 1024; // 100KB default

    /**
     * @brief Setup TensorFlow Lite Micro operations
     */
    core::Status setupOperations();

    /**
     * @brief Validate model configuration
     */
    core::Status validateConfig() const;

    /**
     * @brief Quantize input data from float32 to int8
     */
    core::Status quantizeInput(const float *input_data, int8_t *quantized_data, size_t size);

    /**
     * @brief Dequantize output data from int8 to float32
     */
    core::Status dequantizeOutput(const int8_t *quantized_data, float *output_data, size_t size);

    /**
     * @brief Load default configuration if no header file is available
     */
    core::Status loadDefaultConfig();

    /**
     * @brief Forward callback implementation
     */
    core::Status forwardCallback(const core::Tensor &input, core::Tensor &output);
};

/**
 * @brief Factory function to create UniversalTFLiteEngine with auto-detection
 *
 * @return Unique pointer to engine instance
 */
std::unique_ptr<UniversalTFLiteEngine> createUniversalTFLiteEngine();

/**
 * @brief Factory function to create UniversalTFLiteEngine with specific config
 *
 * @param config Model configuration
 * @return Unique pointer to engine instance
 */
std::unique_ptr<UniversalTFLiteEngine> createUniversalTFLiteEngine(const UniversalTFLiteEngine::ModelConfig &config);

} // namespace hal

// Include auto-generated model configuration if available
#ifdef ACOUSTICSLAB_MODEL_H_
#include "acousticslab_model.h"
#endif

#endif // UNIVERSAL_TFLITE_ENGINE_HPP
