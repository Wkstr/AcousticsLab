#include "tflite_engine.hpp"

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_log.h"
// MALLOC_CAP_DEFAULT and MALLOC_CAP_SPIRAM are already defined in esp_heap_caps.h
#else
#include <cstdlib>
#define heap_caps_malloc(size, caps) malloc(size)
#define heap_caps_free(ptr)          free(ptr)
#define MALLOC_CAP_DEFAULT           0
#define MALLOC_CAP_SPIRAM            0
#endif

// TensorFlow Lite Micro includes
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace hal {

/**
 * @brief Helper function to format tensor shape for logging
 */
std::string formatShape(const std::vector<size_t> &shape)
{
    std::string result;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            result += ", ";
        result += std::to_string(shape[i]);
    }
    return result;
}

/**
 * @brief Internal implementation structure for TFLite engine
 */
struct EngineTFLite::TFLiteImpl
{
    const tflite::Model *model = nullptr;
    std::unique_ptr<tflite::MicroInterpreter> interpreter;
    std::unique_ptr<tflite::MicroMutableOpResolver<20>> op_resolver;
    TfLiteTensor *input_tensor = nullptr;
    TfLiteTensor *output_tensor = nullptr;

    // Model data storage
    std::vector<uint8_t> model_data;
    std::unordered_map<int, std::string> labels;

    TFLiteImpl() : op_resolver(std::make_unique<tflite::MicroMutableOpResolver<20>>()) { }
};

EngineTFLite::EngineTFLite()
    : Engine(Info(1, "TensorFlow Lite Micro Engine", Type::TFLiteMicro, DEFAULT_TFLITE_ENGINE_CONFIGS())),
      _initialized(false), _tensor_arena_size(2048 * 1024), _use_psram(true), _enable_profiling(false),
      _max_op_count(20), _tensor_arena(nullptr)
{
}

EngineTFLite::~EngineTFLite()
{
    deinit();
}

core::Status EngineTFLite::init()
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (_initialized)
    {
        LOG(WARNING, "TFLite engine already initialized");
        return STATUS_OK();
    }

    // Update configuration from current settings
    auto status = updateConfigFromInfo();
    if (!status)
    {
        LOG(ERROR, "Failed to update configuration: %s", status.message().c_str());
        return status;
    }

    // Create implementation
    _impl = std::make_unique<TFLiteImpl>();
    if (!_impl)
    {
        LOG(ERROR, "Failed to create TFLite implementation");
        return STATUS(ENOMEM, "TFLite implementation allocation failed");
    }

    // Allocate tensor arena
    status = allocateTensorArena();
    if (!status)
    {
        LOG(ERROR, "Failed to allocate tensor arena: %s", status.message().c_str());
        return status;
    }

    // Setup operation resolver
    status = setupOpResolver();
    if (!status)
    {
        LOG(ERROR, "Failed to setup operation resolver: %s", status.message().c_str());
        return status;
    }

    _initialized = true;
    _info.status = Status::Idle;

    LOG(INFO, "TFLite engine initialized successfully");
    LOG(INFO, "Configuration: arena_size=%zu, use_psram=%s, profiling=%s", _tensor_arena_size,
        _use_psram ? "true" : "false", _enable_profiling ? "true" : "false");

    return STATUS_OK();
}

core::Status EngineTFLite::deinit()
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_initialized)
    {
        return STATUS_OK();
    }

    // Reset TFLite resources
    if (_impl)
    {
        _impl->interpreter.reset();
        _impl->model = nullptr;
        _impl->input_tensor = nullptr;
        _impl->output_tensor = nullptr;
        _impl.reset();
    }

    // Free tensor arena
    freeTensorArena();

    _initialized = false;
    _info.status = Status::Uninitialized;

    LOG(INFO, "TFLite engine deinitialized");
    return STATUS_OK();
}

core::Status EngineTFLite::updateConfig(const core::ConfigMap &configs)
{
    const std::lock_guard<std::mutex> lock(_lock);

    // Update internal configuration from provided config map
    for (const auto &[key, value]: configs)
    {
        if (key == "tensor_arena_size")
        {
            if (auto val = std::get_if<int>(&value))
            {
                _tensor_arena_size = static_cast<size_t>(*val);
            }
        }
        else if (key == "use_psram")
        {
            if (auto val = std::get_if<bool>(&value))
            {
                _use_psram = *val;
            }
        }
        else if (key == "enable_profiling")
        {
            if (auto val = std::get_if<bool>(&value))
            {
                _enable_profiling = *val;
            }
        }
        else if (key == "max_op_count")
        {
            if (auto val = std::get_if<int>(&value))
            {
                _max_op_count = static_cast<size_t>(*val);
            }
        }
    }

    // If already initialized, need to reinitialize with new config
    if (_initialized)
    {
        LOG(INFO, "Reinitializing TFLite engine with new configuration");
        auto status = deinit();
        if (!status)
        {
            return status;
        }
        return init();
    }

    return STATUS_OK();
}

core::Status EngineTFLite::updateConfigFromInfo()
{
    // Extract configuration from _info.configs
    for (const auto &[key, config_obj]: _info.configs)
    {
        if (key == "tensor_arena_size")
        {
            _tensor_arena_size = static_cast<size_t>(config_obj.getValue<int>(2048 * 1024));
        }
        else if (key == "use_psram")
        {
            _use_psram = config_obj.getValue<bool>(true);
        }
        else if (key == "enable_profiling")
        {
            _enable_profiling = config_obj.getValue<bool>(false);
        }
        else if (key == "max_op_count")
        {
            _max_op_count = static_cast<size_t>(config_obj.getValue<int>(20));
        }
    }

    return STATUS_OK();
}

core::Status EngineTFLite::allocateTensorArena()
{
    uint32_t caps = _use_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_DEFAULT;

    _tensor_arena = static_cast<unsigned char *>(heap_caps_malloc(_tensor_arena_size, caps));
    if (!_tensor_arena)
    {
        LOG(ERROR, "Failed to allocate tensor arena of size %zu bytes", _tensor_arena_size);
        return STATUS(ENOMEM, "Tensor arena allocation failed");
    }

    LOG(DEBUG, "Allocated tensor arena: %zu bytes at %p", _tensor_arena_size, _tensor_arena);
    return STATUS_OK();
}

void EngineTFLite::freeTensorArena()
{
    if (_tensor_arena)
    {
        heap_caps_free(_tensor_arena);
        _tensor_arena = nullptr;
        LOG(DEBUG, "Tensor arena freed");
    }
}

core::Status EngineTFLite::setupOpResolver()
{
    if (!_impl || !_impl->op_resolver)
    {
        LOG(ERROR, "TFLite implementation not initialized");
        return STATUS(EFAULT, "TFLite implementation not initialized");
    }

    // Add common operations used in audio classification models
    _impl->op_resolver->AddConv2D();
    _impl->op_resolver->AddMaxPool2D();
    _impl->op_resolver->AddFullyConnected();
    _impl->op_resolver->AddReshape();
    _impl->op_resolver->AddSoftmax();
    _impl->op_resolver->AddDequantize();
    _impl->op_resolver->AddQuantize();
    _impl->op_resolver->AddAdd();
    _impl->op_resolver->AddMul();
    _impl->op_resolver->AddRelu();
    _impl->op_resolver->AddLogistic();

    LOG(DEBUG, "Operation resolver setup completed");
    return STATUS_OK();
}

core::Status EngineTFLite::setModelData(const void *model_data, size_t model_size)
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_impl)
    {
        LOG(ERROR, "TFLite implementation not initialized");
        return STATUS(ENXIO, "TFLite implementation not initialized");
    }

    if (!model_data || model_size == 0)
    {
        LOG(ERROR, "Invalid model data or size");
        return STATUS(EINVAL, "Invalid model data or size");
    }

    // Store model data in implementation
    _impl->model_data.resize(model_size);
    std::memcpy(_impl->model_data.data(), model_data, model_size);

    LOG(INFO, "Model data set successfully, size: %zu bytes", model_size);
    return STATUS_OK();
}

const core::Status EngineTFLite::loadModel(const core::Model::Info &info, std::shared_ptr<core::Model> &model) noexcept
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_initialized)
    {
        LOG(ERROR, "TFLite engine not initialized");
        return STATUS(ENXIO, "TFLite engine not initialized");
    }

    if (info.type != core::Model::Type::TFLite)
    {
        LOG(ERROR, "Model type mismatch: expected TFLite, got %d", static_cast<int>(info.type));
        return STATUS(EINVAL, "Invalid model type");
    }

    // Use stored model data if available
    if (_impl->model_data.empty())
    {
        LOG(ERROR, "No model data available. Call setModelData() first.");
        return STATUS(EINVAL, "No model data available");
    }

    LOG(INFO, "Loading TFLite model from stored data, size: %zu bytes", _impl->model_data.size());

    // Create model graph with actual model data
    LOG(INFO, "Creating model graph...");
    auto graph = createModelGraph(_impl->model_data.data(), _impl->model_data.size(), info.labels);
    if (!graph)
    {
        LOG(ERROR, "Failed to create model graph");
        return STATUS(EFAULT, "Failed to create model graph");
    }
    LOG(INFO, "Model graph created successfully");

    // Create model info
    auto model_info = std::make_shared<core::Model::Info>(info.type, info.name, info.version,
        std::unordered_map<int, std::string>(info.labels), info.path);

    // Create model with graph
    std::vector<std::shared_ptr<core::Model::Graph>> graphs;
    graphs.push_back(graph);

    model = std::make_shared<core::Model>(model_info, std::move(graphs));

    LOG(INFO, "TFLite model '%s' loaded successfully", info.name.c_str());
    return STATUS_OK();
}

std::shared_ptr<core::Model::Graph> EngineTFLite::createModelGraph(const void *model_data, size_t model_size,
    const std::unordered_map<int, std::string> &labels)
{
    if (!_impl)
    {
        LOG(ERROR, "TFLite implementation not initialized");
        return nullptr;
    }

    if (!model_data || model_size == 0)
    {
        LOG(ERROR, "Invalid model data for graph creation");
        return nullptr;
    }

    // Parse TFLite model
    _impl->model = tflite::GetModel(model_data);
    if (!_impl->model)
    {
        LOG(ERROR, "Failed to parse TFLite model");
        return nullptr;
    }

    if (_impl->model->version() != TFLITE_SCHEMA_VERSION)
    {
        LOG(ERROR, "Model schema version %u not supported (expected %d)",
            static_cast<unsigned int>(_impl->model->version()), TFLITE_SCHEMA_VERSION);
        return nullptr;
    }

    // Operation resolver should already be set up in init()

    // Create interpreter
    _impl->interpreter = std::make_unique<tflite::MicroInterpreter>(_impl->model, *_impl->op_resolver, _tensor_arena,
        _tensor_arena_size);

    if (!_impl->interpreter)
    {
        LOG(ERROR, "Failed to create TFLite interpreter");
        return nullptr;
    }

    // Allocate tensors
    TfLiteStatus allocate_status = _impl->interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        LOG(ERROR, "Failed to allocate tensors: %d", allocate_status);
        return nullptr;
    }

    // Get input and output tensors
    _impl->input_tensor = _impl->interpreter->input(0);
    _impl->output_tensor = _impl->interpreter->output(0);

    if (!_impl->input_tensor || !_impl->output_tensor)
    {
        LOG(ERROR, "Failed to get input/output tensors");
        return nullptr;
    }

    // Log tensor shapes for debugging
    LOG(INFO, "Model input tensor: type=%d, dims=%d, bytes=%zu", _impl->input_tensor->type,
        _impl->input_tensor->dims->size, _impl->input_tensor->bytes);
    for (int i = 0; i < _impl->input_tensor->dims->size; ++i)
    {
        LOG(INFO, "  Input dim[%d] = %d", i, _impl->input_tensor->dims->data[i]);
    }
    LOG(INFO, "Model output tensor: type=%d, dims=%d, bytes=%zu", _impl->output_tensor->type,
        _impl->output_tensor->dims->size, _impl->output_tensor->bytes);
    for (int i = 0; i < _impl->output_tensor->dims->size; ++i)
    {
        LOG(INFO, "  Output dim[%d] = %d", i, _impl->output_tensor->dims->data[i]);
    }

    // Calculate expected sizes
    size_t input_elements_from_dims = 1;
    for (int i = 0; i < _impl->input_tensor->dims->size; ++i)
    {
        input_elements_from_dims *= _impl->input_tensor->dims->data[i];
    }
    LOG(INFO, "Input tensor: calculated elements=%zu, bytes=%zu, bytes/element=%zu", input_elements_from_dims,
        _impl->input_tensor->bytes,
        input_elements_from_dims > 0 ? _impl->input_tensor->bytes / input_elements_from_dims : 0);

    // Create input and output tensor descriptors
    std::vector<core::Tensor> inputs;
    std::vector<core::Tensor> outputs;

    // Create input tensor - handle different dimension cases properly
    size_t total_input_elements = 1;

    for (int i = 0; i < _impl->input_tensor->dims->size; ++i)
    {
        size_t dim = static_cast<size_t>(_impl->input_tensor->dims->data[i]);
        total_input_elements *= dim;
    }

    LOG(INFO, "Model input tensor total elements: %zu", total_input_elements);

    if (_impl->input_tensor->dims->size == 2)
    {
        // 2D tensor: [batch_size, features] or [frames, features_per_frame]
        inputs.emplace_back(core::Tensor::Type::Float32,
            std::initializer_list<size_t> { static_cast<size_t>(_impl->input_tensor->dims->data[0]),
                static_cast<size_t>(_impl->input_tensor->dims->data[1]) });
    }
    else if (_impl->input_tensor->dims->size == 1)
    {
        // 1D tensor: [total_features]
        inputs.emplace_back(core::Tensor::Type::Float32,
            std::initializer_list<size_t> { static_cast<size_t>(_impl->input_tensor->dims->data[0]) });
    }
    else
    {
        // Multi-dimensional tensor: flatten to 1D for simplicity
        inputs.emplace_back(core::Tensor::Type::Float32, std::initializer_list<size_t> { total_input_elements });
        LOG(INFO, "Flattened %dD input tensor to 1D with %zu elements", _impl->input_tensor->dims->size,
            total_input_elements);
    }

    // Create output tensor - assume typical classification output shape
    if (_impl->output_tensor->dims->size == 2)
    {
        outputs.emplace_back(core::Tensor::Type::Float32,
            std::initializer_list<size_t> { static_cast<size_t>(_impl->output_tensor->dims->data[0]),
                static_cast<size_t>(_impl->output_tensor->dims->data[1]) });
    }
    else
    {
        // Fallback for other dimensions
        outputs.emplace_back(core::Tensor::Type::Float32,
            std::initializer_list<size_t> { static_cast<size_t>(_impl->output_tensor->dims->data[0]) });
    }

    // Create quantization parameters
    std::vector<core::Tensor::QuantParams> input_quant_params;
    std::vector<core::Tensor::QuantParams> output_quant_params;
    input_quant_params.emplace_back(_impl->input_tensor->params.scale, _impl->input_tensor->params.zero_point);
    output_quant_params.emplace_back(_impl->output_tensor->params.scale, _impl->output_tensor->params.zero_point);

    // Create graph with forward callback using lambda to capture this instance
    auto forward_callback = [this](core::Model::Graph &graph) -> core::Status { return this->forwardCallback(graph); };
    auto graph = std::make_shared<core::Model::Graph>(0, "main", std::move(inputs), std::move(outputs),
        std::move(input_quant_params), std::move(output_quant_params), std::move(forward_callback));

    LOG(INFO, "TFLite model graph created successfully");
    LOG(INFO, "Input tensor dims: %d, Output tensor dims: %d", _impl->input_tensor->dims->size,
        _impl->output_tensor->dims->size);

    return graph;
}

core::Status EngineTFLite::forwardCallback(core::Model::Graph &graph)
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_initialized || !_impl || !_impl->interpreter || !_impl->input_tensor || !_impl->output_tensor)
    {
        LOG(ERROR, "TFLite engine not properly initialized for inference");
        return STATUS(ENXIO, "TFLite engine not initialized");
    }

    if (graph.inputs() == 0 || graph.outputs() == 0)
    {
        LOG(ERROR, "Graph has no input or output tensors");
        return STATUS(EINVAL, "Invalid graph tensor configuration");
    }

    auto *input_graph_tensor = graph.input(0);
    auto *output_graph_tensor = graph.output(0);

    if (!input_graph_tensor || !output_graph_tensor)
    {
        LOG(ERROR, "Failed to get graph input/output tensors");
        return STATUS(EFAULT, "Graph tensor access failed");
    }

    // Get input data from graph tensor
    const float *input_data = input_graph_tensor->dataAs<float>();
    if (!input_data)
    {
        LOG(ERROR, "Failed to get input data from graph tensor");
        return STATUS(EFAULT, "Input data access failed");
    }

    // Get input tensor info
    TfLiteTensor *input_tensor = _impl->input_tensor;
    TfLiteTensor *output_tensor = _impl->output_tensor;

    // Verify tensor types and handle quantization
    if (input_tensor->type == kTfLiteInt8)
    {
        // Handle quantized input
        float input_scale = input_tensor->params.scale;
        int32_t input_zero_point = input_tensor->params.zero_point;
        size_t num_input_elements = input_tensor->bytes;

        // Quantize input data
        for (size_t i = 0; i < num_input_elements; ++i)
        {
            float float_val = input_data[i];
            int32_t quantized_val = static_cast<int32_t>(std::round(float_val / input_scale)) + input_zero_point;

            // Clamp to int8 range
            if (quantized_val < -128)
                quantized_val = -128;
            if (quantized_val > 127)
                quantized_val = 127;

            input_tensor->data.int8[i] = static_cast<int8_t>(quantized_val);
        }
    }
    else if (input_tensor->type == kTfLiteFloat32)
    {
        // Handle float32 input
        size_t input_size = input_graph_tensor->shape().dot();

        // Use bytes to determine actual input size (like faudioclassify)
        size_t tflite_input_bytes = input_tensor->bytes;
        size_t tflite_input_size = tflite_input_bytes / sizeof(float);

        LOG(DEBUG, "Input tensor: graph size=%zu, TFLite size=%zu (bytes=%zu)", input_size, tflite_input_size,
            tflite_input_bytes);

        // Verify sizes match or handle mismatch
        if (input_size != tflite_input_size)
        {
            LOG(WARNING, "Input size mismatch: graph tensor has %zu elements, TFLite tensor expects %zu elements",
                input_size, tflite_input_size);

            // Debug: Print tensor shapes
            LOG(DEBUG, "Graph tensor shape: [%zu]", input_graph_tensor->shape().dot());
            LOG(DEBUG, "TFLite tensor shape: [");
            for (int i = 0; i < input_tensor->dims->size; ++i)
            {
                LOG(DEBUG, "  dim[%d] = %d", i, input_tensor->dims->data[i]);
            }
            LOG(DEBUG, "]");

            // If graph tensor is larger than TFLite tensor, truncate
            if (input_size > tflite_input_size)
            {
                LOG(WARNING, "Truncating input from %zu to %zu elements", input_size, tflite_input_size);
                input_size = tflite_input_size;
            }
            else
            {
                // If graph tensor is smaller, we can't proceed
                return STATUS(EINVAL, "Input tensor too small");
            }
        }

        // Copy data (using the possibly adjusted input_size)
        std::memcpy(input_tensor->data.f, input_data, input_size * sizeof(float));

        // Debug: Print first few values
        LOG(DEBUG, "Input data (first 10): %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", input_data[0],
            input_data[1], input_data[2], input_data[3], input_data[4], input_data[5], input_data[6], input_data[7],
            input_data[8], input_data[9]);
    }
    else
    {
        LOG(ERROR, "Unsupported input tensor type: %d", input_tensor->type);
        return STATUS(EINVAL, "Unsupported input tensor type");
    }

    // Perform inference
    TfLiteStatus invoke_status = _impl->interpreter->Invoke();
    if (invoke_status != kTfLiteOk)
    {
        LOG(ERROR, "TFLite inference failed with status: %d", invoke_status);
        return STATUS(EFAULT, "TFLite inference failed");
    }

    // Get output data and handle dequantization
    float *output_data = output_graph_tensor->dataAs<float>();
    if (!output_data)
    {
        LOG(ERROR, "Failed to get output data from graph tensor");
        return STATUS(EFAULT, "Output data access failed");
    }

    if (output_tensor->type == kTfLiteInt8)
    {
        // Handle quantized output - dequantize to float
        float output_scale = output_tensor->params.scale;
        int32_t output_zero_point = output_tensor->params.zero_point;
        size_t num_output_elements = output_tensor->bytes;

        for (size_t i = 0; i < num_output_elements; ++i)
        {
            int8_t quantized_val = output_tensor->data.int8[i];
            float float_score = static_cast<float>(quantized_val - output_zero_point) * output_scale;
            output_data[i] = float_score;
        }
    }
    else if (output_tensor->type == kTfLiteFloat32)
    {
        // Handle float32 output
        size_t output_size = output_graph_tensor->shape().dot();
        std::memcpy(output_data, output_tensor->data.f, output_size * sizeof(float));
    }
    else
    {
        LOG(ERROR, "Unsupported output tensor type: %d", output_tensor->type);
        return STATUS(EINVAL, "Unsupported output tensor type");
    }

    LOG(DEBUG, "TFLite inference completed successfully");
    return STATUS_OK();
}

} // namespace hal
