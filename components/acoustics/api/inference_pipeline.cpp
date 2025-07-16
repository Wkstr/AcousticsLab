#include "inference_pipeline.hpp"
#include "hal/engine.hpp"
#include "hal/engines/tflite_engine.hpp"
#include "hal/processor.hpp"

#include <chrono>

namespace api {

InferencePipeline::InferencePipeline(const InferencePipelineConfig &config)
    : _config(config), _initialized(false), _feature_extractor(nullptr), _inference_engine(nullptr), _frame_counter(0)
{
    LOG(DEBUG, "InferencePipeline created with config: audio_samples=%zu, model='%s'", _config.audio_samples,
        _config.model_name.c_str());
}

InferencePipeline::~InferencePipeline()
{
    deinit();
    LOG(DEBUG, "InferencePipeline destroyed");
}

core::Status InferencePipeline::init()
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (_initialized)
    {
        LOG(WARNING, "InferencePipeline already initialized");
        return STATUS_OK();
    }

    // Initialize feature extractor
    auto status = initFeatureExtractor();
    if (!status)
    {
        LOG(ERROR, "Failed to initialize feature extractor: %s", status.message().c_str());
        return status;
    }

    // Initialize inference engine
    status = initInferenceEngine();
    if (!status)
    {
        LOG(ERROR, "Failed to initialize inference engine: %s", status.message().c_str());
        return status;
    }

    // Create working tensors
    // Get the correct feature size from the feature extractor's output specification
    auto feature_spec = _feature_extractor->getOutputSpec();
    size_t feature_size = feature_spec.first[0]; // Get the first dimension of the output shape

    _feature_tensor = std::unique_ptr<core::Tensor>(new core::Tensor(core::Tensor::Type::Float32, { feature_size }));
    if (!_feature_tensor)
    {
        LOG(ERROR, "Failed to create feature tensor");
        return STATUS(ENOMEM, "Feature tensor allocation failed");
    }

    size_t output_size = _config.labels.size();
    if (output_size == 0)
        output_size = 3; // Default to 3 classes

    _output_tensor = std::unique_ptr<core::Tensor>(new core::Tensor(core::Tensor::Type::Float32, { output_size }));
    if (!_output_tensor)
    {
        LOG(ERROR, "Failed to create output tensor");
        return STATUS(ENOMEM, "Output tensor allocation failed");
    }

    _initialized = true;
    LOG(INFO, "InferencePipeline initialized successfully");
    LOG(INFO, "Feature tensor size: %zu, Output tensor size: %zu", feature_size, output_size);

    return STATUS_OK();
}

core::Status InferencePipeline::deinit()
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_initialized)
    {
        return STATUS_OK();
    }

    // Reset working tensors
    _feature_tensor.reset();
    _output_tensor.reset();

    // Reset model
    _model.reset();

    // Note: We don't own the processor and engine instances, they are managed by registries
    _feature_extractor = nullptr;
    _inference_engine = nullptr;

    _initialized = false;
    LOG(INFO, "InferencePipeline deinitialized");

    return STATUS_OK();
}

core::Status InferencePipeline::updateConfig(const InferencePipelineConfig &config)
{
    const std::lock_guard<std::mutex> lock(_lock);

    _config = config;

    // If already initialized, need to reinitialize with new config
    if (_initialized)
    {
        LOG(INFO, "Reinitializing InferencePipeline with new configuration");
        auto status = deinit();
        if (!status)
        {
            return status;
        }
        return init();
    }

    return STATUS_OK();
}

core::Status InferencePipeline::loadModel(const void *model_data, size_t model_size,
    const std::unordered_map<int, std::string> &labels)
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_initialized)
    {
        LOG(ERROR, "InferencePipeline not initialized");
        return STATUS(ENXIO, "InferencePipeline not initialized");
    }

    if (!_inference_engine)
    {
        LOG(ERROR, "Inference engine not available");
        return STATUS(EFAULT, "Inference engine not available");
    }

    // Update labels in config
    _config.labels = labels;

    // Store model data in engine for actual loading
    if (model_data && model_size > 0)
    {
        // Store model data in engine's internal storage using static_cast
        // We know this is a TFLite engine from the configuration
        auto tflite_engine = static_cast<hal::EngineTFLite *>(_inference_engine);
        auto status = tflite_engine->setModelData(model_data, model_size);
        if (!status)
        {
            LOG(ERROR, "Failed to set model data: %s", status.message().c_str());
            return status;
        }
    }

    // Create model info
    core::Model::Info model_info(core::Model::Type::TFLite, _config.model_name, _config.model_version,
        std::unordered_map<int, std::string>(labels), _config.model_path);

    // Load model through engine
    auto status = _inference_engine->loadModel(model_info, _model);
    if (!status)
    {
        LOG(ERROR, "Failed to load model: %s", status.message().c_str());
        return status;
    }

    LOG(INFO, "Model loaded successfully: %s v%s with %zu classes", _config.model_name.c_str(),
        _config.model_version.c_str(), labels.size());

    return STATUS_OK();
}

core::Status InferencePipeline::infer(const float *audio_data, size_t audio_samples, InferenceResult &result)
{
    const std::lock_guard<std::mutex> lock(_lock);

    if (!_initialized)
    {
        LOG(ERROR, "InferencePipeline not initialized");
        return STATUS(ENXIO, "InferencePipeline not initialized");
    }

    if (!audio_data)
    {
        LOG(ERROR, "Audio data is null");
        return STATUS(EINVAL, "Audio data is null");
    }

    if (audio_samples != _config.audio_samples)
    {
        LOG(ERROR, "Audio samples mismatch: expected %zu, got %zu", _config.audio_samples, audio_samples);
        return STATUS(EINVAL, "Audio samples mismatch");
    }

    auto start_time = std::chrono::steady_clock::now();

    // Step 1: Feature extraction
    auto feature_start = std::chrono::steady_clock::now();

    core::Tensor audio_tensor(core::Tensor::Type::Float32, { audio_samples });
    memcpy(audio_tensor.dataAs<float>(), audio_data, audio_samples * sizeof(float));

    auto status = _feature_extractor->process(audio_tensor, *_feature_tensor);
    if (!status)
    {
        LOG(ERROR, "Feature extraction failed: %s", status.message().c_str());
        return status;
    }

    auto feature_end = std::chrono::steady_clock::now();

    // Step 2: Model inference
    auto inference_start = std::chrono::steady_clock::now();

    if (!_model || _model->graphs().empty())
    {
        LOG(ERROR, "Model not loaded");
        return STATUS(EFAULT, "Model not loaded");
    }

    auto graph = _model->graph(0);
    if (!graph)
    {
        LOG(ERROR, "Model graph not available");
        return STATUS(EFAULT, "Model graph not available");
    }

    // Copy feature data to graph input
    if (graph->inputs() == 0)
    {
        LOG(ERROR, "Model has no inputs");
        return STATUS(EFAULT, "Model has no inputs");
    }

    // Get model input tensor and copy feature data
    auto model_input_tensor = graph->input(0);
    if (!model_input_tensor)
    {
        LOG(ERROR, "Failed to get model input tensor");
        return STATUS(EFAULT, "Failed to get model input tensor");
    }

    // Verify input tensor shape matches feature tensor
    const float *feature_data = _feature_tensor->dataAs<float>();
    size_t feature_size = _feature_tensor->shape().dot();

    LOG(DEBUG, "Feature tensor size: %zu, Model input tensor size: %zu", feature_size,
        model_input_tensor->shape().dot());

    if (model_input_tensor->shape().dot() != feature_size)
    {
        LOG(ERROR, "Model input tensor size mismatch: expected %zu, got %zu", feature_size,
            model_input_tensor->shape().dot());
        return STATUS(EINVAL, "Model input tensor size mismatch");
    }

    // Copy feature data to model input tensor
    float *input_data = model_input_tensor->dataAs<float>();
    if (!input_data || !feature_data)
    {
        LOG(ERROR, "Failed to get tensor data pointers");
        return STATUS(EFAULT, "Invalid tensor data");
    }

    memcpy(input_data, feature_data, feature_size * sizeof(float));

    // Debug: Print first few feature values
    LOG(DEBUG, "Features (first 10): %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", feature_data[0],
        feature_data[1], feature_data[2], feature_data[3], feature_data[4], feature_data[5], feature_data[6],
        feature_data[7], feature_data[8], feature_data[9]);

    // Perform model inference
    status = graph->forward();
    if (!status)
    {
        LOG(ERROR, "Model inference failed: %s", status.message().c_str());
        return status;
    }

    auto inference_end = std::chrono::steady_clock::now();

    // Step 3: Extract results
    if (graph->outputs() == 0)
    {
        LOG(ERROR, "Model has no outputs");
        return STATUS(EFAULT, "Model has no outputs");
    }

    auto *output_tensor = graph->output(0);
    if (!output_tensor)
    {
        LOG(ERROR, "Failed to get output tensor");
        return STATUS(EFAULT, "Failed to get output tensor");
    }

    const float *output_data = output_tensor->dataAs<float>();
    size_t output_size = output_tensor->shape().dot();

    result.scores.resize(output_size);
    memcpy(result.scores.data(), output_data, output_size * sizeof(float));
    result.labels = _config.labels;
    result.timestamp = std::chrono::steady_clock::now();
    result.frame_index = _frame_counter++;

    // Update performance statistics
    auto total_end = std::chrono::steady_clock::now();
    double feature_ms = std::chrono::duration<double, std::milli>(feature_end - feature_start).count();
    double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - start_time).count();

    _perf_stats.total_inferences++;
    _perf_stats.avg_feature_extraction_ms
        = (_perf_stats.avg_feature_extraction_ms * (_perf_stats.total_inferences - 1) + feature_ms)
          / _perf_stats.total_inferences;
    _perf_stats.avg_inference_ms = (_perf_stats.avg_inference_ms * (_perf_stats.total_inferences - 1) + inference_ms)
                                   / _perf_stats.total_inferences;
    _perf_stats.avg_total_ms
        = (_perf_stats.avg_total_ms * (_perf_stats.total_inferences - 1) + total_ms) / _perf_stats.total_inferences;

    LOG(DEBUG, "Inference completed: feature=%.2fms, inference=%.2fms, total=%.2fms", feature_ms, inference_ms,
        total_ms);

    return STATUS_OK();
}

core::Status InferencePipeline::inferDataFrame(const core::DataFrame<core::Tensor> &input_frame,
    InferenceResult &result)
{
    const float *audio_data = nullptr;
    size_t audio_samples = input_frame.data.shape().dot();
    std::unique_ptr<float[]> converted_data;

    // Handle different input data types
    if (input_frame.data.dtype() == core::Tensor::Type::Float32)
    {
        // Direct use of Float32 data
        audio_data = input_frame.data.dataAs<float>();
    }
    else if (input_frame.data.dtype() == core::Tensor::Type::Int16)
    {
        // Convert Int16 to Float32
        const int16_t *int16_data = input_frame.data.dataAs<int16_t>();
        converted_data = std::make_unique<float[]>(audio_samples);

        for (size_t i = 0; i < audio_samples; ++i)
        {
            // Normalize Int16 to Float32 range [-1.0, 1.0] (like main repository)
            converted_data[i] = static_cast<float>(int16_data[i]) / 32768.0f;
        }
        audio_data = converted_data.get();

        LOG(DEBUG, "Converted %zu Int16 samples to Float32", audio_samples);
    }
    else
    {
        LOG(ERROR, "Unsupported input tensor type: %d", static_cast<int>(input_frame.data.dtype()));
        return STATUS(EINVAL, "Unsupported input tensor type");
    }

    auto status = infer(audio_data, audio_samples, result);
    if (!status)
    {
        return status;
    }

    // Copy timestamp from input frame if available
    result.timestamp = input_frame.timestamp;

    return STATUS_OK();
}

InferencePipeline::PerformanceStats InferencePipeline::getPerformanceStats() const
{
    const std::lock_guard<std::mutex> lock(_lock);
    return _perf_stats;
}

core::Status InferencePipeline::initFeatureExtractor()
{
    // Get feature extractor from processor registry
    _feature_extractor = hal::ProcessorRegistry::getProcessor(1); // ID 1 for feature extractor
    if (!_feature_extractor)
    {
        LOG(ERROR, "Feature extractor processor not found in registry");
        return STATUS(ENODEV, "Feature extractor not found");
    }

    // Initialize if not already initialized
    if (!_feature_extractor->initialized())
    {
        auto config = createProcessorConfig();
        auto status = _feature_extractor->updateConfig(config);
        if (!status)
        {
            LOG(ERROR, "Failed to configure feature extractor: %s", status.message().c_str());
            return status;
        }

        status = _feature_extractor->init();
        if (!status)
        {
            LOG(ERROR, "Failed to initialize feature extractor: %s", status.message().c_str());
            return status;
        }
    }

    LOG(DEBUG, "Feature extractor initialized");
    return STATUS_OK();
}

core::Status InferencePipeline::initInferenceEngine()
{
    // Get TFLite engine from engine registry
    _inference_engine = hal::EngineRegistry::getEngine(1); // ID 1 for TFLite engine
    if (!_inference_engine)
    {
        LOG(ERROR, "TFLite inference engine not found in registry");
        return STATUS(ENODEV, "TFLite engine not found");
    }

    // Initialize if not already initialized
    if (!_inference_engine->initialized())
    {
        auto config = createModelConfig();
        auto status = _inference_engine->updateConfig(config);
        if (!status)
        {
            LOG(ERROR, "Failed to configure inference engine: %s", status.message().c_str());
            return status;
        }

        status = _inference_engine->init();
        if (!status)
        {
            LOG(ERROR, "Failed to initialize inference engine: %s", status.message().c_str());
            return status;
        }
    }

    LOG(DEBUG, "Inference engine initialized");
    return STATUS_OK();
}

core::ConfigMap InferencePipeline::createModelConfig() const
{
    core::ConfigMap config;
    config["tensor_arena_size"] = static_cast<int>(_config.tensor_arena_size);
    config["use_psram"] = _config.use_psram;
    config["enable_profiling"] = _config.enable_profiling;
    return config;
}

core::ConfigMap InferencePipeline::createProcessorConfig() const
{
    core::ConfigMap config;
    config["audio_samples"] = static_cast<int>(_config.audio_samples);
    config["frame_length"] = static_cast<int>(_config.frame_length);
    config["hop_length"] = static_cast<int>(_config.hop_length);
    config["fft_size"] = static_cast<int>(_config.fft_size);
    config["features_per_frame"] = static_cast<int>(_config.features_per_frame);
    config["use_psram"] = _config.use_psram;
    config["normalize_features"] = _config.normalize_features;
    return config;
}

} // namespace api
