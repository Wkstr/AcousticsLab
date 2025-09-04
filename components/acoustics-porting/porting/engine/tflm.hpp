#pragma once
#if defined(PORTING_LIB_TFLM_ENABLE) && PORTING_LIB_TFLM_ENABLE
#ifndef TFLM_HPP
#define TFLM_HPP

#include "core/logger.hpp"
#include "core/status.hpp"
#include "hal/engine.hpp"

#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <esp_heap_caps.h>
#include <esp_partition.h>
#include <spi_flash_mmap.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <variant>

namespace porting {

#define TFLM_OPS_REQUIRED(X) X(Add) X(Conv2D) X(DepthwiseConv2D) X(MaxPool2D) X(Softmax) X(Reshape) X(Relu) X(Relu6)

#define TFLM_OPS_COUNT_HELPER(OP_NAME) nullptr,
#define TFLM_OPS_COUNT(OPS_REQUIRED)   (std::initializer_list<void *> { OPS_REQUIRED(TFLM_OPS_COUNT_HELPER) }.size())

#define TFLM_OPS_REGISTER_HELPER(OP_NAME) __op_resolver_ptr__->Add##OP_NAME();
#define TFLM_OPS_REGISTER(OPS_REQUIRED, OPS_RESOLVER_PTR)                                                              \
    {                                                                                                                  \
        auto *__op_resolver_ptr__ = OPS_RESOLVER_PTR;                                                                  \
        OPS_REQUIRED(TFLM_OPS_REGISTER_HELPER)                                                                         \
    }

class EngineTFLM final: public hal::Engine
{
public:
    static inline core::ConfigObjectMap DEFAULT_CONFIGS() noexcept
    {
        return { CONFIG_OBJECT_DECL_INTEGER("tensor_arena_size", "Size of tensor arena in bytes", 2048 * 1024,
                     512 * 1024, 4096 * 1024),
            CONFIG_OBJECT_DECL_INTEGER("model_lookup_step", "Size of model lookup step in bytes", 1024 * 1024,
                128 * 1024, 2048 * 1024) };
    }

    EngineTFLM() noexcept : Engine(Info(1, "TFLite Micro", Type::TFLiteMicro, { DEFAULT_CONFIGS() })) { }

    ~EngineTFLM() override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        while (!internalDeInit()) [[unlikely]]
        {
            LOG(WARNING, "Failed to deinitialize TFLite Micro engine, retrying...");
            vTaskDelay(pdMS_TO_TICKS(100));
        }

        if (_mmap_handle) [[likely]]
        {
            _mmap_ptr = nullptr;
            spi_flash_munmap(_mmap_handle);
            _mmap_handle = 0;
        }

        if (_op_resolver) [[likely]]
        {
            _op_resolver.reset();
        }

        if (_tensor_arena) [[likely]]
        {
            heap_caps_free(_tensor_arena);
            _tensor_arena = nullptr;
        }
    }

    core::Status init() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status != Status::Uninitialized)
        {
            return STATUS(ENXIO, "Engine is already initialized or in an invalid state");
        }

        {
            if (!_tensor_arena) [[likely]]
            {
                const size_t align = 64;
                const size_t size = _info.configs["tensor_arena_size"].getValue<int>() + (align - 1);

                _tensor_arena = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
                if (!_tensor_arena) [[unlikely]]
                {
                    LOG(ERROR, "Failed to allocate tensor arena of size %zu bytes", size);
                    return STATUS(ENOMEM, "Failed to allocate tensor arena");
                }
                LOG(INFO, "Allocated tensor arena of size %zu bytes at %p", size, _tensor_arena);
                _tensor_arena_aligned = reinterpret_cast<void *>(
                    (reinterpret_cast<uintptr_t>(_tensor_arena) + (align - 1)) & ~(align - 1));
                _tensor_arena_aligned_size = size
                                             - (reinterpret_cast<uintptr_t>(_tensor_arena_aligned)
                                                 - reinterpret_cast<uintptr_t>(_tensor_arena));
                LOG(INFO, "Aligned tensor arena of size %zu bytes at %p", _tensor_arena_aligned_size,
                    _tensor_arena_aligned);
            }
            std::memset(_tensor_arena, 0, _tensor_arena_aligned_size);
        }

        {
            if (!_op_resolver) [[likely]]
            {
                _op_resolver = std::make_unique<tflite::MicroMutableOpResolver<_ops_count>>();
                if (!_op_resolver) [[unlikely]]
                {
                    LOG(ERROR, "Failed to create MicroMutableOpResolver");
                    return STATUS(ENOMEM, "Failed to create MicroMutableOpResolver");
                }

                TFLM_OPS_REGISTER(TFLM_OPS_REQUIRED, _op_resolver.get());
            }
        }

        {
            auto status = findModels();
            if (!status) [[unlikely]]
            {
                LOG(ERROR, "Failed to find models: %s", status.message().c_str());
                return status;
            }
        }

        _info.status = Status::Idle;

        return STATUS_OK();
    }

    core::Status deinit() noexcept override
    {
        const std::lock_guard<std::mutex> lock(_lock);

        return internalDeInit();
    }

    core::Status updateConfig(const core::ConfigMap &configs) noexcept override
    {
        return STATUS(ENOTSUP, "Update config is not supported for UART transport");
    }

    virtual core::Status loadModel(const std::shared_ptr<core::Model::Info> &info,
        std::shared_ptr<core::Model> &model) noexcept override
    {
        if (!info) [[unlikely]]
        {
            LOG(ERROR, "Model info is null");
            return STATUS(EINVAL, "Model info is null");
        }

        const std::lock_guard<std::mutex> lock(_lock);

        if (_info.status < Status::Idle) [[unlikely]]
        {
            LOG(ERROR, "Engine is not initialized or in an invalid state");
            return STATUS(ENXIO, "Engine is not initialized or in an invalid state");
        }

        if (_loaded_model_info && *_loaded_model_info == *info)
        {
            model = _loaded_model;
            return STATUS_OK();
        }

        if (_interpreter) [[unlikely]]
        {
            LOG(ERROR, "This engine does not support loading multiple models at once");
            return STATUS(ENOTSUP, "This engine does not support loading multiple models at once");
        }

        if (info->type != core::Model::Type::TFLite) [[unlikely]]
        {
            LOG(ERROR, "Unsupported model type: %d", static_cast<int>(info->type));
            return STATUS(ENOTSUP, "Unsupported model type");
        }

        if (!std::holds_alternative<const void *>(info->location)) [[unlikely]]
        {
            LOG(ERROR, "This engine only supports models loaded from memory");
            return STATUS(ENOTSUP, "This engine only supports models loaded from memory");
        }

        const void *model_data = std::get<const void *>(info->location);
        if (!model_data) [[unlikely]]
        {
            LOG(ERROR, "Model data is null");
            return STATUS(EFAULT, "Model data is null");
        }

        auto *model_ptr = tflite::GetModel(model_data);
        if (!model_ptr) [[unlikely]]
        {
            LOG(ERROR, "Failed to get TFLite model from data");
            return STATUS(EFAULT, "Failed to get TFLite model from data");
        }
        if (model_ptr->version() != TFLITE_SCHEMA_VERSION) [[unlikely]]
        {
            LOG(ERROR, "TFLite model version mismatch: expected %d, got %ld", TFLITE_SCHEMA_VERSION,
                model_ptr->version());
            return STATUS(ENOTSUP, "TFLite model version mismatch");
        }

        _interpreter = std::make_unique<tflite::MicroInterpreter>(model_ptr, *_op_resolver,
            static_cast<uint8_t *>(_tensor_arena_aligned), _tensor_arena_aligned_size, nullptr, nullptr, false);
        if (!_interpreter) [[unlikely]]
        {
            LOG(ERROR, "Failed to create MicroInterpreter");
            return STATUS(EFAULT, "Failed to create MicroInterpreter");
        }

        auto status = _interpreter->AllocateTensors();
        if (status != kTfLiteOk) [[unlikely]]
        {
            LOG(ERROR, "Failed to allocate tensors: %d", static_cast<int>(status));
            _interpreter.reset();
            return STATUS(EFAULT, "Failed to allocate tensors");
        }

        auto build_status = buildModel(info, model);
        if (!build_status) [[unlikely]]
        {
            LOG(ERROR, "Failed to build model: %s", build_status.message().c_str());
            _interpreter.reset();
        }

        return build_status;
    }

private:
    core::Status internalDeInit() noexcept
    {
        if (_info.status <= Status::Uninitialized)
        {
            return STATUS_OK();
        }

        _loaded_model.reset();
        _loaded_model_info.reset();

        for (const auto &model_info: _model_infos)
        {
            const auto use_count = model_info.use_count();
            if (use_count > 1) [[unlikely]]
            {
                LOG(WARNING, "Model info '%s' is still in use by %ld references, skipping deinitialization",
                    model_info->name.c_str(), use_count);
                return STATUS(EBUSY, "Model is still in use");
            }
        }
        _model_infos.clear();

        _interpreter.reset();

        _info.status = Status::Uninitialized;

        return STATUS_OK();
    }

    core::Status findModels() noexcept
    {
        if (!_partition)
        {
            _partition
                = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_UNDEFINED, "models");
            if (!_partition) [[unlikely]]
            {
                LOG(ERROR, "Failed to find models partition");
                return STATUS(ENOENT, "Models partition not found");
            }
            if (_partition->size < 1024) [[unlikely]]
            {
                LOG(ERROR, "Models partition size is too small: %ld bytes", _partition->size);
                return STATUS(EINVAL, "Models partition size is too small");
            }
        }

        if (!_mmap_handle)
        {
            esp_err_t ret
                = spi_flash_mmap(_partition->address, _partition->size, SPI_FLASH_MMAP_DATA, &_mmap_ptr, &_mmap_handle);
            if (ret != ESP_OK) [[unlikely]]
            {
                LOG(ERROR, "Failed to map models partition: %s", esp_err_to_name(ret));
                return STATUS(EIO, "Failed to map models partition");
            }
            if (_mmap_ptr == nullptr) [[unlikely]]
            {
                LOG(ERROR, "Mapped pointer is null");
                return STATUS(EFAULT, "Mapped pointer is null");
            }
        }

        const size_t partition_size = _partition->size;
        const size_t lookup_step = _info.configs["model_lookup_step"].getValue<int>();

        for (size_t i = 0, id = 0; i < partition_size; i += lookup_step)
        {
            const void *data = static_cast<const uint8_t *>(_mmap_ptr) + i;
            if (__builtin_bswap32(*(static_cast<const uint32_t *>(data) + 1)) != 0x54464C33)
            {
                continue;
            }

            LOG(INFO, "Found TFLite model at offset %zu, address %p", i, data);

            _model_infos.emplace_front(
                std::make_shared<core::Model::Info>(static_cast<int>(++id), std::string("Model_") + std::to_string(i),
                    core::Model::Type::TFLite, "TFL3", std::unordered_map<int, std::string> {}, data));
        }

        return STATUS_OK();
    }

    core::Status buildModel(const std::shared_ptr<core::Model::Info> &info,
        std::shared_ptr<core::Model> &model) noexcept
    {
        const auto &inputs_size = _interpreter->inputs_size();
        const auto &outputs_size = _interpreter->outputs_size();
        if (inputs_size < 1 || outputs_size < 1) [[unlikely]]
        {
            LOG(ERROR, "Model has no inputs or outputs");
            return STATUS(EINVAL, "Model has no inputs or outputs");
        }

        std::vector<std::unique_ptr<core::Tensor>> input_tensors(inputs_size);
        std::vector<std::unique_ptr<core::Tensor>> output_tensors(outputs_size);
        std::vector<core::Tensor::QuantParams> input_quant_params(inputs_size);
        std::vector<core::Tensor::QuantParams> output_quant_params(outputs_size);

        for (size_t i = 0; i < input_tensors.size(); ++i)
        {
            auto *input = _interpreter->input(i);
            if (!input) [[unlikely]]
            {
                LOG(ERROR, "Failed to get input tensor %zu", i);
                return STATUS(EFAULT, "Failed to get input tensor");
            }

            core::Tensor::Type dtype = core::Tensor::Type::Unknown;
            switch (input->type)
            {
                case kTfLiteFloat32:
                    dtype = core::Tensor::Type::Float32;
                    break;
                case kTfLiteUInt8:
                    dtype = core::Tensor::Type::UInt8;
                    break;
                case kTfLiteInt8:
                    dtype = core::Tensor::Type::Int8;
                    break;
                default:
                    LOG(ERROR, "Unsupported input tensor type %d", input->type);
                    return STATUS(EINVAL, "Unsupported input tensor type");
            }

            auto *dims = input->dims;
            if (!dims) [[unlikely]]
            {
                LOG(ERROR, "Input tensor %zu has no dimensions", i);
                return STATUS(EFAULT, "Input tensor has no dimensions");
            }
            std::vector<int> shape(dims->size);
            for (size_t j = 0; j < shape.size(); ++j)
            {
                shape[j] = dims->data[j];
            }

            auto tensor = core::Tensor::create(dtype, core::Tensor::Shape(std::move(shape)),
                std::shared_ptr<std::byte[]>(reinterpret_cast<std::byte *>(input->data.data), [](std::byte *) { }),
                input->bytes);
            if (!tensor) [[unlikely]]
            {
                LOG(ERROR, "Failed to create input tensor %zu", i);
                return STATUS(EFAULT, "Failed to create input tensor");
            }

            input_tensors[i] = std::move(tensor);
            input_quant_params[i] = input->quantization.type == kTfLiteNoQuantization
                                        ? core::Tensor::QuantParams()
                                        : core::Tensor::QuantParams(input->params.scale, input->params.zero_point);
        }

        for (size_t i = 0; i < output_tensors.size(); ++i)
        {
            auto *output = _interpreter->output(i);
            if (!output) [[unlikely]]
            {
                LOG(ERROR, "Failed to get output tensor %zu", i);
                return STATUS(EFAULT, "Failed to get output tensor");
            }

            core::Tensor::Type dtype = core::Tensor::Type::Unknown;
            switch (output->type)
            {
                case kTfLiteFloat32:
                    dtype = core::Tensor::Type::Float32;
                    break;
                case kTfLiteUInt8:
                    dtype = core::Tensor::Type::UInt8;
                    break;
                case kTfLiteInt8:
                    dtype = core::Tensor::Type::Int8;
                    break;
                default:
                    LOG(ERROR, "Unsupported output tensor type %d", output->type);
                    return STATUS(EINVAL, "Unsupported output tensor type");
            }

            auto *dims = output->dims;
            if (!dims) [[unlikely]]
            {
                LOG(ERROR, "Output tensor %zu has no dimensions", i);
                return STATUS(EFAULT, "Output tensor has no dimensions");
            }
            std::vector<int> shape(dims->size);
            for (size_t j = 0; j < shape.size(); ++j)
            {
                shape[j] = dims->data[j];
            }

            auto tensor = core::Tensor::create(dtype, core::Tensor::Shape(std::move(shape)),
                std::shared_ptr<std::byte[]>(reinterpret_cast<std::byte *>(output->data.data), [](std::byte *) { }),
                output->bytes);
            if (!tensor) [[unlikely]]
            {
                LOG(ERROR, "Failed to create output tensor %zu", i);
                return STATUS(EFAULT, "Failed to create output tensor");
            }

            output_tensors[i] = std::move(tensor);
            output_quant_params[i] = output->quantization.type == kTfLiteNoQuantization
                                         ? core::Tensor::QuantParams()
                                         : core::Tensor::QuantParams(output->params.scale, output->params.zero_point);
        }

        auto graph = core::Model::Graph::create(0, "", std::move(input_tensors), std::move(output_tensors),
            std::move(input_quant_params), std::move(output_quant_params), [this](core::Model::Graph &graph) noexcept {
                const std::lock_guard<std::mutex> lock(this->_lock);

                if (!this->_interpreter) [[unlikely]]
                {
                    LOG(ERROR, "Interpreter is not initialized");
                    return STATUS(EFAULT, "Interpreter is not initialized");
                }

                auto status = this->_interpreter->Invoke();
                if (status != kTfLiteOk) [[unlikely]]
                {
                    LOG(ERROR, "Failed to invoke interpreter: %d", static_cast<int>(status));
                    return STATUS(EFAULT, "Failed to invoke interpreter");
                }

                return STATUS_OK();
            });
        _loaded_model = std::make_shared<core::Model>(info, std::move(graph));
        if (!_loaded_model) [[unlikely]]
        {
            LOG(ERROR, "Failed to create loaded model");
            return STATUS(EFAULT, "Failed to create loaded model");
        }
        _loaded_model_info = info;
        model = _loaded_model;

        return STATUS_OK();
    }

private:
    mutable std::mutex _lock;
    void *_tensor_arena = nullptr;
    void *_tensor_arena_aligned = nullptr;
    size_t _tensor_arena_aligned_size = 0;

    const esp_partition_t *_partition = nullptr;
    spi_flash_mmap_handle_t _mmap_handle = 0;
    const void *_mmap_ptr = nullptr;

    static constexpr inline const size_t _ops_count = TFLM_OPS_COUNT(TFLM_OPS_REQUIRED);
    std::unique_ptr<tflite::MicroMutableOpResolver<_ops_count>> _op_resolver;
    std::unique_ptr<tflite::MicroInterpreter> _interpreter;

    std::shared_ptr<core::Model> _loaded_model;
    std::shared_ptr<core::Model::Info> _loaded_model_info;
};

} // namespace porting

#endif // TFLM_HPP

#endif // PORTING_LIB_TFLM_ENABLE
