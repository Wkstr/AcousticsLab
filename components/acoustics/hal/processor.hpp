#pragma once
#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include "core/config_object.hpp"
#include "core/data_frame.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace hal {

class Processor;

class ProcessorRegistry
{
public:
    using ProcessorMap = std::unordered_map<int, Processor *>;

    ProcessorRegistry() = default;
    ~ProcessorRegistry() = default;

    inline static Processor *getProcessor(int id) noexcept
    {
        auto it = _processors.find(id);
        if (it != _processors.end()) [[likely]]
        {
            return it->second;
        }
        return nullptr;
    }

    static const ProcessorMap &getProcessorMap() noexcept
    {
        return _processors;
    }

protected:
    friend class Processor;

    static core::Status registerProcessor(Processor *processor) noexcept;

private:
    static ProcessorMap _processors;
};

class Processor
{
public:
    enum class Type {
        Unknown = 0,
        Custom,
        FeatureExtractor,
        AudioPreprocessor,
        SignalFilter,
        Transformer,
    };

    enum class Status : size_t {
        Unknown = 0,
        Uninitialized,
        Idle,
        Processing,
    };

    struct Info final
    {
        Info(int id, std::string_view name, Type type, core::ConfigObjectMap &&configs) noexcept
            : id(id), name(name), type(type), status(Status::Unknown), configs(std::move(configs))
        {
        }

        ~Info() = default;

        const int id;
        const std::string_view name;
        const Type type;
        volatile Status status;
        core::ConfigObjectMap configs;
    };

    virtual ~Processor() = default;

    virtual core::Status init() = 0;
    virtual core::Status deinit() = 0;

    inline bool initialized() const noexcept
    {
        return _info.status >= Status::Idle;
    }

    const Info &info() const noexcept
    {
        syncInfo(_info);
        return _info;
    }

    virtual core::Status updateConfig(const core::ConfigMap &configs) = 0;

    /**
     * @brief Process input tensor and produce output tensor
     * 
     * @param input Input tensor containing data to process
     * @param output Output tensor to store processed data
     * @return core::Status indicating success or failure
     */
    virtual core::Status process(const core::Tensor &input, core::Tensor &output) = 0;

    /**
     * @brief Process input data frame and produce output data frame
     * 
     * @param input_frame Input data frame
     * @param output_frame Output data frame
     * @return core::Status indicating success or failure
     */
    virtual core::Status processDataFrame(const core::DataFrame<core::Tensor> &input_frame, 
                                        core::DataFrame<core::Tensor> &output_frame) = 0;

    /**
     * @brief Get expected input tensor shape and type
     * 
     * @return Expected input tensor configuration
     */
    virtual std::pair<core::Tensor::Shape, core::Tensor::Type> getInputSpec() const = 0;

    /**
     * @brief Get expected output tensor shape and type
     * 
     * @return Expected output tensor configuration
     */
    virtual std::pair<core::Tensor::Shape, core::Tensor::Type> getOutputSpec() const = 0;

protected:
    Processor(Info &&info) noexcept : _info(std::move(info))
    {
        LOG(DEBUG, "Registering processor: ID=%d, Name='%s', Type=%d", _info.id, _info.name.data(),
            static_cast<int>(_info.type));
        [[maybe_unused]] auto status = ProcessorRegistry::registerProcessor(this);
        if (!status) [[unlikely]]
        {
            LOG(ERROR, "Failed to register processor: %s", status.message().c_str());
        }
        _info.status = Status::Uninitialized;
        LOG(INFO, "Processor '%s' registered successfully", _info.name.data());
    }

    virtual void syncInfo(Info &info) const noexcept { }

    mutable Info _info;
    mutable std::mutex _lock;
};

} // namespace hal

namespace bridge {

extern void __REGISTER_PROCESSORS__();

} // namespace bridge

#endif // PROCESSOR_HPP
