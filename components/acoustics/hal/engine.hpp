#pragma once
#ifndef ENGINE_HPP
#define ENGINE_HPP

#include "core/config_object.hpp"
#include "core/data_frame.hpp"
#include "core/logger.hpp"
#include "core/model.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace hal {

class Engine;

class EngineRegistry final
{
public:
    using EngineMap = std::unordered_map<int, Engine *>;

    EngineRegistry() = default;
    ~EngineRegistry() = default;

    inline static Engine *getEngine(int id) noexcept
    {
        auto it = _engines.find(id);
        if (it != _engines.end()) [[likely]]
        {
            return it->second;
        }
        return nullptr;
    }

    static const EngineMap &getEngineMap() noexcept
    {
        return _engines;
    }

protected:
    friend class Engine;

    static core::Status registerEngine(Engine *engine) noexcept;

private:
    static EngineMap _engines;
};

class Engine
{
public:
    enum class Type {
        Unknown = 0,
        Custom,
        TFLiteMicro,
        LiteRT,
        LibTorch,
        ONNXRuntime,
        OpenVINO,
        TensorRT,
        HailoRT,
    };

    enum class Status : size_t {
        Unknown = 0,
        Uninitialized,
        Idle,
        Locked,
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

    virtual ~Engine() = default;

    virtual core::Status init() noexcept = 0;
    virtual core::Status deinit() noexcept = 0;

    inline bool initialized() const noexcept
    {
        return _info.status >= Status::Idle;
    }

    const Info &info() const noexcept
    {
        syncInfo(_info);
        return _info;
    }

    virtual core::Status updateConfig(const core::ConfigMap &configs) noexcept = 0;

    const std::forward_list<std::shared_ptr<core::Model::Info>> &modelInfos() const noexcept
    {
        return _model_infos;
    }

    template<typename T, std::enable_if<std::is_invocable_r_v<bool, T, const core::Model::Info &>, bool> = true>
    std::shared_ptr<core::Model::Info> modelInfo(T &&pred) const noexcept
    {
        auto it = std::find_if(_model_infos.begin(), _model_infos.end(),
            [pred = std::forward<T>(pred)](
                const std::shared_ptr<core::Model::Info> &info) mutable { return info ? pred(*info) : false; });
        if (it != _model_infos.end()) [[likely]]
        {
            return *it;
        }
        return {};
    }

    virtual core::Status loadModel(const std::shared_ptr<core::Model::Info> &info,
        std::shared_ptr<core::Model> &model) noexcept
        = 0;

protected:
    Engine(Info &&info) noexcept : _info(std::move(info))
    {
        LOG(DEBUG, "Registering engine: ID=%d, Name='%s', Type=%d", _info.id, _info.name.data(),
            static_cast<int>(_info.type));
        [[maybe_unused]] auto status = EngineRegistry::registerEngine(this);
        if (!status) [[unlikely]]
        {
            LOG(ERROR, "Failed to register engine: %s", status.message().c_str());
        }
        _info.status = Status::Uninitialized;
        LOG(INFO, "Engine '%s' registered successfully", _info.name.data());
    }

    virtual void syncInfo(Info &info) const noexcept { }

    mutable Info _info;

    std::forward_list<std::shared_ptr<core::Model::Info>> _model_infos;
};

} // namespace hal

namespace bridge {

extern void __REGISTER_ENGINES__();

}

#endif
