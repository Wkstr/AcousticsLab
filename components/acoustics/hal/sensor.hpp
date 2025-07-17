#pragma once
#ifndef SENSOR_HPP
#define SENSOR_HPP

#include "core/config_object.hpp"
#include "core/data_frame.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>

namespace hal {

class Sensor;

class SensorRegistry
{
public:
    using SensorMap = std::unordered_map<int, Sensor *>;

    SensorRegistry() = default;
    ~SensorRegistry() = default;

    inline static Sensor *getSensor(int id) noexcept
    {
        auto it = _sensors.find(id);
        if (it != _sensors.end())
        {
            return it->second;
        }
        return nullptr;
    }

    static const SensorMap &getSensorMap() noexcept
    {
        return _sensors;
    }

protected:
    friend class Sensor;

    static core::Status registerSensor(Sensor *sensor) noexcept;

private:
    static SensorMap _sensors;
};

class Sensor
{
public:
    enum class Type {
        Unknown = 0,
        Accelerometer,
        Gyroscope,
        Magnetometer,
        TemperatureSensor,
        PressureSensor,
        HumiditySensor,
        LightSensor,
        ProximitySensor,
        Microphone,
        Camera,
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

    virtual ~Sensor() = default;

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

    virtual inline size_t dataAvailable() const noexcept = 0;
    virtual inline size_t dataClear() noexcept = 0;

    virtual inline core::Status readDataFrame(core::DataFrame<std::shared_ptr<core::Tensor>> &data_frame,
        size_t batch_size)
        = 0;

protected:
    Sensor(Info &&info) noexcept : _info(std::move(info))
    {
        LOG(DEBUG, "Registering sensor: ID=%d, Name='%s', Type=%d", _info.id, _info.name.data(),
            static_cast<int>(_info.type));
        [[maybe_unused]] auto status = SensorRegistry::registerSensor(this);
        if (!status)
        {
            LOG(ERROR, "Failed to register sensor: %s", status.message().c_str());
        }
        _info.status = Status::Uninitialized;
        LOG(DEBUG, "Sensor '%s' registered successfully", _info.name.data());
    }

    virtual void syncInfo(Info &info) const noexcept { }

    mutable Info _info;
};

} // namespace hal

namespace bridge {

extern void __REGISTER_SENSORS__();

} // namespace bridge

#endif // SENSOR_H
