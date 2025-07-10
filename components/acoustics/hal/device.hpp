#pragma once
#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "core/config_object.hpp"
#include "core/data_frame.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "core/tensor.hpp"

#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace hal {

class Device;

class DeviceRegistry
{
public:
    DeviceRegistry() = default;
    ~DeviceRegistry() = default;

    inline static Device *getDevice() noexcept
    {
        return _device;
    }

protected:
    friend class Device;

    static core::Status registerDevice(Device *device) noexcept;

    static Device *_device;
};

class Device
{
public:
    enum class Status : size_t {
        Unknown = 0,
        Uninitialized,
        Ready,
    };

    struct Info final
    {
        Info(std::string_view id, std::string_view model, std::string_view version, size_t total_memory,
            std::string name = "", size_t boot_count = 0,
            std::chrono::system_clock::time_point last_boot_time = std::chrono::system_clock::now(),
            size_t free_memory = 0) noexcept
            : id(id), model(model), version(version), total_memory(total_memory), name(name), boot_count(boot_count),
              last_boot_time(last_boot_time), free_memory(free_memory), status(Status::Unknown)
        {
            if (id.empty() || model.empty() || version.empty())
            {
                LOG(ERROR, "Device ID, model, and version cannot be empty");
            }
            if (total_memory == 0)
            {
                LOG(ERROR, "Total memory must be greater than zero");
            }
        }

        ~Info() = default;

        const std::string_view id;
        const std::string_view model;
        const std::string_view version;
        const size_t total_memory;

        std::string name;
        size_t boot_count;
        std::chrono::system_clock::time_point last_boot_time;
        size_t free_memory;
        volatile Status status;
    };

    virtual ~Device() = default;

    virtual core::Status init() = 0;
    virtual core::Status deinit() = 0;

    inline bool initialized() const noexcept
    {
        return _info.status >= Status::Ready;
    }

    virtual inline uint32_t timestamp() const noexcept = 0;
    virtual inline void sleep(size_t duration_ms) const noexcept = 0;
    virtual inline void reset() const noexcept = 0;

    inline const Info &info() noexcept
    {
        syncInfo(_info);
        return _info;
    }

    core::Status updateDeviceName(std::string new_name) noexcept
    {
        if (!syncDeviceName(new_name))
        {
            return STATUS(EINVAL, "Failed to sync device name");
        }
        _info.name = std::move(new_name);
        LOG(INFO, "Device name updated to: '%s'", _info.name.data());
        return STATUS_OK();
    }

    enum class GPIOOpType {
        Config,
        Write,
        Read,
    };

    virtual inline int gpio(GPIOOpType op, int pin, int value = 0) noexcept = 0;

    enum class StorageType {
        Internal,
        External,
    };

    virtual core::Status store(StorageType where, std::string path, const void *data, size_t size) = 0;
    virtual core::Status load(StorageType where, std::string path, void *data, size_t &size) = 0;
    virtual core::Status exists(StorageType where, std::string path) = 0;
    virtual core::Status remove(StorageType where, std::string path) = 0;
    virtual core::Status erase(StorageType where) = 0;

protected:
    Device(Info &&info) noexcept : _info(std::move(info))
    {
        LOG(DEBUG, "Registering device: ID='%s', Model='%s', Version='%s'", _info.id.data(), _info.model.data(),
            _info.version.data());
        [[maybe_unused]] auto status = DeviceRegistry::registerDevice(this);
        if (!status)
        {
            LOG(ERROR, "Failed to register device: %s", status.message().c_str());
        }
        _info.status = Status::Uninitialized;
        LOG(DEBUG, "Device '%s' registered successfully", _info.id.data());
    }

    virtual void syncInfo(Info &info) noexcept { }
    virtual bool syncDeviceName(const std::string &new_name) noexcept
    {
        return true;
    }

    mutable Info _info;
};

} // namespace hal

namespace bridge {

extern void __REGISTER_DEVICE__();

} // namespace bridge

#endif // DEVICE_HPP
