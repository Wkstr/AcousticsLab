#pragma once
#ifndef CMD_START_HPP
#define CMD_START_HPP

#include "common.hpp"
#include "task_gedad.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/version.hpp"

#include "hal/device.hpp"
#include "hal/sensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace v0 {

class CmdStart final: public api::Command
{
    static inline constexpr const char *invoke_on_start_path = CONTEXT_PREFIX CONTEXT_VERSION "/invoke_on_start";

public:
    static inline void preInitHook()
    {
        auto device = hal::DeviceRegistry::getDevice();
        if (!device || !device->initialized()) [[unlikely]]
        {
            LOG(ERROR, "Device not registered or not initialized");
            return;
        }

        bool invoke_on_start = false;
        size_t size = sizeof(invoke_on_start);
        auto status = device->load(hal::Device::StorageType::Internal, invoke_on_start_path, &invoke_on_start, size);
        if (status && size == sizeof(invoke_on_start))
        {
            CmdStart::_invoke_on_start = invoke_on_start;
        }
    }

    CmdStart()
        : Command("START", "Get device information and start the sample/invoke stream",
              core::ConfigObjectMap { CONFIG_OBJECT_DECL_STRING("action",
                  "Choose streaming action: sample, invoke, stop_invoke, enable or disable", "") })
    {
        _rc = &rcCallback;
    }

    ~CmdStart() noexcept override = default;

    class Task final: public api::Task
    {
    public:
        Task(api::Context &context, hal::Transport &transport, size_t id, hal::Sensor *sensor, std::string tag,
            const volatile size_t &external_task_id) noexcept
            : api::Task(context, transport, id, v0::defaults::task_priority), _sensor(sensor), _tag(tag),
              _external_task_id(external_task_id), _internal_task_id(_external_task_id), _report_interval_ms(100),
              _start_time(std::chrono::steady_clock::now()), _last_report_time(_start_time), _current_id(0)
        {
            v0::shared::is_sampling = true;
            if (sensor) [[likely]]
            {
                sensor->dataClear();
            }
            {
                std::lock_guard<std::mutex> lock(shared::buffer_mutex);
                v0::shared::buffer_head = 0;
                v0::shared::buffer_tail = 0;
                v0::shared::buffer_base_ts = _start_time;
                _current_id = v0::shared::buffer_tail;
            }
        }

        ~Task() noexcept override
        {
            v0::shared::is_sampling = false;
        }

        core::Status operator()(api::Executor &executor) override
        {
            v0::shared::is_sampling = true;

            if (_external_task_id != _internal_task_id) [[unlikely]]
            {
                return STATUS_OK();
            }
            if (!_sensor || !_sensor->initialized()) [[unlikely]]
            {
                return replyWithStatus(STATUS(EINVAL, "Accelerometer sensor not initialized"),
                    std::chrono::steady_clock::now());
            }

            size_t available = _sensor->dataAvailable();
            if (available < 1)
            {
                return executor.submit(getptr(), getNextDataDelay(available));
            }

            {
                core::DataFrame<std::shared_ptr<core::Tensor>> data_frame;
                auto status = _sensor->readDataFrame(data_frame, available);
                if (!status) [[unlikely]]
                {
                    return replyWithStatus(status, std::chrono::steady_clock::now());
                }
                if (!data_frame.data) [[unlikely]]
                {
                    LOG(ERROR, "Data frame is null");
                    return replyWithStatus(STATUS(EFAULT, "Data frame is null"), std::chrono::steady_clock::now());
                }
                const auto &shape = data_frame.data->shape();
                available = shape[0];
                if (available < 1 || data_frame.data->dtype() != core::Tensor::Type::Float32) [[unlikely]]
                {
                    LOG(ERROR, "Data frame shape or type mismatch: shape=(%zu,%zu), dtype=%d", shape[0], shape[1],
                        static_cast<int>(data_frame.data->dtype()));
                    return replyWithStatus(STATUS(EINVAL, "Data frame shape or type mismatch"),
                        std::chrono::steady_clock::now());
                }
                const auto size = shape[0] * shape[1];
                const auto axes = shape[1];
                const auto buffer_axes = v0::shared::buffer.size();

                const auto gravity_alpha = v0::shared::gravity_correction_alpha.load();
                const auto gravity_beta = v0::shared::gravity_correction_beta.load();
                {
                    const std::lock_guard<std::mutex> lock(v0::shared::buffer_mutex);
                    const size_t head = v0::shared::buffer_head;
                    const size_t tail = v0::shared::buffer_tail;
                    const size_t next_head = head + shape[0];
                    if (static_cast<size_t>(next_head - tail) > v0::shared::buffer_size) [[unlikely]]
                    {
                        v0::shared::buffer_tail = next_head - v0::shared::buffer_size;
                    }
                    for (size_t i = 0, h = head; i < size; i += axes, ++h)
                    {
                        const auto data_i = &data_frame.data->data<float>()[i];
                        const auto index = h & v0::shared::buffer_size_mask;
                        for (size_t j = 0; j < buffer_axes; ++j)
                        {
                            v0::shared::buffer[j][index] = (data_i[j] * gravity_alpha) + gravity_beta;
                        }
                    }
                    v0::shared::buffer_head = next_head;
                }
            }

            const auto current = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(current - _last_report_time).count()
                >= _report_interval_ms) [[unlikely]]
            {
                _last_report_time = current;
                replyWithStatus(STATUS_OK(), current);
            }

            return executor.submit(getptr(), getNextDataDelay(_sensor->dataAvailable()));
        }

    private:
        inline core::Status replyWithStatus(core::Status status, std::chrono::steady_clock::time_point current) noexcept
        {
            auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
                std::bind(&v0::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v0::defaults::flush_callback, std::ref(_transport)));

            writer["type"] += v0::ResponseType::Stream;
            writer["name"] += "AccelerometerData";
            writer["tag"] << _tag;
            writer["code"] += status.code();
            writer["msg"] << status.message();
            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["id"] += v0::shared::buffer_head;
                data["ts"] += std::chrono::duration_cast<std::chrono::milliseconds>(current - _start_time).count();
                {
                    const std::lock_guard<std::mutex> lock(v0::shared::buffer_mutex);
                    const size_t available = static_cast<size_t>(v0::shared::buffer_head - _current_id);

                    auto data_array = data["data"].writer<core::ArrayWriter>();
                    for (size_t i = 0; i < v0::shared::buffer.size(); ++i)
                    {
                        auto axes_data = data_array.writer<core::ArrayWriter>();
                        const auto &buffer_i = v0::shared::buffer[i];
                        for (size_t j = 0; j < available; ++j)
                        {
                            axes_data << buffer_i[(_current_id + j) & v0::shared::buffer_size_mask];
                        }
                    }

                    _current_id += available;
                }
            }

            return status;
        }

        inline size_t getNextDataDelay(size_t current_available) const noexcept
        {
            if (current_available >= 1) [[unlikely]]
            {
                return 0;
            }
            return v0::shared::sample_interval_ms;
        }

        hal::Sensor *_sensor;
        const std::string _tag;
        const volatile size_t &_external_task_id;
        const size_t _internal_task_id;
        const size_t _report_interval_ms;
        const std::chrono::steady_clock::time_point _start_time;
        std::chrono::steady_clock::time_point _last_report_time;
        size_t _current_id;
    };

    std::shared_ptr<api::Task> operator()(api::Context &context, hal::Transport &transport, const core::ConfigMap &args,
        size_t id) override
    {
        std::string cmd_tag = "";
        if (auto it = args.find("@cmd_tag"); it != args.end())
        {
            auto tag = std::get_if<std::string>(&it->second);
            if (tag != nullptr)
                cmd_tag = *tag;
        }

        bool sample_enabled = false;
        bool invoke_enabled = false;

        auto status = STATUS_OK();
        auto device = hal::DeviceRegistry::getDevice();
        if (!device || !device->initialized()) [[unlikely]]
        {
            status = STATUS(ENODEV, "Device is not registered or initialized");
            goto Reply;
        }
        status = makeSensorReady();
        if (!status) [[unlikely]]
        {
            goto Reply;
        }

        if (auto it = args.find("@0"); it != args.end())
        {
            if (auto action = std::get_if<std::string>(&it->second); action != nullptr)
            {
                if (*action == "sample")
                {
                    sample_enabled = true;
                }
                else if (*action == "invoke")
                {
                    invoke_enabled = true;
                }
                else if (*action == "stop_invoke")
                {
                    _internal_invoke_task_id += 1;
                }
                else if (*action == "enable")
                {
                    if (!_invoke_on_start)
                    {
                        const bool invoke_on_start = true;
                        status = device->store(hal::Device::StorageType::Internal, invoke_on_start_path,
                            &invoke_on_start, sizeof(invoke_on_start));
                        if (status)
                        {
                            _invoke_on_start = invoke_on_start;
                        }
                    }
                }
                else if (*action == "disable")
                {
                    if (_invoke_on_start)
                    {
                        const bool invoke_on_start = false;
                        status = device->store(hal::Device::StorageType::Internal, invoke_on_start_path,
                            &invoke_on_start, sizeof(invoke_on_start));
                        if (status)
                        {
                            _invoke_on_start = invoke_on_start;
                        }
                    }
                }
                else
                {
                    status = STATUS(EINVAL, "Invalid action: " + *action);
                    goto Reply;
                }
            }
        }

    Reply: {
        auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
            std::bind(&v0::defaults::write_callback, std::ref(transport), std::placeholders::_1, std::placeholders::_2),
            std::bind(&v0::defaults::flush_callback, std::ref(transport)));

        writer["type"] += v0::ResponseType::Direct;
        writer["name"] += _name;

        writer["tag"] << cmd_tag;
        writer["code"] += status.code();
        writer["msg"] << status.message();
        {
            size_t boot_count = 0;
            std::chrono::system_clock::time_point last_boot_time = std::chrono::system_clock::now();
            size_t free_memory = 0;
            auto data = writer["data"].writer<core::ObjectWriter>();
            {
                auto device_data = data["device"].writer<core::ObjectWriter>();
                if (device) [[likely]]
                {
                    auto device_info = device->info();
                    device_data["name"] << device_info.name;
                    device_data["id"] << device_info.id;
                    device_data["model"] << device_info.model;
                    device_data["hwVer"] << device_info.version;
                    device_data["swVer"] << CORE_VERSION;
                    device_data["atVer"] << v0::defaults::at_version;
                    device_data["fwType"] << v0::defaults::firmware_type;

                    boot_count = device_info.boot_count;
                    last_boot_time = device_info.last_boot_time;
                    free_memory = device_info.free_memory;
                }
            }
            {
                auto status = data["status"].writer<core::ObjectWriter>();
                status["bootCount"] += boot_count;
                status["uptime"] += std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - last_boot_time)
                                        .count();
                status["freeMemory"] += free_memory;
                status["lastTrained"] += v0::shared::last_trained.load();
                status["isTraining"] += v0::shared::is_training.load();
                status["isSampling"] += v0::shared::is_sampling.load();
                status["isInvoking"] += v0::shared::is_invoking.load();
                status["invokeOnStart"] += _invoke_on_start;
            }
        }
    }

        if (sample_enabled)
        {
            _internal_sample_task_id += 1;
            return std::shared_ptr<api::Task>(
                new CmdStart::Task(context, transport, id, _sensor, cmd_tag, _internal_sample_task_id));
        }
        if (invoke_enabled)
        {
            _internal_invoke_task_id += 1;
            return std::shared_ptr<api::Task>(
                new TaskGEDAD::Invoke(context, transport, id, cmd_tag, _internal_invoke_task_id));
        }

        return nullptr;
    }

    static core::Status makeSensorReady() noexcept
    {
        auto status = STATUS_OK();
        auto sensor = hal::SensorRegistry::getSensor(v0::defaults::accelerometer_id);
        if (!sensor) [[unlikely]]
        {
            status = STATUS(ENODEV, "Accelerometer sensor not found");
            return status;
        }
        else if (!sensor->initialized()) [[unlikely]]
        {
            status = sensor->init();
        }
        if (_sensor != sensor) [[unlikely]]
        {
            _sensor = sensor;
        }
        return status;
    }

    static core::Status rcCallback(api::Context &context, api::Executor &executor, size_t id) noexcept
    {
        if (!_invoke_on_start)
        {
            return STATUS_OK();
        }

        auto *console = context.console();
        if (!console) [[unlikely]]
        {
            return STATUS(ENODEV, "Console transport is not initialized");
        }

        auto status = makeSensorReady();
        if (!status) [[unlikely]]
        {
            return status;
        }

        status = executor.submit(std::shared_ptr<api::Task>(
            new CmdStart::Task(context, *console, id, _sensor, "RC", _internal_sample_task_id)));
        if (!status) [[unlikely]]
        {
            return status;
        }

        status = executor.submit(
            std::shared_ptr<api::Task>(new TaskGEDAD::Invoke(context, *console, id, "RC", _internal_invoke_task_id)));
        return status;
    }

protected:
    static volatile size_t _internal_sample_task_id;
    static volatile size_t _internal_invoke_task_id;

private:
    static hal::Sensor *_sensor;
    static bool _invoke_on_start;
};

inline volatile size_t CmdStart::_internal_sample_task_id = 0;
inline volatile size_t CmdStart::_internal_invoke_task_id = 0;

inline hal::Sensor *CmdStart::_sensor = nullptr;
inline bool CmdStart::_invoke_on_start = false;

} // namespace v0

#endif // CMD_START_HPP
