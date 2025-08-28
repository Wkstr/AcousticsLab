#pragma once
#ifndef CMD_START_HPP
#define CMD_START_HPP

#include "common.hpp"
#include "task_sc.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/version.hpp"

#include "hal/device.hpp"
#include "hal/sensor.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>

namespace v1 {

class CmdStart final: public api::Command
{
public:
    CmdStart()
        : Command("START", "Get device information and start the sample/invoke stream",
              core::ConfigObjectMap { CONFIG_OBJECT_DECL_STRING("action",
                  "Choose streaming action: sample, invoke, stop_invoke, enable or disable", "") })
    {
    }

    ~CmdStart() noexcept override = default;

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

        if (auto it = args.find("@0"); it != args.end())
        {
            if (auto action = std::get_if<std::string>(&it->second); action != nullptr)
            {
                if (*action == "sample")
                {
                    sample_enabled = true;
                    status = makeSensorReady();
                    if (!status) [[unlikely]]
                    {
                        goto Reply;
                    }
                }
                else if (*action == "invoke")
                {
                    invoke_enabled = true;
                }
                else if (*action == "stop_invoke")
                {
                    _internal_invoke_task_id += 1;
                }
                else
                {
                    status = STATUS(EINVAL, "Invalid action: " + *action);
                    goto Reply;
                }
            }
        }

    Reply: {
        auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
            std::bind(&v1::defaults::write_callback, std::ref(transport), std::placeholders::_1, std::placeholders::_2),
            std::bind(&v1::defaults::flush_callback, std::ref(transport)));

        writer["type"] += v1::ResponseType::Direct;
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
                    device_data["atVer"] << v1::defaults::at_version;
                    device_data["fwType"] << v1::defaults::firmware_type;

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
                status["isSampling"] += v1::shared::is_sampling.load();
                status["isInvoking"] += v1::shared::is_invoking.load();
            }
        }
    }

        if (sample_enabled)
        {
            _internal_sample_task_id += 1;
            return std::shared_ptr<api::Task>(
                new v1::TaskSC::Sample(context, transport, id, _sensor, cmd_tag, _internal_sample_task_id));
        }
        if (invoke_enabled)
        {
            // TODO: implement invoke task
        }

        return nullptr;
    }

    static core::Status makeSensorReady() noexcept
    {
        auto status = STATUS_OK();
        const auto &sensor_map = hal::SensorRegistry::getSensorMap();
        auto it = std::find_if(sensor_map.begin(), sensor_map.end(),
            [](const auto &pair) { return pair.second->info().type == hal::Sensor::Type::Microphone; });
        if (it == sensor_map.end() || !it->second) [[unlikely]]
        {
            status = STATUS(ENODEV, "Microphone sensor not found");
            return status;
        }
        auto sensor = it->second;
        if (!sensor->initialized()) [[unlikely]]
        {
            sensor->deinit();
            status = sensor->init();
        }
        if (_sensor != sensor) [[unlikely]]
        {
            _sensor = sensor;
        }
        return status;
    }

protected:
    static volatile size_t _internal_sample_task_id;
    static volatile size_t _internal_invoke_task_id;

private:
    static hal::Sensor *_sensor;
};

inline volatile size_t CmdStart::_internal_sample_task_id = 0;
inline volatile size_t CmdStart::_internal_invoke_task_id = 0;

inline hal::Sensor *CmdStart::_sensor = nullptr;

} // namespace v1

#endif
