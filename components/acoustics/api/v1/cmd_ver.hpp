#pragma once
#ifndef CMD_VER_HPP
#define CMD_VER_HPP

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/version.hpp"

#include "hal/device.hpp"

namespace v1 {

class CmdVer final: public api::Command
{
public:
    CmdVer() : api::Command("VER?", "Get device version information", core::ConfigObjectMap {}) { }

    ~CmdVer() noexcept override = default;

    std::shared_ptr<api::Task> operator()(api::Context &context, hal::Transport &transport, const core::ConfigMap &args,
        size_t id) override
    {
        std::string cmd_tag;
        if (auto it = args.find("@cmd_tag"); it != args.end())
        {
            auto tag = std::get_if<std::string>(&it->second);
            if (tag != nullptr)
                cmd_tag = *tag;
        }

        core::Status status = STATUS_OK();
        auto device = hal::DeviceRegistry::getDevice();
        if (!device || !device->initialized())
        {
            status = STATUS(ENODEV, "Device is not registered or initialized");
        }

        {
            auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
                std::bind(&v1::defaults::write_callback, std::ref(transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v1::defaults::flush_callback, std::ref(transport)));

            writer["type"] += v1::ResponseType::Direct;
            writer["name"] += _name;
            writer["tag"] << cmd_tag;
            writer["code"] += status.code();
            writer["msg"] += status.message();
            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                if (device) [[likely]]
                {
                    auto device_info = device->info();
                    data["name"] << device_info.name;
                    data["id"] << device_info.id;
                    data["model"] << device_info.model;
                    data["hwVer"] << device_info.version;
                    data["swVer"] << CORE_VERSION;
                    data["atVer"] << v1::defaults::at_version;
                    data["fwType"] << v1::defaults::firmware_type;
                }
            }
        }

        return nullptr;
    }
};

} // namespace v1

#endif
