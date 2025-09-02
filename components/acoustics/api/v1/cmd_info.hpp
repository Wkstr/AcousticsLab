#pragma once
#ifndef CMD_INFO
#define CMD_INFO

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "core/crc/crc16.hpp"

#include "hal/device.hpp"

#include <mutex>

namespace v1 {

class CmdInfo final: public api::Command
{
public:
    static inline constexpr const char *info_path = CONTEXT_PREFIX CONTEXT_VERSION "/info";

    CmdInfo()
        : api::Command("INFO", "Get/set device information",
              core::ConfigObjectMap { CONFIG_OBJECT_DECL_STRING("info", "The info string to store", "") })
    {
    }

    ~CmdInfo() noexcept override = default;

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

        bool is_load = false;
        std::string info_str;
        size_t info_str_len = 0;
        auto crc_gen = core::CRC16XMODEM();

        auto status = STATUS_OK();
        auto device = hal::DeviceRegistry::getDevice();
        if (!device || !device->initialized())
        {
            status = STATUS(ENODEV, "Device is not registered or initialized");
            goto Reply;
        }

        if (auto it = args.find("@0"); it != args.end())
        {
            auto str = std::get_if<std::string>(&it->second);
            if (str != nullptr)
            {
                info_str_len = str->length();
                crc_gen.update(reinterpret_cast<const uint8_t *>(str->c_str()), info_str_len);

                std::lock_guard<std::mutex> lock(_info_rw_mutex);
                status = device->store(hal::Device::StorageType::Internal, info_path, str->c_str(), info_str_len);
            }
        }
        else
        {
            is_load = true;
            {
                std::lock_guard<std::mutex> lock(_info_rw_mutex);

                if (!device->load(hal::Device::StorageType::Internal, info_path, nullptr, info_str_len))
                    goto Reply;
                info_str.resize(info_str_len);
                status = device->load(hal::Device::StorageType::Internal, info_path, info_str.data(), info_str_len);
                if (!status)
                    goto Reply;
            }
            crc_gen.update(reinterpret_cast<const uint8_t *>(info_str.data()), info_str_len);
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
            auto data = writer["data"].writer<core::ObjectWriter>();
            if (is_load)
                data["info"] << info_str;
            data["len"] += info_str_len;
            data["crc"] += crc_gen.finalize();
        }
    }
        return nullptr;
    }

private:
    std::mutex _info_rw_mutex;
};

} // namespace v1

#endif
