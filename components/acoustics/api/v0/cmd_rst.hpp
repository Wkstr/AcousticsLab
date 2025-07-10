#pragma once
#ifndef CMD_RST_HPP
#define CMD_RST_HPP

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "hal/device.hpp"

namespace v0 {

class CmdRst final: public api::Command
{
public:
    CmdRst() : api::Command("RST", "Reset the device", core::ConfigObjectMap {}) { }

    ~CmdRst() noexcept override = default;

    std::shared_ptr<api::Task> operator()(api::Context &context, hal::Transport &transport, const core::ConfigMap &args,
        size_t id) override
    {
        auto device = hal::DeviceRegistry::getDevice();
        if (!device) [[unlikely]]
        {
            return nullptr;
        }
        device->reset();

        return nullptr;
    }
};

} // namespace v0

#endif
