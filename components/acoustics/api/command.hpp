#pragma once
#ifndef COMMAND_HPP
#define COMMAND_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"

#include "hal/transport.hpp"

#include "context.hpp"
#include "executor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace api {

class Command;

class CommandRegistry final
{
public:
    using CommandMap = std::unordered_map<std::string_view, Command *>;

    CommandRegistry() = default;
    ~CommandRegistry() = default;

    static Command *getCommand(std::string_view name) noexcept
    {
        auto it = _commands.find(name);
        if (it != _commands.end()) [[likely]]
        {
            return it->second;
        }
        return nullptr;
    }

    static const CommandMap &getCommandMap() noexcept
    {
        return _commands;
    }

private:
    friend class Command;

    static core::Status registerCommand(Command *command) noexcept;

    static CommandMap _commands;
};

class Command
{
public:
    using RC = core::Status (*)(Context &context, Executor &executor, size_t id) noexcept;

    virtual ~Command() noexcept { }

    std::string_view name() const noexcept
    {
        return _name;
    }

    std::string_view description() const noexcept
    {
        return _description;
    }

    const core::ConfigObjectMap &args() const noexcept
    {
        const std::lock_guard<std::mutex> lock(_args_mutex);
        return _args;
    }

    virtual std::shared_ptr<Task> operator()(Context &context, hal::Transport &transport, const core::ConfigMap &args,
        size_t id)
        = 0;

    RC rc() noexcept
    {
        return _rc;
    }

protected:
    Command(std::string_view name, std::string_view description, core::ConfigObjectMap &&args, RC rc = nullptr) noexcept
        : _name(name), _description(description), _args(std::move(args)), _rc(rc), _args_mutex()
    {
        [[maybe_unused]] auto status = CommandRegistry::registerCommand(this);
        if (!status)
        {
            LOG(ERROR, "Failed to register command '%s': %s", _name.c_str(), status.message().c_str());
        }

        LOG(DEBUG, "Command '%s' registered successfully", _name.c_str());
    }

    const std::string _name;
    const std::string _description;
    core::ConfigObjectMap _args;
    RC _rc;
    mutable std::mutex _args_mutex;
};

} // namespace api

#endif // AT_COMMAND_HPP
