#include "command.hpp"
#include "core/logger.hpp"

namespace api {

core::Status CommandRegistry::registerCommand(Command *command) noexcept
{
    if (!command)
    {
        return STATUS(EINVAL, "Command is null");
    }
    if (_commands.contains(command->name()))
    {
        return STATUS(EEXIST, "Command with name already exists");
    }

    _commands.emplace(command->name(), command);

    return STATUS_OK();
}

CommandRegistry::CommandMap CommandRegistry::_commands = {};

} // namespace api
