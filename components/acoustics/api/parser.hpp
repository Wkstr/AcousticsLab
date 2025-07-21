#pragma once
#ifndef PARSER_HPP
#define PARSER_HPP

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"

#include "command.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace api {

class Parser final
{
public:
    static core::Status parseCommand(std::string_view command_str, const CommandRegistry::CommandMap &commands,
        Command *&command_ptr, core::ConfigMap &config_objects) noexcept
    {
        {
            const size_t start = command_str.find_first_not_of(" \n\r\t");
            const size_t end = command_str.find_last_not_of(" \n\r\t");
            if (start == std::string_view::npos || end == std::string_view::npos) [[unlikely]]
            {
                return STATUS(ENODATA, "Command string is empty or contains only invalid characters");
            }
            command_str = command_str.substr(start, end - start + 1);
        }

        if (!command_str.starts_with("AT+")) [[unlikely]]
        {
            return STATUS(EINVAL, "Command string must start with 'AT+'");
        }
        else if (command_str.size() < 4) [[unlikely]]
        {
            return STATUS(EINVAL, "Command string is too short");
        }

        std::string_view cmd_prefix;
        std::string_view cmd_suffix;

        const size_t eq_pos = command_str.find_first_of('=');
        if (eq_pos != std::string_view::npos)
        {
            cmd_prefix = command_str.substr(0, eq_pos);
            cmd_suffix = command_str.substr(eq_pos + 1, command_str.size());
        }
        else
        {
            cmd_prefix = command_str;
            cmd_suffix = "";
        }

        std::string_view cmd_body = cmd_prefix.substr(3);
        std::string_view cmd_tag;
        std::string_view cmd_name;

        const size_t tag_pos = cmd_body.rfind('@');
        if (tag_pos != std::string_view::npos)
        {
            cmd_tag = cmd_body.substr(0, tag_pos);
            cmd_name = cmd_body.substr(tag_pos + 1, cmd_body.size());
        }
        else
        {
            cmd_tag = "";
            cmd_name = cmd_body;
        }

        const auto it = commands.find(cmd_name);
        if (it == commands.end())
        {
            return STATUS(ESRCH, "Command not found: " + std::string(cmd_name));
        }
        command_ptr = it->second;
        config_objects["@cmd_tag"] = std::string(cmd_tag);
        config_objects["@cmd_name"] = std::string(cmd_name);

        const size_t n_args = std::min(it->second->args().size(), _args_index.size());
        if (n_args == 0)
        {
            return STATUS_OK();
        }

        size_t arg_index = 0;
        std::string arg = "";
        for (size_t i = 0; i < cmd_suffix.size() && arg_index < n_args; ++i)
        {
            const char c = cmd_suffix[i];
            if (c == ',')
            {
                LOG(DEBUG, "Parsed argument '%s' at index %zu", arg.c_str(), arg_index);
                config_objects[_args_index[arg_index++]] = arg;
                arg.clear();
            }
            else if (c == '\\') [[unlikely]]
            {
                if (++i >= cmd_suffix.size()) [[unlikely]]
                {
                    return STATUS(EINVAL, "Invalid escape sequence in command suffix");
                }
                arg += cmd_suffix[i];
            }
            else [[likely]]
            {
                arg += c;
            }
        }
        if (arg_index < n_args && !arg.empty())
        {
            LOG(DEBUG, "Parsed argument '%s' at index %zu", arg.c_str(), arg_index);
            config_objects[_args_index[arg_index++]] = arg;
        }

        return STATUS_OK();
    }

private:
    static constexpr const std::array<std::string_view, 20> _args_index = { "@0", "@1", "@2", "@3", "@4", "@5", "@6",
        "@7", "@8", "@9", "@10", "@11", "@12", "@13", "@14", "@15", "@16", "@17", "@18", "@19" };
};

} // namespace api

#endif // PARSER_HPP
