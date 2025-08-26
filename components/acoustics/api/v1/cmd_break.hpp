#pragma once
#ifndef CMD_BREAK_HPP
#define CMD_BREAK_HPP

#include "common.hpp"

#include "api/command.hpp"
#include "api/executor.hpp"

#include "hal/device.hpp"

namespace v1 {

class CmdBreak final: public api::Command
{
public:
    CmdBreak() : api::Command("BREAK", "Break the all running tasks", core::ConfigObjectMap {}) { }

    ~CmdBreak() noexcept override = default;

    class Task final: public api::Task
    {
    public:
        Task(api::Context &context, hal::Transport &transport, size_t id, const std::string &tag) noexcept
            : api::Task(context, transport, id, v1::defaults::task_priority + 1), _tag(tag)
        {
        }

        ~Task() noexcept override { }

        core::Status operator()(api::Executor &executor) override
        {
            const auto removed = executor.clear();

            {
                auto writer = v1::defaults::serializer->writer(v1::defaults::wait_callback,
                    std::bind(&v1::defaults::write_callback, std::ref(_transport), std::placeholders::_1,
                        std::placeholders::_2),
                    std::bind(&v1::defaults::flush_callback, std::ref(_transport)));

                writer["type"] += v1::ResponseType::Direct;
                writer["name"] += "BREAK";
                writer["tag"] << _tag;
                writer["code"] += 0;
                writer["msg"] += "";
                {
                    auto data = writer["data"].writer<core::ObjectWriter>();
                    data["removed"] += removed;
                    data["transport"] << _transport.info().name;
                    data["transport_id"] += _transport.info().id;
                }
            }

            return STATUS_OK();
        }

    private:
        const std::string _tag;
    };

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
        return std::make_shared<Task>(context, transport, id, cmd_tag);
    }
};

} // namespace v1

#endif
