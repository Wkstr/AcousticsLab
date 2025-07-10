#pragma once
#ifndef CMD_TRAINGEDAD_HPP
#define CMD_TRAINGEDAD_HPP

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

class CmdTrainGEDAD final: public api::Command
{
public:
    CmdTrainGEDAD()
        : Command("TRAINGEDAD", "Train the Gyroscope Euclidean Distance Anomaly Detection",
              core::ConfigObjectMap {
                  { "action", core::ConfigObject::createString("action",
                                  "Use 'train' to start train, 'save' to save the trained parameters to flash") },
              })
    {
    }

    ~CmdTrainGEDAD() noexcept override = default;

    static inline void preInitHook()
    {
        v0::TaskGEDAD::Storage::preInitHook();
    }

    inline std::shared_ptr<api::Task> operator()(api::Context &context, hal::Transport &transport,
        const core::ConfigMap &args, size_t id) override
    {
        std::string cmd_tag;
        if (auto it = args.find("@cmd_tag"); it != args.end())
        {
            auto tag = std::get_if<std::string>(&it->second);
            if (tag != nullptr)
                cmd_tag = *tag;
        }

        std::shared_ptr<api::Task> task = nullptr;

        auto status = STATUS_OK();
        if (auto it = args.find("@0"); it != args.end())
        {
            if (auto invoke = std::get_if<std::string>(&it->second); invoke != nullptr)
            {
                if (*invoke == "train")
                {
                    const size_t samples_required = v0::shared::window_size;
                    task = std::shared_ptr<api::Task>(
                        new TaskGEDAD::Train(context, transport, id, cmd_tag, samples_required));
                }
                else if (*invoke == "save")
                {
                    task = std::shared_ptr<api::Task>(new TaskGEDAD::Storage(context, transport, id, cmd_tag));
                }
                else
                {
                    status = STATUS(EINVAL, "Invalid action: " + *invoke);
                }
            }
        }
        else
        {
            status = STATUS(EINVAL, "Required argument is missing");
        }

        {
            auto writer = v0::defaults::serializer->writer(v0::defaults::wait_callback,
                std::bind(&v0::defaults::write_callback, std::ref(transport), std::placeholders::_1,
                    std::placeholders::_2),
                std::bind(&v0::defaults::flush_callback, std::ref(transport)));

            writer["type"] += v0::ResponseType::Direct;
            writer["name"] += _name;
            writer["tag"] << cmd_tag;
            writer["code"] += status.code();
            writer["msg"] += status.message();
            {
                auto data = writer["data"].writer<core::ObjectWriter>();
                data["lastTrained"] += v0::shared::last_trained.load();
                data["isTraining"] += v0::shared::is_training.load();
                data["isSampling"] += v0::shared::is_sampling.load();
            }
        }

        return task;
    }
};

} // namespace v0

#endif
