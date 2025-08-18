#pragma once
#ifndef NODE_INPUT
#define NODE_INPUT

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "module/module_io.hpp"
#include "module/module_node.hpp"

#include <memory>

namespace module::node {

class MNInput final: public module::MNode
{
public:
    static inline constexpr const std::string_view node_name = "input";

    ~MNInput() override = default;

    static std::shared_ptr<MNode> create(const core::ConfigMap &configs, MIOS *inputs, MIOS *outputs, int priority)
    {
        if (!inputs || inputs->empty()) [[unlikely]]
        {
            LOG(ERROR, "Inputs cannot be null or empty for MNInput node");
            return {};
        }
        if (inputs != outputs) [[unlikely]]
        {
            LOG(ERROR, "Inputs and outputs must be the same for MNInput node");
            return {};
        }

        return std::make_shared<MNInput>(*inputs, *outputs, priority);
    }

private:
    template<typename IS, typename OS>
    constexpr explicit MNInput(IS &&inputs, OS &&outputs, int priority) noexcept
        : module::MNode(node_name, std::forward<IS>(inputs), std::forward<OS>(outputs), priority)
    {
    }

    inline core::Status forward(const MIOS &inputs, MIOS &outputs) noexcept override
    {
        return STATUS_OK();
    }
};

} // namespace module::node

#endif
