#pragma once
#ifndef NODE_OUTPUT
#define NODE_OUTPUT

#include "core/config_object.hpp"
#include "core/logger.hpp"
#include "core/status.hpp"
#include "module/module_io.hpp"
#include "module/module_node.hpp"

#include <memory>

namespace module::node {

class MNOutput final: public module::MNode
{
public:
    static inline constexpr const std::string_view node_name = "output";

    template<typename IS, typename OS>
    constexpr explicit MNOutput(IS &&inputs, OS &&outputs, int priority) noexcept
        : module::MNode(std::string(node_name), std::forward<IS>(inputs), std::forward<OS>(outputs), priority)
    {
    }

    ~MNOutput() override = default;

    static std::shared_ptr<MNode> create(const core::ConfigMap &configs, const MIOS *inputs, const MIOS *outputs,
        int priority)
    {
        if (!outputs || outputs->empty()) [[unlikely]]
        {
            LOG(ERROR, "Outputs cannot be null or empty for MNOutput node");
            return {};
        }
        if (inputs != outputs) [[unlikely]]
        {
            LOG(ERROR, "Inputs and outputs must be the same for MNOutput node");
            return {};
        }

        return std::make_shared<MNOutput>(*inputs, *outputs, priority);
    }

private:
    inline core::Status forward(const MIOS &inputs, MIOS &outputs) noexcept override
    {
        return STATUS_OK();
    }
};

} // namespace module::node

#endif
