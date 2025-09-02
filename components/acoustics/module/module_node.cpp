#include "module_node.hpp"

#include "node/node_input.hpp"
#include "node/node_output.hpp"

namespace module {

core::Status MNodeBuilderRegistry::registerNodeBuilder(std::string_view name, NodeBuilder builder,
    bool replace) noexcept
{
    if (name.empty() || !builder) [[unlikely]]
    {
        LOG(ERROR, "Invalid node name or builder function");
        return STATUS(EINVAL, "Node name or builder function is invalid");
    }

    auto it = _nodes.find(name);
    if (it != _nodes.end()) [[unlikely]]
    {
        if (!replace)
        {
            return STATUS(EEXIST, "Node with this name already exists");
        }
        it->second = std::move(builder);
    }
    else
    {
        _nodes.emplace(name, std::move(builder));
    }

    return STATUS_OK();
}

MNodeBuilderRegistry::NodeBuilderMap MNodeBuilderRegistry::_nodes = {};

} // namespace module

namespace bridge {

void __REGISTER_PREDEFINED_MODULE_NODE_BUILDER__()
{
    module::MNodeBuilderRegistry::registerNodeBuilder(module::node::MNInput::node_name, &module::node::MNInput::create);
    module::MNodeBuilderRegistry::registerNodeBuilder(module::node::MNOutput::node_name,
        &module::node::MNOutput::create);
}

__attribute__((weak)) void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__() { }

__attribute__((weak)) void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__() { }

} // namespace bridge
