#include "core/logger.hpp"
#include "module/module_node.hpp"
#include "node/inference_node.hpp"

namespace bridge {

void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__()
{
    static const char *TAG = "NodeBridge";

    LOG(INFO, "Registering internal module node builders");

    // Register InferenceNode
    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("InferenceNode",
        algorithms::node::createInferenceNode, false);

    if (!status)
    {
        LOG(ERROR, "Failed to register InferenceNode: %s", status.message().c_str());
    }
    else
    {
        LOG(INFO, "Successfully registered InferenceNode");
    }

    // Log all registered nodes for debugging
    const auto &node_map = module::MNodeBuilderRegistry::getNodeBuilderMap();
    LOG(INFO, "Total registered internal nodes: %zu", node_map.size());
    for (const auto &[name, builder]: node_map)
    {
        LOG(DEBUG, "  - %s", name.data());
    }
}

} // namespace bridge
