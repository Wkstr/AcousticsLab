#include "core/logger.hpp"
#include "module/module_node.hpp"
#include "node/speech_commands_pre_node.hpp"

namespace bridge {

void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__()
{
    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("SpeechCommandsPreprocess",
        porting::algorithms::node::SpeechCommandsPreprocess::create, false);

    if (!status)
    {
        LOG(ERROR, "Failed to register SpeechCommandsPreprocess: %s", status.message().c_str());
    }
}

} // namespace bridge
