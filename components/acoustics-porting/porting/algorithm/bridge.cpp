#include "core/logger.hpp"
#include "module/module_node.hpp"
#include "node/speech_commands_preprocess.hpp"

namespace bridge {

void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__()
{
    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("SpeechCommandsPreprocess",
        porting::algorithm::node::SpeechCommandsPreprocess::create, true);

    if (!status)
    {
        LOG(ERROR, "Failed to register SpeechCommandsPreprocess: %s", status.message().c_str());
    }
}

} // namespace bridge
