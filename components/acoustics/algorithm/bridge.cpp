#include "core/logger.hpp"
#include "dag/sound_classification.hpp"
#include "module/module_dag.hpp"
#include "module/module_node.hpp"
#include "node/speech_commands_inference.hpp"

namespace bridge {

void __REGISTER_INTERNAL_MODULE_NODE_BUILDER__()
{
    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("SpeechCommands",
        algorithm::node::SpeechCommands::create, false);

    if (!status)
    {
        LOG(ERROR, "Failed to register SpeechCommands: %s", status.message().c_str());
    }
}

void __REGISTER_MODULE_DAG_BUILDER__()
{
    auto status = module::MDAGBuilderRegistry::registerDAGBuilder("SoundClassification",
        algorithm::dag::createSoundClassification);

    if (!status)
    {
        LOG(ERROR, "Failed to register SoundClassification: %s", status.message().c_str());
    }
}

} // namespace bridge
