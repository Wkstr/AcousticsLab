#include "core/logger.hpp"
#include "module/module_builder.hpp"
#include "module/module_node.hpp"
#if defined(PORTING_LIB_DL_FFT_ENABLE) && PORTING_LIB_DL_FFT_ENABLE
#include "node/speech_commands_preprocess.hpp"
#endif

namespace bridge {

void __REGISTER_EXTERNAL_MODULE_NODE_BUILDER__()
{
#if defined(PORTING_LIB_DL_FFT_ENABLE) && PORTING_LIB_DL_FFT_ENABLE
    auto status = module::MNodeBuilderRegistry::registerNodeBuilder("SpeechCommandsPreprocess",
        porting::algorithm::node::SpeechCommandsPreprocess::create, true);

    if (!status)
    {
        LOG(ERROR, "Failed to register SpeechCommandsPreprocess: %s", status.message().c_str());
    }
#endif // PORTING_LIB_DL_FFT_ENABLE
}

} // namespace bridge
